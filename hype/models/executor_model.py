"""
Executor Model for HyPE system.

This module implements the Executor Model which uses LoRA (Low-Rank Adaptation)
for test-time fine-tuning to correct actions that don't align with principles.
"""

import os
import sys
import torch
from peft import LoraConfig, get_peft_model, PeftModel
from typing import Optional, List
import logging

from ..core.config import ModelConfig
from ..core.data_models import Principle, Action
from .base_model import BaseModelLoader
from ..utils.error_handlers import ErrorHandler, ModelGenerationError

# 🔥 禁用 bitsandbytes 量化
os.environ['BITSANDBYTES_NOWELCOME'] = '1'
os.environ['DISABLE_BITSANDBYTES'] = '1'

# 🔥 阻止 bitsandbytes 被导入
import importlib.util
if importlib.util.find_spec('bitsandbytes') is not None:
    sys.modules['bitsandbytes'] = None  # type: ignore

logger = logging.getLogger(__name__)


class ExecutorModel:
    """
    Executor Model with LoRA adapter for test-time fine-tuning.
    
    The Executor Model:
    - Uses LoRA adapter on attention layers for efficient adaptation
    - Performs test-time fine-tuning to correct misaligned actions
    - Generates actions that better align with retrieved principles
    - Maintains base model weights while updating only LoRA parameters
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize Executor Model.
        
        Args:
            config: Model configuration
        """
        self.config = config
        self.base_loader = BaseModelLoader(config)
        self.device = self.base_loader.device
        self.peft_model = None
        self.lora_config = None
        self.error_handler = ErrorHandler()
        
        logger.info("Initialized ExecutorModel")
    
    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        Load the base model and add LoRA adapter.
        
        Args:
            model_path: Optional path to local model
        """
        # Load base model
        self.base_loader.load_model(model_path)
        
        # Configure LoRA
        self.lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Add LoRA adapter to base model
        base_model = self.base_loader.get_model()
        self.peft_model = get_peft_model(base_model, self.lora_config)
        
        # Log trainable parameters
        trainable_params = sum(p.numel() for p in self.peft_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.peft_model.parameters())
        
        logger.info(f"Added LoRA adapter: {trainable_params:,} trainable params "
                   f"({100 * trainable_params / total_params:.2f}% of total)")
    
    def generate_action(
        self,
        task: str,
        state: str,
        hypothesis: str,
        principles: Optional[List[Principle]] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Generate an action using the Executor Model.
        
        Args:
            task: Task description
            state: Current state description
            hypothesis: Hypothesis to instantiate
            principles: Optional list of relevant principles
            temperature: Sampling temperature
            
        Returns:
            Generated action description
        """
        if self.peft_model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Define generation function
        def _generate():
            # Format principles as text
            principle_texts = [p.text for p in principles] if principles else None
            
            # Format prompt
            prompt = self.base_loader.format_prompt(
                task=task,
                state=state,
                hypothesis=hypothesis,
                principles=principle_texts,
                prompt_type="action"
            )
            
            # Tokenize
            tokenizer = self.base_loader.get_tokenizer()
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_length
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.peft_model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=temperature or self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            # Decode
            generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
            action_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            return action_text.strip()
        
        # Define fallback function
        def _fallback(context):
            logger.warning("Using fallback: generating generic action")
            return f"Execute action based on hypothesis: {hypothesis}"
        
        # Try generation with error handling
        try:
            return _generate()
        except Exception as e:
            logger.warning(f"Action generation failed: {e}")
            return self.error_handler.handle_generation_error(
                error=e,
                generation_fn=lambda **kwargs: _generate(),
                context={
                    'task': task,
                    'state': state,
                    'hypothesis': hypothesis,
                    'principles': principles,
                    'temperature': temperature or self.config.temperature
                },
                fallback_fn=_fallback
            )
    
    def test_time_correction(
        self,
        misaligned_action: str,
        task: str,
        state: str,
        hypothesis: str,
        principles: List[Principle],
        correction_steps: Optional[int] = None,
        learning_rate: Optional[float] = None
    ) -> str:
        """
        Apply test-time LoRA fine-tuning to correct a misaligned action.
        
        Args:
            misaligned_action: Action with low alignment score
            task: Task description
            state: Current state description
            hypothesis: Hypothesis that led to the action
            principles: Principles to align with
            correction_steps: Number of gradient steps (uses config default if None)
            learning_rate: Learning rate for correction (uses config default if None)
            
        Returns:
            Corrected action description
        """
        if self.peft_model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        correction_steps = correction_steps or self.config.correction_steps if hasattr(self.config, 'correction_steps') else 5
        learning_rate = learning_rate or self.config.correction_learning_rate if hasattr(self.config, 'correction_learning_rate') else 1e-4
        
        # Format correction prompt
        principle_texts = [p.text for p in principles]
        correction_prompt = self._format_correction_prompt(
            task=task,
            state=state,
            hypothesis=hypothesis,
            misaligned_action=misaligned_action,
            principles=principle_texts
        )
        
        # Set model to training mode
        self.peft_model.train()
        
        # Only optimize LoRA parameters
        optimizer = torch.optim.AdamW(
            [p for p in self.peft_model.parameters() if p.requires_grad],
            lr=learning_rate
        )
        
        # Tokenize correction prompt
        tokenizer = self.base_loader.get_tokenizer()
        inputs = tokenizer(
            correction_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length
        ).to(self.device)
        
        # Apply gradient updates
        for step in range(correction_steps):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.peft_model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            logger.debug(f"Correction step {step + 1}/{correction_steps}, loss: {loss.item():.4f}")
        
        # Set back to eval mode
        self.peft_model.eval()
        
        # Generate corrected action
        corrected_action = self.generate_action(
            task=task,
            state=state,
            hypothesis=hypothesis,
            principles=principles
        )
        
        return corrected_action
    
    def _format_correction_prompt(
        self,
        task: str,
        state: str,
        hypothesis: str,
        misaligned_action: str,
        principles: List[str]
    ) -> str:
        """
        Format prompt for correction fine-tuning.
        
        Args:
            task: Task description
            state: Current state
            hypothesis: Hypothesis
            misaligned_action: Action that needs correction
            principles: Principles to align with
            
        Returns:
            Formatted correction prompt
        """
        prompt = f"Task: {task}\n"
        prompt += f"Current State: {state}\n"
        prompt += f"Hypothesis: {hypothesis}\n\n"
        
        prompt += "The following action does not align well with the principles:\n"
        prompt += f"Misaligned Action: {misaligned_action}\n\n"
        
        prompt += "Relevant Principles:\n"
        for i, principle in enumerate(principles, 1):
            prompt += f"{i}. {principle}\n"
        
        prompt += "\nGenerate a corrected action that better follows the principles.\n"
        prompt += "Corrected Action:"
        
        return prompt
    
    def reset_lora_weights(self) -> None:
        """
        Reset LoRA adapter weights to initial state.
        
        Useful for starting fresh correction attempts.
        """
        if self.peft_model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Reinitialize LoRA adapter
        base_model = self.base_loader.get_model()
        self.peft_model = get_peft_model(base_model, self.lora_config)
        
        logger.info("Reset LoRA adapter weights")
    
    def get_lora_parameters(self):
        """
        Get only the LoRA adapter parameters.
        
        Returns:
            Iterator over LoRA parameters
        """
        if self.peft_model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        return [p for p in self.peft_model.parameters() if p.requires_grad]
    
    def train_mode(self) -> None:
        """Set model to training mode."""
        if self.peft_model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        self.peft_model.train()
    
    def eval_mode(self) -> None:
        """Set model to evaluation mode."""
        if self.peft_model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        self.peft_model.eval()
    
    def save_model(self, save_path: str) -> None:
        """
        Save the Executor Model (base model + LoRA adapter).
        
        Args:
            save_path: Directory to save model
        """
        if self.peft_model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Save PEFT model (includes LoRA adapter)
        self.peft_model.save_pretrained(save_path)
        
        # Save tokenizer
        tokenizer = self.base_loader.get_tokenizer()
        tokenizer.save_pretrained(save_path)
        
        logger.info(f"Saved Executor Model to {save_path}")
    
    def load_from_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load Executor Model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint directory
        """
        # Load base model first
        self.base_loader.load_model()
        
        # Load PEFT model with LoRA adapter
        base_model = self.base_loader.get_model()
        self.peft_model = PeftModel.from_pretrained(base_model, checkpoint_path)
        
        logger.info(f"Loaded Executor Model from {checkpoint_path}")
    
    def get_base_model(self):
        """
        Get the underlying base model.
        
        Returns:
            Base language model
        """
        return self.base_loader.get_model()
    
    def get_peft_model(self):
        """
        Get the PEFT model with LoRA adapter.
        
        Returns:
            PEFT model
        """
        if self.peft_model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self.peft_model
