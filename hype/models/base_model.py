"""
Base model loader for HyPE system.

This module provides utilities for loading and managing the base language model
(Llama-3-8B-Instruct or Qwen-2.5-7B-Instruct) with device management and
prompt formatting.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, List, Dict, Any
import logging
from pathlib import Path

from ..core.config import ModelConfig


logger = logging.getLogger(__name__)


class BaseModelLoader:
    """
    Loader and manager for base language models.
    
    Handles:
    - Model loading from HuggingFace
    - Device management (CPU/GPU)
    - Prompt formatting utilities
    - Generation with configurable parameters
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize base model loader.
        
        Args:
            config: Model configuration
        """
        self.config = config
        self.device = self._setup_device()
        self.model = None
        self.tokenizer = None
        
        logger.info(f"Initialized BaseModelLoader with device: {self.device}")
    
    def _setup_device(self) -> torch.device:
        """
        Setup computation device based on configuration.
        
        Returns:
            torch.device for computation
        """
        if self.config.device == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        elif self.config.device == "mps" and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using MPS device (Apple Silicon)")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU device")
        
        return device
    
    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        Load the base language model and tokenizer.
        
        Args:
            model_path: Optional path to local model. If None, uses config.base_model_name
        """
        model_name = model_path or self.config.base_model_name
        
        logger.info(f"Loading model: {model_name}")
        print(f"   📦 Loading model: {model_name}", flush=True)
        
        try:
            # Load tokenizer
            print(f"   ⏳ Loading tokenizer...", flush=True)
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                padding_side="left"
            )
            print(f"   ✅ Tokenizer loaded", flush=True)
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with float32 for training stability
            print(f"   ⏳ Loading model weights (this may take 1-2 minutes)...", flush=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float32,  # Use float32 for better training stability
                device_map="auto" if self.device.type == "cuda" else None,
            )
            print(f"   ✅ Model weights loaded", flush=True)
            
            # Move to device if not using device_map
            if self.device.type != "cuda":
                print(f"   ⏳ Moving model to {self.device}...", flush=True)
                self.model = self.model.to(self.device)
                print(f"   ✅ Model moved to device", flush=True)
            
            self.model.eval()
            
            num_params = sum(p.numel() for p in self.model.parameters()) / 1e9
            logger.info(f"Successfully loaded model: {model_name}")
            logger.info(f"Model parameters: {num_params:.2f}B")
            print(f"   ✅ Model ready ({num_params:.2f}B parameters)", flush=True)
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            print(f"   ❌ Failed to load model: {e}", flush=True)
            raise
    
    def format_prompt(
        self,
        task: str,
        state: Optional[str] = None,
        principles: Optional[List[str]] = None,
        hypothesis: Optional[str] = None,
        prompt_type: str = "general"
    ) -> str:
        """
        Format a prompt for the model based on context.
        
        Args:
            task: Task description
            state: Current state description
            principles: List of relevant principles
            hypothesis: Optional hypothesis for action instantiation
            prompt_type: Type of prompt ("hypothesis", "action", "value", "general")
            
        Returns:
            Formatted prompt string
        """
        if prompt_type == "hypothesis":
            return self._format_hypothesis_prompt(task, state, principles)
        elif prompt_type == "action":
            return self._format_action_prompt(task, state, hypothesis, principles)
        elif prompt_type == "value":
            return self._format_value_prompt(task, state, hypothesis, principles)
        else:
            return self._format_general_prompt(task, state, principles)
    
    def _format_hypothesis_prompt(
        self,
        task: str,
        state: Optional[str],
        principles: Optional[List[str]]
    ) -> str:
        """Format prompt for hypothesis generation."""
        prompt = f"Task: {task}\n"
        
        if state:
            prompt += f"Current State: {state}\n"
        
        if principles:
            prompt += "Relevant Principles:\n"
            for i, principle in enumerate(principles, 1):
                prompt += f"{i}. {principle}\n"
        
        prompt += "\nGenerate a high-level hypothesis for the next strategic step.\nHypothesis:"
        
        return prompt
    
    def _format_action_prompt(
        self,
        task: str,
        state: Optional[str],
        hypothesis: Optional[str],
        principles: Optional[List[str]]
    ) -> str:
        """Format prompt for action instantiation."""
        prompt = f"Task: {task}\n"
        
        if state:
            prompt += f"Current State: {state}\n"
        
        if hypothesis:
            prompt += f"Hypothesis: {hypothesis}\n"
        
        if principles:
            prompt += "Relevant Principles:\n"
            for i, principle in enumerate(principles, 1):
                prompt += f"{i}. {principle}\n"
        
        prompt += "\nInstantiate the hypothesis into a concrete action.\nAction:"
        
        return prompt
    
    def _format_value_prompt(
        self,
        task: str,
        state: Optional[str],
        hypothesis: Optional[str],
        principles: Optional[List[str]]
    ) -> str:
        """Format prompt for value estimation."""
        prompt = f"Task: {task}\n"
        
        if state:
            prompt += f"Current State: {state}\n"
        
        if hypothesis:
            prompt += f"Hypothesis: {hypothesis}\n"
        
        if principles:
            prompt += "Relevant Principles:\n"
            for i, principle in enumerate(principles, 1):
                prompt += f"{i}. {principle}\n"
        
        prompt += "\nEstimated Value:"
        
        return prompt
    
    def _format_general_prompt(
        self,
        task: str,
        state: Optional[str],
        principles: Optional[List[str]]
    ) -> str:
        """Format general prompt."""
        prompt = f"Task: {task}\n"
        
        if state:
            prompt += f"Current State: {state}\n"
        
        if principles:
            prompt += "Relevant Principles:\n"
            for i, principle in enumerate(principles, 1):
                prompt += f"{i}. {principle}\n"
        
        return prompt
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        num_return_sequences: int = 1,
    ) -> List[str]:
        """
        Generate text from the model.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
            num_return_sequences: Number of sequences to generate
            
        Returns:
            List of generated text strings
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Use config defaults if not specified
        max_new_tokens = max_new_tokens or self.config.max_length
        temperature = temperature or self.config.temperature
        top_p = top_p or self.config.top_p
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode outputs
        generated_texts = []
        for output in outputs:
            # Remove input tokens from output
            generated_ids = output[inputs['input_ids'].shape[1]:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            generated_texts.append(text.strip())
        
        return generated_texts
    
    def get_model(self) -> AutoModelForCausalLM:
        """
        Get the loaded model.
        
        Returns:
            The loaded model
            
        Raises:
            RuntimeError: If model not loaded
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self.model
    
    def get_tokenizer(self) -> AutoTokenizer:
        """
        Get the loaded tokenizer.
        
        Returns:
            The loaded tokenizer
            
        Raises:
            RuntimeError: If tokenizer not loaded
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load_model() first.")
        return self.tokenizer
    
    def save_model(self, save_path: str) -> None:
        """
        Save the model and tokenizer to disk.
        
        Args:
            save_path: Directory to save model
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving model to {save_path}")
        
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        logger.info("Model saved successfully")
    
    def load_from_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model from a saved checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint directory
        """
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        self.load_model(checkpoint_path)
