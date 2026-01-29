"""
Value Model for HyPE system.

This module implements the Value Model which consists of the base language model
with an additional scalar output head for value prediction.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from typing import Optional, List
import logging

from ..core.config import ModelConfig
from ..core.data_models import Principle
from .base_model import BaseModelLoader
from ..utils.error_handlers import ErrorHandler, ModelGenerationError


logger = logging.getLogger(__name__)


class ValueHead(nn.Module):
    """
    Scalar value head for value prediction.
    
    Takes the hidden states from the base model and produces a single
    scalar value estimate.
    """
    
    def __init__(self, hidden_dim: int):
        """
        Initialize value head.
        
        Args:
            hidden_dim: Dimension of hidden states from base model
        """
        super().__init__()
        self.value_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through value head.
        
        Args:
            hidden_states: Hidden states from base model [batch_size, seq_len, hidden_dim]
            
        Returns:
            Value predictions [batch_size, 1]
        """
        # Take the last token's hidden state
        last_hidden = hidden_states[:, -1, :]  # [batch_size, hidden_dim]
        
        # Project to scalar value
        value = self.value_head(last_hidden)  # [batch_size, 1]
        
        return value


class ValueModel:
    """
    Value Model for hypothesis evaluation.
    
    Consists of:
    - Base language model for encoding
    - Scalar value head for value prediction
    
    Used by H-MCTS to evaluate hypotheses during planning.
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize Value Model.
        
        Args:
            config: Model configuration
        """
        self.config = config
        self.base_loader = BaseModelLoader(config)
        self.value_head = None
        self.device = self.base_loader.device
        self.error_handler = ErrorHandler()
        
        logger.info("Initialized ValueModel")
    
    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        Load the base model and initialize value head.
        
        Args:
            model_path: Optional path to local model
        """
        # Load base model
        self.base_loader.load_model(model_path)
        
        # Update device after model loading (in case device_map changed it)
        self.device = self.base_loader.device
        
        # Get hidden dimension from base model
        base_model = self.base_loader.get_model()
        hidden_dim = base_model.config.hidden_size
        
        # Get actual device from model (in case of device_map="auto")
        if hasattr(base_model, 'device'):
            self.device = base_model.device
        elif hasattr(base_model, 'hf_device_map'):
            # Model is distributed, use first device
            first_device = list(base_model.hf_device_map.values())[0]
            self.device = torch.device(first_device)
        
        # Enable gradient checkpointing to save memory during training
        if hasattr(base_model, 'gradient_checkpointing_enable'):
            base_model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing for memory efficiency")
        
        # Get model dtype (match value head to base model dtype)
        model_dtype = base_model.dtype if hasattr(base_model, 'dtype') else torch.float32
        
        # Initialize value head on the correct device with matching dtype
        self.value_head = ValueHead(hidden_dim).to(self.device).to(model_dtype)
        
        logger.info(f"Initialized value head with hidden_dim={hidden_dim}, dtype={model_dtype}, device={self.device}")
    
    def predict_value(
        self,
        task: str,
        state: str,
        hypothesis: str,
        principles: Optional[List[Principle]] = None
    ) -> float:
        """
        Predict the expected value of a hypothesis.
        
        Args:
            task: Task description
            state: Current state description
            hypothesis: Hypothesis to evaluate
            principles: Optional list of relevant principles
            
        Returns:
            Predicted value (scalar)
        """
        if self.value_head is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Define prediction function
        def _predict():
            # Format principles as text
            principle_texts = [p.text for p in principles] if principles else None
            
            # Format prompt for value prediction
            prompt = self.base_loader.format_prompt(
                task=task,
                state=state,
                principles=principle_texts,
                hypothesis=hypothesis,
                prompt_type="value"
            )
            
            # Get hidden states from base model
            hidden_states = self._get_hidden_states(prompt)
            
            # Predict value
            with torch.no_grad():
                value = self.value_head(hidden_states)
            
            return value.item()
        
        # Define fallback function
        def _fallback(context):
            logger.warning("Using fallback: returning neutral value")
            return 0.0
        
        # Try prediction with error handling
        try:
            return _predict()
        except Exception as e:
            logger.warning(f"Value prediction failed: {e}")
            return self.error_handler.handle_generation_error(
                error=e,
                generation_fn=lambda **kwargs: _predict(),
                context={
                    'task': task,
                    'state': state,
                    'hypothesis': hypothesis,
                    'principles': principles
                },
                fallback_fn=_fallback
            )
    
    def _get_hidden_states(self, prompt: str) -> torch.Tensor:
        """
        Get hidden states from base model for a prompt.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Hidden states tensor [1, seq_len, hidden_dim]
        """
        tokenizer = self.base_loader.get_tokenizer()
        base_model = self.base_loader.get_model()
        
        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length
        ).to(self.device)
        
        # Get hidden states
        with torch.no_grad():
            outputs = base_model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]  # Last layer
        
        return hidden_states
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the complete Value Model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Value predictions [batch_size, 1]
        """
        if self.value_head is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        base_model = self.base_loader.get_model()
        
        # Get hidden states from base model
        # Use output_hidden_states=True to get last layer hidden states
        outputs = base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False  # Disable KV cache to save memory
        )
        hidden_states = outputs.hidden_states[-1]
        
        # Clear intermediate outputs to free memory
        del outputs
        
        # Predict value
        value = self.value_head(hidden_states)
        
        # Clear hidden states
        del hidden_states
        
        return value
    
    def train_mode(self) -> None:
        """Set model to training mode."""
        if self.value_head is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        self.base_loader.get_model().train()
        self.value_head.train()
    
    def eval_mode(self) -> None:
        """Set model to evaluation mode."""
        if self.value_head is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        self.base_loader.get_model().eval()
        self.value_head.eval()
    
    def get_parameters(self):
        """
        Get all trainable parameters.
        
        Returns:
            Iterator over parameters
        """
        if self.value_head is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Return both base model and value head parameters
        base_params = self.base_loader.get_model().parameters()
        value_params = self.value_head.parameters()
        
        return list(base_params) + list(value_params)
    
    def save_model(self, save_path: str) -> None:
        """
        Save the Value Model (base model + value head).
        
        Args:
            save_path: Directory to save model
        """
        if self.value_head is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Save base model
        self.base_loader.save_model(save_path)
        
        # Save value head
        value_head_path = f"{save_path}/value_head.pt"
        torch.save(self.value_head.state_dict(), value_head_path)
        
        logger.info(f"Saved Value Model to {save_path}")
    
    def load_from_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load Value Model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint directory
        """
        # Load base model
        self.base_loader.load_from_checkpoint(checkpoint_path)
        
        # Get hidden dimension
        base_model = self.base_loader.get_model()
        hidden_dim = base_model.config.hidden_size
        
        # Initialize value head
        self.value_head = ValueHead(hidden_dim).to(self.device)
        
        # Load value head weights
        value_head_path = f"{checkpoint_path}/value_head.pt"
        self.value_head.load_state_dict(torch.load(value_head_path, map_location=self.device))
        
        logger.info(f"Loaded Value Model from {checkpoint_path}")
    
    def get_base_model(self) -> AutoModelForCausalLM:
        """
        Get the base language model.
        
        Returns:
            Base model
        """
        return self.base_loader.get_model()
    
    def get_value_head(self) -> ValueHead:
        """
        Get the value head.
        
        Returns:
            Value head module
        """
        if self.value_head is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self.value_head
