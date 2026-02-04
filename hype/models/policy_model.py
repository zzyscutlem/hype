"""
Policy Model for HyPE system.

This module implements the Policy Model which generates hypotheses and
instantiates actions. It is initialized from the base model and fine-tuned
via Direct Preference Optimization (DPO).
"""

import torch
from typing import Optional, List, Dict, Any
import logging

from ..core.config import ModelConfig
from ..core.data_models import Principle, Action
from .base_model import BaseModelLoader
from ..utils.error_handlers import ErrorHandler, ModelGenerationError


logger = logging.getLogger(__name__)


class PolicyModel:
    """
    Policy Model for hypothesis generation and action instantiation.
    
    The Policy Model:
    - Generates high-level hypotheses during H-MCTS planning
    - Instantiates hypotheses into concrete actions
    - Is conditioned on task, state, and retrieved principles
    - Is optimized via DPO using principle-guided preferences
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize Policy Model.
        
        Args:
            config: Model configuration
        """
        self.config = config
        self.base_loader = BaseModelLoader(config)
        self.device = self.base_loader.device
        self.error_handler = ErrorHandler()
        
        logger.info("Initialized PolicyModel")
    
    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        Load the base model for policy.
        
        Args:
            model_path: Optional path to local model or checkpoint
        """
        self.base_loader.load_model(model_path)
        logger.info("Loaded Policy Model from base model")
    
    def generate_hypothesis(
        self,
        task: str,
        state: str,
        principles: Optional[List[Principle]] = None,
        num_hypotheses: int = 1,
        temperature: Optional[float] = None
    ) -> List[str]:
        """
        Generate high-level hypotheses for planning.
        
        Args:
            task: Task description
            state: Current state description
            principles: Optional list of relevant principles
            num_hypotheses: Number of hypotheses to generate
            temperature: Sampling temperature (uses config default if None)
            
        Returns:
            List of generated hypotheses
        """
        # Define generation function
        def _generate():
            # Format principles as text
            principle_texts = [p.text for p in principles] if principles else None
            
            # Format prompt for hypothesis generation
            prompt = self.base_loader.format_prompt(
                task=task,
                state=state,
                principles=principle_texts,
                prompt_type="hypothesis"
            )
            
            # Generate hypotheses
            hypotheses = self.base_loader.generate(
                prompt=prompt,
                max_new_tokens=200,  # Hypotheses should be concise
                temperature=temperature or self.config.temperature,
                num_return_sequences=num_hypotheses,
                do_sample=True
            )
            
            return hypotheses
        
        # Define fallback function
        def _fallback(context):
            logger.warning("Using fallback: generating generic hypothesis")
            return [f"Analyze the current state and take the next logical step toward: {task}"]
        
        # Try generation with error handling
        try:
            return _generate()
        except Exception as e:
            logger.warning(f"Hypothesis generation failed: {e}")
            return self.error_handler.handle_generation_error(
                error=e,
                generation_fn=lambda **kwargs: _generate(),
                context={
                    'task': task,
                    'state': state,
                    'principles': principles,
                    'temperature': temperature or self.config.temperature
                },
                fallback_fn=_fallback
            )
    
    def instantiate_action(
        self,
        task: str,
        state: str,
        hypothesis: str,
        principles: Optional[List[Principle]] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Instantiate a hypothesis into a concrete action.
        
        Args:
            task: Task description
            state: Current state description
            hypothesis: High-level hypothesis to instantiate
            principles: Optional list of relevant principles
            temperature: Sampling temperature (uses config default if None)
            
        Returns:
            Generated action description
        """
        # Define generation function
        def _generate():
            # Format principles as text
            principle_texts = [p.text for p in principles] if principles else None
            
            # Format prompt for action instantiation
            prompt = self.base_loader.format_prompt(
                task=task,
                state=state,
                hypothesis=hypothesis,
                principles=principle_texts,
                prompt_type="action"
            )
            
            # Generate action
            actions = self.base_loader.generate(
                prompt=prompt,
                max_new_tokens=150,  # Actions should be specific
                temperature=temperature or self.config.temperature,
                num_return_sequences=1,
                do_sample=True
            )
            
            return actions[0]
        
        # Define fallback function
        def _fallback(context):
            logger.warning("Using fallback: generating generic action")
            return f"Execute the next step based on hypothesis: {hypothesis}"
        
        # Try generation with error handling
        try:
            return _generate()
        except Exception as e:
            logger.warning(f"Action instantiation failed: {e}")
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
    
    def generate_action_with_format(
        self,
        task: str,
        state: str,
        hypothesis: str,
        principles: Optional[List[Principle]] = None,
        action_format: Optional[Dict[str, Any]] = None,
        environment_type: str = "general",
        temperature: Optional[float] = None
    ) -> Action:
        """
        Generate an action with structured format.
        
        Args:
            task: Task description
            state: Current state description
            hypothesis: High-level hypothesis to instantiate
            principles: Optional list of relevant principles
            action_format: Optional format specification for the action
            environment_type: Type of environment (toolbench, api_bank, alfworld)
            temperature: Sampling temperature
            
        Returns:
            Structured Action object
        """
        # Generate action description
        action_desc = self.instantiate_action(
            task=task,
            state=state,
            hypothesis=hypothesis,
            principles=principles,
            temperature=temperature
        )
        
        # Parse action description into structured format
        action = self._parse_action_description(action_desc, action_format, environment_type)
        
        return action
    
    def _parse_action_description(
        self,
        description: str,
        action_format: Optional[Dict[str, Any]] = None,
        environment_type: str = "general"
    ) -> Action:
        """
        Parse action description into structured Action object.
        
        Args:
            description: Natural language action description
            action_format: Optional format specification
            environment_type: Type of environment (toolbench, api_bank, alfworld)
            
        Returns:
            Structured Action object
        """
        # Determine action type based on environment
        if environment_type == "api_bank":
            action_type = "api_call"
            # Try to extract API parameters from description
            parameters = self._extract_api_bank_params(description)
        elif environment_type == "toolbench":
            action_type = "tool_use"
            parameters = self._extract_toolbench_params(description)
        elif environment_type == "alfworld":
            action_type = "navigation"  # Default, could be "interaction"
            parameters = self._extract_alfworld_params(description)
        else:
            action_type = "general"
            parameters = {"description": description}
        
        # Override with format if provided
        if action_format:
            if "type" in action_format:
                action_type = action_format["type"]
            if "parameters" in action_format:
                parameters.update(action_format["parameters"])
        
        return Action(
            type=action_type,
            parameters=parameters,
            description=description
        )
    
    def _extract_api_bank_params(self, description: str) -> Dict[str, Any]:
        """
        Extract API-Bank parameters from action description.
        
        Args:
            description: Action description
            
        Returns:
            Dictionary with api_name, method, and parameters
        """
        # Default parameters
        params = {
            "api_name": "unknown_api",
            "method": "GET",
            "parameters": {}
        }
        
        # Try to extract API name (simple heuristic)
        desc_lower = description.lower()
        if "api" in desc_lower or "request" in desc_lower:
            # Look for common API patterns
            if "get" in desc_lower or "fetch" in desc_lower or "retrieve" in desc_lower:
                params["method"] = "GET"
            elif "post" in desc_lower or "create" in desc_lower or "submit" in desc_lower:
                params["method"] = "POST"
            elif "put" in desc_lower or "update" in desc_lower:
                params["method"] = "PUT"
            elif "delete" in desc_lower or "remove" in desc_lower:
                params["method"] = "DELETE"
            
            # Try to extract API name from description
            # This is a simple heuristic - in practice would use more sophisticated parsing
            words = description.split()
            for i, word in enumerate(words):
                if word.lower() in ["api", "call", "request"] and i + 1 < len(words):
                    params["api_name"] = words[i + 1].strip("()[]{}:,.")
                    break
        
        return params
    
    def _extract_toolbench_params(self, description: str) -> Dict[str, Any]:
        """
        Extract ToolBench parameters from action description.
        
        Args:
            description: Action description
            
        Returns:
            Dictionary with tool and args
        """
        params = {
            "tool": "unknown_tool",
            "args": {}
        }
        
        # Try to extract tool name
        desc_lower = description.lower()
        if "tool" in desc_lower or "use" in desc_lower:
            words = description.split()
            for i, word in enumerate(words):
                if word.lower() in ["tool", "use"] and i + 1 < len(words):
                    params["tool"] = words[i + 1].strip("()[]{}:,.")
                    break
        
        return params
    
    def _extract_alfworld_params(self, description: str) -> Dict[str, Any]:
        """
        Extract ALFWorld parameters from action description.
        
        Args:
            description: Action description
            
        Returns:
            Dictionary with command
        """
        # For ALFWorld, the description itself is the command
        return {
            "command": description
        }
    
    def batch_generate_hypotheses(
        self,
        tasks: List[str],
        states: List[str],
        principles_list: Optional[List[List[Principle]]] = None,
        num_hypotheses: int = 1,
        temperature: Optional[float] = None
    ) -> List[List[str]]:
        """
        Generate hypotheses for multiple inputs in batch.
        
        Args:
            tasks: List of task descriptions
            states: List of state descriptions
            principles_list: Optional list of principle lists for each input
            num_hypotheses: Number of hypotheses per input
            temperature: Sampling temperature
            
        Returns:
            List of hypothesis lists, one per input
        """
        if len(tasks) != len(states):
            raise ValueError("tasks and states must have same length")
        
        if principles_list and len(principles_list) != len(tasks):
            raise ValueError("principles_list must have same length as tasks")
        
        all_hypotheses = []
        
        for i, (task, state) in enumerate(zip(tasks, states)):
            principles = principles_list[i] if principles_list else None
            hypotheses = self.generate_hypothesis(
                task=task,
                state=state,
                principles=principles,
                num_hypotheses=num_hypotheses,
                temperature=temperature
            )
            all_hypotheses.append(hypotheses)
        
        return all_hypotheses
    
    def train_mode(self) -> None:
        """Set model to training mode."""
        self.base_loader.get_model().train()
    
    def eval_mode(self) -> None:
        """Set model to evaluation mode."""
        self.base_loader.get_model().eval()
    
    def get_parameters(self):
        """
        Get all trainable parameters.
        
        Returns:
            Iterator over parameters
        """
        return self.base_loader.get_model().parameters()
    
    def get_base_model(self):
        """
        Get the underlying base model.
        
        Returns:
            Base language model
        """
        return self.base_loader.get_model()
    
    def get_tokenizer(self):
        """
        Get the tokenizer.
        
        Returns:
            Tokenizer
        """
        return self.base_loader.get_tokenizer()
    
    def save_model(self, save_path: str) -> None:
        """
        Save the Policy Model.
        
        Args:
            save_path: Directory to save model
        """
        self.base_loader.save_model(save_path)
        logger.info(f"Saved Policy Model to {save_path}")
    
    def load_from_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load Policy Model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint directory
        """
        self.base_loader.load_from_checkpoint(checkpoint_path)
        logger.info(f"Loaded Policy Model from {checkpoint_path}")
    
    def compute_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log probabilities for DPO training.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Target token IDs [batch_size, seq_len]
            
        Returns:
            Log probabilities [batch_size]
        """
        model = self.base_loader.get_model()
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # Get logits
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]
        
        # Compute log probabilities
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        # Gather log probs for target tokens
        target_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Mask padding tokens and sum
        mask = (labels != -100).float()
        sequence_log_probs = (target_log_probs * mask).sum(dim=-1)
        
        return sequence_log_probs
