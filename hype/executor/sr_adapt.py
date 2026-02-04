"""
SR-Adapt (Semantic-Reflective Adaptation) module for HyPE system.

This module implements the SR-Adapt validator which ensures executed actions
align with principles through dual guardrails:
1. Syntax validation: Ensures actions are well-formed for the environment
2. Semantic alignment: Ensures actions align with retrieved principles

The module also provides LoRA-based correction for misaligned actions.
"""

import logging
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer

from ..core.data_models import Action, Principle, State
from ..core.config import ModelConfig, PrincipleMemoryConfig
from ..models.executor_model import ExecutorModel
from ..utils.error_handlers import ErrorHandler, EnvironmentExecutionError


logger = logging.getLogger(__name__)


class SyntaxValidator:
    """
    Environment-specific syntax validation for actions.
    
    This class provides syntax checking for different environment types
    to ensure actions are well-formed before execution.
    """
    
    def __init__(self):
        """Initialize syntax validator."""
        self.validators = {
            "toolbench": self._validate_toolbench,
            "api_bank": self._validate_api_bank,
            "alfworld": self._validate_alfworld,
        }
        logger.info("Initialized SyntaxValidator")
    
    def validate(self, action: Action, environment_type: str) -> Tuple[bool, Optional[str]]:
        """
        Validate action syntax for the given environment type.
        
        Args:
            action: Action to validate
            environment_type: Type of environment (toolbench, api_bank, alfworld)
            
        Returns:
            Tuple of (is_valid, error_message)
            - is_valid: True if action is syntactically correct
            - error_message: None if valid, otherwise description of the error
        """
        if environment_type not in self.validators:
            logger.warning(f"Unknown environment type: {environment_type}, skipping validation")
            return True, None
        
        validator_func = self.validators[environment_type]
        return validator_func(action)
    
    def _validate_toolbench(self, action: Action) -> Tuple[bool, Optional[str]]:
        """
        Validate ToolBench action syntax.
        
        ToolBench actions must have:
        - type: "tool_use"
        - parameters: {"tool": str, "args": dict}
        
        Args:
            action: Action to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check action type
        if action.type != "tool_use":
            return False, f"Invalid action type for ToolBench: {action.type}, expected 'tool_use'"
        
        # Check required parameters
        if "tool" not in action.parameters:
            return False, "Missing required parameter: 'tool'"
        
        if not isinstance(action.parameters["tool"], str):
            return False, f"Parameter 'tool' must be string, got {type(action.parameters['tool'])}"
        
        if "args" not in action.parameters:
            return False, "Missing required parameter: 'args'"
        
        if not isinstance(action.parameters["args"], dict):
            return False, f"Parameter 'args' must be dict, got {type(action.parameters['args'])}"
        
        # Check tool name is not empty
        if not action.parameters["tool"].strip():
            return False, "Tool name cannot be empty"
        
        return True, None
    
    def _validate_api_bank(self, action: Action) -> Tuple[bool, Optional[str]]:
        """
        Validate API-Bank action syntax.
        
        API-Bank actions must have:
        - type: "api_call"
        - parameters: {"api_name": str, "method": str, "parameters": dict}
        
        Args:
            action: Action to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check action type
        if action.type != "api_call":
            return False, f"Invalid action type for API-Bank: {action.type}, expected 'api_call'"
        
        # Check required parameters
        required_params = ["api_name", "method", "parameters"]
        for param in required_params:
            if param not in action.parameters:
                return False, f"Missing required parameter: '{param}'"
        
        # Check parameter types
        if not isinstance(action.parameters["api_name"], str):
            return False, f"Parameter 'api_name' must be string, got {type(action.parameters['api_name'])}"
        
        if not isinstance(action.parameters["method"], str):
            return False, f"Parameter 'method' must be string, got {type(action.parameters['method'])}"
        
        if not isinstance(action.parameters["parameters"], dict):
            return False, f"Parameter 'parameters' must be dict, got {type(action.parameters['parameters'])}"
        
        # Check API name and method are not empty
        if not action.parameters["api_name"].strip():
            return False, "API name cannot be empty"
        
        if not action.parameters["method"].strip():
            return False, "API method cannot be empty"
        
        # Validate HTTP method
        valid_methods = ["GET", "POST", "PUT", "DELETE", "PATCH"]
        if action.parameters["method"].upper() not in valid_methods:
            return False, f"Invalid HTTP method: {action.parameters['method']}, must be one of {valid_methods}"
        
        return True, None
    
    def _validate_alfworld(self, action: Action) -> Tuple[bool, Optional[str]]:
        """
        Validate ALFWorld action syntax.
        
        ALFWorld actions must have:
        - type: "navigation" or "interaction"
        - parameters: {"command": str}
        
        Args:
            action: Action to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check action type
        valid_types = ["navigation", "interaction"]
        if action.type not in valid_types:
            return False, f"Invalid action type for ALFWorld: {action.type}, expected one of {valid_types}"
        
        # Check required parameters
        if "command" not in action.parameters:
            return False, "Missing required parameter: 'command'"
        
        if not isinstance(action.parameters["command"], str):
            return False, f"Parameter 'command' must be string, got {type(action.parameters['command'])}"
        
        # Check command is not empty
        if not action.parameters["command"].strip():
            return False, "Command cannot be empty"
        
        # Validate command format for navigation actions
        if action.type == "navigation":
            valid_nav_commands = ["go to", "move to", "walk to", "turn", "look"]
            command_lower = action.parameters["command"].lower()
            if not any(cmd in command_lower for cmd in valid_nav_commands):
                return False, f"Navigation command must contain one of: {valid_nav_commands}"
        
        # Validate command format for interaction actions
        if action.type == "interaction":
            valid_int_commands = ["take", "put", "open", "close", "toggle", "use", "examine"]
            command_lower = action.parameters["command"].lower()
            if not any(cmd in command_lower for cmd in valid_int_commands):
                return False, f"Interaction command must contain one of: {valid_int_commands}"
        
        return True, None


class SRAdapt:
    """
    Semantic-Reflective Adaptation module for action validation and correction.
    
    This class implements the dual guardrail system:
    1. Syntax validation: Ensures actions are well-formed
    2. Semantic alignment: Ensures actions align with principles
    
    When actions fail semantic alignment, LoRA-based correction is applied.
    
    Attributes:
        config: Model configuration
        memory_config: Principle memory configuration
        syntax_validator: Syntax validation component
        executor_model: Executor model with LoRA for correction
        embedding_model: Model for computing semantic embeddings
        alignment_threshold: Minimum alignment score required
    """
    
    def __init__(
        self,
        config: ModelConfig,
        memory_config: Optional[PrincipleMemoryConfig] = None,
        alignment_threshold: float = 0.7
    ):
        """
        Initialize SR-Adapt module.
        
        Args:
            config: Model configuration
            memory_config: Principle memory configuration
            alignment_threshold: Minimum alignment score (0-1) required for execution
        """
        self.config = config
        self.memory_config = memory_config or PrincipleMemoryConfig()
        self.alignment_threshold = alignment_threshold
        
        # Initialize components
        self.syntax_validator = SyntaxValidator()
        self.executor_model = None  # Lazy initialization
        self.embedding_model = None  # Lazy initialization
        self.error_handler = ErrorHandler()
        
        logger.info(f"Initialized SRAdapt with alignment_threshold={alignment_threshold}")
    
    def _ensure_models_loaded(self):
        """Ensure executor and embedding models are loaded."""
        if self.executor_model is None:
            logger.info("Loading executor model...")
            self.executor_model = ExecutorModel(self.config)
            self.executor_model.load_model()
        
        if self.embedding_model is None:
            logger.info(f"Loading embedding model: {self.memory_config.embedding_model}")
            
            # Force use of local cache
            import os
            from pathlib import Path
            
            model_name = self.memory_config.embedding_model
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
            
            # Convert model name to cache directory format
            # e.g., "BAAI/bge-large-en-v1.5" -> "models--BAAI--bge-large-en-v1.5"
            cache_model_name = "models--" + model_name.replace("/", "--")
            model_cache_path = cache_dir / cache_model_name
            
            if model_cache_path.exists():
                logger.info(f"Found local cache: {model_cache_path}")
                
                # Find the snapshot directory (contains actual model files)
                snapshots_dir = model_cache_path / "snapshots"
                if snapshots_dir.exists():
                    # Get the first (and usually only) snapshot
                    snapshot_dirs = list(snapshots_dir.iterdir())
                    if snapshot_dirs:
                        snapshot_path = snapshot_dirs[0]
                        logger.info(f"Using snapshot: {snapshot_path.name}")
                        
                        # Force offline mode
                        os.environ['HF_HUB_OFFLINE'] = '1'
                        os.environ['TRANSFORMERS_OFFLINE'] = '1'
                        
                        try:
                            # Load directly from snapshot path
                            self.embedding_model = SentenceTransformer(
                                str(snapshot_path),
                                device=self.config.device
                            )
                            logger.info(f"✅ Successfully loaded embedding model from local cache")
                            return
                        except Exception as e:
                            logger.error(f"Failed to load from snapshot: {e}")
                            raise RuntimeError(
                                f"Cannot load embedding model from local cache. "
                                f"Please ensure {model_cache_path} contains valid model files."
                            )
                    else:
                        raise RuntimeError(f"No snapshots found in {snapshots_dir}")
                else:
                    raise RuntimeError(f"Snapshots directory not found: {snapshots_dir}")
            else:
                raise RuntimeError(
                    f"Embedding model not found in local cache: {model_cache_path}\n"
                    f"Please download the model first or check the cache path."
                )
    
    def validate_syntax(
        self,
        action: Action,
        environment_type: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate action syntax for the given environment.
        
        Args:
            action: Action to validate
            environment_type: Type of environment (toolbench, api_bank, alfworld)
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        return self.syntax_validator.validate(action, environment_type)


    
    def compute_embedding(self, text: str) -> np.ndarray:
        """
        Compute semantic embedding for text.
        
        Args:
            text: Text to embed
            
        Returns:
            1024-dimensional embedding vector
        """
        self._ensure_models_loaded()
        embedding = self.embedding_model.encode(text, convert_to_numpy=True)
        return embedding
    
    def semantic_alignment_check(
        self,
        action: Action,
        principles: List[Principle]
    ) -> float:
        """
        Compute semantic alignment score between action and principles.
        
        The alignment score is computed as the cosine similarity between
        the action's embedding and a credit-weighted average of principle embeddings.
        
        Formula:
            alignment_score = cosine_similarity(
                embed(action_description),
                weighted_avg([embed(p.text) for p in principles], weights=credit_scores)
            )
        
        Args:
            action: Action to check alignment for
            principles: List of relevant principles
            
        Returns:
            Alignment score in range [0, 1], where 1 is perfect alignment
        """
        if not principles:
            logger.warning("No principles provided for alignment check, returning 0.0")
            return 0.0
        
        self._ensure_models_loaded()
        
        # Compute action embedding
        action_embedding = self.compute_embedding(action.description)
        
        # Compute credit-weighted average of principle embeddings
        principle_embeddings = np.array([p.embedding for p in principles])
        credit_scores = np.array([p.credit_score for p in principles])
        
        # Normalize credit scores to sum to 1
        if credit_scores.sum() > 0:
            weights = credit_scores / credit_scores.sum()
        else:
            # If all credit scores are 0, use uniform weights
            weights = np.ones(len(principles)) / len(principles)
        
        # Compute weighted average embedding
        weighted_principle_embedding = np.average(principle_embeddings, axis=0, weights=weights)
        
        # Compute cosine similarity
        alignment_score = self._cosine_similarity(action_embedding, weighted_principle_embedding)
        
        # Ensure score is in [0, 1] range
        alignment_score = max(0.0, min(1.0, alignment_score))
        
        logger.debug(f"Alignment score: {alignment_score:.4f} (threshold: {self.alignment_threshold})")
        
        return alignment_score
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity in range [-1, 1]
        """
        # Normalize vectors
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)
        
        # Compute dot product
        similarity = np.dot(vec1_norm, vec2_norm)
        
        return float(similarity)
    
    def lora_correction(
        self,
        action: Action,
        principles: List[Principle],
        alignment_score: float,
        task: str,
        state: State,
        hypothesis: str,
        max_attempts: int = 3
    ) -> Action:
        """
        Apply LoRA-based correction to improve action alignment with principles.
        
        This method uses test-time fine-tuning of the Executor Model's LoRA adapter
        to generate a corrected action that better aligns with the given principles.
        
        Args:
            action: Original action with low alignment
            principles: Principles to align with
            alignment_score: Current alignment score
            task: Task description
            state: Current state
            hypothesis: Hypothesis that led to this action
            max_attempts: Maximum number of correction attempts
            
        Returns:
            Corrected action with improved alignment
            
        Raises:
            RuntimeError: If correction fails to improve alignment after max_attempts
        """
        self._ensure_models_loaded()
        
        logger.info(f"Applying LoRA correction (original alignment: {alignment_score:.4f})")
        
        best_action = action
        best_score = alignment_score
        
        for attempt in range(max_attempts):
            try:
                # Apply test-time correction
                corrected_action_text = self.executor_model.test_time_correction(
                    misaligned_action=action.description,
                    task=task,
                    state=str(state.observation),
                    hypothesis=hypothesis,
                    principles=principles
                )
                
                # Create corrected action object
                corrected_action = Action(
                    type=action.type,
                    parameters=action.parameters.copy(),
                    description=corrected_action_text
                )
                
                # Compute new alignment score
                corrected_score = self.semantic_alignment_check(corrected_action, principles)
                
                logger.info(f"Correction attempt {attempt + 1}: alignment {alignment_score:.4f} -> {corrected_score:.4f}")
                
                # Check if improvement was achieved
                if corrected_score > best_score:
                    best_action = corrected_action
                    best_score = corrected_score
                    
                    # If we've reached acceptable alignment, return
                    if corrected_score >= self.alignment_threshold:
                        logger.info(f"Correction successful: alignment improved to {corrected_score:.4f}")
                        return corrected_action
                
                # Reset LoRA weights for next attempt
                if attempt < max_attempts - 1:
                    self.executor_model.reset_lora_weights()
                    
            except Exception as e:
                logger.error(f"Error during correction attempt {attempt + 1}: {e}")
                if attempt == max_attempts - 1:
                    raise
        
        # Return best action found, even if it didn't reach threshold
        if best_score > alignment_score:
            logger.warning(f"Correction improved alignment to {best_score:.4f} but did not reach threshold {self.alignment_threshold}")
            return best_action
        else:
            logger.warning(f"Correction failed to improve alignment after {max_attempts} attempts")
            return action
    
    def validate_and_execute(
        self,
        action: Action,
        principles: List[Principle],
        environment: Any,
        environment_type: str,
        task: str,
        state: State,
        hypothesis: str,
        apply_correction: bool = True
    ) -> Tuple[State, float, bool, Dict[str, Any]]:
        """
        Validate action through dual guardrails and execute if valid.
        
        This method implements the complete SR-Adapt workflow:
        1. Syntax validation: Check if action is well-formed
        2. Semantic alignment: Check if action aligns with principles
        3. LoRA correction: If alignment is low, apply correction
        4. Execution: Execute the validated action in the environment
        
        Args:
            action: Action to validate and execute
            principles: Relevant principles for alignment checking
            environment: Environment simulator
            environment_type: Type of environment (toolbench, api_bank, alfworld)
            task: Task description
            state: Current state
            hypothesis: Hypothesis that led to this action
            apply_correction: Whether to apply LoRA correction if alignment is low
            
        Returns:
            Tuple of (next_state, reward, done, metadata)
            - next_state: State after action execution
            - reward: Reward received
            - done: Whether episode is complete
            - metadata: Dictionary with validation info:
                - syntax_valid: bool
                - alignment_score: float
                - corrected: bool
                - error: Optional[str]
        """
        metadata = {
            "syntax_valid": False,
            "alignment_score": 0.0,
            "corrected": False,
            "error": None
        }
        
        # Step 1: Syntax validation
        logger.info("Step 1: Syntax validation")
        syntax_valid, syntax_error = self.validate_syntax(action, environment_type)
        metadata["syntax_valid"] = syntax_valid
        
        if not syntax_valid:
            logger.warning(f"Syntax validation failed: {syntax_error}")
            metadata["error"] = f"Syntax error: {syntax_error}"
            # Return failure state without executing
            return state, -1.0, True, metadata
        
        logger.info("Syntax validation passed")
        
        # Step 2: Semantic alignment check
        logger.info("Step 2: Semantic alignment check")
        alignment_score = self.semantic_alignment_check(action, principles)
        metadata["alignment_score"] = alignment_score
        
        # Step 3: LoRA correction if needed
        final_action = action
        if alignment_score < self.alignment_threshold:
            logger.warning(f"Alignment score {alignment_score:.4f} below threshold {self.alignment_threshold}")
            
            if apply_correction:
                logger.info("Step 3: Applying LoRA correction")
                try:
                    corrected_action = self.lora_correction(
                        action=action,
                        principles=principles,
                        alignment_score=alignment_score,
                        task=task,
                        state=state,
                        hypothesis=hypothesis
                    )
                    
                    # Re-validate syntax of corrected action
                    corrected_syntax_valid, corrected_syntax_error = self.validate_syntax(
                        corrected_action, environment_type
                    )
                    
                    if corrected_syntax_valid:
                        final_action = corrected_action
                        metadata["corrected"] = True
                        metadata["alignment_score"] = self.semantic_alignment_check(
                            corrected_action, principles
                        )
                        logger.info(f"Correction applied, new alignment: {metadata['alignment_score']:.4f}")
                    else:
                        logger.warning(f"Corrected action failed syntax validation: {corrected_syntax_error}")
                        metadata["error"] = f"Corrected action syntax error: {corrected_syntax_error}"
                        return state, -1.0, True, metadata
                        
                except Exception as e:
                    logger.error(f"LoRA correction failed: {e}")
                    metadata["error"] = f"Correction error: {str(e)}"
                    return state, -1.0, True, metadata
            else:
                logger.warning("Correction disabled, blocking execution due to low alignment")
                metadata["error"] = f"Alignment score {alignment_score:.4f} below threshold"
                return state, -1.0, True, metadata
        else:
            logger.info(f"Alignment check passed: {alignment_score:.4f} >= {self.alignment_threshold}")
        
        # Step 4: Execute action
        logger.info("Step 4: Executing validated action")
        
        # Define correction function for error handler
        def correction_fn(failed_action):
            return self.lora_correction(
                action=failed_action,
                principles=principles,
                alignment_score=metadata["alignment_score"],
                task=task,
                state=state,
                hypothesis=hypothesis
            )
        
        try:
            next_state, reward, done = environment.execute(final_action)
            logger.info(f"Action executed successfully, reward: {reward}, done: {done}")
            return next_state, reward, done, metadata
            
        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            
            # Use error handler to handle execution error
            try:
                next_state, reward, done = self.error_handler.handle_environment_error(
                    error=e,
                    action=final_action,
                    state=state,
                    correction_fn=correction_fn if apply_correction else None,
                    environment=environment
                )
                
                # Update metadata if correction was applied
                if reward != -1.0:
                    metadata["corrected"] = True
                    metadata["error"] = f"Recovered from error: {str(e)}"
                else:
                    metadata["error"] = f"Unrecoverable execution error: {str(e)}"
                
                return next_state, reward, done, metadata
                
            except Exception as recovery_error:
                logger.error(f"Error recovery failed: {recovery_error}")
                metadata["error"] = f"Execution and recovery failed: {str(recovery_error)}"
                return state, -1.0, True, metadata
    
    def set_alignment_threshold(self, threshold: float):
        """
        Update the alignment threshold.
        
        Args:
            threshold: New alignment threshold (0-1)
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be in [0, 1], got {threshold}")
        
        self.alignment_threshold = threshold
        logger.info(f"Updated alignment threshold to {threshold}")
    
    def get_alignment_threshold(self) -> float:
        """
        Get the current alignment threshold.
        
        Returns:
            Current alignment threshold
        """
        return self.alignment_threshold
