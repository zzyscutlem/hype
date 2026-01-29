"""
Error handling utilities for HyPE system.

This module provides error handlers for various failure scenarios:
- Model generation errors
- Environment execution errors
- Database operation errors
- Training errors
"""

import time
import logging
from typing import Optional, List, Dict, Any, Tuple, Callable
from dataclasses import dataclass
import torch
import numpy as np

from ..core.data_models import Action, State, Principle


logger = logging.getLogger(__name__)


@dataclass
class ErrorContext:
    """Context information for error handling."""
    component: str  # Component where error occurred
    operation: str  # Operation that failed
    attempt: int  # Current attempt number
    max_attempts: int  # Maximum retry attempts
    metadata: Dict[str, Any]  # Additional context


class ModelGenerationError(Exception):
    """Exception raised when model generation fails."""
    pass


class EnvironmentExecutionError(Exception):
    """Exception raised when environment execution fails."""
    pass


class DatabaseOperationError(Exception):
    """Exception raised when database operations fail."""
    pass


class TrainingDivergenceError(Exception):
    """Exception raised when training diverges."""
    pass


class ErrorHandler:
    """
    Central error handler for HyPE system.
    
    Provides retry logic, fallback strategies, and error recovery
    for different types of failures.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize error handler.
        
        Args:
            config: Optional configuration for error handling
        """
        self.config = config or {}
        
        # Default retry settings
        self.max_retries = self.config.get('max_retries', 3)
        self.base_backoff = self.config.get('base_backoff', 1.0)
        self.max_backoff = self.config.get('max_backoff', 30.0)
        
        # Temperature adjustment for generation retries
        self.temperature_increment = self.config.get('temperature_increment', 0.1)
        self.max_temperature = self.config.get('max_temperature', 1.5)
        
        logger.info("Initialized ErrorHandler")
    
    def handle_generation_error(
        self,
        error: Exception,
        generation_fn: Callable,
        context: Dict[str, Any],
        max_retries: Optional[int] = None,
        fallback_fn: Optional[Callable] = None
    ) -> Optional[Any]:
        """
        Handle model generation errors with retry and fallback.
        
        Args:
            error: The exception that occurred
            generation_fn: Function to retry for generation
            context: Context dict with generation parameters
            max_retries: Maximum retry attempts (uses default if None)
            fallback_fn: Optional fallback function if retries fail
            
        Returns:
            Generated output or fallback result
            
        Raises:
            ModelGenerationError: If all retries and fallback fail
        """
        max_retries = max_retries or self.max_retries
        base_temperature = context.get('temperature', 0.7)
        
        logger.warning(f"Generation error: {error}. Attempting retries...")
        
        for attempt in range(max_retries):
            try:
                # Adjust temperature for diversity
                adjusted_temp = min(
                    base_temperature + (attempt * self.temperature_increment),
                    self.max_temperature
                )
                
                # Update context with adjusted temperature
                retry_context = context.copy()
                retry_context['temperature'] = adjusted_temp
                
                logger.info(f"Retry attempt {attempt + 1}/{max_retries} "
                           f"with temperature={adjusted_temp:.2f}")
                
                # Retry generation
                result = generation_fn(**retry_context)
                
                # Validate result
                if self._validate_generation_result(result):
                    logger.info(f"Generation succeeded on attempt {attempt + 1}")
                    return result
                else:
                    logger.warning(f"Generated invalid result on attempt {attempt + 1}")
                    
            except Exception as e:
                logger.warning(f"Retry {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    # Last attempt failed
                    break
        
        # All retries failed, try fallback
        if fallback_fn is not None:
            try:
                logger.info("All retries failed, using fallback strategy")
                result = fallback_fn(context)
                return result
            except Exception as e:
                logger.error(f"Fallback strategy also failed: {e}")
        
        # Everything failed
        raise ModelGenerationError(
            f"Generation failed after {max_retries} retries and fallback"
        )
    
    def handle_environment_error(
        self,
        error: Exception,
        action: Action,
        state: State,
        correction_fn: Optional[Callable] = None,
        environment: Optional[Any] = None
    ) -> Tuple[State, float, bool]:
        """
        Handle environment execution errors.
        
        Args:
            error: The exception that occurred
            action: Action that failed
            state: Current state
            correction_fn: Optional function to correct the action
            environment: Optional environment to retry execution
            
        Returns:
            Tuple of (next_state, reward, done)
        """
        logger.warning(f"Environment execution error: {error}")
        
        # Check if it's a syntax error that can be corrected
        if self._is_syntax_error(error) and correction_fn is not None:
            try:
                logger.info("Attempting action correction via SR-Adapt")
                corrected_action = correction_fn(action)
                
                # Retry with corrected action
                if environment is not None:
                    next_state, reward, done = environment.execute(corrected_action)
                    logger.info("Corrected action executed successfully")
                    return next_state, reward, done
                    
            except Exception as e:
                logger.error(f"Action correction failed: {e}")
        
        # Unrecoverable error - terminate trajectory
        logger.error("Unrecoverable environment error, terminating trajectory")
        return state, -1.0, True
    
    def handle_database_error(
        self,
        error: Exception,
        query_fn: Callable,
        query_args: Dict[str, Any],
        max_retries: Optional[int] = None,
        fallback_fn: Optional[Callable] = None
    ) -> Any:
        """
        Handle database operation errors with exponential backoff.
        
        Args:
            error: The exception that occurred
            query_fn: Function to retry for database query
            query_args: Arguments for the query function
            max_retries: Maximum retry attempts (uses default if None)
            fallback_fn: Optional fallback function (e.g., random sampling)
            
        Returns:
            Query result or fallback result
            
        Raises:
            DatabaseOperationError: If all retries and fallback fail
        """
        max_retries = max_retries or self.max_retries
        
        logger.warning(f"Database error: {error}. Attempting retries with backoff...")
        
        for attempt in range(max_retries):
            try:
                # Exponential backoff
                if attempt > 0:
                    backoff_time = min(
                        self.base_backoff * (2 ** attempt),
                        self.max_backoff
                    )
                    logger.info(f"Waiting {backoff_time:.1f}s before retry {attempt + 1}")
                    time.sleep(backoff_time)
                
                # Retry query
                logger.info(f"Database retry attempt {attempt + 1}/{max_retries}")
                result = query_fn(**query_args)
                
                logger.info(f"Database query succeeded on attempt {attempt + 1}")
                return result
                
            except Exception as e:
                logger.warning(f"Database retry {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    # Last attempt failed
                    break
        
        # All retries failed, try fallback
        if fallback_fn is not None:
            try:
                logger.info("All database retries failed, using fallback strategy")
                result = fallback_fn(query_args)
                return result
            except Exception as e:
                logger.error(f"Database fallback strategy also failed: {e}")
        
        # Everything failed
        raise DatabaseOperationError(
            f"Database operation failed after {max_retries} retries and fallback"
        )
    
    def handle_training_error(
        self,
        loss: float,
        gradients: Optional[Dict[str, torch.Tensor]],
        checkpoint_fn: Optional[Callable] = None,
        revert_fn: Optional[Callable] = None
    ) -> bool:
        """
        Handle training errors (divergence, NaN loss, exploding gradients).
        
        Args:
            loss: Current training loss
            gradients: Dictionary of parameter gradients
            checkpoint_fn: Optional function to save checkpoint
            revert_fn: Optional function to revert to previous checkpoint
            
        Returns:
            True if training should continue, False if should stop
        """
        # Check for NaN loss
        if torch.isnan(torch.tensor(loss)) or np.isnan(loss):
            logger.error("Training diverged: NaN loss detected")
            
            if revert_fn is not None:
                try:
                    logger.info("Reverting to previous checkpoint")
                    revert_fn()
                    return False  # Stop current training, will restart from checkpoint
                except Exception as e:
                    logger.error(f"Failed to revert checkpoint: {e}")
            
            raise TrainingDivergenceError("Training loss became NaN")
        
        # Check for exploding gradients
        if gradients is not None:
            max_grad_norm = self._compute_max_gradient_norm(gradients)
            
            if max_grad_norm > 100.0:  # Threshold for exploding gradients
                logger.error(f"Training diverged: Exploding gradients "
                           f"(max norm: {max_grad_norm:.2f})")
                
                if revert_fn is not None:
                    try:
                        logger.info("Reverting to previous checkpoint")
                        revert_fn()
                        return False
                    except Exception as e:
                        logger.error(f"Failed to revert checkpoint: {e}")
                
                raise TrainingDivergenceError(
                    f"Gradients exploded (max norm: {max_grad_norm:.2f})"
                )
        
        # Check for loss increasing consistently (potential divergence)
        # This would require tracking loss history, which should be done by caller
        
        return True  # Training can continue
    
    def _validate_generation_result(self, result: Any) -> bool:
        """
        Validate that generation result is usable.
        
        Args:
            result: Generated result to validate
            
        Returns:
            True if result is valid, False otherwise
        """
        if result is None:
            return False
        
        if isinstance(result, str):
            # Check for empty or whitespace-only strings
            if not result.strip():
                return False
            
            # Check for common error patterns
            error_patterns = [
                "error",
                "failed",
                "invalid",
                "cannot",
                "unable to"
            ]
            result_lower = result.lower()
            if any(pattern in result_lower for pattern in error_patterns):
                # May be an error message rather than valid generation
                logger.warning(f"Generated text contains error pattern: {result[:100]}")
                return False
        
        elif isinstance(result, list):
            # Check for empty list
            if len(result) == 0:
                return False
            
            # Validate each item
            return all(self._validate_generation_result(item) for item in result)
        
        return True
    
    def _is_syntax_error(self, error: Exception) -> bool:
        """
        Check if error is a syntax error that can be corrected.
        
        Args:
            error: Exception to check
            
        Returns:
            True if it's a correctable syntax error
        """
        # Check exception type
        if isinstance(error, (SyntaxError, ValueError, TypeError)):
            return True
        
        # Check error message for syntax-related keywords
        error_msg = str(error).lower()
        syntax_keywords = [
            "syntax",
            "invalid",
            "malformed",
            "parse",
            "format",
            "expected"
        ]
        
        return any(keyword in error_msg for keyword in syntax_keywords)
    
    def _compute_max_gradient_norm(
        self,
        gradients: Dict[str, torch.Tensor]
    ) -> float:
        """
        Compute maximum gradient norm across all parameters.
        
        Args:
            gradients: Dictionary of parameter gradients
            
        Returns:
            Maximum gradient norm
        """
        max_norm = 0.0
        
        for name, grad in gradients.items():
            if grad is not None:
                grad_norm = torch.norm(grad).item()
                max_norm = max(max_norm, grad_norm)
        
        return max_norm


# Convenience functions for common error handling patterns

def retry_with_backoff(
    fn: Callable,
    max_retries: int = 3,
    base_backoff: float = 1.0,
    max_backoff: float = 30.0,
    **kwargs
) -> Any:
    """
    Retry a function with exponential backoff.
    
    Args:
        fn: Function to retry
        max_retries: Maximum number of retries
        base_backoff: Base backoff time in seconds
        max_backoff: Maximum backoff time in seconds
        **kwargs: Arguments to pass to function
        
    Returns:
        Function result
        
    Raises:
        Exception: Last exception if all retries fail
    """
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return fn(**kwargs)
        except Exception as e:
            last_exception = e
            
            if attempt < max_retries - 1:
                backoff_time = min(
                    base_backoff * (2 ** attempt),
                    max_backoff
                )
                logger.info(f"Retry {attempt + 1}/{max_retries} after {backoff_time:.1f}s")
                time.sleep(backoff_time)
    
    raise last_exception


def safe_execute(
    fn: Callable,
    default_value: Any = None,
    log_errors: bool = True,
    **kwargs
) -> Any:
    """
    Execute a function safely, returning default value on error.
    
    Args:
        fn: Function to execute
        default_value: Value to return on error
        log_errors: Whether to log errors
        **kwargs: Arguments to pass to function
        
    Returns:
        Function result or default value
    """
    try:
        return fn(**kwargs)
    except Exception as e:
        if log_errors:
            logger.error(f"Error in {fn.__name__}: {e}")
        return default_value
