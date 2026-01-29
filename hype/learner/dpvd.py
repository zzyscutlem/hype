"""
Dense Principle-Value Distillation (DPVD) module for HyPE system.

This module implements:
- Dense reward computation from principle credit scores
- Value Model training with dense rewards
- Replay buffer for trajectory storage
- Training trigger based on buffer size
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from typing import List, Tuple, Optional, Dict, Any
import logging
import numpy as np
from collections import deque

from ..core.data_models import Trajectory, TrajectoryStep, Principle, TrainingExample
from ..models.value_model import ValueModel
from ..utils.error_handlers import ErrorHandler, TrainingDivergenceError


logger = logging.getLogger(__name__)


class DPVD:
    """
    Dense Principle-Value Distillation module.
    
    Constructs dense rewards from principle credit scores and trains
    the Value Model to predict these rewards.
    """
    
    def __init__(
        self,
        value_model: ValueModel,
        beta: float = 0.1,
        learning_rate: float = 1e-5,
        batch_size: int = 32,
        epochs: int = 3,
        gamma: float = 0.99
    ):
        """
        Initialize DPVD module.
        
        Args:
            value_model: Value Model to train
            beta: Principle weight coefficient for dense rewards
            learning_rate: Learning rate for AdamW optimizer
            batch_size: Training batch size
            epochs: Number of training epochs
            gamma: Discount factor for future returns
        """
        self.value_model = value_model
        self.beta = beta
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.gamma = gamma
        self.error_handler = ErrorHandler()
        self.checkpoint_state = None  # For checkpoint reversion
        
        logger.info(
            f"Initialized DPVD with beta={beta}, lr={learning_rate}, "
            f"batch_size={batch_size}, epochs={epochs}"
        )
    
    def compute_dense_rewards(
        self,
        trajectory: Trajectory,
        principles_used: Optional[List[List[Principle]]] = None
    ) -> List[float]:
        """
        Compute step-wise dense rewards from principle credits.
        
        Formula: dense_reward(step_t) = sparse_reward(step_t) + β * Σ(credit_score(p))
        
        Args:
            trajectory: Sequence of (state, action, reward) tuples
            principles_used: Principles applied at each step (uses trajectory.principles_used if None)
            
        Returns:
            Dense reward for each step
        """
        if principles_used is None:
            principles_used = trajectory.principles_used
        
        if len(principles_used) == 0:
            # No principles annotated, use trajectory steps length
            principles_used = [[] for _ in trajectory.steps]
        
        if len(principles_used) != len(trajectory.steps):
            raise ValueError(
                f"principles_used length ({len(principles_used)}) must match "
                f"trajectory steps length ({len(trajectory.steps)})"
            )
        
        dense_rewards = []
        
        for step, step_principles in zip(trajectory.steps, principles_used):
            # Get sparse reward from environment
            sparse_reward = step.reward
            
            # Compute principle credit sum
            principle_credit_sum = sum(p.credit_score for p in step_principles)
            
            # Compute dense reward
            dense_reward = sparse_reward + self.beta * principle_credit_sum
            
            dense_rewards.append(dense_reward)
        
        logger.debug(
            f"Computed dense rewards for trajectory {trajectory.id}: "
            f"{len(dense_rewards)} steps"
        )
        
        return dense_rewards
    
    def compute_returns(
        self,
        rewards: List[float],
        gamma: Optional[float] = None
    ) -> List[float]:
        """
        Compute discounted returns from rewards.
        
        Args:
            rewards: List of rewards
            gamma: Discount factor (uses self.gamma if None)
            
        Returns:
            List of discounted returns
        """
        if gamma is None:
            gamma = self.gamma
        
        returns = []
        G = 0.0
        
        # Compute returns backwards
        for reward in reversed(rewards):
            G = reward + gamma * G
            returns.insert(0, G)
        
        return returns
    
    def construct_training_batch(
        self,
        trajectories: List[Trajectory]
    ) -> Tuple[List[str], List[float]]:
        """
        Prepare training data for Value Model.
        
        Args:
            trajectories: List of trajectories with principle annotations
            
        Returns:
            Tuple of (input_prompts, target_values)
        """
        input_prompts = []
        target_values = []
        
        for trajectory in trajectories:
            # Compute dense rewards
            dense_rewards = self.compute_dense_rewards(trajectory)
            
            # Compute returns (targets for value prediction)
            returns = self.compute_returns(dense_rewards)
            
            # Create training examples for each step
            for step, target_value, step_principles in zip(
                trajectory.steps,
                returns,
                trajectory.principles_used if trajectory.principles_used else [[] for _ in trajectory.steps]
            ):
                # Format state as string
                state_str = str(step.state.observation)
                
                # Format hypothesis (if available)
                hypothesis = step.hypothesis if step.hypothesis else "Continue task"
                
                # Create input prompt for Value Model
                # This should match the format used during inference
                prompt = self._format_value_input(
                    task=trajectory.task,
                    state=state_str,
                    hypothesis=hypothesis,
                    principles=step_principles
                )
                
                input_prompts.append(prompt)
                target_values.append(target_value)
        
        logger.debug(
            f"Constructed training batch: {len(input_prompts)} examples "
            f"from {len(trajectories)} trajectories"
        )
        
        return input_prompts, target_values
    
    def _format_value_input(
        self,
        task: str,
        state: str,
        hypothesis: str,
        principles: List[Principle]
    ) -> str:
        """
        Format input for Value Model prediction.
        
        Args:
            task: Task description
            state: State description
            hypothesis: Hypothesis to evaluate
            principles: Relevant principles
            
        Returns:
            Formatted prompt string
        """
        # Format principles
        if principles:
            principle_text = "\n".join([f"- {p.text}" for p in principles])
        else:
            principle_text = "None"
        
        # Create prompt
        prompt = f"""Task: {task}
Current State: {state}
Hypothesis: {hypothesis}
Relevant Principles:
{principle_text}

Estimated Value:"""
        
        return prompt
    
    def train_value_model(
        self,
        trajectories: List[Trajectory],
        validation_split: float = 0.1
    ) -> Dict[str, List[float]]:
        """
        Train Value Model to predict dense rewards.
        
        Args:
            trajectories: List of trajectories with principle annotations
            validation_split: Fraction of data to use for validation
            
        Returns:
            Dictionary with training history (train_loss, val_loss per epoch)
        """
        if len(trajectories) == 0:
            raise ValueError("Cannot train with empty trajectory list")
        
        logger.info(f"Training Value Model on {len(trajectories)} trajectories")
        
        # Save checkpoint before training
        self._save_checkpoint()
        
        # Construct training data
        input_prompts, target_values = self.construct_training_batch(trajectories)
        
        # Split into train/validation
        n_val = int(len(input_prompts) * validation_split)
        if n_val > 0:
            val_prompts = input_prompts[-n_val:]
            val_targets = target_values[-n_val:]
            train_prompts = input_prompts[:-n_val]
            train_targets = target_values[:-n_val]
        else:
            train_prompts = input_prompts
            train_targets = target_values
            val_prompts = []
            val_targets = []
        
        # Set model to training mode
        self.value_model.train_mode()
        
        # Get model parameters
        parameters = self.value_model.get_parameters()
        
        # Initialize optimizer
        optimizer = AdamW(parameters, lr=self.learning_rate)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        # Training loop
        for epoch in range(self.epochs):
            epoch_losses = []
            
            # Create batches
            n_batches = (len(train_prompts) + self.batch_size - 1) // self.batch_size
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(train_prompts))
                
                batch_prompts = train_prompts[start_idx:end_idx]
                batch_targets = train_targets[start_idx:end_idx]
                
                # Forward pass
                try:
                    batch_loss = self._train_batch(
                        batch_prompts,
                        batch_targets,
                        optimizer
                    )
                    
                    # Check for training divergence
                    gradients = {name: param.grad for name, param in self.value_model.get_base_model().named_parameters() if param.grad is not None}
                    
                    should_continue = self.error_handler.handle_training_error(
                        loss=batch_loss,
                        gradients=gradients,
                        checkpoint_fn=lambda: self._save_checkpoint(),
                        revert_fn=lambda: self._revert_checkpoint()
                    )
                    
                    if not should_continue:
                        logger.warning("Training stopped due to divergence, checkpoint reverted")
                        return history
                    
                    epoch_losses.append(batch_loss)
                    
                except TrainingDivergenceError as e:
                    logger.error(f"Training diverged: {e}")
                    self._revert_checkpoint()
                    raise
                
                # Aggressive memory cleanup after each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Compute epoch metrics
            avg_train_loss = np.mean(epoch_losses)
            history['train_loss'].append(avg_train_loss)
            
            # Validation
            if val_prompts:
                val_loss = self._evaluate(val_prompts, val_targets)
                history['val_loss'].append(val_loss)
                logger.info(
                    f"Epoch {epoch+1}/{self.epochs}: "
                    f"train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}"
                )
            else:
                history['val_loss'].append(0.0)
                logger.info(
                    f"Epoch {epoch+1}/{self.epochs}: "
                    f"train_loss={avg_train_loss:.4f}"
                )
            
            # Cleanup after each epoch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Set model back to eval mode
        self.value_model.eval_mode()
        
        logger.info("Value Model training complete")
        
        return history
    
    def _train_batch(
        self,
        prompts: List[str],
        targets: List[float],
        optimizer: torch.optim.Optimizer
    ) -> float:
        """
        Train on a single batch.
        
        Args:
            prompts: Batch of input prompts
            targets: Batch of target values
            optimizer: Optimizer
            
        Returns:
            Batch loss
        """
        # Tokenize inputs
        tokenizer = self.value_model.base_loader.get_tokenizer()
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.value_model.config.max_length
        ).to(self.value_model.device)
        
        # Get model dtype to match target tensor
        model_dtype = self.value_model.get_value_head().value_head.weight.dtype
        
        # Convert targets to tensor with matching dtype
        target_tensor = torch.tensor(
            targets,
            dtype=model_dtype,
            device=self.value_model.device
        ).unsqueeze(1)  # [batch_size, 1]
        
        # Forward pass
        predictions = self.value_model.forward(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
        
        # Compute MSE loss
        loss = nn.functional.mse_loss(predictions, target_tensor)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Get loss value before cleanup
        loss_value = loss.item()
        
        # Clean up tensors to prevent memory leak
        del inputs
        del target_tensor
        del predictions
        del loss
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return loss_value
    
    def _evaluate(
        self,
        prompts: List[str],
        targets: List[float]
    ) -> float:
        """
        Evaluate on validation data.
        
        Args:
            prompts: Validation prompts
            targets: Validation targets
            
        Returns:
            Average validation loss
        """
        self.value_model.eval_mode()
        
        # Tokenize inputs
        tokenizer = self.value_model.base_loader.get_tokenizer()
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.value_model.config.max_length
        ).to(self.value_model.device)
        
        # Get model dtype to match target tensor
        model_dtype = self.value_model.get_value_head().value_head.weight.dtype
        
        # Convert targets to tensor with matching dtype
        target_tensor = torch.tensor(
            targets,
            dtype=model_dtype,
            device=self.value_model.device
        ).unsqueeze(1)
        
        # Forward pass (no gradients)
        with torch.no_grad():
            predictions = self.value_model.forward(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
            
            # Compute MSE loss
            loss = nn.functional.mse_loss(predictions, target_tensor)
        
        # Get loss value before cleanup
        loss_value = loss.item()
        
        # Clean up tensors to prevent memory leak
        del inputs
        del target_tensor
        del predictions
        del loss
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        self.value_model.train_mode()
        
        return loss_value
    
    def _save_checkpoint(self):
        """Save current model state for potential reversion."""
        import copy
        logger.info("Saving training checkpoint")
        
        # Save model state dict (deep copy to avoid reference issues)
        base_model = self.value_model.get_base_model()
        value_head = self.value_model.get_value_head()
        
        self.checkpoint_state = {
            'base_model': copy.deepcopy(base_model.state_dict()),
            'value_head': copy.deepcopy(value_head.state_dict())
        }
    
    def _revert_checkpoint(self):
        """Revert model to saved checkpoint state."""
        if self.checkpoint_state is None:
            logger.warning("No checkpoint available to revert to")
            return
        
        logger.info("Reverting to previous checkpoint")
        
        # Restore model state
        base_model = self.value_model.get_base_model()
        value_head = self.value_model.get_value_head()
        
        base_model.load_state_dict(self.checkpoint_state['base_model'])
        value_head.load_state_dict(self.checkpoint_state['value_head'])
        
        logger.info("Checkpoint restored successfully")


class ReplayBuffer:
    """
    Replay buffer for trajectory storage.
    
    Stores trajectories with principle annotations and dense rewards
    for training the Value Model.
    """
    
    def __init__(self, max_size: int = 10000):
        """
        Initialize replay buffer.
        
        Args:
            max_size: Maximum number of trajectories to store
        """
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        
        logger.info(f"Initialized ReplayBuffer with max_size={max_size}")
    
    def add(self, trajectory: Trajectory) -> None:
        """
        Add a trajectory to the buffer.
        
        Args:
            trajectory: Trajectory to add
        """
        # Validate trajectory has required fields
        if not trajectory.steps:
            raise ValueError("Cannot add trajectory with no steps")
        
        # Add to buffer (automatically removes oldest if at capacity)
        self.buffer.append(trajectory)
        
        logger.debug(
            f"Added trajectory {trajectory.id} to buffer "
            f"(size: {len(self.buffer)}/{self.max_size})"
        )
    
    def sample(self, batch_size: int) -> List[Trajectory]:
        """
        Sample a batch of trajectories.
        
        Args:
            batch_size: Number of trajectories to sample
            
        Returns:
            List of sampled trajectories
        """
        if batch_size > len(self.buffer):
            logger.warning(
                f"Requested batch_size ({batch_size}) > buffer size ({len(self.buffer)}), "
                f"returning all trajectories"
            )
            batch_size = len(self.buffer)
        
        # Random sampling without replacement
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        sampled = [self.buffer[i] for i in indices]
        
        logger.debug(f"Sampled {len(sampled)} trajectories from buffer")
        
        return sampled
    
    def get_all(self) -> List[Trajectory]:
        """
        Get all trajectories in the buffer.
        
        Returns:
            List of all trajectories
        """
        return list(self.buffer)
    
    def size(self) -> int:
        """
        Get current buffer size.
        
        Returns:
            Number of trajectories in buffer
        """
        return len(self.buffer)
    
    def is_full(self) -> bool:
        """
        Check if buffer is at capacity.
        
        Returns:
            True if buffer is full
        """
        return len(self.buffer) >= self.max_size
    
    def clear(self) -> None:
        """Clear all trajectories from buffer."""
        self.buffer.clear()
        logger.info("Cleared replay buffer")


class TrainingTrigger:
    """
    Training trigger based on buffer size.
    
    Monitors replay buffer and triggers Value Model training when
    sufficient data is accumulated.
    """
    
    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        trigger_threshold: int = 100,
        min_trajectories: int = 10
    ):
        """
        Initialize training trigger.
        
        Args:
            replay_buffer: Replay buffer to monitor
            trigger_threshold: Buffer size threshold to trigger training
            min_trajectories: Minimum trajectories required for training
        """
        self.replay_buffer = replay_buffer
        self.trigger_threshold = trigger_threshold
        self.min_trajectories = min_trajectories
        self.last_training_size = 0
        
        logger.info(
            f"Initialized TrainingTrigger with threshold={trigger_threshold}, "
            f"min_trajectories={min_trajectories}"
        )
    
    def should_trigger(self) -> bool:
        """
        Check if training should be triggered.
        
        Returns:
            True if training should be triggered
        """
        current_size = self.replay_buffer.size()
        
        # Check minimum requirement
        if current_size < self.min_trajectories:
            return False
        
        # Check if threshold reached since last training
        new_data = current_size - self.last_training_size
        
        should_train = new_data >= self.trigger_threshold
        
        if should_train:
            logger.info(
                f"Training trigger activated: {new_data} new trajectories "
                f"(threshold: {self.trigger_threshold})"
            )
        
        return should_train
    
    def mark_training_complete(self) -> None:
        """Mark that training has been completed."""
        self.last_training_size = self.replay_buffer.size()
        logger.debug(f"Training complete, marked buffer size: {self.last_training_size}")
    
    def reset(self) -> None:
        """Reset the trigger state."""
        self.last_training_size = 0
        logger.info("Reset training trigger")
