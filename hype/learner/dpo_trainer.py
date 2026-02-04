"""
DPO (Direct Preference Optimization) Trainer for Policy Model.

This module implements DPO training for the Policy Model using principle-guided
preferences. It constructs preference pairs from trajectories based on cumulative
principle credit scores and optimizes the policy to prefer high-credit trajectories.
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any
import logging
from dataclasses import dataclass

from ..core.data_models import Trajectory, Principle
from ..models.policy_model import PolicyModel
from ..utils.error_handlers import ErrorHandler, TrainingDivergenceError


logger = logging.getLogger(__name__)


@dataclass
class PreferencePair:
    """
    A pair of trajectories for DPO training.
    
    Attributes:
        preferred: Trajectory with higher principle credit
        dispreferred: Trajectory with lower principle credit
        preferred_credit: Cumulative principle credit for preferred trajectory
        dispreferred_credit: Cumulative principle credit for dispreferred trajectory
    """
    preferred: Trajectory
    dispreferred: Trajectory
    preferred_credit: float
    dispreferred_credit: float
    
    def __post_init__(self):
        """Validate preference pair."""
        if self.preferred_credit < self.dispreferred_credit:
            raise ValueError(
                f"Preferred trajectory must have higher credit "
                f"({self.preferred_credit}) than dispreferred ({self.dispreferred_credit})"
            )


class PreferencePairConstructor:
    """
    Constructs preference pairs from trajectories based on principle credit scores.
    
    This class ranks trajectories by their cumulative principle credit and creates
    (preferred, dispreferred) pairs for DPO training.
    """
    
    def __init__(self, min_credit_gap: float = 0.1):
        """
        Initialize preference pair constructor.
        
        Args:
            min_credit_gap: Minimum credit difference required for a valid pair
        """
        self.min_credit_gap = min_credit_gap
        logger.info(f"Initialized PreferencePairConstructor with min_credit_gap={min_credit_gap}")
    
    def compute_trajectory_credit(self, trajectory: Trajectory) -> float:
        """
        Compute cumulative principle credit for a trajectory.
        
        Args:
            trajectory: Trajectory with principle annotations
            
        Returns:
            Sum of credit scores from all principles used in the trajectory
        """
        total_credit = 0.0
        
        # Sum credit scores from all principles used at each step
        for step_principles in trajectory.principles_used:
            for principle in step_principles:
                total_credit += principle.credit_score
        
        return total_credit
    
    def rank_trajectories(
        self,
        trajectories: List[Trajectory]
    ) -> List[Tuple[Trajectory, float]]:
        """
        Rank trajectories by cumulative principle credit.
        
        Args:
            trajectories: List of trajectories to rank
            
        Returns:
            List of (trajectory, credit) tuples sorted by credit (descending)
        """
        # Compute credit for each trajectory
        trajectory_credits = [
            (traj, self.compute_trajectory_credit(traj))
            for traj in trajectories
        ]
        
        # Sort by credit score (descending)
        ranked = sorted(trajectory_credits, key=lambda x: x[1], reverse=True)
        
        logger.info(f"Ranked {len(trajectories)} trajectories by principle credit")
        
        return ranked
    
    def construct_pairs(
        self,
        trajectories: List[Trajectory],
        max_pairs: Optional[int] = None
    ) -> List[PreferencePair]:
        """
        Construct preference pairs from trajectories.
        
        Creates pairs by comparing trajectories with different credit scores.
        Higher credit trajectories are marked as preferred.
        
        Args:
            trajectories: List of trajectories to create pairs from
            max_pairs: Maximum number of pairs to create (None for all possible)
            
        Returns:
            List of preference pairs
        """
        if len(trajectories) < 2:
            logger.warning("Need at least 2 trajectories to construct pairs")
            return []
        
        # Rank trajectories by credit
        ranked = self.rank_trajectories(trajectories)
        
        pairs = []
        
        # Create pairs by comparing higher-credit with lower-credit trajectories
        for i in range(len(ranked)):
            for j in range(i + 1, len(ranked)):
                preferred_traj, preferred_credit = ranked[i]
                dispreferred_traj, dispreferred_credit = ranked[j]
                
                # Check if credit gap is sufficient
                credit_gap = preferred_credit - dispreferred_credit
                if credit_gap >= self.min_credit_gap:
                    pair = PreferencePair(
                        preferred=preferred_traj,
                        dispreferred=dispreferred_traj,
                        preferred_credit=preferred_credit,
                        dispreferred_credit=dispreferred_credit
                    )
                    pairs.append(pair)
                    
                    # Stop if we've reached max_pairs
                    if max_pairs and len(pairs) >= max_pairs:
                        break
            
            if max_pairs and len(pairs) >= max_pairs:
                break
        
        logger.info(f"Constructed {len(pairs)} preference pairs from {len(trajectories)} trajectories")
        
        return pairs
    
    def construct_pairs_stratified(
        self,
        trajectories: List[Trajectory],
        pairs_per_trajectory: int = 3
    ) -> List[PreferencePair]:
        """
        Construct preference pairs with stratified sampling.
        
        For each trajectory, pairs it with multiple lower-ranked trajectories
        to ensure balanced training data.
        
        Args:
            trajectories: List of trajectories
            pairs_per_trajectory: Number of pairs to create per trajectory
            
        Returns:
            List of preference pairs
        """
        if len(trajectories) < 2:
            return []
        
        ranked = self.rank_trajectories(trajectories)
        pairs = []
        
        # For each trajectory (except the last)
        for i in range(len(ranked) - 1):
            preferred_traj, preferred_credit = ranked[i]
            
            # Pair with next few lower-ranked trajectories
            num_pairs = min(pairs_per_trajectory, len(ranked) - i - 1)
            
            for j in range(i + 1, i + 1 + num_pairs):
                dispreferred_traj, dispreferred_credit = ranked[j]
                
                credit_gap = preferred_credit - dispreferred_credit
                if credit_gap >= self.min_credit_gap:
                    pair = PreferencePair(
                        preferred=preferred_traj,
                        dispreferred=dispreferred_traj,
                        preferred_credit=preferred_credit,
                        dispreferred_credit=dispreferred_credit
                    )
                    pairs.append(pair)
        
        logger.info(
            f"Constructed {len(pairs)} stratified preference pairs "
            f"from {len(trajectories)} trajectories"
        )
        
        return pairs



class DPOLoss:
    """
    Computes Direct Preference Optimization (DPO) loss.
    
    DPO optimizes the policy to prefer trajectories with higher principle credit
    by maximizing the log-likelihood ratio between preferred and dispreferred
    trajectories, regularized by KL divergence from a reference model.
    """
    
    def __init__(self, beta: float = 0.1):
        """
        Initialize DPO loss.
        
        Args:
            beta: KL penalty coefficient (higher = stronger regularization)
        """
        self.beta = beta
        logger.info(f"Initialized DPOLoss with beta={beta}")
    
    def compute_sequence_log_prob(
        self,
        model: PolicyModel,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log probability of a sequence under the model.
        
        Args:
            model: Policy model
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Target token IDs [batch_size, seq_len]
            
        Returns:
            Log probabilities [batch_size]
        """
        return model.compute_log_probs(input_ids, attention_mask, labels)
    
    def compute_loss(
        self,
        policy_model: PolicyModel,
        reference_model: PolicyModel,
        preferred_input_ids: torch.Tensor,
        preferred_attention_mask: torch.Tensor,
        preferred_labels: torch.Tensor,
        dispreferred_input_ids: torch.Tensor,
        dispreferred_attention_mask: torch.Tensor,
        dispreferred_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute DPO loss for a batch of preference pairs.
        
        The DPO loss is:
        L = -log(σ(β * (log π_θ(y_w|x) - log π_θ(y_l|x) 
                        - log π_ref(y_w|x) + log π_ref(y_l|x))))
        
        where:
        - y_w = preferred trajectory
        - y_l = dispreferred trajectory
        - π_θ = policy model
        - π_ref = reference model
        - σ = sigmoid function
        - β = KL penalty coefficient
        
        Args:
            policy_model: Current policy model being trained
            reference_model: Reference model (frozen)
            preferred_input_ids: Input IDs for preferred trajectories
            preferred_attention_mask: Attention mask for preferred
            preferred_labels: Labels for preferred trajectories
            dispreferred_input_ids: Input IDs for dispreferred trajectories
            dispreferred_attention_mask: Attention mask for dispreferred
            dispreferred_labels: Labels for dispreferred trajectories
            
        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Compute log probs for preferred trajectories
        policy_preferred_logprobs = self.compute_sequence_log_prob(
            policy_model,
            preferred_input_ids,
            preferred_attention_mask,
            preferred_labels
        )
        
        with torch.no_grad():
            reference_preferred_logprobs = self.compute_sequence_log_prob(
                reference_model,
                preferred_input_ids,
                preferred_attention_mask,
                preferred_labels
            )
        
        # Compute log probs for dispreferred trajectories
        policy_dispreferred_logprobs = self.compute_sequence_log_prob(
            policy_model,
            dispreferred_input_ids,
            dispreferred_attention_mask,
            dispreferred_labels
        )
        
        with torch.no_grad():
            reference_dispreferred_logprobs = self.compute_sequence_log_prob(
                reference_model,
                dispreferred_input_ids,
                dispreferred_attention_mask,
                dispreferred_labels
            )
        
        # Compute log ratio
        # log π_θ(y_w|x) - log π_θ(y_l|x) - log π_ref(y_w|x) + log π_ref(y_l|x)
        policy_logratios = policy_preferred_logprobs - policy_dispreferred_logprobs
        reference_logratios = reference_preferred_logprobs - reference_dispreferred_logprobs
        
        logits = policy_logratios - reference_logratios
        
        # DPO loss: -log(sigmoid(beta * logits))
        # Equivalent to: log(1 + exp(-beta * logits))
        loss = -F.logsigmoid(self.beta * logits).mean()
        
        # Compute metrics
        with torch.no_grad():
            # Implicit reward: beta * (log π_θ - log π_ref)
            preferred_rewards = self.beta * (
                policy_preferred_logprobs - reference_preferred_logprobs
            )
            dispreferred_rewards = self.beta * (
                policy_dispreferred_logprobs - reference_dispreferred_logprobs
            )
            
            # Accuracy: how often does policy prefer the preferred trajectory
            accuracy = (logits > 0).float().mean()
            
            metrics = {
                'loss': loss.item(),
                'accuracy': accuracy.item(),
                'preferred_reward_mean': preferred_rewards.mean().item(),
                'dispreferred_reward_mean': dispreferred_rewards.mean().item(),
                'reward_margin': (preferred_rewards - dispreferred_rewards).mean().item(),
                'logits_mean': logits.mean().item(),
                'logits_std': logits.std().item()
            }
        
        return loss, metrics
    
    def compute_loss_from_trajectories(
        self,
        policy_model: PolicyModel,
        reference_model: PolicyModel,
        preference_pairs: List[PreferencePair],
        tokenizer: Any
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute DPO loss from trajectory preference pairs.
        
        This is a convenience method that handles tokenization and batching.
        
        Args:
            policy_model: Current policy model
            reference_model: Reference model
            preference_pairs: List of preference pairs
            tokenizer: Tokenizer for encoding trajectories
            
        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Convert trajectories to text sequences
        preferred_texts = [
            self._trajectory_to_text(pair.preferred)
            for pair in preference_pairs
        ]
        dispreferred_texts = [
            self._trajectory_to_text(pair.dispreferred)
            for pair in preference_pairs
        ]
        
        # Tokenize
        preferred_encodings = tokenizer(
            preferred_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        dispreferred_encodings = tokenizer(
            dispreferred_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Move to device
        device = policy_model.device
        preferred_input_ids = preferred_encodings['input_ids'].to(device)
        preferred_attention_mask = preferred_encodings['attention_mask'].to(device)
        dispreferred_input_ids = dispreferred_encodings['input_ids'].to(device)
        dispreferred_attention_mask = dispreferred_encodings['attention_mask'].to(device)
        
        # Labels are the same as input_ids for language modeling
        preferred_labels = preferred_input_ids.clone()
        dispreferred_labels = dispreferred_input_ids.clone()
        
        # Compute loss
        return self.compute_loss(
            policy_model=policy_model,
            reference_model=reference_model,
            preferred_input_ids=preferred_input_ids,
            preferred_attention_mask=preferred_attention_mask,
            preferred_labels=preferred_labels,
            dispreferred_input_ids=dispreferred_input_ids,
            dispreferred_attention_mask=dispreferred_attention_mask,
            dispreferred_labels=dispreferred_labels
        )
    
    def _trajectory_to_text(self, trajectory: Trajectory) -> str:
        """
        Convert trajectory to text representation for DPO training.
        
        Args:
            trajectory: Trajectory to convert
            
        Returns:
            Text representation of the trajectory
        """
        # Format trajectory as a sequence of state-action pairs
        text_parts = [f"Task: {trajectory.task}"]
        
        for i, step in enumerate(trajectory.steps):
            state_desc = str(step.state.observation)
            action_desc = step.action.description
            
            if step.hypothesis:
                text_parts.append(f"Step {i+1} Hypothesis: {step.hypothesis}")
            
            text_parts.append(f"Step {i+1} State: {state_desc}")
            text_parts.append(f"Step {i+1} Action: {action_desc}")
            text_parts.append(f"Step {i+1} Reward: {step.reward}")
        
        text_parts.append(f"Final Reward: {trajectory.final_reward}")
        
        return "\n".join(text_parts)



class DPOTrainer:
    """
    Trainer for Policy Model using Direct Preference Optimization.
    
    This class orchestrates the DPO training process:
    1. Constructs preference pairs from trajectories
    2. Computes DPO loss using policy and reference models
    3. Updates policy model weights via gradient descent
    """
    
    def __init__(
        self,
        policy_model: PolicyModel,
        reference_model: Optional[PolicyModel] = None,
        beta: float = 0.1,
        learning_rate: float = 1e-6,
        min_credit_gap: float = 0.1,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 1
    ):
        """
        Initialize DPO trainer.
        
        Args:
            policy_model: Policy model to train
            reference_model: Reference model (if None, uses copy of policy_model)
            beta: KL penalty coefficient for DPO loss
            learning_rate: Learning rate for optimizer
            min_credit_gap: Minimum credit gap for preference pairs
            batch_size: Batch size for training
            gradient_accumulation_steps: Steps to accumulate gradients
        """
        self.policy_model = policy_model
        
        # Create reference model if not provided
        if reference_model is None:
            logger.info("Creating reference model as deep copy of policy model")
            # 🔥 关键修复：创建深拷贝，而不是引用同一个对象
            # 这样 reference model 和 policy model 是独立的
            import copy
            self.reference_model = copy.deepcopy(policy_model)
            logger.info("Reference model created successfully")
        else:
            self.reference_model = reference_model
        
        # Freeze reference model
        try:
            self.reference_model.eval_mode()
            for param in self.reference_model.get_parameters():
                param.requires_grad = False
            logger.info("Reference model frozen successfully")
        except (TypeError, AttributeError):
            # Handle mock objects or models without get_parameters
            logger.warning("Could not freeze reference model parameters")
        
        self.beta = beta
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Initialize components
        self.pair_constructor = PreferencePairConstructor(min_credit_gap=min_credit_gap)
        self.dpo_loss = DPOLoss(beta=beta)
        self.error_handler = ErrorHandler()
        self.checkpoint_state = None  # For checkpoint reversion
        
        # Initialize optimizer
        try:
            self.optimizer = torch.optim.AdamW(
                self.policy_model.get_parameters(),
                lr=learning_rate,
                weight_decay=0.01
            )
        except (TypeError, AttributeError) as e:
            # Handle Mock objects in tests
            logger.warning(f"Could not initialize optimizer: {e}")
            self.optimizer = None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        
        logger.info(
            f"Initialized DPOTrainer with lr={learning_rate}, "
            f"beta={beta}, batch_size={batch_size}"
        )
    
    def train_policy_model(
        self,
        trajectories: List[Trajectory],
        num_epochs: int = 3,
        max_pairs: Optional[int] = None,
        stratified: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train policy model using DPO on trajectory preferences.
        
        Args:
            trajectories: List of trajectories with principle annotations
            num_epochs: Number of training epochs
            max_pairs: Maximum number of preference pairs (None for all)
            stratified: Whether to use stratified pair construction
            
        Returns:
            Dictionary of training metrics over time
        """
        logger.info(f"Starting DPO training with {len(trajectories)} trajectories")
        
        # Construct preference pairs
        if stratified:
            preference_pairs = self.pair_constructor.construct_pairs_stratified(
                trajectories,
                pairs_per_trajectory=3
            )
        else:
            preference_pairs = self.pair_constructor.construct_pairs(
                trajectories,
                max_pairs=max_pairs
            )
        
        if len(preference_pairs) == 0:
            logger.warning("No preference pairs constructed, skipping training")
            return {}
        
        logger.info(f"Constructed {len(preference_pairs)} preference pairs")
        
        # Training metrics
        metrics_history = {
            'loss': [],
            'accuracy': [],
            'preferred_reward': [],
            'dispreferred_reward': [],
            'reward_margin': []
        }
        
        # Set model to training mode
        self.policy_model.train_mode()
        self.reference_model.eval_mode()
        
        # Get tokenizer
        tokenizer = self.policy_model.get_tokenizer()
        
        # Training loop
        for epoch in range(num_epochs):
            self.epoch = epoch
            epoch_metrics = self._train_epoch(
                preference_pairs,
                tokenizer
            )
            
            # Log epoch metrics
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Loss: {epoch_metrics['loss']:.4f}, "
                f"Accuracy: {epoch_metrics['accuracy']:.4f}, "
                f"Reward Margin: {epoch_metrics['reward_margin']:.4f}"
            )
            
            # Store metrics
            for key in metrics_history:
                if key in epoch_metrics:
                    metrics_history[key].append(epoch_metrics[key])
        
        # Set model back to eval mode
        self.policy_model.eval_mode()
        
        logger.info("DPO training completed")
        
        return metrics_history
    
    def _train_epoch(
        self,
        preference_pairs: List[PreferencePair],
        tokenizer: Any
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            preference_pairs: List of preference pairs
            tokenizer: Tokenizer for encoding
            
        Returns:
            Dictionary of epoch metrics
        """
        # Save checkpoint before epoch
        self._save_checkpoint()
        
        total_loss = 0.0
        total_accuracy = 0.0
        total_preferred_reward = 0.0
        total_dispreferred_reward = 0.0
        total_reward_margin = 0.0
        num_batches = 0
        
        # Shuffle pairs
        import random
        shuffled_pairs = preference_pairs.copy()
        random.shuffle(shuffled_pairs)
        
        # Process in batches
        for i in range(0, len(shuffled_pairs), self.batch_size):
            batch_pairs = shuffled_pairs[i:i + self.batch_size]
            
            try:
                # Compute loss for batch
                loss, metrics = self.dpo_loss.compute_loss_from_trajectories(
                    policy_model=self.policy_model,
                    reference_model=self.reference_model,
                    preference_pairs=batch_pairs,
                    tokenizer=tokenizer
                )
                
                # Normalize loss by gradient accumulation steps
                loss = loss / self.gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Update weights if we've accumulated enough gradients
                if (num_batches + 1) % self.gradient_accumulation_steps == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(
                        self.policy_model.get_parameters(),
                        max_norm=1.0
                    )
                    
                    # Check for training divergence
                    gradients = {name: param.grad for name, param in self.policy_model.get_base_model().named_parameters() if param.grad is not None}
                    
                    should_continue = self.error_handler.handle_training_error(
                        loss=metrics['loss'],
                        gradients=gradients,
                        checkpoint_fn=lambda: self._save_checkpoint(),
                        revert_fn=lambda: self._revert_checkpoint()
                    )
                    
                    if not should_continue:
                        logger.warning("Training stopped due to divergence, checkpoint reverted")
                        return {
                            'loss': total_loss / max(num_batches, 1),
                            'accuracy': total_accuracy / max(num_batches, 1),
                            'preferred_reward': total_preferred_reward / max(num_batches, 1),
                            'dispreferred_reward': total_dispreferred_reward / max(num_batches, 1),
                            'reward_margin': total_reward_margin / max(num_batches, 1)
                        }
                    
                    # Optimizer step
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    self.global_step += 1
                
                # Accumulate metrics
                total_loss += metrics['loss']
                total_accuracy += metrics['accuracy']
                total_preferred_reward += metrics['preferred_reward_mean']
                total_dispreferred_reward += metrics['dispreferred_reward_mean']
                total_reward_margin += metrics['reward_margin']
                num_batches += 1
                
            except TrainingDivergenceError as e:
                logger.error(f"Training diverged: {e}")
                self._revert_checkpoint()
                raise
        
        # Compute average metrics
        epoch_metrics = {
            'loss': total_loss / num_batches,
            'accuracy': total_accuracy / num_batches,
            'preferred_reward': total_preferred_reward / num_batches,
            'dispreferred_reward': total_dispreferred_reward / num_batches,
            'reward_margin': total_reward_margin / num_batches
        }
        
        return epoch_metrics
    
    def save_checkpoint(self, checkpoint_path: str) -> None:
        """
        Save training checkpoint.
        
        Args:
            checkpoint_path: Path to save checkpoint
        """
        checkpoint = {
            'policy_model_state': self.policy_model.get_base_model().state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'beta': self.beta,
            'learning_rate': self.learning_rate
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
        """
        checkpoint = torch.load(checkpoint_path)
        
        self.policy_model.get_base_model().load_state_dict(
            checkpoint['policy_model_state']
        )
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    def _save_checkpoint(self):
        """Save current model state for potential reversion."""
        import copy
        logger.debug("Saving training checkpoint")
        
        # Save model and optimizer state (deep copy to avoid reference issues)
        self.checkpoint_state = {
            'policy_model': copy.deepcopy(self.policy_model.get_base_model().state_dict()),
            'optimizer': copy.deepcopy(self.optimizer.state_dict()),
            'global_step': self.global_step,
            'epoch': self.epoch
        }
    
    def _revert_checkpoint(self):
        """Revert model to saved checkpoint state."""
        if self.checkpoint_state is None:
            logger.warning("No checkpoint available to revert to")
            return
        
        logger.info("Reverting to previous checkpoint")
        
        # Restore model and optimizer state
        self.policy_model.get_base_model().load_state_dict(self.checkpoint_state['policy_model'])
        self.optimizer.load_state_dict(self.checkpoint_state['optimizer'])
        self.global_step = self.checkpoint_state['global_step']
        self.epoch = self.checkpoint_state['epoch']
        
        logger.info("Checkpoint restored successfully")
