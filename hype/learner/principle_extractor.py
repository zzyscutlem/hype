"""
Principle Extraction module for HyPE system.

This module implements:
- Trajectory analysis to identify key decision points
- Principle generation using the base model
- Credit assignment based on reward improvement and application frequency
"""

import logging
import uuid
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import numpy as np

from ..core.data_models import Trajectory, TrajectoryStep, Principle
from ..models.base_model import BaseModelLoader
from ..memory.principle_memory import PrincipleMemory


logger = logging.getLogger(__name__)


class PrincipleExtractor:
    """
    Principle Extractor for learning from successful trajectories.
    
    This class analyzes trajectories to:
    - Identify key decision points
    - Generate natural language principles
    - Assign credit scores based on contribution
    """
    
    def __init__(
        self,
        base_loader: BaseModelLoader,
        principle_memory: PrincipleMemory,
        success_threshold: float = 0.5,
        min_trajectory_length: int = 3,
        reward_improvement_weight: float = 0.7,
        frequency_weight: float = 0.3
    ):
        """
        Initialize Principle Extractor.
        
        Args:
            base_loader: Base model for principle generation
            principle_memory: Principle memory for storing extracted principles
            success_threshold: Minimum reward to consider trajectory successful
            min_trajectory_length: Minimum steps required for extraction
            reward_improvement_weight: Weight for reward improvement in credit
            frequency_weight: Weight for application frequency in credit
        """
        self.base_loader = base_loader
        self.principle_memory = principle_memory
        self.success_threshold = success_threshold
        self.min_trajectory_length = min_trajectory_length
        self.reward_improvement_weight = reward_improvement_weight
        self.frequency_weight = frequency_weight
        
        logger.info(
            f"Initialized PrincipleExtractor with success_threshold={success_threshold}, "
            f"min_trajectory_length={min_trajectory_length}"
        )
    
    def analyze_trajectory(self, trajectory: Trajectory) -> List[Dict[str, Any]]:
        """
        Analyze a trajectory to identify key decision points.
        
        Key decision points are steps where:
        - Reward changes significantly
        - State changes substantially
        - Critical actions are taken
        
        Args:
            trajectory: Trajectory to analyze
            
        Returns:
            List of decision point dictionaries with metadata
        """
        if len(trajectory.steps) < self.min_trajectory_length:
            logger.info(
                f"Trajectory {trajectory.id} too short ({len(trajectory.steps)} steps), "
                f"minimum is {self.min_trajectory_length}"
            )
            return []
        
        decision_points = []
        
        # Analyze each step
        for i, step in enumerate(trajectory.steps):
            # Check if this is a key decision point
            is_key_point = False
            reasons = []
            
            # 1. Significant reward change
            if step.reward != 0:
                is_key_point = True
                reasons.append(f"reward_change={step.reward}")
            
            # 2. Episode termination
            if step.done:
                is_key_point = True
                reasons.append("episode_end")
            
            # 3. First and last steps are always key points
            if i == 0:
                is_key_point = True
                reasons.append("initial_step")
            elif i == len(trajectory.steps) - 1:
                is_key_point = True
                reasons.append("final_step")
            
            # 4. Steps with hypotheses (strategic decisions)
            if step.hypothesis:
                is_key_point = True
                reasons.append("hypothesis_present")
            
            if is_key_point:
                decision_point = {
                    'step_index': i,
                    'step': step,
                    'reasons': reasons,
                    'reward': step.reward,
                    'cumulative_reward': sum(s.reward for s in trajectory.steps[:i+1])
                }
                decision_points.append(decision_point)
        
        logger.info(
            f"Identified {len(decision_points)} key decision points in trajectory {trajectory.id} "
            f"(steps={len(trajectory.steps)})"
        )
        
        return decision_points
    
    def extract_principles(
        self,
        trajectory: Trajectory,
        baseline_reward: float = 0.0
    ) -> List[Principle]:
        """
        Extract principles from a successful trajectory.
        
        Args:
            trajectory: Successful trajectory to extract from
            baseline_reward: Baseline reward for credit computation
            
        Returns:
            List of extracted principles
        """
        # Check if trajectory is successful
        if trajectory.final_reward < self.success_threshold:
            logger.debug(
                f"Trajectory {trajectory.id} not successful "
                f"(reward={trajectory.final_reward} < {self.success_threshold})"
            )
            return []
        
        # Analyze trajectory for key decision points
        decision_points = self.analyze_trajectory(trajectory)
        
        if not decision_points:
            logger.warning(
                f"No key decision points found in trajectory {trajectory.id} "
                f"(steps={len(trajectory.steps)}, min_length={self.min_trajectory_length})"
            )
            return []
        
        logger.info(
            f"Found {len(decision_points)} key decision points in trajectory {trajectory.id}"
        )
        
        # Generate principles from decision points
        principles = []
        
        for dp in decision_points:
            # Generate principle text
            principle_text = self._generate_principle_text(
                trajectory=trajectory,
                decision_point=dp
            )
            
            if not principle_text or len(principle_text.strip()) < 10:
                logger.debug(f"Generated principle too short, skipping")
                continue
            
            # Compute initial credit score
            credit_score = self._compute_initial_credit(
                trajectory=trajectory,
                decision_point=dp,
                baseline_reward=baseline_reward
            )
            
            # Compute embedding
            embedding = self.principle_memory.compute_embedding(principle_text)
            
            # Create principle
            principle = Principle(
                id=str(uuid.uuid4()),
                text=principle_text,
                embedding=embedding,
                credit_score=credit_score,
                application_count=1,  # First application
                created_at=datetime.now(),
                last_used=datetime.now(),
                source_trajectory_id=trajectory.id
            )
            
            principles.append(principle)
        
        logger.info(
            f"Extracted {len(principles)} principles from trajectory {trajectory.id}"
        )
        
        return principles
    
    def _generate_principle_text(
        self,
        trajectory: Trajectory,
        decision_point: Dict[str, Any]
    ) -> str:
        """
        Generate natural language principle text using the base model.
        
        Args:
            trajectory: Source trajectory
            decision_point: Decision point metadata
            
        Returns:
            Natural language principle description
        """
        step = decision_point['step']
        
        # Create prompt for principle generation
        prompt = self._format_principle_generation_prompt(
            task=trajectory.task,
            step=step,
            reward=decision_point['reward'],
            cumulative_reward=decision_point['cumulative_reward'],
            reasons=decision_point['reasons']
        )
        
        logger.debug(f"Generating principle with prompt length: {len(prompt)}")
        
        # Generate principle using base model
        try:
            generated_texts = self.base_loader.generate(
                prompt=prompt,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                num_return_sequences=1
            )
            
            if generated_texts:
                principle_text = generated_texts[0].strip()
                
                # Clean up the generated text
                principle_text = self._clean_principle_text(principle_text)
                
                logger.info(f"Generated principle: {principle_text[:100]}...")
                return principle_text
            else:
                logger.warning("Model generated empty principle")
                return ""
                
        except Exception as e:
            logger.error(f"Failed to generate principle: {e}", exc_info=True)
            return ""
    
    def _format_principle_generation_prompt(
        self,
        task: str,
        step: TrajectoryStep,
        reward: float,
        cumulative_reward: float,
        reasons: List[str]
    ) -> str:
        """
        Format prompt for principle generation.
        
        Args:
            task: Task description
            step: Trajectory step
            reward: Step reward
            cumulative_reward: Cumulative reward up to this step
            reasons: Reasons why this is a key decision point
            
        Returns:
            Formatted prompt
        """
        # Format state
        state_str = str(step.state.observation)
        if len(state_str) > 200:
            state_str = state_str[:200] + "..."
        
        # Format action
        action_str = f"{step.action.type}: {step.action.description}"
        
        # Format hypothesis if present
        hypothesis_str = step.hypothesis if step.hypothesis else "None"
        
        prompt = f"""Task: {task}

State: {state_str}

Hypothesis: {hypothesis_str}

Action Taken: {action_str}

Outcome:
- Step Reward: {reward}
- Cumulative Reward: {cumulative_reward}
- Key Decision: {', '.join(reasons)}

Based on this successful decision, generate a reusable principle that captures the strategy used.
The principle should be:
1. General enough to apply to similar situations
2. Specific enough to be actionable
3. Written as "When [condition], then [action]" or "To [goal], [strategy]"

Principle:"""
        
        return prompt
    
    def _clean_principle_text(self, text: str) -> str:
        """
        Clean and format generated principle text.
        
        Args:
            text: Raw generated text
            
        Returns:
            Cleaned principle text
        """
        # Strip leading/trailing whitespace first
        text = text.strip()
        
        # Remove common prefixes
        prefixes_to_remove = [
            "Principle:",
            "The principle is:",
            "Strategy:",
            "Rule:",
        ]
        
        for prefix in prefixes_to_remove:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        
        # Remove quotes
        text = text.strip('"\'')
        
        # Ensure first letter is capitalized
        if text:
            text = text[0].upper() + text[1:]
        
        # Remove trailing punctuation if not sentence-ending
        if text and text[-1] in [',', ';', ':']:
            text = text[:-1]
        
        # Add period if missing
        if text and text[-1] not in ['.', '!', '?']:
            text = text + '.'
        
        return text
    
    def _compute_initial_credit(
        self,
        trajectory: Trajectory,
        decision_point: Dict[str, Any],
        baseline_reward: float
    ) -> float:
        """
        Compute initial credit score for a principle.
        
        Credit is based on:
        - Reward improvement over baseline
        - Position in trajectory (later decisions weighted more)
        
        Args:
            trajectory: Source trajectory
            decision_point: Decision point metadata
            baseline_reward: Baseline reward for comparison
            
        Returns:
            Initial credit score
        """
        # Reward improvement component
        reward_improvement = trajectory.final_reward - baseline_reward
        reward_improvement = max(0.0, reward_improvement)  # Ensure non-negative
        
        # Normalize to [0, 1] range (assuming rewards in [0, 1])
        normalized_improvement = min(1.0, reward_improvement)
        
        # Position weight (later decisions more important)
        step_index = decision_point['step_index']
        total_steps = len(trajectory.steps)
        position_weight = (step_index + 1) / total_steps
        
        # Combine components
        credit = (
            self.reward_improvement_weight * normalized_improvement +
            self.frequency_weight * position_weight
        )
        
        # Ensure credit is in reasonable range [0, 1]
        credit = max(0.0, min(1.0, credit))
        
        logger.debug(
            f"Computed initial credit: {credit:.3f} "
            f"(improvement={normalized_improvement:.3f}, position={position_weight:.3f})"
        )
        
        return credit
    
    def update_principle_credit(
        self,
        principle_id: str,
        reward_improvement: float,
        application_frequency: int = 1
    ) -> None:
        """
        Update credit score for an existing principle.
        
        Args:
            principle_id: ID of principle to update
            reward_improvement: Reward improvement from this application
            application_frequency: Number of times applied (default 1)
        """
        # Compute credit delta
        normalized_improvement = max(0.0, min(1.0, reward_improvement))
        
        # Frequency bonus (diminishing returns)
        frequency_bonus = np.log1p(application_frequency) / 10.0
        
        credit_delta = (
            self.reward_improvement_weight * normalized_improvement +
            self.frequency_weight * frequency_bonus
        )
        
        # Update in memory
        try:
            self.principle_memory.update_credit(principle_id, credit_delta)
            logger.debug(
                f"Updated principle {principle_id} credit by {credit_delta:.3f}"
            )
        except Exception as e:
            logger.error(f"Failed to update principle credit: {e}")
    
    def extract_and_insert(
        self,
        trajectories: List[Trajectory],
        baseline_reward: float = 0.0
    ) -> int:
        """
        Extract principles from multiple trajectories and insert into memory.
        
        Args:
            trajectories: List of trajectories to extract from
            baseline_reward: Baseline reward for credit computation
            
        Returns:
            Number of new principles inserted (not counting merged duplicates)
        """
        total_inserted = 0
        total_extracted = 0
        
        for trajectory in trajectories:
            # Extract principles
            principles = self.extract_principles(trajectory, baseline_reward)
            total_extracted += len(principles)
            
            # Insert into memory (with deduplication)
            for principle in principles:
                try:
                    is_new = self.principle_memory.insert(principle)
                    if is_new:
                        total_inserted += 1
                except Exception as e:
                    logger.error(
                        f"Failed to insert principle from trajectory {trajectory.id}: {e}"
                    )
        
        logger.info(
            f"Extracted {total_extracted} principles from {len(trajectories)} trajectories, "
            f"inserted {total_inserted} new principles (rest merged with duplicates)"
        )
        
        return total_inserted
