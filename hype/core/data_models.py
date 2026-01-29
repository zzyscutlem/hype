"""
Core data models for the HyPE agent system.

This module defines the fundamental data structures used throughout the system:
- State: Environment state representation
- Action: Agent action representation
- Principle: Reusable knowledge stored in memory
- Trajectory: Complete task execution trajectory
- HypothesisNode: Node in H-MCTS search tree
- TrainingExample: Training data for models
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
import numpy as np


@dataclass
class State:
    """
    Environment state representation.
    
    Attributes:
        observation: Environment-specific observation data
        history: List of previous actions taken
        timestamp: When this state was observed
    """
    observation: Dict[str, Any]
    history: List[str]
    timestamp: datetime
    
    def __post_init__(self):
        """Validate state after initialization."""
        if not isinstance(self.observation, dict):
            raise TypeError("observation must be a dictionary")
        if not isinstance(self.history, list):
            raise TypeError("history must be a list")


@dataclass
class Action:
    """
    Agent action representation.
    
    Attributes:
        type: Action type (e.g., "api_call", "tool_use", "navigation")
        parameters: Action-specific parameters
        description: Natural language description of the action
    """
    type: str
    parameters: Dict[str, Any]
    description: str
    
    def __post_init__(self):
        """Validate action after initialization."""
        if not self.type:
            raise ValueError("action type cannot be empty")
        if not isinstance(self.parameters, dict):
            raise TypeError("parameters must be a dictionary")


@dataclass
class Principle:
    """
    Principle stored in memory.
    
    Attributes:
        id: Unique identifier
        text: Natural language principle description
        embedding: Semantic embedding vector (1024-dim from BGE-large-en)
        credit_score: Accumulated credit indicating principle quality
        application_count: Number of times this principle has been applied
        created_at: When the principle was created
        last_used: When the principle was last used
        source_trajectory_id: Optional ID of the trajectory this came from
    """
    id: str
    text: str
    embedding: np.ndarray
    credit_score: float
    application_count: int
    created_at: datetime
    last_used: datetime
    source_trajectory_id: Optional[str] = None
    
    def __post_init__(self):
        """Validate principle after initialization."""
        if not self.text:
            raise ValueError("principle text cannot be empty")
        if self.embedding.shape != (1024,):
            raise ValueError(f"embedding must be 1024-dimensional, got {self.embedding.shape}")
        if self.credit_score < 0:
            raise ValueError("credit_score cannot be negative")
        if self.application_count < 0:
            raise ValueError("application_count cannot be negative")


@dataclass
class TrajectoryStep:
    """
    Single step in a trajectory.
    
    Attributes:
        state: Current state
        action: Action taken
        reward: Reward received
        next_state: Resulting state after action
        done: Whether the episode is complete
        hypothesis: Optional hypothesis that led to this action
    """
    state: State
    action: Action
    reward: float
    next_state: State
    done: bool
    hypothesis: Optional[str] = None


@dataclass
class Trajectory:
    """
    Complete task execution trajectory.
    
    Attributes:
        id: Unique trajectory identifier
        task: Task description
        steps: Sequence of trajectory steps
        final_reward: Final cumulative reward
        success: Whether the task was completed successfully
        principles_used: Principles applied at each step
    """
    id: str
    task: str
    steps: List[TrajectoryStep]
    final_reward: float
    success: bool
    principles_used: List[List[Principle]] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate trajectory after initialization."""
        if not self.task:
            raise ValueError("task description cannot be empty")
        if len(self.steps) == 0:
            raise ValueError("trajectory must have at least one step")


@dataclass
class HypothesisNode:
    """
    Node in H-MCTS search tree.
    
    Attributes:
        hypothesis: Natural language hypothesis
        state: Environment state at this node
        parent: Parent node in the tree
        children: Child nodes
        visit_count: Number of times this node has been visited
        total_value: Accumulated value from all visits
        principles: Principles retrieved for this hypothesis
        ucb_score: Upper Confidence Bound score for selection
    """
    hypothesis: str
    state: State
    parent: Optional['HypothesisNode']
    children: List['HypothesisNode'] = field(default_factory=list)
    visit_count: int = 0
    total_value: float = 0.0
    principles: List[Principle] = field(default_factory=list)
    ucb_score: float = 0.0
    
    def __post_init__(self):
        """Validate hypothesis node after initialization."""
        if not self.hypothesis:
            raise ValueError("hypothesis cannot be empty")
        if self.visit_count < 0:
            raise ValueError("visit_count cannot be negative")
    
    @property
    def q_value(self) -> float:
        """
        Compute Q-value (average value) for this node.
        
        Returns:
            Average value, or 0 if never visited
        """
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf node."""
        return len(self.children) == 0
    
    def is_root(self) -> bool:
        """Check if this is the root node."""
        return self.parent is None


@dataclass
class TrainingExample:
    """
    Training example for models.
    
    Attributes:
        input_text: Input text for the model
        target: Target output (float for Value Model, str for Policy Model)
        metadata: Additional metadata about this example
    """
    input_text: str
    target: Any  # float for Value Model, str for Policy Model
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate training example after initialization."""
        if not self.input_text:
            raise ValueError("input_text cannot be empty")
