"""
Base adapter for environment integration.

Defines the interface that all environment adapters must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any, Optional
from datetime import datetime
import numpy as np

from hype.core.data_models import State, Action, Trajectory, TrajectoryStep


class BaseAdapter(ABC):
    """
    Base class for environment adapters.
    
    Each environment (ToolBench, API-Bank, ALFWorld) needs an adapter
    that implements these methods to connect with the HyPE system.
    """
    
    def __init__(self, data_path: str = None):
        """
        Initialize adapter.
        
        Args:
            data_path: Path to environment data/dataset
        """
        self.data_path = data_path
        self.current_env = None
    
    @abstractmethod
    def load_tasks(self, num_tasks: int) -> List[Dict[str, Any]]:
        """
        Load tasks from the environment's dataset.
        
        Args:
            num_tasks: Number of tasks to load
            
        Returns:
            List of task dictionaries with environment-specific data
        """
        pass
    
    @abstractmethod
    def task_to_state(self, task_data: Dict[str, Any]) -> State:
        """
        Convert environment task to HyPE State.
        
        Args:
            task_data: Task data from environment
            
        Returns:
            HyPE State object representing the initial task state
        """
        pass
    
    @abstractmethod
    def create_environment(self, task_data: Dict[str, Any]) -> Any:
        """
        Create/initialize environment instance for a task.
        
        Args:
            task_data: Task data
            
        Returns:
            Environment instance
        """
        pass
    
    @abstractmethod
    def execute_action(
        self, 
        action: Action, 
        env: Any,
        current_state: State
    ) -> Tuple[State, float, bool, Dict[str, Any]]:
        """
        Execute action in the environment.
        
        Args:
            action: HyPE Action to execute
            env: Environment instance
            current_state: Current state
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        pass
    
    @abstractmethod
    def evaluate_success(
        self, 
        trajectory: Trajectory,
        task_data: Dict[str, Any]
    ) -> bool:
        """
        Evaluate if task was completed successfully.
        
        Args:
            trajectory: Complete trajectory of task execution
            task_data: Original task data
            
        Returns:
            True if task was successful, False otherwise
        """
        pass
    
    def action_to_env_format(self, action: Action) -> Any:
        """
        Convert HyPE Action to environment-specific format.
        
        Args:
            action: HyPE Action
            
        Returns:
            Environment-specific action format
        """
        # Default implementation - override if needed
        return action.description
    
    def get_task_description(self, task_data: Dict[str, Any]) -> str:
        """
        Extract task description from task data.
        
        Args:
            task_data: Task data
            
        Returns:
            Human-readable task description
        """
        return task_data.get('description', str(task_data))
