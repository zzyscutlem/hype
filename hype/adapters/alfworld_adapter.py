"""
ALFWorld environment adapter.

Connects HyPE system with ALFWorld embodied AI tasks.
"""

import os
import json
import random
from typing import List, Dict, Tuple, Any
from datetime import datetime

from hype.core.data_models import State, Action, Trajectory
from .base_adapter import BaseAdapter


class ALFWorldAdapter(BaseAdapter):
    """Adapter for ALFWorld environment."""
    
    def __init__(self, data_path: str = None):
        """
        Initialize ALFWorld adapter.
        
        Args:
            data_path: Path to ALFWorld data directory
        """
        if data_path is None:
            data_path = "/share/home/202520143336/project/alfworld"
        super().__init__(data_path)
        
        self.max_steps = 30
        self.step_count = 0
    
    def load_tasks(self, num_tasks: int) -> List[Dict[str, Any]]:
        """
        Load ALFWorld tasks.
        
        Args:
            num_tasks: Number of tasks to load
            
        Returns:
            List of task dictionaries
        """
        tasks = []
        
        # Try to load from actual ALFWorld data
        data_dir = os.path.join(self.data_path, "alfworld", "data")
        if os.path.exists(data_dir):
            try:
                # Look for task files
                for root, dirs, files in os.walk(data_dir):
                    json_files = [f for f in files if f.endswith('.json')]
                    for i, file in enumerate(json_files[:num_tasks]):
                        with open(os.path.join(root, file), 'r') as f:
                            data = json.load(f)
                            tasks.append({
                                'id': f"alfworld_{i}",
                                'description': data.get('task', f"ALFWorld task {i}"),
                                'data': data,
                                'type': 'alfworld'
                            })
                    
                    if len(tasks) >= num_tasks:
                        break
                
                if len(tasks) >= num_tasks:
                    return tasks[:num_tasks]
            except Exception as e:
                print(f"Warning: Could not load ALFWorld data: {e}")
        
        # Fallback: Create synthetic tasks
        synthetic_tasks = [
            "Put a clean mug in the coffee machine",
            "Place a heated apple slice in the garbage bin",
            "Put a cool tomato in the microwave",
            "Place a clean plate on the dining table",
            "Put a heated potato in the refrigerator",
            "Place a clean knife in the drawer",
            "Put a cool lettuce in the garbage bin",
            "Place a heated egg in the sink",
            "Put a clean bowl on the counter",
            "Place a cool bread in the refrigerator"
        ]
        
        for i in range(num_tasks):
            task_desc = synthetic_tasks[i % len(synthetic_tasks)]
            tasks.append({
                'id': f"alfworld_synthetic_{i}",
                'description': task_desc,
                'data': {'task': task_desc, 'synthetic': True},
                'type': 'alfworld'
            })
        
        return tasks[:num_tasks]
    
    def task_to_state(self, task_data: Dict[str, Any]) -> State:
        """
        Convert ALFWorld task to HyPE State.
        
        Args:
            task_data: Task data
            
        Returns:
            Initial State for the task
        """
        return State(
            observation={
                'task': task_data['description'],
                'task_id': task_data['id'],
                'step': 0,
                'location': 'start',
                'inventory': [],
                'visible_objects': self._get_initial_objects(task_data),
                'available_actions': self._get_available_actions(),
                'context': task_data.get('data', {})
            },
            history=[],
            timestamp=datetime.now()
        )
    
    def create_environment(self, task_data: Dict[str, Any]) -> Any:
        """
        Create ALFWorld environment instance.
        
        Args:
            task_data: Task data
            
        Returns:
            Environment instance
        """
        self.step_count = 0
        return {
            'task': task_data,
            'state': 'initialized',
            'location': 'start',
            'inventory': [],
            'actions_taken': [],
            'object_states': self._initialize_object_states(task_data)
        }
    
    def execute_action(
        self,
        action: Action,
        env: Any,
        current_state: State
    ) -> Tuple[State, float, bool, Dict[str, Any]]:
        """
        Execute action in ALFWorld environment.
        
        Args:
            action: Action to execute
            env: Environment instance
            current_state: Current state
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        self.step_count += 1
        
        # Simulate action execution
        action_result = self._simulate_action(action, env, current_state)
        
        # Update environment
        env['actions_taken'].append(action.description)
        self._update_environment_state(env, action, action_result)
        
        # Create next state
        next_state = State(
            observation={
                'task': current_state.observation['task'],
                'task_id': current_state.observation['task_id'],
                'step': self.step_count,
                'location': env['location'],
                'inventory': env['inventory'].copy(),
                'last_action': action.description,
                'last_result': action_result,
                'visible_objects': self._get_visible_objects(env),
                'available_actions': self._get_available_actions(),
                'context': current_state.observation['context']
            },
            history=current_state.history + [action.description],
            timestamp=datetime.now()
        )
        
        # Calculate reward
        reward = self._calculate_reward(action, action_result, env)
        
        # Check if done
        done = self._check_done(env, self.step_count, action_result)
        
        info = {
            'action_result': action_result,
            'location': env['location'],
            'inventory': env['inventory'],
            'step': self.step_count
        }
        
        return next_state, reward, done, info
    
    def evaluate_success(
        self,
        trajectory: Trajectory,
        task_data: Dict[str, Any]
    ) -> bool:
        """
        Evaluate if ALFWorld task was successful.
        
        Args:
            trajectory: Complete trajectory
            task_data: Original task data
            
        Returns:
            True if successful
        """
        if len(trajectory.steps) == 0:
            return False
        
        # Check if task completed successfully
        last_step = trajectory.steps[-1]
        
        # Look for success indicators in the last action result
        if hasattr(last_step, 'reward') and last_step.reward > 0.5:
            return True
        
        # Check final reward
        if trajectory.final_reward > 1.0:
            return True
        
        # For synthetic tasks, use heuristics
        if task_data.get('data', {}).get('synthetic', False):
            # Consider successful if took reasonable number of actions and got positive reward
            return (len(trajectory.steps) >= 3 and 
                   len(trajectory.steps) < self.max_steps and
                   trajectory.final_reward > 0.5)
        
        return trajectory.success
    
    def _get_initial_objects(self, task_data: Dict[str, Any]) -> List[str]:
        """Get initial visible objects."""
        # Extract objects from task description
        task = task_data['description'].lower()
        objects = []
        
        common_objects = ['mug', 'apple', 'tomato', 'plate', 'potato', 
                         'knife', 'lettuce', 'egg', 'bowl', 'bread']
        
        for obj in common_objects:
            if obj in task:
                objects.append(obj)
        
        return objects if objects else ['object']
    
    def _get_available_actions(self) -> List[str]:
        """Get list of available actions."""
        return [
            'go to', 'take', 'put', 'open', 'close',
            'toggle', 'clean', 'heat', 'cool', 'examine'
        ]
    
    def _initialize_object_states(self, task_data: Dict[str, Any]) -> Dict[str, Dict]:
        """Initialize object states."""
        objects = self._get_initial_objects(task_data)
        states = {}
        
        for obj in objects:
            states[obj] = {
                'location': 'counter',
                'state': 'normal',
                'temperature': 'room',
                'clean': False
            }
        
        return states
    
    def _simulate_action(self, action: Action, env: Any, state: State) -> str:
        """Simulate action execution."""
        action_type = action.type.lower()
        
        # Simple simulation based on action type
        if 'go' in action_type or 'move' in action_type:
            return "Moved to new location"
        elif 'take' in action_type or 'pick' in action_type:
            return "Picked up object"
        elif 'put' in action_type or 'place' in action_type:
            return "Placed object"
        elif 'open' in action_type:
            return "Opened container"
        elif 'close' in action_type:
            return "Closed container"
        elif 'clean' in action_type:
            return "Cleaned object"
        elif 'heat' in action_type:
            return "Heated object"
        elif 'cool' in action_type:
            return "Cooled object"
        else:
            return "Action executed"
    
    def _update_environment_state(self, env: Any, action: Action, result: str):
        """Update environment state based on action."""
        action_type = action.type.lower()
        
        # Update location
        if 'go' in action_type:
            locations = ['kitchen', 'living room', 'bedroom', 'bathroom']
            env['location'] = random.choice(locations)
        
        # Update inventory
        if 'take' in action_type and len(env['inventory']) < 3:
            obj = action.parameters.get('object', 'object')
            if obj not in env['inventory']:
                env['inventory'].append(obj)
        elif 'put' in action_type and env['inventory']:
            env['inventory'].pop()
    
    def _get_visible_objects(self, env: Any) -> List[str]:
        """Get currently visible objects."""
        # Objects visible depend on location
        all_objects = list(env['object_states'].keys())
        return all_objects[:3]  # Simplified: show first 3 objects
    
    def _calculate_reward(self, action: Action, result: str, env: Any) -> float:
        """Calculate reward for action."""
        base_reward = 0.1
        
        # Bonus for productive actions
        if any(word in action.type.lower() for word in ['take', 'put', 'clean', 'heat', 'cool']):
            base_reward += 0.2
        
        # Bonus for successful results
        if 'success' in result.lower() or 'completed' in result.lower():
            base_reward += 0.3
        
        # Penalty for too many steps
        if self.step_count > self.max_steps * 0.8:
            base_reward -= 0.1
        
        return base_reward
    
    def _check_done(self, env: Any, step_count: int, action_result: str) -> bool:
        """Check if task is done."""
        # Done if max steps reached
        if step_count >= self.max_steps:
            return True
        
        # Done if task appears complete (simplified heuristic)
        if len(env['actions_taken']) >= 5:
            return True
        
        return False
