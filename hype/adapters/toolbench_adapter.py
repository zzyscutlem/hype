"""
ToolBench environment adapter.

Connects HyPE system with ToolBench tool-use tasks.
"""

import os
import json
import random
from typing import List, Dict, Tuple, Any
from datetime import datetime

from hype.core.data_models import State, Action, Trajectory
from .base_adapter import BaseAdapter


class ToolBenchAdapter(BaseAdapter):
    """Adapter for ToolBench environment."""
    
    def __init__(self, data_path: str = None):
        """
        Initialize ToolBench adapter.
        
        Args:
            data_path: Path to ToolBench data directory
        """
        if data_path is None:
            data_path = "/share/home/202520143336/project/ToolBench/data_example"
        super().__init__(data_path)
        
        self.max_steps = 20  # Maximum steps per task
        self.step_count = 0
    
    def load_tasks(self, num_tasks: int) -> List[Dict[str, Any]]:
        """
        Load ToolBench tasks.
        
        For now, creates synthetic tasks. In production, would load from
        actual ToolBench dataset.
        
        Args:
            num_tasks: Number of tasks to load
            
        Returns:
            List of task dictionaries
        """
        tasks = []
        
        # Try to load from actual data if available
        data_dir = os.path.join(self.data_path, "toolenv")
        if os.path.exists(data_dir):
            # Load real tasks from ToolBench dataset
            try:
                files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
                if files:
                    for i, file in enumerate(files[:num_tasks]):
                        with open(os.path.join(data_dir, file), 'r') as f:
                            data = json.load(f)
                            tasks.append({
                                'id': f"toolbench_{i}",
                                'description': data.get('query', f"ToolBench task {i}"),
                                'data': data,
                                'type': 'toolbench'
                            })
                    
                    if len(tasks) >= num_tasks:
                        return tasks[:num_tasks]
            except Exception as e:
                print(f"Warning: Could not load ToolBench data: {e}")
        
        # Fallback: Create synthetic tasks
        synthetic_tasks = [
            "Use the weather API to get the current temperature in New York",
            "Search for the latest news about artificial intelligence",
            "Calculate the sum of numbers from 1 to 100 using a calculator tool",
            "Translate 'Hello, how are you?' to Spanish using a translation tool",
            "Get the current stock price of Apple Inc.",
            "Find the capital city of France using a geography tool",
            "Convert 100 USD to EUR using a currency converter",
            "Get the definition of the word 'serendipity' from a dictionary",
            "Find the distance between New York and Los Angeles",
            "Get the current time in Tokyo, Japan"
        ]
        
        for i in range(num_tasks):
            task_desc = synthetic_tasks[i % len(synthetic_tasks)]
            tasks.append({
                'id': f"toolbench_synthetic_{i}",
                'description': task_desc,
                'data': {'query': task_desc, 'synthetic': True},
                'type': 'toolbench'
            })
        
        return tasks[:num_tasks]
    
    def task_to_state(self, task_data: Dict[str, Any]) -> State:
        """
        Convert ToolBench task to HyPE State.
        
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
                'available_tools': self._get_available_tools(task_data),
                'context': task_data.get('data', {})
            },
            history=[],
            timestamp=datetime.now()
        )
    
    def create_environment(self, task_data: Dict[str, Any]) -> Any:
        """
        Create ToolBench environment instance.
        
        Args:
            task_data: Task data
            
        Returns:
            Environment instance (simplified for now)
        """
        self.step_count = 0
        return {
            'task': task_data,
            'state': 'initialized',
            'tools_used': [],
            'results': []
        }
    
    def execute_action(
        self,
        action: Action,
        env: Any,
        current_state: State
    ) -> Tuple[State, float, bool, Dict[str, Any]]:
        """
        Execute action in ToolBench environment.
        
        Args:
            action: Action to execute
            env: Environment instance
            current_state: Current state
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        self.step_count += 1
        
        # Simulate tool execution
        tool_result = self._simulate_tool_execution(action, env)
        
        # Update environment
        env['tools_used'].append(action.type)
        env['results'].append(tool_result)
        
        # Create next state
        next_state = State(
            observation={
                'task': current_state.observation['task'],
                'task_id': current_state.observation['task_id'],
                'step': self.step_count,
                'last_action': action.description,
                'last_result': tool_result,
                'available_tools': current_state.observation['available_tools'],
                'context': current_state.observation['context']
            },
            history=current_state.history + [action.description],
            timestamp=datetime.now()
        )
        
        # Calculate reward (simplified)
        reward = self._calculate_reward(action, tool_result, env)
        
        # Check if done
        done = self._check_done(env, self.step_count)
        
        info = {
            'tool_result': tool_result,
            'tools_used': len(env['tools_used']),
            'step': self.step_count
        }
        
        return next_state, reward, done, info
    
    def evaluate_success(
        self,
        trajectory: Trajectory,
        task_data: Dict[str, Any]
    ) -> bool:
        """
        Evaluate if ToolBench task was successful.
        
        Args:
            trajectory: Complete trajectory
            task_data: Original task data
            
        Returns:
            True if successful
        """
        # For synthetic tasks, consider successful if:
        # 1. Trajectory has steps
        # 2. Final reward is positive
        # 3. Task completed (not just max steps reached)
        
        if len(trajectory.steps) == 0:
            return False
        
        if trajectory.final_reward <= 0:
            return False
        
        # Check if task actually completed vs just ran out of steps
        last_step = trajectory.steps[-1]
        if last_step.done and self.step_count < self.max_steps:
            return True
        
        # For synthetic tasks, use heuristics
        if task_data.get('data', {}).get('synthetic', False):
            # Consider successful if used at least one tool and got positive reward
            return trajectory.final_reward > 0.5
        
        return trajectory.success
    
    def _get_available_tools(self, task_data: Dict[str, Any]) -> List[str]:
        """Get list of available tools for task."""
        # Default tool set
        return [
            'search', 'calculator', 'translator', 'weather',
            'stock_price', 'geography', 'currency_converter',
            'dictionary', 'distance_calculator', 'time_zone'
        ]
    
    def _simulate_tool_execution(self, action: Action, env: Any) -> str:
        """Simulate tool execution (placeholder)."""
        tool_type = action.type
        
        # Simple simulation based on tool type
        simulations = {
            'search': 'Found relevant information',
            'calculator': 'Calculation result: 42',
            'translator': 'Translation: Hola, ¿cómo estás?',
            'weather': 'Temperature: 72°F, Sunny',
            'stock_price': 'Current price: $150.25',
            'geography': 'Capital: Paris',
            'currency_converter': 'Converted amount: 85.50 EUR',
            'dictionary': 'Definition: The occurrence of events by chance',
            'distance_calculator': 'Distance: 2,789 miles',
            'time_zone': 'Current time: 14:30 JST'
        }
        
        return simulations.get(tool_type, f'Executed {tool_type}')
    
    def _calculate_reward(self, action: Action, result: str, env: Any) -> float:
        """Calculate reward for action."""
        # Simple reward: positive for tool use, bonus for completion
        base_reward = 0.1
        
        # Bonus if result looks successful
        if 'error' not in result.lower() and 'failed' not in result.lower():
            base_reward += 0.2
        
        # Penalty for too many steps
        if self.step_count > self.max_steps * 0.8:
            base_reward -= 0.1
        
        return base_reward
    
    def _check_done(self, env: Any, step_count: int) -> bool:
        """Check if task is done."""
        # Done if max steps reached or task appears complete
        if step_count >= self.max_steps:
            return True
        
        # Simple heuristic: done if used multiple tools
        if len(env['tools_used']) >= 3:
            return True
        
        return False
