"""
API-Bank environment adapter.

Connects HyPE system with API-Bank API calling tasks.
"""

import os
import json
import random
from typing import List, Dict, Tuple, Any
from datetime import datetime

from hype.core.data_models import State, Action, Trajectory
from .base_adapter import BaseAdapter


class APIBankAdapter(BaseAdapter):
    """Adapter for API-Bank environment."""
    
    def __init__(self, data_path: str = None):
        """
        Initialize API-Bank adapter.
        
        Args:
            data_path: Path to API-Bank data directory
        """
        if data_path is None:
            data_path = "/share/home/202520143336/project/DAMO-ConvAI/api-bank"
        super().__init__(data_path)
        
        self.max_steps = 15
        self.step_count = 0
        self.api_call_count = 0
    
    def load_tasks(self, num_tasks: int) -> List[Dict[str, Any]]:
        """
        Load API-Bank tasks.
        
        Args:
            num_tasks: Number of tasks to load
            
        Returns:
            List of task dictionaries
        """
        tasks = []
        
        # Try to load from actual API-Bank data
        data_file = os.path.join(self.data_path, "data", "test.json")
        if os.path.exists(data_file):
            try:
                with open(data_file, 'r') as f:
                    data = json.load(f)
                    for i, item in enumerate(data[:num_tasks]):
                        tasks.append({
                            'id': f"apibank_{i}",
                            'description': item.get('query', f"API-Bank task {i}"),
                            'data': item,
                            'type': 'apibank'
                        })
                
                if len(tasks) >= num_tasks:
                    return tasks[:num_tasks]
            except Exception as e:
                print(f"Warning: Could not load API-Bank data: {e}")
        
        # Fallback: Create synthetic tasks
        synthetic_tasks = [
            "Book a flight from New York to London for next Monday",
            "Order a pizza with pepperoni and mushrooms for delivery",
            "Schedule a meeting with John for tomorrow at 2 PM",
            "Send an email to team@company.com with project update",
            "Create a calendar event for the quarterly review",
            "Reserve a table for 4 people at Italian restaurant tonight",
            "Update user profile with new phone number",
            "Get list of available hotels in Paris for next week",
            "Cancel subscription for premium service",
            "Transfer $100 to savings account"
        ]
        
        for i in range(num_tasks):
            task_desc = synthetic_tasks[i % len(synthetic_tasks)]
            tasks.append({
                'id': f"apibank_synthetic_{i}",
                'description': task_desc,
                'data': {'query': task_desc, 'synthetic': True},
                'type': 'apibank'
            })
        
        return tasks[:num_tasks]
    
    def task_to_state(self, task_data: Dict[str, Any]) -> State:
        """
        Convert API-Bank task to HyPE State.
        
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
                'available_apis': self._get_available_apis(task_data),
                'api_calls_made': 0,
                'context': task_data.get('data', {})
            },
            history=[],
            timestamp=datetime.now()
        )
    
    def create_environment(self, task_data: Dict[str, Any]) -> Any:
        """
        Create API-Bank environment instance.
        
        Args:
            task_data: Task data
            
        Returns:
            Environment instance
        """
        self.step_count = 0
        self.api_call_count = 0
        return {
            'task': task_data,
            'state': 'initialized',
            'api_calls': [],
            'responses': []
        }
    
    def execute_action(
        self,
        action: Action,
        env: Any,
        current_state: State
    ) -> Tuple[State, float, bool, Dict[str, Any]]:
        """
        Execute action in API-Bank environment.
        
        Args:
            action: Action to execute
            env: Environment instance
            current_state: Current state
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        self.step_count += 1
        
        # Check if action is an API call
        is_api_call = action.type.startswith('api_') or 'call' in action.type.lower()
        if is_api_call:
            self.api_call_count += 1
        
        # Simulate API execution
        api_response = self._simulate_api_call(action, env)
        
        # Update environment
        env['api_calls'].append({
            'action': action.description,
            'type': action.type,
            'parameters': action.parameters
        })
        env['responses'].append(api_response)
        
        # Create next state
        next_state = State(
            observation={
                'task': current_state.observation['task'],
                'task_id': current_state.observation['task_id'],
                'step': self.step_count,
                'last_action': action.description,
                'last_response': api_response,
                'available_apis': current_state.observation['available_apis'],
                'api_calls_made': self.api_call_count,
                'context': current_state.observation['context']
            },
            history=current_state.history + [action.description],
            timestamp=datetime.now()
        )
        
        # Calculate reward
        reward = self._calculate_reward(action, api_response, is_api_call)
        
        # Check if done
        done = self._check_done(env, self.step_count, self.api_call_count)
        
        info = {
            'api_response': api_response,
            'api_calls_made': self.api_call_count,
            'step': self.step_count,
            'is_api_call': is_api_call
        }
        
        return next_state, reward, done, info
    
    def evaluate_success(
        self,
        trajectory: Trajectory,
        task_data: Dict[str, Any]
    ) -> bool:
        """
        Evaluate if API-Bank task was successful.
        
        Args:
            trajectory: Complete trajectory
            task_data: Original task data
            
        Returns:
            True if successful
        """
        if len(trajectory.steps) == 0:
            return False
        
        # Check if made at least one API call
        api_calls = sum(1 for step in trajectory.steps 
                       if 'api' in step.action.type.lower())
        
        if api_calls == 0:
            return False
        
        # Check final reward
        if trajectory.final_reward <= 0:
            return False
        
        # For synthetic tasks, consider successful if made API calls and got positive reward
        if task_data.get('data', {}).get('synthetic', False):
            return api_calls > 0 and trajectory.final_reward > 0.5
        
        return trajectory.success
    
    def _get_available_apis(self, task_data: Dict[str, Any]) -> List[str]:
        """Get list of available APIs for task."""
        return [
            'flight_booking', 'hotel_booking', 'restaurant_reservation',
            'email_service', 'calendar_service', 'payment_service',
            'user_management', 'notification_service', 'search_service',
            'weather_service'
        ]
    
    def _simulate_api_call(self, action: Action, env: Any) -> Dict[str, Any]:
        """Simulate API call execution."""
        api_type = action.type
        
        # Simulate different API responses
        responses = {
            'flight_booking': {'status': 'success', 'booking_id': 'FL12345', 'price': 450},
            'hotel_booking': {'status': 'success', 'booking_id': 'HT67890', 'price': 120},
            'restaurant_reservation': {'status': 'confirmed', 'reservation_id': 'RS11111'},
            'email_service': {'status': 'sent', 'message_id': 'EM22222'},
            'calendar_service': {'status': 'created', 'event_id': 'EV33333'},
            'payment_service': {'status': 'completed', 'transaction_id': 'TX44444'},
            'user_management': {'status': 'updated', 'user_id': 'USR55555'},
            'notification_service': {'status': 'delivered', 'notification_id': 'NT66666'},
            'search_service': {'status': 'success', 'results': ['item1', 'item2']},
            'weather_service': {'status': 'success', 'temperature': 72, 'condition': 'sunny'}
        }
        
        return responses.get(api_type, {'status': 'success', 'message': 'API call completed'})
    
    def _calculate_reward(self, action: Action, response: Dict, is_api_call: bool) -> float:
        """Calculate reward for action."""
        base_reward = 0.1
        
        # Bonus for API calls
        if is_api_call:
            base_reward += 0.3
        
        # Bonus for successful responses
        if response.get('status') in ['success', 'completed', 'confirmed', 'sent', 'created']:
            base_reward += 0.2
        
        # Penalty for too many steps
        if self.step_count > self.max_steps * 0.8:
            base_reward -= 0.1
        
        return base_reward
    
    def _check_done(self, env: Any, step_count: int, api_calls: int) -> bool:
        """Check if task is done."""
        # Done if max steps reached
        if step_count >= self.max_steps:
            return True
        
        # Done if made sufficient API calls
        if api_calls >= 2:
            return True
        
        return False
