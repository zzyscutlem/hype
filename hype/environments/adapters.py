"""
Environment adapters for different benchmark environments.

This module provides adapters that translate between the HyPE system's
internal representations (State, Action) and environment-specific formats
for ToolBench, API-Bank, and ALFWorld.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List
from datetime import datetime

from hype.core import State, Action


class EnvironmentAdapter(ABC):
    """
    Base class for environment-specific adapters.
    
    Adapters handle the translation between HyPE's internal representations
    and environment-specific formats. Each environment has different:
    - Observation formats
    - Action APIs
    - Validation rules
    """
    
    @abstractmethod
    def parse_state(self, raw_observation: Any) -> State:
        """
        Convert environment observation to State.
        
        Args:
            raw_observation: Environment-specific observation format
            
        Returns:
            State object with standardized format
            
        Raises:
            ValueError: If observation format is invalid
        """
        pass
    
    @abstractmethod
    def format_action(self, action: Action) -> Any:
        """
        Convert Action to environment-specific format.
        
        Args:
            action: HyPE Action object
            
        Returns:
            Environment-specific action format
            
        Raises:
            ValueError: If action cannot be formatted
        """
        pass
    
    @abstractmethod
    def validate_action(self, action: Action) -> bool:
        """
        Check if action is valid for this environment.
        
        Args:
            action: Action to validate
            
        Returns:
            True if action is valid, False otherwise
        """
        pass


class ToolBenchAdapter(EnvironmentAdapter):
    """
    Adapter for ToolBench environment.
    
    ToolBench is a tool-use benchmark where agents interact with various
    tools through API calls. Actions specify tool names and arguments.
    
    Expected observation format:
        {
            "observation": str,  # Current observation text
            "action_history": List[str],  # Previous actions
            "available_tools": List[str],  # Available tool names
            "tool_descriptions": Dict[str, str]  # Tool documentation
        }
    
    Expected action format:
        {
            "tool_name": str,  # Name of tool to use
            "arguments": Dict[str, Any]  # Tool arguments
        }
    """
    
    def parse_state(self, raw_observation: Any) -> State:
        """
        Parse ToolBench observation into State.
        
        Args:
            raw_observation: Dict with observation, action_history, etc.
            
        Returns:
            State object
            
        Raises:
            ValueError: If observation format is invalid
        """
        if not isinstance(raw_observation, dict):
            raise ValueError(f"ToolBench observation must be dict, got {type(raw_observation)}")
        
        # Extract action history
        history = raw_observation.get("action_history", [])
        if not isinstance(history, list):
            raise ValueError(f"action_history must be list, got {type(history)}")
        
        # Create state with all observation data
        return State(
            observation=raw_observation,
            history=history,
            timestamp=datetime.now()
        )
    
    def format_action(self, action: Action) -> Dict[str, Any]:
        """
        Format Action for ToolBench environment.
        
        Args:
            action: HyPE Action object
            
        Returns:
            Dict with tool_name and arguments
            
        Raises:
            ValueError: If action is missing required fields
        """
        if "tool" not in action.parameters:
            raise ValueError("ToolBench action must have 'tool' parameter")
        
        tool_name = action.parameters["tool"]
        arguments = action.parameters.get("args", {})
        
        if not isinstance(arguments, dict):
            raise ValueError(f"Tool arguments must be dict, got {type(arguments)}")
        
        return {
            "tool_name": tool_name,
            "arguments": arguments
        }
    
    def validate_action(self, action: Action) -> bool:
        """
        Validate ToolBench action.
        
        Args:
            action: Action to validate
            
        Returns:
            True if action is valid
        """
        # Check action type
        if action.type not in ["tool_use", "tool_call"]:
            return False
        
        # Check required parameters
        if "tool" not in action.parameters:
            return False
        
        # Check tool name is non-empty string
        tool_name = action.parameters["tool"]
        if not isinstance(tool_name, str) or not tool_name:
            return False
        
        # Check arguments is dict
        arguments = action.parameters.get("args", {})
        if not isinstance(arguments, dict):
            return False
        
        return True


class APIBankAdapter(EnvironmentAdapter):
    """
    Adapter for API-Bank environment.
    
    API-Bank is a benchmark for API interaction tasks where agents make
    RESTful API calls with specific endpoints, methods, and parameters.
    
    Expected observation format:
        {
            "observation": str,  # Current observation text
            "action_history": List[str],  # Previous actions
            "available_apis": List[Dict],  # Available API specs
            "api_responses": List[Dict]  # Previous API responses
        }
    
    Expected action format:
        {
            "api_name": str,  # API endpoint name
            "method": str,  # HTTP method (GET, POST, etc.)
            "parameters": Dict[str, Any]  # API parameters
        }
    """
    
    def parse_state(self, raw_observation: Any) -> State:
        """
        Parse API-Bank observation into State.
        
        Args:
            raw_observation: Dict with observation, action_history, etc.
            
        Returns:
            State object
            
        Raises:
            ValueError: If observation format is invalid
        """
        if not isinstance(raw_observation, dict):
            raise ValueError(f"API-Bank observation must be dict, got {type(raw_observation)}")
        
        # Extract action history
        history = raw_observation.get("action_history", [])
        if not isinstance(history, list):
            raise ValueError(f"action_history must be list, got {type(history)}")
        
        # Create state with all observation data
        return State(
            observation=raw_observation,
            history=history,
            timestamp=datetime.now()
        )
    
    def format_action(self, action: Action) -> Dict[str, Any]:
        """
        Format Action for API-Bank environment.
        
        Args:
            action: HyPE Action object
            
        Returns:
            Dict with api_name, method, and parameters
            
        Raises:
            ValueError: If action is missing required fields
        """
        if "api_name" not in action.parameters:
            raise ValueError("API-Bank action must have 'api_name' parameter")
        
        if "method" not in action.parameters:
            raise ValueError("API-Bank action must have 'method' parameter")
        
        api_name = action.parameters["api_name"]
        method = action.parameters["method"]
        parameters = action.parameters.get("parameters", {})
        
        if not isinstance(parameters, dict):
            raise ValueError(f"API parameters must be dict, got {type(parameters)}")
        
        return {
            "api_name": api_name,
            "method": method,
            "parameters": parameters
        }
    
    def validate_action(self, action: Action) -> bool:
        """
        Validate API-Bank action.
        
        Args:
            action: Action to validate
            
        Returns:
            True if action is valid
        """
        # Check action type
        if action.type not in ["api_call", "api_request"]:
            return False
        
        # Check required parameters
        if "api_name" not in action.parameters:
            return False
        
        if "method" not in action.parameters:
            return False
        
        # Check api_name is non-empty string
        api_name = action.parameters["api_name"]
        if not isinstance(api_name, str) or not api_name:
            return False
        
        # Check method is valid HTTP method
        method = action.parameters["method"]
        valid_methods = ["GET", "POST", "PUT", "DELETE", "PATCH"]
        if method not in valid_methods:
            return False
        
        # Check parameters is dict
        parameters = action.parameters.get("parameters", {})
        if not isinstance(parameters, dict):
            return False
        
        return True


class ALFWorldAdapter(EnvironmentAdapter):
    """
    Adapter for ALFWorld environment.
    
    ALFWorld is an embodied agent benchmark where agents navigate and
    interact with household environments using text commands.
    
    Expected observation format:
        {
            "observation": str,  # Current observation text
            "action_history": List[str],  # Previous actions
            "inventory": List[str],  # Items in inventory
            "location": str  # Current location
        }
    
    Expected action format:
        {
            "command": str  # Text command (e.g., "go to kitchen", "take apple")
        }
    """
    
    def parse_state(self, raw_observation: Any) -> State:
        """
        Parse ALFWorld observation into State.
        
        Args:
            raw_observation: Dict with observation, action_history, etc.
            
        Returns:
            State object
            
        Raises:
            ValueError: If observation format is invalid
        """
        if not isinstance(raw_observation, dict):
            raise ValueError(f"ALFWorld observation must be dict, got {type(raw_observation)}")
        
        # Extract action history
        history = raw_observation.get("action_history", [])
        if not isinstance(history, list):
            raise ValueError(f"action_history must be list, got {type(history)}")
        
        # Create state with all observation data
        return State(
            observation=raw_observation,
            history=history,
            timestamp=datetime.now()
        )
    
    def format_action(self, action: Action) -> Dict[str, str]:
        """
        Format Action for ALFWorld environment.
        
        Args:
            action: HyPE Action object
            
        Returns:
            Dict with command string
            
        Raises:
            ValueError: If action is missing required fields
        """
        if "command" not in action.parameters:
            raise ValueError("ALFWorld action must have 'command' parameter")
        
        command = action.parameters["command"]
        
        if not isinstance(command, str):
            raise ValueError(f"Command must be string, got {type(command)}")
        
        return {
            "command": command
        }
    
    def validate_action(self, action: Action) -> bool:
        """
        Validate ALFWorld action.
        
        Args:
            action: Action to validate
            
        Returns:
            True if action is valid
        """
        # Check action type
        if action.type not in ["navigation", "interaction", "embodied_action"]:
            return False
        
        # Check required parameters
        if "command" not in action.parameters:
            return False
        
        # Check command is non-empty string
        command = action.parameters["command"]
        if not isinstance(command, str) or not command:
            return False
        
        # Check command starts with valid verb
        valid_verbs = [
            "go", "take", "put", "open", "close", "toggle",
            "examine", "look", "inventory", "use"
        ]
        
        command_lower = command.lower().strip()
        if not any(command_lower.startswith(verb) for verb in valid_verbs):
            return False
        
        return True
