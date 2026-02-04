"""
HyPE System orchestrator for online inference workflow.

This module implements the HyPESystem class which coordinates all components
for online task execution:
- Principle Memory for knowledge retrieval
- H-MCTS for hypothesis-driven planning
- Policy Model for action generation
- SR-Adapt for action validation and correction
- Trajectory collection for offline learning

The system implements the complete online inference workflow:
1. Retrieve relevant principles from memory
2. Plan using H-MCTS to generate best hypothesis
3. Instantiate hypothesis into concrete action
4. Validate and potentially correct action using SR-Adapt
5. Execute action in environment
6. Collect trajectory data for offline evolution
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import uuid

from .core.data_models import (
    State, Action, Principle, Trajectory, TrajectoryStep, HypothesisNode
)
from .core.config import ModelConfig, PrincipleMemoryConfig
from .memory.principle_memory import PrincipleMemory
from .planner.hmcts import HMCTS
from .models.policy_model import PolicyModel
from .models.value_model import ValueModel
from .executor.sr_adapt import SRAdapt
from .environments.adapters import EnvironmentAdapter
from .learner.principle_extractor import PrincipleExtractor
from .learner.dpvd import DPVD, ReplayBuffer, TrainingTrigger
from .learner.dpo_trainer import DPOTrainer


logger = logging.getLogger(__name__)


class HyPESystem:
    """
    HyPE System orchestrator for online inference.
    
    This class coordinates all components to execute tasks through:
    - Principle retrieval from memory
    - Hypothesis-driven planning with H-MCTS
    - Action instantiation with Policy Model
    - Semantic validation with SR-Adapt
    - Environment execution
    - Trajectory collection
    
    Attributes:
        config: Model configuration
        memory_config: Principle memory configuration
        principle_memory: Principle memory for knowledge storage/retrieval
        policy_model: Policy model for hypothesis and action generation
        value_model: Value model for hypothesis evaluation
        hmcts: H-MCTS planner
        sr_adapt: SR-Adapt validator
        trajectories: Collected trajectories for offline learning
    """
    
    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        memory_config: Optional[PrincipleMemoryConfig] = None,
        hmcts_budget: int = 50,
        hmcts_exploration_constant: float = 1.414,
        hmcts_max_depth: int = 5,
        alignment_threshold: float = 0.7,
        max_steps_per_task: int = 20,
        # Offline evolution parameters
        replay_buffer_size: int = 10000,
        training_trigger_threshold: int = 100,
        min_trajectories_for_training: int = 3,  # 🔥 从 10 改为 3，允许更早触发训练
        dpvd_beta: float = 0.1,
        dpvd_learning_rate: float = 1e-5,
        dpvd_batch_size: int = 1,  # 🔥 从 32 改为 1 以避免 CUDA OOM
        dpvd_epochs: int = 3,
        dpo_beta: float = 0.1,
        dpo_learning_rate: float = 1e-6,
        dpo_batch_size: int = 4,
        dpo_epochs: int = 3,
        principle_success_threshold: float = 0.5
    ):
        """
        Initialize HyPE System.
        
        Args:
            config: Model configuration
            memory_config: Principle memory configuration
            hmcts_budget: Number of MCTS iterations per planning step
            hmcts_exploration_constant: UCB exploration constant
            hmcts_max_depth: Maximum tree depth for H-MCTS
            alignment_threshold: Minimum alignment score for SR-Adapt
            max_steps_per_task: Maximum steps per task execution
            replay_buffer_size: Maximum size of replay buffer
            training_trigger_threshold: Buffer size threshold to trigger training
            min_trajectories_for_training: Minimum trajectories required for training
            dpvd_beta: Principle weight coefficient for dense rewards
            dpvd_learning_rate: Learning rate for Value Model training
            dpvd_batch_size: Batch size for Value Model training
            dpvd_epochs: Number of epochs for Value Model training
            dpo_beta: KL penalty coefficient for DPO
            dpo_learning_rate: Learning rate for Policy Model training
            dpo_batch_size: Batch size for Policy Model training
            dpo_epochs: Number of epochs for Policy Model training
            principle_success_threshold: Minimum reward to extract principles
        """
        self.config = config or ModelConfig()
        self.memory_config = memory_config or PrincipleMemoryConfig()
        self.hmcts_budget = hmcts_budget
        self.max_steps_per_task = max_steps_per_task
        
        # Initialize components (lazy loading)
        self.principle_memory = None
        self.policy_model = None
        self.value_model = None
        self.hmcts = None
        self.sr_adapt = None
        
        # Offline evolution components
        self.principle_extractor = None
        self.dpvd = None
        self.dpo_trainer = None
        self.replay_buffer = None
        self.training_trigger = None
        
        # Store configuration for component initialization
        self._hmcts_exploration_constant = hmcts_exploration_constant
        self._hmcts_max_depth = hmcts_max_depth
        self._alignment_threshold = alignment_threshold
        
        # Offline evolution configuration
        self._replay_buffer_size = replay_buffer_size
        self._training_trigger_threshold = training_trigger_threshold
        self._min_trajectories_for_training = min_trajectories_for_training
        self._dpvd_beta = dpvd_beta
        self._dpvd_learning_rate = dpvd_learning_rate
        self._dpvd_batch_size = dpvd_batch_size
        self._dpvd_epochs = dpvd_epochs
        self._dpo_beta = dpo_beta
        self._dpo_learning_rate = dpo_learning_rate
        self._dpo_batch_size = dpo_batch_size
        self._dpo_epochs = dpo_epochs
        self._principle_success_threshold = principle_success_threshold
        
        # Trajectory collection
        self.trajectories: List[Trajectory] = []
        
        # System state
        self._initialized = False
        self._evolution_enabled = False
        
        logger.info(
            f"Initialized HyPESystem with hmcts_budget={hmcts_budget}, "
            f"alignment_threshold={alignment_threshold}, "
            f"max_steps={max_steps_per_task}, "
            f"replay_buffer_size={replay_buffer_size}, "
            f"training_threshold={training_trigger_threshold}"
        )
    
    def initialize(self, enable_evolution: bool = True):
        """
        Initialize all system components.
        
        This method:
        1. Connects to Principle Memory
        2. Loads Policy Model
        3. Loads Value Model
        4. Initializes H-MCTS planner
        5. Initializes SR-Adapt validator
        6. (Optional) Initializes offline evolution components
        
        Args:
            enable_evolution: Whether to initialize offline evolution components
        
        Raises:
            RuntimeError: If initialization fails
        """
        if self._initialized:
            logger.warning("System already initialized")
            return
        
        try:
            logger.info("Initializing HyPE System components...")
            print("\n" + "="*70, flush=True)
            print("🚀 Initializing HyPE System Components", flush=True)
            print("="*70, flush=True)
            
            # 1. Initialize Principle Memory
            logger.info("Step 1/5: Connecting to Principle Memory")
            print("\n📋 Step 1/5: Connecting to Principle Memory", flush=True)
            self.principle_memory = PrincipleMemory(self.memory_config)
            print("   ⏳ Connecting to Milvus...", flush=True)
            self.principle_memory.connect()
            stats = self.principle_memory.get_stats()
            logger.info(f"Principle Memory connected: {stats['num_principles']} principles")
            print(f"   ✅ Connected ({stats['num_principles']} principles stored)", flush=True)
            
            # 2. Load Policy Model
            logger.info("Step 2/5: Loading Policy Model")
            print("\n🤖 Step 2/5: Loading Policy Model", flush=True)
            print("   ⏳ Initializing Policy Model...", flush=True)
            self.policy_model = PolicyModel(self.config)
            self.policy_model.load_model()
            logger.info("Policy Model loaded")
            print("   ✅ Policy Model ready", flush=True)
            
            # 3. Load Value Model
            logger.info("Step 3/5: Loading Value Model")
            print("\n💎 Step 3/5: Loading Value Model", flush=True)
            print("   ⏳ Initializing Value Model...", flush=True)
            self.value_model = ValueModel(self.config)
            self.value_model.load_model()
            logger.info("Value Model loaded")
            print("   ✅ Value Model ready", flush=True)
            
            # 4. Initialize H-MCTS
            logger.info("Step 4/5: Initializing H-MCTS planner")
            print("\n🌲 Step 4/5: Initializing H-MCTS Planner", flush=True)
            print("   ⏳ Setting up H-MCTS...", flush=True)
            self.hmcts = HMCTS(
                policy_model=self.policy_model,
                value_model=self.value_model,
                exploration_constant=self._hmcts_exploration_constant,
                max_depth=self._hmcts_max_depth
            )
            logger.info("H-MCTS initialized")
            print("   ✅ H-MCTS ready", flush=True)
            
            # 5. Initialize SR-Adapt
            logger.info("Step 5/5: Initializing SR-Adapt validator")
            print("\n🔍 Step 5/5: Initializing SR-Adapt Validator", flush=True)
            print("   ⏳ Setting up SR-Adapt...", flush=True)
            self.sr_adapt = SRAdapt(
                config=self.config,
                memory_config=self.memory_config,
                alignment_threshold=self._alignment_threshold
            )
            logger.info("SR-Adapt initialized")
            print("   ✅ SR-Adapt ready", flush=True)
            
            self._initialized = True
            
            # 6. Initialize offline evolution components if requested
            if enable_evolution:
                print("\n🧬 Initializing offline evolution components...", flush=True)
                self._initialize_evolution_components()
            
            logger.info("HyPE System initialization complete")
            print("\n" + "="*70, flush=True)
            print("✅ HyPE System Initialization Complete!", flush=True)
            print("="*70 + "\n", flush=True)
            
        except Exception as e:
            logger.error(f"Failed to initialize HyPE System: {e}")
            print(f"\n❌ Initialization failed: {e}", flush=True)
            self.shutdown()
            raise RuntimeError(f"System initialization failed: {e}")
    
    def shutdown(self):
        """
        Shutdown system and cleanup resources.
        
        This method:
        - Disconnects from Principle Memory
        - Clears model caches
        - Resets system state
        """
        logger.info("Shutting down HyPE System...")
        
        if self.principle_memory is not None:
            try:
                self.principle_memory.disconnect()
                logger.info("Disconnected from Principle Memory")
            except Exception as e:
                logger.error(f"Error disconnecting from Principle Memory: {e}")
        
        # Clear references to allow garbage collection
        self.principle_memory = None
        self.policy_model = None
        self.value_model = None
        self.hmcts = None
        self.sr_adapt = None
        
        self._initialized = False
        logger.info("HyPE System shutdown complete")
    
    def is_initialized(self) -> bool:
        """
        Check if system is initialized.
        
        Returns:
            True if system is ready for task execution
        """
        return self._initialized
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get system statistics.
        
        Returns:
            Dictionary with system statistics
        """
        if not self._initialized:
            return {"status": "not_initialized"}
        
        memory_stats = self.principle_memory.get_stats()
        
        return {
            "status": "initialized",
            "num_principles": memory_stats["num_principles"],
            "num_trajectories_collected": len(self.trajectories),
            "hmcts_budget": self.hmcts_budget,
            "alignment_threshold": self._alignment_threshold,
            "max_steps_per_task": self.max_steps_per_task
        }
    
    def clear_trajectories(self):
        """Clear collected trajectories."""
        self.trajectories.clear()
        logger.info("Cleared collected trajectories")
    
    def get_trajectories(self) -> List[Trajectory]:
        """
        Get collected trajectories.
        
        Returns:
            List of collected trajectories
        """
        return self.trajectories.copy()
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        if self._initialized:
            self.shutdown()
    
    def execute_task(
        self,
        task: str,
        initial_state: State,
        environment: Any,
        environment_adapter: EnvironmentAdapter,
        environment_type: str,
        collect_trajectory: bool = True
    ) -> Tuple[Trajectory, bool]:
        """
        Execute a task through the complete online inference workflow.
        
        This method implements the full online workflow:
        1. Retrieve relevant principles from memory
        2. Plan using H-MCTS to generate best hypothesis
        3. Instantiate hypothesis into concrete action
        4. Validate and potentially correct action using SR-Adapt
        5. Execute action in environment
        6. Repeat until task completion or max steps reached
        
        Args:
            task: Task description
            initial_state: Initial environment state
            environment: Environment simulator
            environment_adapter: Adapter for environment-specific operations
            environment_type: Type of environment (toolbench, api_bank, alfworld)
            collect_trajectory: Whether to collect trajectory for offline learning
            
        Returns:
            Tuple of (trajectory, success)
            - trajectory: Complete execution trajectory
            - success: Whether task was completed successfully
            
        Raises:
            RuntimeError: If system is not initialized
        """
        if not self._initialized:
            raise RuntimeError("System not initialized. Call initialize() first.")
        
        logger.info(f"Starting task execution: {task[:100]}...")
        
        # Initialize trajectory
        trajectory_id = str(uuid.uuid4())
        steps: List[TrajectoryStep] = []
        principles_used: List[List[Principle]] = []
        
        # Current state
        current_state = initial_state
        cumulative_reward = 0.0
        done = False
        
        # Execute task loop
        for step_num in range(self.max_steps_per_task):
            logger.info(f"Step {step_num + 1}/{self.max_steps_per_task}")
            
            try:
                # Step 1: Retrieve relevant principles
                logger.info("Step 1: Retrieving relevant principles")
                principles = self._retrieve_principles(task, current_state)
                logger.info(f"Retrieved {len(principles)} principles")
                
                # Step 2: Plan using H-MCTS
                logger.info("Step 2: Planning with H-MCTS")
                best_hypothesis_node = self._plan_with_hmcts(
                    task, current_state, principles
                )
                hypothesis = best_hypothesis_node.hypothesis
                logger.info(f"Selected hypothesis: {hypothesis[:100]}...")
                
                # Step 3: Instantiate action
                logger.info("Step 3: Instantiating action")
                action = self._instantiate_action(
                    task, current_state, hypothesis, principles, environment_adapter
                )
                logger.info(f"Generated action: {action.description[:100]}...")
                
                # Step 4: Validate and execute with SR-Adapt
                logger.info("Step 4: Validating and executing action")
                next_state, reward, done, metadata = self._validate_and_execute(
                    action, principles, environment, environment_type,
                    task, current_state, hypothesis
                )
                
                logger.info(
                    f"Action executed: reward={reward:.3f}, done={done}, "
                    f"syntax_valid={metadata['syntax_valid']}, "
                    f"alignment={metadata['alignment_score']:.3f}, "
                    f"corrected={metadata['corrected']}"
                )
                
                # Step 5: Record trajectory step
                trajectory_step = TrajectoryStep(
                    state=current_state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                    hypothesis=hypothesis
                )
                steps.append(trajectory_step)
                principles_used.append(principles)
                
                # Update state and cumulative reward
                cumulative_reward += reward
                current_state = next_state
                
                # Check if task is complete
                if done:
                    logger.info(f"Task completed after {step_num + 1} steps")
                    break
                
                # Check for execution errors
                if metadata.get("error"):
                    logger.warning(f"Execution error: {metadata['error']}")
                    # Continue to next step unless it's a terminal error
                    if "syntax error" in metadata["error"].lower():
                        logger.error("Terminal syntax error, stopping execution")
                        done = True
                        break
                
            except Exception as e:
                logger.error(f"Error during step {step_num + 1}: {e}")
                # Create error trajectory step
                error_action = Action(
                    type="error",
                    parameters={"error": str(e)},
                    description=f"Error: {str(e)}"
                )
                trajectory_step = TrajectoryStep(
                    state=current_state,
                    action=error_action,
                    reward=-1.0,
                    next_state=current_state,
                    done=True,
                    hypothesis="Error occurred"
                )
                steps.append(trajectory_step)
                principles_used.append([])
                cumulative_reward += -1.0
                done = True
                break
        
        # Create trajectory
        success = cumulative_reward > 0 and done
        trajectory = Trajectory(
            id=trajectory_id,
            task=task,
            steps=steps,
            final_reward=cumulative_reward,
            success=success,
            principles_used=principles_used
        )
        
        # Collect trajectory if requested
        if collect_trajectory:
            self.trajectories.append(trajectory)
            logger.info(f"Collected trajectory {trajectory_id} (success={success})")
        
        logger.info(
            f"Task execution complete: {len(steps)} steps, "
            f"reward={cumulative_reward:.3f}, success={success}"
        )
        
        return trajectory, success
    
    def _retrieve_principles(
        self,
        task: str,
        state: State
    ) -> List[Principle]:
        """
        Retrieve relevant principles from memory.
        
        Args:
            task: Task description
            state: Current state
            
        Returns:
            List of retrieved principles
        """
        # Combine task and state for query
        query = f"{task}\n{self._format_state(state)}"
        
        # Retrieve principles
        principles = self.principle_memory.retrieve(
            query=query,
            top_k=self.memory_config.top_k
        )
        
        return principles
    
    def _plan_with_hmcts(
        self,
        task: str,
        state: State,
        principles: List[Principle]
    ) -> HypothesisNode:
        """
        Plan using H-MCTS to find best hypothesis.
        
        Args:
            task: Task description
            state: Current state
            principles: Retrieved principles
            
        Returns:
            Best hypothesis node
        """
        best_hypothesis = self.hmcts.search(
            task=task,
            state=state,
            principles=principles,
            budget=self.hmcts_budget
        )
        
        return best_hypothesis
    
    def _instantiate_action(
        self,
        task: str,
        state: State,
        hypothesis: str,
        principles: List[Principle],
        environment_adapter: EnvironmentAdapter
    ) -> Action:
        """
        Instantiate hypothesis into concrete action.
        
        Args:
            task: Task description
            state: Current state
            hypothesis: Hypothesis to instantiate
            principles: Retrieved principles
            environment_adapter: Environment adapter for action formatting
            
        Returns:
            Instantiated action
        """
        # Format state as string
        state_str = self._format_state(state)
        
        # Generate action description
        action_description = self.policy_model.instantiate_action(
            task=task,
            state=state_str,
            hypothesis=hypothesis,
            principles=principles
        )
        
        # Parse action description into Action object
        # This is a simplified version - in practice, would need more sophisticated parsing
        action = self._parse_action_description(
            action_description, environment_adapter
        )
        
        return action
    
    def _parse_action_description(
        self,
        description: str,
        environment_adapter: EnvironmentAdapter
    ) -> Action:
        """
        Parse action description into Action object.
        
        Args:
            description: Action description from model
            environment_adapter: Environment adapter
            
        Returns:
            Action object
        """
        # Determine environment type from adapter class name
        adapter_class_name = environment_adapter.__class__.__name__
        
        # Set default action type based on environment
        if "APIBank" in adapter_class_name:
            default_type = "api_call"
            default_params = {"api_name": "unknown", "method": "GET", "parameters": {}}
        elif "ToolBench" in adapter_class_name:
            default_type = "tool_use"
            default_params = {"tool": "unknown", "args": {}}
        elif "ALFWorld" in adapter_class_name:
            default_type = "navigation"
            default_params = {"command": description}
        else:
            default_type = "generic"
            default_params = {}
        
        # Create action with environment-specific type
        action = Action(
            type=default_type,
            parameters=default_params.copy(),
            description=description
        )
        
        # Try to extract more specific parameters from description
        description_lower = description.lower()
        
        if action.type == "api_call":
            # Extract API method if mentioned
            if "post" in description_lower or "create" in description_lower:
                action.parameters["method"] = "POST"
            elif "put" in description_lower or "update" in description_lower:
                action.parameters["method"] = "PUT"
            elif "delete" in description_lower or "remove" in description_lower:
                action.parameters["method"] = "DELETE"
            else:
                action.parameters["method"] = "GET"
            
            # Try to extract API name
            words = description.split()
            for i, word in enumerate(words):
                if word.lower() in ["api", "call", "request"] and i + 1 < len(words):
                    action.parameters["api_name"] = words[i + 1].strip("()[]{}:,.")
                    break
        
        elif action.type == "tool_use":
            # Try to extract tool name
            words = description.split()
            for i, word in enumerate(words):
                if word.lower() in ["tool", "use"] and i + 1 < len(words):
                    action.parameters["tool"] = words[i + 1].strip("()[]{}:,.")
                    break
        
        elif action.type in ["navigation", "interaction"]:
            # For ALFWorld, use description as command
            action.parameters["command"] = description
            
            # Refine type based on keywords
            if any(kw in description_lower for kw in ["go", "move", "walk", "turn", "look"]):
                action.type = "navigation"
            elif any(kw in description_lower for kw in ["take", "put", "open", "close", "toggle", "use"]):
                action.type = "interaction"
        
        return action
    
    def _validate_and_execute(
        self,
        action: Action,
        principles: List[Principle],
        environment: Any,
        environment_type: str,
        task: str,
        state: State,
        hypothesis: str
    ) -> Tuple[State, float, bool, Dict[str, Any]]:
        """
        Validate and execute action using SR-Adapt.
        
        Args:
            action: Action to validate and execute
            principles: Retrieved principles
            environment: Environment simulator
            environment_type: Type of environment
            task: Task description
            state: Current state
            hypothesis: Hypothesis that led to this action
            
        Returns:
            Tuple of (next_state, reward, done, metadata)
        """
        return self.sr_adapt.validate_and_execute(
            action=action,
            principles=principles,
            environment=environment,
            environment_type=environment_type,
            task=task,
            state=state,
            hypothesis=hypothesis,
            apply_correction=True
        )
    
    def _format_state(self, state: State) -> str:
        """
        Format state as string for model input.
        
        Args:
            state: State object
            
        Returns:
            String representation of state
        """
        obs_str = str(state.observation)
        history_str = ", ".join(state.history[-3:]) if state.history else "No previous actions"
        
        return f"Observation: {obs_str}\nRecent actions: {history_str}"
    
    def get_trajectory_by_id(self, trajectory_id: str) -> Optional[Trajectory]:
        """
        Get a specific trajectory by ID.
        
        Args:
            trajectory_id: Trajectory ID to retrieve
            
        Returns:
            Trajectory if found, None otherwise
        """
        for trajectory in self.trajectories:
            if trajectory.id == trajectory_id:
                return trajectory
        return None
    
    def get_successful_trajectories(self) -> List[Trajectory]:
        """
        Get all successful trajectories.
        
        Returns:
            List of successful trajectories
        """
        return [t for t in self.trajectories if t.success]
    
    def get_failed_trajectories(self) -> List[Trajectory]:
        """
        Get all failed trajectories.
        
        Returns:
            List of failed trajectories
        """
        return [t for t in self.trajectories if not t.success]
    
    def get_trajectory_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about collected trajectories.
        
        Returns:
            Dictionary with trajectory statistics
        """
        if not self.trajectories:
            return {
                "total": 0,
                "successful": 0,
                "failed": 0,
                "success_rate": 0.0,
                "avg_steps": 0.0,
                "avg_reward": 0.0
            }
        
        successful = self.get_successful_trajectories()
        failed = self.get_failed_trajectories()
        
        total_steps = sum(len(t.steps) for t in self.trajectories)
        total_reward = sum(t.final_reward for t in self.trajectories)
        
        return {
            "total": len(self.trajectories),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(self.trajectories),
            "avg_steps": total_steps / len(self.trajectories),
            "avg_reward": total_reward / len(self.trajectories),
            "total_steps": total_steps,
            "total_reward": total_reward
        }
    
    def export_trajectories(self, filepath: str):
        """
        Export collected trajectories to a file.
        
        Args:
            filepath: Path to export file (JSON format)
        """
        import json
        from datetime import datetime
        
        # Convert trajectories to serializable format
        export_data = {
            "export_time": datetime.now().isoformat(),
            "num_trajectories": len(self.trajectories),
            "statistics": self.get_trajectory_statistics(),
            "trajectories": []
        }
        
        for trajectory in self.trajectories:
            traj_data = {
                "id": trajectory.id,
                "task": trajectory.task,
                "final_reward": trajectory.final_reward,
                "success": trajectory.success,
                "num_steps": len(trajectory.steps),
                "steps": [
                    {
                        "state": {
                            "observation": step.state.observation,
                            "history": step.state.history,
                            "timestamp": step.state.timestamp.isoformat()
                        },
                        "action": {
                            "type": step.action.type,
                            "parameters": step.action.parameters,
                            "description": step.action.description
                        },
                        "reward": step.reward,
                        "done": step.done,
                        "hypothesis": step.hypothesis
                    }
                    for step in trajectory.steps
                ],
                "principles_used": [
                    [
                        {
                            "id": p.id,
                            "text": p.text,
                            "credit_score": p.credit_score
                        }
                        for p in principles
                    ]
                    for principles in trajectory.principles_used
                ]
            }
            export_data["trajectories"].append(traj_data)
        
        # Write to file
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported {len(self.trajectories)} trajectories to {filepath}")
    
    def import_trajectories(self, filepath: str):
        """
        Import trajectories from a file.
        
        Args:
            filepath: Path to import file (JSON format)
        """
        import json
        from datetime import datetime
        import numpy as np
        
        with open(filepath, 'r') as f:
            import_data = json.load(f)
        
        for traj_data in import_data["trajectories"]:
            # Reconstruct trajectory steps
            steps = []
            for step_data in traj_data["steps"]:
                state = State(
                    observation=step_data["state"]["observation"],
                    history=step_data["state"]["history"],
                    timestamp=datetime.fromisoformat(step_data["state"]["timestamp"])
                )
                
                action = Action(
                    type=step_data["action"]["type"],
                    parameters=step_data["action"]["parameters"],
                    description=step_data["action"]["description"]
                )
                
                next_state = State(
                    observation=step_data["state"]["observation"],
                    history=step_data["state"]["history"],
                    timestamp=datetime.fromisoformat(step_data["state"]["timestamp"])
                )
                
                step = TrajectoryStep(
                    state=state,
                    action=action,
                    reward=step_data["reward"],
                    next_state=next_state,
                    done=step_data["done"],
                    hypothesis=step_data.get("hypothesis")
                )
                steps.append(step)
            
            # Reconstruct principles (with placeholder embeddings)
            principles_used = []
            for principles_data in traj_data["principles_used"]:
                principles = []
                for p_data in principles_data:
                    principle = Principle(
                        id=p_data["id"],
                        text=p_data["text"],
                        embedding=np.zeros(1024),  # Placeholder
                        credit_score=p_data["credit_score"],
                        application_count=0,
                        created_at=datetime.now(),
                        last_used=datetime.now()
                    )
                    principles.append(principle)
                principles_used.append(principles)
            
            # Create trajectory
            trajectory = Trajectory(
                id=traj_data["id"],
                task=traj_data["task"],
                steps=steps,
                final_reward=traj_data["final_reward"],
                success=traj_data["success"],
                principles_used=principles_used
            )
            
            self.trajectories.append(trajectory)
        
        logger.info(f"Imported {len(import_data['trajectories'])} trajectories from {filepath}")
    
    # ========================================================================
    # Offline Evolution Workflow
    # ========================================================================
    
    def _initialize_evolution_components(self):
        """
        Initialize offline evolution components.
        
        This method initializes:
        - Principle Extractor for learning from trajectories
        - DPVD for Value Model training
        - DPO Trainer for Policy Model training
        - Replay Buffer for trajectory storage
        - Training Trigger for automatic training
        """
        logger.info("Initializing offline evolution components...")
        
        # Initialize Principle Extractor
        logger.info("Initializing Principle Extractor")
        self.principle_extractor = PrincipleExtractor(
            base_loader=self.policy_model.base_loader,
            principle_memory=self.principle_memory,
            success_threshold=self._principle_success_threshold,
            min_trajectory_length=1  # 允许从单步轨迹提取 principle
        )
        
        # Initialize DPVD
        logger.info("Initializing DPVD")
        self.dpvd = DPVD(
            value_model=self.value_model,
            beta=self._dpvd_beta,
            learning_rate=self._dpvd_learning_rate,
            batch_size=self._dpvd_batch_size,
            epochs=self._dpvd_epochs
        )
        
        # Initialize Replay Buffer
        logger.info("Initializing Replay Buffer")
        self.replay_buffer = ReplayBuffer(max_size=self._replay_buffer_size)
        
        # Initialize Training Trigger
        logger.info("Initializing Training Trigger")
        self.training_trigger = TrainingTrigger(
            replay_buffer=self.replay_buffer,
            trigger_threshold=self._training_trigger_threshold,
            min_trajectories=self._min_trajectories_for_training
        )
        
        # Initialize DPO Trainer
        logger.info("Initializing DPO Trainer")
        self.dpo_trainer = DPOTrainer(
            policy_model=self.policy_model,
            reference_model=None,  # Will use copy of policy model
            beta=self._dpo_beta,
            learning_rate=self._dpo_learning_rate,
            batch_size=self._dpo_batch_size
        )
        
        self._evolution_enabled = True
        logger.info("Offline evolution components initialized")
    
    def is_evolution_enabled(self) -> bool:
        """
        Check if offline evolution is enabled.
        
        Returns:
            True if evolution components are initialized
        """
        return self._evolution_enabled
    
    def add_trajectory_to_buffer(self, trajectory: Trajectory):
        """
        Add a trajectory to the replay buffer.
        
        Args:
            trajectory: Trajectory to add
        
        Raises:
            RuntimeError: If evolution is not enabled
        """
        if not self._evolution_enabled:
            raise RuntimeError("Evolution not enabled. Call initialize(enable_evolution=True)")
        
        self.replay_buffer.add(trajectory)
        logger.debug(f"Added trajectory {trajectory.id} to replay buffer")
    
    def extract_principles_from_trajectories(
        self,
        trajectories: Optional[List[Trajectory]] = None,
        filter_successful: bool = True,
        baseline_reward: float = 0.0
    ) -> int:
        """
        Extract principles from trajectories and insert into memory.
        
        This implements the principle extraction pipeline:
        1. Filter successful trajectories (if requested)
        2. Extract principles from each trajectory
        3. Insert principles with semantic deduplication
        
        Args:
            trajectories: List of trajectories (uses collected if None)
            filter_successful: Whether to filter for successful trajectories only
            baseline_reward: Baseline reward for credit computation
            
        Returns:
            Number of new principles inserted
            
        Raises:
            RuntimeError: If evolution is not enabled
        """
        if not self._evolution_enabled:
            raise RuntimeError("Evolution not enabled. Call initialize(enable_evolution=True)")
        
        # Use collected trajectories if none provided
        if trajectories is None:
            trajectories = self.trajectories
        
        if not trajectories:
            logger.warning("No trajectories to extract principles from")
            return 0
        
        logger.info(f"Starting principle extraction from {len(trajectories)} trajectories")
        
        # Filter successful trajectories if requested
        if filter_successful:
            successful_trajectories = [
                t for t in trajectories
                if t.success and t.final_reward >= self._principle_success_threshold
            ]
            logger.info(
                f"Filtered to {len(successful_trajectories)} successful trajectories "
                f"(threshold={self._principle_success_threshold})"
            )
        else:
            successful_trajectories = trajectories
        
        if not successful_trajectories:
            logger.warning("No successful trajectories to extract from")
            return 0
        
        # Extract and insert principles
        num_inserted = self.principle_extractor.extract_and_insert(
            trajectories=successful_trajectories,
            baseline_reward=baseline_reward
        )
        
        logger.info(f"Principle extraction complete: {num_inserted} new principles inserted")
        
        return num_inserted
    
    def train_value_model_from_buffer(
        self,
        num_trajectories: Optional[int] = None,
        validation_split: float = 0.1
    ) -> Dict[str, List[float]]:
        """
        Train Value Model using trajectories from replay buffer.
        
        This implements the Value Model training pipeline:
        1. Sample trajectories from replay buffer
        2. Compute dense rewards using DPVD
        3. Train Value Model on dense rewards
        4. Update H-MCTS to use new Value Model
        
        Args:
            num_trajectories: Number of trajectories to use (None for all)
            validation_split: Fraction of data for validation
            
        Returns:
            Training history dictionary
            
        Raises:
            RuntimeError: If evolution is not enabled
        """
        if not self._evolution_enabled:
            raise RuntimeError("Evolution not enabled. Call initialize(enable_evolution=True)")
        
        logger.info("Starting Value Model training from replay buffer")
        
        # Get trajectories from buffer
        if num_trajectories is None:
            trajectories = self.replay_buffer.get_all()
        else:
            trajectories = self.replay_buffer.sample(num_trajectories)
        
        if not trajectories:
            logger.warning("No trajectories in replay buffer")
            return {}
        
        logger.info(f"Training Value Model on {len(trajectories)} trajectories")
        
        # Train Value Model using DPVD
        history = self.dpvd.train_value_model(
            trajectories=trajectories,
            validation_split=validation_split
        )
        
        # Update H-MCTS to use new Value Model
        # The Value Model is already updated in-place, so H-MCTS will use it
        logger.info("Value Model training complete, H-MCTS will use updated model")
        
        return history
    
    def train_policy_model_from_buffer(
        self,
        num_trajectories: Optional[int] = None,
        num_epochs: int = None,
        stratified: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train Policy Model using DPO on trajectories from replay buffer.
        
        This implements the Policy Model training pipeline:
        1. Sample trajectories from replay buffer
        2. Construct preference pairs based on principle credit
        3. Train Policy Model using DPO
        4. Update Policy Model for hypothesis generation
        
        Args:
            num_trajectories: Number of trajectories to use (None for all)
            num_epochs: Number of training epochs (uses config if None)
            stratified: Whether to use stratified pair construction
            
        Returns:
            Training history dictionary
            
        Raises:
            RuntimeError: If evolution is not enabled
        """
        if not self._evolution_enabled:
            raise RuntimeError("Evolution not enabled. Call initialize(enable_evolution=True)")
        
        logger.info("Starting Policy Model training from replay buffer")
        
        # Get trajectories from buffer
        if num_trajectories is None:
            trajectories = self.replay_buffer.get_all()
        else:
            trajectories = self.replay_buffer.sample(num_trajectories)
        
        if not trajectories:
            logger.warning("No trajectories in replay buffer")
            return {}
        
        logger.info(f"Training Policy Model on {len(trajectories)} trajectories")
        
        # Use configured epochs if not specified
        if num_epochs is None:
            num_epochs = self._dpo_epochs
        
        # Train Policy Model using DPO
        history = self.dpo_trainer.train_policy_model(
            trajectories=trajectories,
            num_epochs=num_epochs,
            stratified=stratified
        )
        
        # Policy Model is already updated in-place
        logger.info("Policy Model training complete, will use updated model for generation")
        
        return history
    
    def run_offline_evolution(
        self,
        trajectories: Optional[List[Trajectory]] = None,
        extract_principles: bool = True,
        train_value_model: bool = True,
        train_policy_model: bool = True,
        baseline_reward: float = 0.0,
        add_to_buffer: bool = True
    ) -> Dict[str, Any]:
        """
        Run complete offline evolution cycle.
        
        This orchestrates the full offline evolution workflow:
        1. Extract principles from successful trajectories
        2. Add trajectories to replay buffer
        3. Train Value Model with dense rewards
        4. Train Policy Model with DPO
        
        The evolution cycle improves the system by:
        - Growing the principle knowledge base
        - Improving value estimation for planning
        - Improving action generation quality
        
        Args:
            trajectories: Trajectories to evolve from (uses collected if None)
            extract_principles: Whether to extract principles
            train_value_model: Whether to train Value Model
            train_policy_model: Whether to train Policy Model
            baseline_reward: Baseline reward for principle credit
            add_to_buffer: Whether to add trajectories to replay buffer
            
        Returns:
            Dictionary with evolution results and metrics
            
        Raises:
            RuntimeError: If evolution is not enabled or system not initialized
        """
        if not self._initialized:
            raise RuntimeError("System not initialized. Call initialize() first.")
        
        if not self._evolution_enabled:
            raise RuntimeError("Evolution not enabled. Call initialize(enable_evolution=True)")
        
        logger.info("=" * 80)
        logger.info("Starting Offline Evolution Cycle")
        logger.info("=" * 80)
        
        # Use collected trajectories if none provided
        if trajectories is None:
            trajectories = self.trajectories
        
        if not trajectories:
            logger.warning("No trajectories to evolve from")
            return {
                "status": "no_trajectories",
                "num_trajectories": 0
            }
        
        logger.info(f"Evolution cycle with {len(trajectories)} trajectories")
        
        # 🔥 关键优化：训练前释放所有不必要的显存
        import torch
        if torch.cuda.is_available():
            logger.info("🧹 Cleaning up GPU memory before training...")
            print("\n🧹 Cleaning up GPU memory before training...", flush=True)
            
            # 1. 卸载 Executor Model（最大的显存占用）
            if hasattr(self, 'sr_adapt') and hasattr(self.sr_adapt, 'executor_model'):
                if self.sr_adapt.executor_model is not None:
                    logger.info("   - Unloading Executor Model")
                    print("   - Unloading Executor Model", flush=True)
                    del self.sr_adapt.executor_model
                    self.sr_adapt.executor_model = None
            
            # 2. 清理 HMCTS 的缓存
            if hasattr(self, 'hmcts') and self.hmcts is not None:
                if hasattr(self.hmcts, 'value_cache'):
                    logger.info("   - Clearing HMCTS cache")
                    print("   - Clearing HMCTS cache", flush=True)
                    self.hmcts.value_cache.clear()
            
            # 3. 强制 Python 垃圾回收
            import gc
            gc.collect()
            
            # 4. 清空 CUDA 缓存
            torch.cuda.empty_cache()
            
            # 5. 报告显存状态
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            free = (torch.cuda.get_device_properties(0).total_memory / 1e9) - allocated
            
            logger.info(f"   ✅ GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {free:.2f}GB free")
            print(f"   ✅ GPU memory: {allocated:.2f}GB allocated, {free:.2f}GB free", flush=True)
        
        results = {
            "status": "success",
            "num_trajectories": len(trajectories),
            "principles_extracted": 0,
            "value_training_history": {},
            "policy_training_history": {},
            "errors": []
        }
        
        # Phase 1: Principle Extraction
        if extract_principles:
            logger.info("-" * 80)
            logger.info("Phase 1: Principle Extraction")
            logger.info("-" * 80)
            
            try:
                num_principles = self.extract_principles_from_trajectories(
                    trajectories=trajectories,
                    filter_successful=True,
                    baseline_reward=baseline_reward
                )
                results["principles_extracted"] = num_principles
                logger.info(f"Phase 1 complete: {num_principles} principles extracted")
            except Exception as e:
                logger.error(f"Phase 1 failed: {e}")
                results["errors"].append(f"Principle extraction: {str(e)}")
        
        # Add trajectories to replay buffer
        if add_to_buffer:
            logger.info("-" * 80)
            logger.info("Adding trajectories to replay buffer")
            logger.info("-" * 80)
            
            try:
                for trajectory in trajectories:
                    self.add_trajectory_to_buffer(trajectory)
                logger.info(f"Added {len(trajectories)} trajectories to buffer")
                logger.info(f"Buffer size: {self.replay_buffer.size()}/{self._replay_buffer_size}")
            except Exception as e:
                logger.error(f"Failed to add trajectories to buffer: {e}")
                results["errors"].append(f"Buffer addition: {str(e)}")
        
        # Phase 2: Value Model Training
        if train_value_model:
            logger.info("-" * 80)
            logger.info("Phase 2: Value Model Training")
            logger.info("-" * 80)
            
            try:
                # Check if we have enough data
                if self.replay_buffer.size() < self._min_trajectories_for_training:
                    logger.warning(
                        f"Insufficient trajectories for training: "
                        f"{self.replay_buffer.size()} < {self._min_trajectories_for_training}"
                    )
                    results["errors"].append("Insufficient data for Value Model training")
                else:
                    value_history = self.train_value_model_from_buffer()
                    results["value_training_history"] = value_history
                    logger.info("Phase 2 complete: Value Model trained")
            except Exception as e:
                logger.error(f"Phase 2 failed: {e}")
                results["errors"].append(f"Value Model training: {str(e)}")
        
        # Phase 3: Policy Model Training
        if train_policy_model:
            logger.info("-" * 80)
            logger.info("Phase 3: Policy Model Training")
            logger.info("-" * 80)
            
            try:
                # Check if we have enough data
                if self.replay_buffer.size() < self._min_trajectories_for_training:
                    logger.warning(
                        f"Insufficient trajectories for training: "
                        f"{self.replay_buffer.size()} < {self._min_trajectories_for_training}"
                    )
                    results["errors"].append("Insufficient data for Policy Model training")
                else:
                    policy_history = self.train_policy_model_from_buffer()
                    results["policy_training_history"] = policy_history
                    logger.info("Phase 3 complete: Policy Model trained")
            except Exception as e:
                logger.error(f"Phase 3 failed: {e}")
                results["errors"].append(f"Policy Model training: {str(e)}")
        
        # Mark training complete in trigger
        if train_value_model or train_policy_model:
            self.training_trigger.mark_training_complete()
        
        logger.info("=" * 80)
        logger.info("Offline Evolution Cycle Complete")
        logger.info(f"Principles extracted: {results['principles_extracted']}")
        logger.info(f"Value Model trained: {bool(results['value_training_history'])}")
        logger.info(f"Policy Model trained: {bool(results['policy_training_history'])}")
        if results["errors"]:
            logger.warning(f"Errors encountered: {len(results['errors'])}")
        logger.info("=" * 80)
        
        return results
    
    def should_trigger_evolution(self) -> bool:
        """
        Check if offline evolution should be triggered.
        
        Returns:
            True if training trigger conditions are met
            
        Raises:
            RuntimeError: If evolution is not enabled
        """
        if not self._evolution_enabled:
            raise RuntimeError("Evolution not enabled. Call initialize(enable_evolution=True)")
        
        return self.training_trigger.should_trigger()
    
    def get_evolution_stats(self) -> Dict[str, Any]:
        """
        Get statistics about offline evolution components.
        
        Returns:
            Dictionary with evolution statistics
            
        Raises:
            RuntimeError: If evolution is not enabled
        """
        if not self._evolution_enabled:
            return {
                "evolution_enabled": False
            }
        
        return {
            "evolution_enabled": True,
            "replay_buffer_size": self.replay_buffer.size(),
            "replay_buffer_capacity": self._replay_buffer_size,
            "buffer_utilization": self.replay_buffer.size() / self._replay_buffer_size,
            "training_threshold": self._training_trigger_threshold,
            "min_trajectories": self._min_trajectories_for_training,
            "should_trigger": self.training_trigger.should_trigger(),
            "dpvd_config": {
                "beta": self._dpvd_beta,
                "learning_rate": self._dpvd_learning_rate,
                "batch_size": self._dpvd_batch_size,
                "epochs": self._dpvd_epochs
            },
            "dpo_config": {
                "beta": self._dpo_beta,
                "learning_rate": self._dpo_learning_rate,
                "batch_size": self._dpo_batch_size,
                "epochs": self._dpo_epochs
            }
        }
