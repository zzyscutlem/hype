# HyPE Agent System - API Documentation

This document provides comprehensive API documentation for all public classes and methods in the HyPE Agent System.

## Table of Contents

1. [Core Components](#core-components)
   - [HyPESystem](#hypesystem)
   - [Configuration](#configuration)
   - [Data Models](#data-models)
2. [Planning](#planning)
   - [H-MCTS](#h-mcts)
3. [Memory](#memory)
   - [Principle Memory](#principle-memory)
4. [Execution](#execution)
   - [SR-Adapt](#sr-adapt)
5. [Learning](#learning)
   - [DPVD](#dpvd)
   - [DPO Trainer](#dpo-trainer)
   - [Principle Extractor](#principle-extractor)
6. [Models](#models)
   - [Policy Model](#policy-model)
   - [Value Model](#value-model)
   - [Executor Model](#executor-model)
7. [Environment Adapters](#environment-adapters)
8. [Utilities](#utilities)

---

## Core Components

### HyPESystem

Main orchestrator class for the HyPE Agent System.

#### Class: `HyPESystem`

**Location:** `hype.system`

**Description:** Coordinates all components for online task execution and offline evolution.

#### Constructor

```python
HyPESystem(
    config: Optional[ModelConfig] = None,
    memory_config: Optional[PrincipleMemoryConfig] = None,
    hmcts_budget: int = 50,
    hmcts_exploration_constant: float = 1.414,
    hmcts_max_depth: int = 5,
    alignment_threshold: float = 0.7,
    max_steps_per_task: int = 20,
    replay_buffer_size: int = 10000,
    training_trigger_threshold: int = 100,
    min_trajectories_for_training: int = 10,
    dpvd_beta: float = 0.1,
    dpvd_learning_rate: float = 1e-5,
    dpvd_batch_size: int = 32,
    dpvd_epochs: int = 3,
    dpo_beta: float = 0.1,
    dpo_learning_rate: float = 1e-6,
    dpo_batch_size: int = 4,
    dpo_epochs: int = 3,
    principle_success_threshold: float = 0.5
)
```

**Parameters:**
- `config`: Model configuration
- `memory_config`: Principle memory configuration
- `hmcts_budget`: Number of MCTS iterations per planning step
- `hmcts_exploration_constant`: UCB exploration constant
- `hmcts_max_depth`: Maximum tree depth for H-MCTS
- `alignment_threshold`: Minimum alignment score for SR-Adapt
- `max_steps_per_task`: Maximum steps per task execution
- `replay_buffer_size`: Maximum size of replay buffer
- `training_trigger_threshold`: Buffer size threshold to trigger training
- `min_trajectories_for_training`: Minimum trajectories required for training
- `dpvd_beta`: Principle weight coefficient for dense rewards
- `dpvd_learning_rate`: Learning rate for Value Model training
- `dpvd_batch_size`: Batch size for Value Model training
- `dpvd_epochs`: Number of epochs for Value Model training
- `dpo_beta`: KL penalty coefficient for DPO
- `dpo_learning_rate`: Learning rate for Policy Model training
- `dpo_batch_size`: Batch size for Policy Model training
- `dpo_epochs`: Number of epochs for Policy Model training
- `principle_success_threshold`: Minimum reward to extract principles

#### Methods

##### `initialize(enable_evolution: bool = True)`

Initialize all system components.

**Parameters:**
- `enable_evolution`: Whether to initialize offline evolution components

**Raises:**
- `RuntimeError`: If initialization fails

**Example:**
```python
system = HyPESystem()
system.initialize(enable_evolution=True)
```

##### `shutdown()`

Shutdown system and cleanup resources.

**Example:**
```python
system.shutdown()
```

##### `execute_task(...) -> Tuple[Trajectory, bool]`

Execute a task through the complete online inference workflow.

**Parameters:**
- `task` (str): Task description
- `initial_state` (State): Initial environment state
- `environment` (Any): Environment simulator
- `environment_adapter` (EnvironmentAdapter): Adapter for environment-specific operations
- `environment_type` (str): Type of environment (toolbench, api_bank, alfworld)
- `collect_trajectory` (bool): Whether to collect trajectory for offline learning

**Returns:**
- `Tuple[Trajectory, bool]`: (trajectory, success)

**Raises:**
- `RuntimeError`: If system is not initialized

**Example:**
```python
trajectory, success = system.execute_task(
    task="Complete the task",
    initial_state=initial_state,
    environment=env,
    environment_adapter=adapter,
    environment_type="toolbench",
    collect_trajectory=True
)
```

##### `run_offline_evolution()`

Run complete offline evolution workflow.

Executes:
1. Principle extraction from successful trajectories
2. Value Model training with dense rewards
3. Policy Model training with DPO

**Raises:**
- `RuntimeError`: If system is not initialized or evolution not enabled

**Example:**
```python
system.run_offline_evolution()
```

##### `get_system_stats() -> Dict[str, Any]`

Get system statistics.

**Returns:**
- Dictionary with system statistics

**Example:**
```python
stats = system.get_system_stats()
print(f"Principles: {stats['num_principles']}")
```

##### `get_trajectory_statistics() -> Dict[str, Any]`

Get statistics about collected trajectories.

**Returns:**
- Dictionary with trajectory statistics including:
  - `total`: Total number of trajectories
  - `successful`: Number of successful trajectories
  - `failed`: Number of failed trajectories
  - `success_rate`: Success rate (0.0-1.0)
  - `avg_steps`: Average steps per trajectory
  - `avg_reward`: Average reward per trajectory

**Example:**
```python
stats = system.get_trajectory_statistics()
print(f"Success rate: {stats['success_rate']:.1%}")
```

##### `export_trajectories(filepath: str)`

Export collected trajectories to a JSON file.

**Parameters:**
- `filepath`: Path to export file

**Example:**
```python
system.export_trajectories("trajectories.json")
```

##### `import_trajectories(filepath: str)`

Import trajectories from a JSON file.

**Parameters:**
- `filepath`: Path to import file

**Example:**
```python
system.import_trajectories("trajectories.json")
```

---

### Configuration

#### Class: `HyPEConfig`

**Location:** `hype.core.config`

**Description:** Main configuration class for the HyPE system.

#### Class Methods

##### `from_yaml(path: str) -> HyPEConfig`

Load configuration from YAML file.

**Parameters:**
- `path`: Path to YAML configuration file

**Returns:**
- `HyPEConfig` instance

**Raises:**
- `FileNotFoundError`: If config file doesn't exist
- `ValueError`: If config validation fails

**Example:**
```python
config = HyPEConfig.from_yaml("config.yaml")
```

##### `from_json(path: str) -> HyPEConfig`

Load configuration from JSON file.

**Parameters:**
- `path`: Path to JSON configuration file

**Returns:**
- `HyPEConfig` instance

**Example:**
```python
config = HyPEConfig.from_json("config.json")
```

#### Instance Methods

##### `validate()`

Validate all configuration sections.

**Raises:**
- `ValueError`: If any validation errors are found

**Example:**
```python
config.validate()
```

##### `save_yaml(path: str)`

Save configuration to YAML file.

**Parameters:**
- `path`: Path to save YAML file

**Example:**
```python
config.save_yaml("config_backup.yaml")
```

##### `save_json(path: str)`

Save configuration to JSON file.

**Parameters:**
- `path`: Path to save JSON file

**Example:**
```python
config.save_json("config_backup.json")
```

---

### Data Models

#### Class: `State`

**Location:** `hype.core.data_models`

**Description:** Environment state representation.

**Attributes:**
- `observation` (Dict[str, Any]): Environment-specific observation
- `history` (List[str]): Action history
- `timestamp` (datetime): State timestamp

**Example:**
```python
state = State(
    observation={"obs": "value"},
    history=["action1", "action2"],
    timestamp=datetime.now()
)
```

#### Class: `Action`

**Location:** `hype.core.data_models`

**Description:** Agent action representation.

**Attributes:**
- `type` (str): Action type (e.g., "api_call", "tool_use")
- `parameters` (Dict[str, Any]): Action parameters
- `description` (str): Natural language description

**Example:**
```python
action = Action(
    type="tool_use",
    parameters={"tool": "calculator", "args": {"x": 5}},
    description="Use calculator to compute 5 + 3"
)
```

#### Class: `Principle`

**Location:** `hype.core.data_models`

**Description:** Principle stored in memory.

**Attributes:**
- `id` (str): Unique identifier
- `text` (str): Natural language principle description
- `embedding` (np.ndarray): Semantic embedding (1024-dim)
- `credit_score` (float): Accumulated credit
- `application_count` (int): Number of times applied
- `created_at` (datetime): Creation timestamp
- `last_used` (datetime): Last usage timestamp
- `source_trajectory_id` (Optional[str]): Source trajectory ID

**Example:**
```python
principle = Principle(
    id="p_001",
    text="When solving math problems, break them into steps",
    embedding=np.random.randn(1024),
    credit_score=0.8,
    application_count=5,
    created_at=datetime.now(),
    last_used=datetime.now(),
    source_trajectory_id="traj_001"
)
```

#### Class: `Trajectory`

**Location:** `hype.core.data_models`

**Description:** Complete task execution trajectory.

**Attributes:**
- `id` (str): Unique identifier
- `task` (str): Task description
- `steps` (List[TrajectoryStep]): Trajectory steps
- `final_reward` (float): Final cumulative reward
- `success` (bool): Whether task was successful
- `principles_used` (List[List[Principle]]): Principles per step

**Example:**
```python
trajectory = Trajectory(
    id="traj_001",
    task="Complete the task",
    steps=[step1, step2],
    final_reward=1.5,
    success=True,
    principles_used=[[p1, p2], [p3]]
)
```

---

## Planning

### H-MCTS

#### Class: `HMCTS`

**Location:** `hype.planner.hmcts`

**Description:** Hypothesis-driven Monte Carlo Tree Search planner.

#### Constructor

```python
HMCTS(
    policy_model: PolicyModel,
    value_model: ValueModel,
    exploration_constant: float = 1.414,
    max_depth: int = 5
)
```

**Parameters:**
- `policy_model`: Policy model for hypothesis generation
- `value_model`: Value model for hypothesis evaluation
- `exploration_constant`: UCB exploration constant (c)
- `max_depth`: Maximum tree depth

#### Methods

##### `search(...) -> HypothesisNode`

Perform MCTS search to find best hypothesis.

**Parameters:**
- `task` (str): Task description
- `state` (State): Current environment state
- `principles` (List[Principle]): Retrieved principles
- `budget` (int): Number of search iterations

**Returns:**
- Best hypothesis node based on visit count

**Example:**
```python
hmcts = HMCTS(policy_model, value_model)
best_hypothesis = hmcts.search(
    task="Complete task",
    state=current_state,
    principles=principles,
    budget=100
)
```

---

## Memory

### Principle Memory

#### Class: `PrincipleMemory`

**Location:** `hype.memory.principle_memory`

**Description:** Vector database for storing and retrieving principles.

#### Constructor

```python
PrincipleMemory(config: PrincipleMemoryConfig)
```

**Parameters:**
- `config`: Principle memory configuration

#### Methods

##### `connect()`

Connect to Milvus vector database.

**Raises:**
- `ConnectionError`: If connection fails

**Example:**
```python
memory = PrincipleMemory(config)
memory.connect()
```

##### `disconnect()`

Disconnect from database.

**Example:**
```python
memory.disconnect()
```

##### `insert(principle: Principle) -> bool`

Insert principle with semantic deduplication.

**Parameters:**
- `principle`: Principle to insert

**Returns:**
- True if inserted, False if merged with duplicate

**Example:**
```python
principle = Principle(...)
inserted = memory.insert(principle)
```

##### `retrieve(query: str, top_k: int) -> List[Principle]`

Retrieve principles using credit-weighted semantic search.

**Parameters:**
- `query`: Query string
- `top_k`: Number of principles to retrieve

**Returns:**
- List of retrieved principles

**Example:**
```python
principles = memory.retrieve("solve math problem", top_k=5)
```

##### `update_credit(principle_id: str, credit_delta: float)`

Update principle credit score.

**Parameters:**
- `principle_id`: Principle ID
- `credit_delta`: Credit change (can be negative)

**Example:**
```python
memory.update_credit("p_001", 0.5)
```

##### `prune_low_credit(threshold: float) -> int`

Remove principles below credit threshold.

**Parameters:**
- `threshold`: Minimum credit score

**Returns:**
- Number of principles removed

**Example:**
```python
removed = memory.prune_low_credit(0.1)
```

---

## Execution

### SR-Adapt

#### Class: `SRAdapt`

**Location:** `hype.executor.sr_adapt`

**Description:** Semantic-Reflective Adaptation validator with dual guardrails.

#### Constructor

```python
SRAdapt(
    config: ModelConfig,
    memory_config: PrincipleMemoryConfig,
    alignment_threshold: float = 0.7,
    correction_steps: int = 5,
    correction_lr: float = 1e-4,
    max_correction_attempts: int = 3
)
```

**Parameters:**
- `config`: Model configuration
- `memory_config`: Principle memory configuration
- `alignment_threshold`: Minimum alignment score
- `correction_steps`: Number of LoRA fine-tuning steps
- `correction_lr`: Learning rate for correction
- `max_correction_attempts`: Maximum correction attempts

#### Methods

##### `validate_and_execute(...) -> Tuple[State, float, bool, Dict]`

Validate action through dual guardrails and execute.

**Parameters:**
- `action` (Action): Generated action
- `principles` (List[Principle]): Retrieved principles
- `environment` (Any): Target environment
- `environment_type` (str): Environment type
- `task` (str): Task description
- `state` (State): Current state
- `hypothesis` (str): Hypothesis that led to action
- `apply_correction` (bool): Whether to apply LoRA correction

**Returns:**
- Tuple of (next_state, reward, done, metadata)

**Example:**
```python
sr_adapt = SRAdapt(config, memory_config)
next_state, reward, done, metadata = sr_adapt.validate_and_execute(
    action=action,
    principles=principles,
    environment=env,
    environment_type="toolbench",
    task="Complete task",
    state=current_state,
    hypothesis="Try approach X",
    apply_correction=True
)
```

---

## Learning

### DPVD

#### Class: `DPVD`

**Location:** `hype.learner.dpvd`

**Description:** Dense Principle-Value Distillation learner.

#### Constructor

```python
DPVD(
    value_model: ValueModel,
    beta: float = 0.1,
    learning_rate: float = 1e-5,
    batch_size: int = 32,
    num_epochs: int = 3
)
```

**Parameters:**
- `value_model`: Value model to train
- `beta`: Principle weight coefficient (β)
- `learning_rate`: Learning rate
- `batch_size`: Batch size
- `num_epochs`: Number of training epochs

#### Methods

##### `compute_dense_rewards(...) -> List[float]`

Compute step-wise dense rewards from principle credits.

**Parameters:**
- `trajectory` (Trajectory): Trajectory
- `principles_used` (List[List[Principle]]): Principles per step

**Returns:**
- Dense reward for each step

**Example:**
```python
dpvd = DPVD(value_model)
dense_rewards = dpvd.compute_dense_rewards(trajectory, principles_used)
```

##### `train_value_model(...)`

Train Value Model to predict dense rewards.

**Parameters:**
- `trajectories` (List[Trajectory]): Training trajectories
- `dense_rewards` (List[List[float]]): Dense rewards per trajectory

**Example:**
```python
dpvd.train_value_model(trajectories, dense_rewards)
```

---

### DPO Trainer

#### Class: `DPOTrainer`

**Location:** `hype.learner.dpo_trainer`

**Description:** Direct Preference Optimization trainer for Policy Model.

#### Constructor

```python
DPOTrainer(
    policy_model: PolicyModel,
    reference_model: PolicyModel,
    beta: float = 0.1,
    learning_rate: float = 1e-6,
    batch_size: int = 4,
    num_epochs: int = 3
)
```

**Parameters:**
- `policy_model`: Policy model to train
- `reference_model`: Reference model for KL penalty
- `beta`: KL penalty coefficient
- `learning_rate`: Learning rate
- `batch_size`: Batch size
- `num_epochs`: Number of training epochs

#### Methods

##### `construct_preference_pairs(...) -> List[Tuple]`

Construct preference pairs from trajectories.

**Parameters:**
- `trajectories` (List[Trajectory]): Trajectories to rank

**Returns:**
- List of (preferred, dispreferred) trajectory pairs

**Example:**
```python
dpo_trainer = DPOTrainer(policy_model, reference_model)
pairs = dpo_trainer.construct_preference_pairs(trajectories)
```

##### `train(...)`

Train Policy Model using DPO.

**Parameters:**
- `preference_pairs` (List[Tuple]): Preference pairs

**Example:**
```python
dpo_trainer.train(preference_pairs)
```

---

### Principle Extractor

#### Class: `PrincipleExtractor`

**Location:** `hype.learner.principle_extractor`

**Description:** Extracts principles from successful trajectories.

#### Constructor

```python
PrincipleExtractor(
    base_model: Any,
    embedding_model: Any,
    success_threshold: float = 0.5
)
```

**Parameters:**
- `base_model`: Base model for principle generation
- `embedding_model`: Embedding model for semantic embeddings
- `success_threshold`: Minimum reward for extraction

#### Methods

##### `extract_principles(...) -> List[Principle]`

Extract principles from trajectory.

**Parameters:**
- `trajectory` (Trajectory): Trajectory to analyze

**Returns:**
- List of extracted principles

**Example:**
```python
extractor = PrincipleExtractor(base_model, embedding_model)
principles = extractor.extract_principles(trajectory)
```

---

## Models

### Policy Model

#### Class: `PolicyModel`

**Location:** `hype.models.policy_model`

**Description:** Policy model for hypothesis and action generation.

#### Methods

##### `generate_hypothesis(...) -> str`

Generate hypothesis for current state.

**Parameters:**
- `task` (str): Task description
- `state` (str): Current state
- `principles` (List[Principle]): Retrieved principles

**Returns:**
- Generated hypothesis

**Example:**
```python
hypothesis = policy_model.generate_hypothesis(
    task="Complete task",
    state="Current state",
    principles=principles
)
```

##### `instantiate_action(...) -> str`

Instantiate hypothesis into concrete action.

**Parameters:**
- `task` (str): Task description
- `state` (str): Current state
- `hypothesis` (str): Hypothesis to instantiate
- `principles` (List[Principle]): Retrieved principles

**Returns:**
- Action description

**Example:**
```python
action_desc = policy_model.instantiate_action(
    task="Complete task",
    state="Current state",
    hypothesis="Try approach X",
    principles=principles
)
```

---

### Value Model

#### Class: `ValueModel`

**Location:** `hype.models.value_model`

**Description:** Value model for hypothesis evaluation.

#### Methods

##### `predict_value(...) -> float`

Predict expected value for hypothesis.

**Parameters:**
- `task` (str): Task description
- `state` (str): Current state
- `hypothesis` (str): Hypothesis to evaluate
- `principles` (List[Principle]): Retrieved principles

**Returns:**
- Predicted value

**Example:**
```python
value = value_model.predict_value(
    task="Complete task",
    state="Current state",
    hypothesis="Try approach X",
    principles=principles
)
```

---

## Environment Adapters

### Base Adapter

#### Class: `EnvironmentAdapter`

**Location:** `hype.environments.adapters`

**Description:** Base class for environment-specific adapters.

#### Methods

##### `parse_state(raw_observation: Any) -> State`

Convert environment observation to State.

**Parameters:**
- `raw_observation`: Raw environment observation

**Returns:**
- State object

##### `format_action(action: Action) -> Any`

Convert Action to environment-specific format.

**Parameters:**
- `action`: Action object

**Returns:**
- Environment-specific action format

##### `validate_action(action: Action) -> bool`

Check if action is valid for environment.

**Parameters:**
- `action`: Action to validate

**Returns:**
- True if valid, False otherwise

---

## Utilities

### Checkpointing

#### Class: `CheckpointManager`

**Location:** `hype.utils.checkpointing`

**Description:** Manages model checkpoints and system state.

#### Methods

##### `save_checkpoint(...) -> str`

Save system checkpoint.

**Parameters:**
- `system` (HyPESystem): System to checkpoint
- `metadata` (Dict): Additional metadata

**Returns:**
- Path to saved checkpoint

**Example:**
```python
manager = CheckpointManager("./checkpoints")
path = manager.save_checkpoint(system, {"epoch": 10})
```

##### `load_checkpoint(system: HyPESystem, path: str)`

Load system checkpoint.

**Parameters:**
- `system`: System to load into
- `path`: Path to checkpoint file

**Example:**
```python
manager.load_checkpoint(system, "checkpoint_latest.pt")
```

---

### Logging

#### Function: `setup_logging`

**Location:** `hype.utils.logging_config`

**Description:** Setup logging configuration.

**Parameters:**
- `log_dir` (str): Directory for log files
- `log_level` (str): Logging level
- `log_to_file` (bool): Whether to log to file

**Example:**
```python
from hype.utils.logging_config import setup_logging

setup_logging(
    log_dir="./logs",
    log_level="INFO",
    log_to_file=True
)
```

---

## Complete Usage Example

```python
from hype.core.config import HyPEConfig
from hype.system import HyPESystem
from hype.core.data_models import State
from hype.environments.adapters import ToolBenchAdapter
from datetime import datetime

# Load configuration
config = HyPEConfig.from_yaml("config.yaml")

# Initialize system
system = HyPESystem(
    config=config.model,
    memory_config=config.principle_memory,
    hmcts_budget=config.hmcts.search_budget
)

# Initialize components
system.initialize(enable_evolution=True)

# Create initial state
initial_state = State(
    observation={"obs": "Starting task"},
    history=[],
    timestamp=datetime.now()
)

# Execute task
trajectory, success = system.execute_task(
    task="Complete the task",
    initial_state=initial_state,
    environment=env,
    environment_adapter=ToolBenchAdapter(),
    environment_type="toolbench",
    collect_trajectory=True
)

# Run offline evolution
system.run_offline_evolution()

# Get statistics
stats = system.get_trajectory_statistics()
print(f"Success rate: {stats['success_rate']:.1%}")

# Cleanup
system.shutdown()
```

---

## Error Handling

All methods may raise the following exceptions:

- `RuntimeError`: System not initialized or operation failed
- `ValueError`: Invalid parameter values
- `ConnectionError`: Database connection failed
- `FileNotFoundError`: Configuration or checkpoint file not found

Always wrap operations in try-except blocks:

```python
try:
    system.initialize()
    trajectory, success = system.execute_task(...)
except RuntimeError as e:
    logger.error(f"System error: {e}")
except Exception as e:
    logger.error(f"Unexpected error: {e}")
finally:
    system.shutdown()
```

---

## Best Practices

1. **Always initialize before use:**
   ```python
   system.initialize()
   ```

2. **Use context managers:**
   ```python
   with HyPESystem() as system:
       trajectory, success = system.execute_task(...)
   ```

3. **Handle errors gracefully:**
   ```python
   try:
       system.execute_task(...)
   except Exception as e:
       logger.error(f"Error: {e}")
   ```

4. **Save checkpoints regularly:**
   ```python
   if task_num % 50 == 0:
       checkpoint_manager.save_checkpoint(system)
   ```

5. **Monitor system statistics:**
   ```python
   stats = system.get_system_stats()
   logger.info(f"Principles: {stats['num_principles']}")
   ```

---

For more examples, see the `examples/` directory.
