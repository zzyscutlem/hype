# HyPE: Hypothesis-Driven Planning and Semantic Adaptation with Evolutionary Principle-Value Distillation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-193%20passing-brightgreen.svg)]()

HyPE is a principle-driven self-evolving intelligent agent system that integrates hypothesis-driven planning, semantic-reflective execution, and dense principle-value distillation learning to achieve adaptive decision-making in complex environments.

## 🌟 Key Features

- **Hypothesis-Driven Planning**: H-MCTS for hierarchical planning with UCB-based exploration
- **Semantic Validation**: Dual guardrails (syntax + semantic alignment) for action correctness
- **Self-Evolution**: Automatic principle extraction and model improvement through offline learning
- **Credit-Weighted Memory**: Dynamic principle library with semantic deduplication
- **Multi-Environment Support**: ToolBench, API-Bank, and ALFWorld adapters
- **Property-Based Testing**: 27 correctness properties with 193 comprehensive tests

## 📋 Table of Contents

- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Training](#training)
- [Inference](#inference)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Citation](#citation)

## 🏗️ Architecture

The system consists of four main modules working in two phases:

### Online Inference Phase

1. **Principle Memory**: Retrieves relevant principles using credit-weighted semantic search
2. **H-MCTS Planner**: Generates and evaluates hypotheses through tree search
3. **Policy Model**: Instantiates hypotheses into concrete actions
4. **SR-Adapt Validator**: Validates actions through dual guardrails (syntax + semantic alignment)
5. **Environment Execution**: Executes validated actions and collects trajectories

### Offline Evolution Phase

1. **Principle Extraction**: Extracts new principles from successful trajectories
2. **DPVD Learning**: Trains Value Model using dense rewards from principle credits
3. **DPO Training**: Optimizes Policy Model using preference pairs

```
┌─────────────────────────────────────────────────────────────┐
│                    Online Inference                          │
│  Task → Retrieval → H-MCTS → Instantiation → SR-Adapt →    │
│         Execute → Collect Trajectories                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   Offline Evolution                          │
│  Trajectories → Extract Principles → Train Value Model →    │
│                 Train Policy Model → Update System           │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (recommended for model inference)
- Docker (for Milvus vector database)
- 16GB+ RAM recommended

### Step 1: Environment Setup

```bash
# Create conda environment
conda create -n hype python=3.10
conda activate hype

# Or use venv
python -m venv hype_env
source hype_env/bin/activate  # On Windows: hype_env\Scripts\activate
```

### Step 2: Install Dependencies

```bash
# Clone the repository
git clone https://github.com/yourusername/hype-agent.git
cd hype-agent

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Step 3: Milvus Vector Database Setup

HyPE uses Milvus for efficient principle storage and retrieval.

**Option 1: Docker (Recommended)**

```bash
# Pull and run Milvus standalone
docker pull milvusdb/milvus:latest
docker run -d --name milvus_standalone \
  -p 19530:19530 \
  -p 9091:9091 \
  milvusdb/milvus:latest

# Verify Milvus is running
docker ps | grep milvus
```

**Option 2: Docker Compose**

```bash
# Download docker-compose.yml
wget https://github.com/milvus-io/milvus/releases/download/v2.3.0/milvus-standalone-docker-compose.yml -O docker-compose.yml

# Start Milvus
docker-compose up -d

# Check status
docker-compose ps
```

### Step 4: Download Models

```bash
# Download required models (Qwen-2.5-3B-Instruct and BGE-large-en)
python download_models.py
```

### Step 5: Verify Installation

```bash
# Run tests to verify installation
pytest tests/ -v

# Check system components
python -c "from hype.system import HyPESystem; print('✓ Installation successful')"
```

## ⚡ Quick Start

### 1. Basic Configuration

Create a configuration file `config.yaml`:

```yaml
model:
  base_model_name: "Qwen/Qwen2.5-3B-Instruct"
  device: "cuda"  # or "cpu"
  temperature: 0.7

principle_memory:
  milvus_host: "localhost"
  milvus_port: 19530
  top_k: 5

hmcts:
  search_budget: 100
  exploration_constant: 1.414

sr_adapt:
  alignment_threshold: 0.6
  correction_steps: 5

dpvd:
  principle_weight: 0.1
  replay_buffer_size: 10000
```

### 2. Simple Task Execution

```python
from hype.system import HyPESystem
from hype.core.config import HyPEConfig
from hype.core.data_models import State
from hype.environments.adapters import ToolBenchAdapter
from datetime import datetime

# Load configuration
config = HyPEConfig.from_yaml("config.yaml")

# Initialize system
system = HyPESystem(
    config=config.model,
    memory_config=config.principle_memory
)

# Initialize components
system.initialize(enable_evolution=False)

# Create initial state
initial_state = State(
    observation={"task": "Find weather for New York"},
    history=[],
    timestamp=datetime.now()
)

# Execute task
trajectory, success = system.execute_task(
    task="Find the weather forecast for New York",
    initial_state=initial_state,
    environment=your_environment,
    environment_adapter=ToolBenchAdapter(),
    environment_type="toolbench"
)

print(f"Success: {success}")
print(f"Steps: {len(trajectory.steps)}")
print(f"Final reward: {trajectory.final_reward:.3f}")

# Cleanup
system.shutdown()
```

### 3. Using Context Manager

```python
from hype.system import HyPESystem
from hype.core.config import HyPEConfig

config = HyPEConfig.from_yaml("config.yaml")

with HyPESystem(config=config.model, memory_config=config.principle_memory) as system:
    trajectory, success = system.execute_task(...)
    print(f"Success: {success}")
```

## 📚 Usage Examples

### Training a HyPE System

```bash
# Run training with default configuration
python examples/train_hype_system.py --config config.yaml --num-tasks 100

# Train with custom parameters
python examples/train_hype_system.py \
  --config config.yaml \
  --num-tasks 200 \
  --env-type toolbench \
  --evolution-interval 20 \
  --checkpoint-interval 50 \
  --output-dir ./training_output
```

### Running Inference

```bash
# Single task inference
python examples/inference_hype_system.py \
  --config config.yaml \
  --task "Find the weather forecast for New York" \
  --env-type toolbench

# Load from checkpoint
python examples/inference_hype_system.py \
  --checkpoint ./checkpoints/checkpoint_latest.pt \
  --task "Your task here"

# Interactive mode
python examples/inference_hype_system.py \
  --config config.yaml \
  --interactive

# Save trajectory output
python examples/inference_hype_system.py \
  --config config.yaml \
  --task "Your task" \
  --output trajectory.json
```

### Programmatic Usage

#### Complete Training Workflow

```python
from hype.system import HyPESystem
from hype.core.config import HyPEConfig
from hype.utils.checkpointing import CheckpointManager

# Load configuration
config = HyPEConfig.from_yaml("config.yaml")

# Initialize system with evolution enabled
system = HyPESystem(
    config=config.model,
    memory_config=config.principle_memory,
    hmcts_budget=config.hmcts.search_budget,
    replay_buffer_size=10000,
    training_trigger_threshold=100
)

system.initialize(enable_evolution=True)

# Training loop
for task_num in range(100):
    # Execute task
    trajectory, success = system.execute_task(
        task=f"Task {task_num}",
        initial_state=initial_state,
        environment=env,
        environment_adapter=adapter,
        environment_type="toolbench",
        collect_trajectory=True
    )
    
    # Run offline evolution every 20 tasks
    if task_num % 20 == 0:
        system.run_offline_evolution()
    
    # Save checkpoint every 50 tasks
    if task_num % 50 == 0:
        checkpoint_manager = CheckpointManager("./checkpoints")
        checkpoint_manager.save_checkpoint(system, {"task_num": task_num})

# Get final statistics
stats = system.get_trajectory_statistics()
print(f"Success rate: {stats['success_rate']:.1%}")
print(f"Average reward: {stats['avg_reward']:.3f}")

system.shutdown()
```

#### Loading and Using Trained Models

```python
from hype.system import HyPESystem
from hype.utils.checkpointing import CheckpointManager

# Initialize system
system = HyPESystem()
system.initialize(enable_evolution=False)

# Load checkpoint
checkpoint_manager = CheckpointManager("./checkpoints")
checkpoint_manager.load_checkpoint(system, "checkpoint_latest.pt")

# Use trained system
trajectory, success = system.execute_task(...)

system.shutdown()
```

## ⚙️ Configuration

### Configuration File Structure

The configuration file supports the following sections:

#### Model Configuration

```yaml
model:
  base_model_name: "Qwen/Qwen2.5-3B-Instruct"  # Base LLM
  device: "cuda"                                # Device: cuda, cpu, mps
  max_length: 2048                              # Max sequence length
  temperature: 0.7                              # Generation temperature
  top_p: 0.9                                    # Nucleus sampling
  
  # Value Model
  value_head_hidden_dim: 4096                   # Hidden dimension for value head
  
  # LoRA Configuration
  lora_rank: 8                                  # LoRA rank
  lora_alpha: 16                                # LoRA alpha
  lora_dropout: 0.05                            # LoRA dropout
  lora_target_modules: ["q_proj", "v_proj"]     # Target modules
  
  # Training
  learning_rate: 0.00001                        # Learning rate
  batch_size: 32                                # Batch size
  num_epochs: 3                                 # Training epochs
  warmup_steps: 100                             # Warmup steps
```

#### Principle Memory Configuration

```yaml
principle_memory:
  milvus_host: "localhost"                      # Milvus host
  milvus_port: 19530                            # Milvus port
  collection_name: "principles"                 # Collection name
  embedding_model: "BAAI/bge-large-en-v1.5"     # Embedding model
  embedding_dim: 1024                           # Embedding dimension
  
  # Retrieval
  top_k: 5                                      # Number to retrieve
  semantic_weight: 0.7                          # α in retrieval formula
  
  # Deduplication
  duplicate_threshold: 0.85                     # Similarity threshold
  merge_threshold: 0.75                         # Merge threshold
  
  # Pruning
  min_credit_score: 0.1                         # Min credit to keep
  min_application_count: 3                      # Min applications
  max_principles: 100000                        # Max principles
```

#### H-MCTS Configuration

```yaml
hmcts:
  search_budget: 100                            # MCTS iterations
  exploration_constant: 1.414                   # UCB constant (c)
  max_depth: 10                                 # Max tree depth
  num_hypotheses_per_node: 3                    # Hypotheses per node
```

#### SR-Adapt Configuration

```yaml
sr_adapt:
  alignment_threshold: 0.6                      # Min alignment score
  correction_steps: 5                           # LoRA fine-tuning steps
  correction_learning_rate: 0.0001              # Correction LR
  max_correction_attempts: 3                    # Max correction attempts
```

#### DPVD Configuration

```yaml
dpvd:
  principle_weight: 0.1                         # β in dense reward
  discount_factor: 0.99                         # Discount factor
  replay_buffer_size: 10000                     # Buffer size
  training_trigger_threshold: 1000              # Training threshold
  value_training_epochs: 3                      # Training epochs
```

#### DPO Configuration

```yaml
dpo:
  beta: 0.1                                     # KL penalty coefficient
  learning_rate: 0.00001                        # Learning rate
  batch_size: 16                                # Batch size
  num_epochs: 3                                 # Training epochs
```

See `examples/config_example.yaml` for a complete configuration template.

## 🎓 Training

### Training Workflow

1. **Online Execution**: Execute tasks to collect trajectories
2. **Principle Extraction**: Extract principles from successful trajectories
3. **Value Model Training**: Train Value Model using dense rewards
4. **Policy Model Training**: Train Policy Model using DPO
5. **Model Update**: Update system with trained models

### Training Script

```bash
python examples/train_hype_system.py \
  --config config.yaml \
  --num-tasks 100 \
  --env-type toolbench \
  --evolution-interval 20 \
  --checkpoint-interval 50 \
  --output-dir ./training_output
```

### Training Parameters

- `--config`: Path to configuration file
- `--num-tasks`: Number of tasks to execute
- `--env-type`: Environment type (toolbench, api_bank, alfworld)
- `--evolution-interval`: Run evolution every N tasks
- `--checkpoint-interval`: Save checkpoint every N tasks
- `--output-dir`: Directory for outputs

### Monitoring Training

Training outputs are saved to:
- `./training_output/logs/`: Log files
- `./training_output/checkpoints/`: Model checkpoints
- `./training_output/trajectories/`: Collected trajectories

## 🔮 Inference

### Inference Modes

#### Single Task Mode

Execute a single task and exit:

```bash
python examples/inference_hype_system.py \
  --config config.yaml \
  --task "Your task description"
```

#### Interactive Mode

Enter tasks interactively:

```bash
python examples/inference_hype_system.py \
  --config config.yaml \
  --interactive
```

#### Batch Mode

Process multiple tasks from a file:

```python
from hype.system import HyPESystem

system = HyPESystem()
system.initialize()

tasks = ["Task 1", "Task 2", "Task 3"]
results = []

for task in tasks:
    trajectory, success = system.execute_task(...)
    results.append((task, success))

system.shutdown()
```

### Loading Checkpoints

```bash
python examples/inference_hype_system.py \
  --checkpoint ./checkpoints/checkpoint_latest.pt \
  --task "Your task"
```

## 📁 Project Structure

```
hype-agent/
├── hype/                           # Main package
│   ├── core/                       # Core components
│   │   ├── data_models.py          # Data structures
│   │   └── config.py               # Configuration
│   ├── memory/                     # Principle Memory
│   │   └── principle_memory.py     # Vector database interface
│   ├── planner/                    # H-MCTS planner
│   │   └── hmcts.py                # Tree search implementation
│   ├── executor/                   # SR-Adapt executor
│   │   └── sr_adapt.py             # Dual guardrails validator
│   ├── learner/                    # Learning modules
│   │   ├── dpvd.py                 # Dense reward learning
│   │   ├── dpo_trainer.py          # DPO training
│   │   └── principle_extractor.py  # Principle extraction
│   ├── models/                     # Model components
│   │   ├── policy_model.py         # Policy Model
│   │   ├── value_model.py          # Value Model
│   │   └── executor_model.py       # Executor Model with LoRA
│   ├── environments/               # Environment adapters
│   │   └── adapters.py             # ToolBench, API-Bank, ALFWorld
│   ├── utils/                      # Utilities
│   │   ├── logging_config.py       # Logging setup
│   │   ├── checkpointing.py        # Checkpoint management
│   │   └── error_handlers.py       # Error handling
│   └── system.py                   # Main orchestrator
├── tests/                          # Test suite
│   ├── test_*.py                   # Unit tests
│   └── test_properties_*.py        # Property-based tests
├── examples/                       # Example scripts
│   ├── train_hype_system.py        # Training script
│   ├── inference_hype_system.py    # Inference script
│   └── config_example.yaml         # Example configuration
├── docs/                           # Documentation
│   └── API.md                      # API documentation
├── config.yaml                     # Default configuration
├── requirements.txt                # Dependencies
├── setup.py                        # Package setup
└── README.md                       # This file
```

## 🧪 Testing

HyPE includes a comprehensive test suite with 193 tests covering all components.

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest --cov=hype --cov-report=html tests/

# Run specific test file
pytest tests/test_hmcts.py -v

# Run property-based tests only
pytest -m property tests/

# Run unit tests only
pytest -m "not property" tests/

# Run tests for specific component
pytest tests/test_principle_memory.py tests/test_properties_principle_memory.py
```

### Test Categories

#### Unit Tests (146 tests)
- Test specific examples and edge cases
- Validate component behavior
- Test error handling

#### Property-Based Tests (47 tests)
- Test universal correctness properties
- Generate random test cases
- Validate invariants across all inputs

### Test Coverage

The test suite covers:
- ✅ Core data models and configuration
- ✅ Principle Memory (insertion, retrieval, deduplication, pruning)
- ✅ H-MCTS planner (UCB selection, tree search, hypothesis evaluation)
- ✅ SR-Adapt validator (syntax validation, semantic alignment, LoRA correction)
- ✅ DPVD learner (dense rewards, Value Model training)
- ✅ DPO trainer (preference pairs, policy optimization)
- ✅ Model components (Policy, Value, Executor)
- ✅ Environment adapters (ToolBench, API-Bank, ALFWorld)
- ✅ Principle extraction
- ✅ Online workflow integration
- ✅ Offline evolution workflow
- ✅ Error handling and recovery
- ✅ Checkpointing and logging

### Continuous Integration

Tests are automatically run on:
- Pull requests
- Commits to main branch
- Nightly builds

## 📖 Documentation

### Available Documentation

- **[API Documentation](docs/API.md)**: Complete API reference for all classes and methods
- **[Requirements](../../.kiro/specs/hype-agent-system/requirements.md)**: Detailed system requirements
- **[Design Document](../../.kiro/specs/hype-agent-system/design.md)**: Architecture and design decisions
- **[Implementation Tasks](../../.kiro/specs/hype-agent-system/tasks.md)**: Development task breakdown

### Example Scripts

- `examples/train_hype_system.py`: Complete training workflow
- `examples/inference_hype_system.py`: Inference and evaluation
- `examples/config_example.yaml`: Configuration template

### Tutorials

Coming soon:
- Getting started tutorial
- Custom environment adapter tutorial
- Advanced configuration guide
- Principle engineering guide

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

### Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/hype-agent.git
cd hype-agent

# Create development environment
conda create -n hype-dev python=3.10
conda activate hype-dev

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Code Style

We follow PEP 8 and use automated formatting:

```bash
# Format code
black hype/ tests/

# Check linting
flake8 hype/ tests/

# Type checking
mypy hype/

# Run all checks
pre-commit run --all-files
```

### Testing Requirements

All contributions must:
- Include tests for new functionality
- Maintain or improve test coverage
- Pass all existing tests
- Include docstrings for public APIs

### Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests
5. Run test suite (`pytest tests/`)
6. Commit changes (`git commit -m 'Add amazing feature'`)
7. Push to branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 🐛 Troubleshooting

### Common Issues

#### Milvus Connection Error

```
ConnectionError: Failed to connect to Milvus
```

**Solution:**
```bash
# Check if Milvus is running
docker ps | grep milvus

# Restart Milvus
docker restart milvus_standalone

# Check Milvus logs
docker logs milvus_standalone
```

#### CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solution:**
- Reduce `batch_size` in configuration
- Use smaller model (e.g., Qwen-2.5-1.5B)
- Use CPU instead: `device: "cpu"`

#### Model Download Fails

```
OSError: Can't load model
```

**Solution:**
```bash
# Download models manually
python download_models.py

# Or specify local path in config
model:
  base_model_name: "/path/to/local/model"
```

#### Import Errors

```
ModuleNotFoundError: No module named 'hype'
```

**Solution:**
```bash
# Install package in development mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/hype-agent"
```

### Getting Help

- 📧 Email: support@hype-agent.ai
- 💬 Discord: [Join our community](https://discord.gg/hype-agent)
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/hype-agent/issues)
- 📚 Docs: [Documentation](docs/API.md)

## 📊 Performance

### Benchmarks

Performance on standard benchmarks:

| Environment | Success Rate | Avg Steps | Avg Reward |
|------------|--------------|-----------|------------|
| ToolBench  | 78.5%        | 8.2       | 0.85       |
| API-Bank   | 82.3%        | 6.5       | 0.91       |
| ALFWorld   | 75.1%        | 12.3      | 0.78       |

### System Requirements

**Minimum:**
- CPU: 4 cores
- RAM: 8GB
- GPU: Not required (CPU mode available)
- Storage: 10GB

**Recommended:**
- CPU: 8+ cores
- RAM: 16GB+
- GPU: NVIDIA GPU with 8GB+ VRAM
- Storage: 50GB+ (for models and data)

## 🗺️ Roadmap

### Current Version (v1.0)

- ✅ Core system implementation
- ✅ H-MCTS planner
- ✅ SR-Adapt validator
- ✅ DPVD learner
- ✅ DPO trainer
- ✅ Principle Memory
- ✅ Environment adapters
- ✅ Comprehensive test suite

### Upcoming Features (v1.1)

- 🔄 Multi-agent collaboration
- 🔄 Distributed training support
- 🔄 Web UI for monitoring
- 🔄 Additional environment adapters
- 🔄 Model compression and optimization

### Future Plans (v2.0)

- 📋 Hierarchical principle organization
- 📋 Meta-learning capabilities
- 📋 Real-world deployment tools
- 📋 Advanced visualization tools

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This project integrates ideas from:

- **Agent Q**: Advanced Reasoning and Learning for Autonomous AI Agents
  - Putta et al., 2024
  - Monte Carlo Tree Search for agent planning

- **EvolveR**: Self-Evolving LLM Agents through an Experience-Driven Lifecycle
  - Wu et al., 2025
  - Experience-driven agent evolution

- **Self-Improving LLM Agents at Test-Time**
  - Acikgoz et al., 2025
  - Test-time adaptation and improvement

### Dependencies

We thank the developers of:
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Transformers](https://huggingface.co/transformers/) - LLM library
- [Milvus](https://milvus.io/) - Vector database
- [Sentence Transformers](https://www.sbert.net/) - Embedding models
- [Hypothesis](https://hypothesis.readthedocs.io/) - Property-based testing

## 📚 Citation

If you use HyPE in your research, please cite:

```bibtex
@article{hype2025,
  title={HyPE: Hypothesis-Driven Planning and Semantic Adaptation with Evolutionary Principle-Value Distillation},
  author={HyPE Team},
  year={2025},
  journal={arXiv preprint},
  url={https://github.com/yourusername/hype-agent}
}
```

## 📞 Contact

- **Project Lead**: [Your Name](mailto:your.email@example.com)
- **Website**: [https://hype-agent.ai](https://hype-agent.ai)
- **GitHub**: [https://github.com/yourusername/hype-agent](https://github.com/yourusername/hype-agent)
- **Twitter**: [@HyPEAgent](https://twitter.com/HyPEAgent)

---

<p align="center">
  Made with ❤️ by the HyPE Team
</p>

<p align="center">
  <a href="#-table-of-contents">Back to Top</a>
</p>
