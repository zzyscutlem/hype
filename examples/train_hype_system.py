#!/usr/bin/env python3
"""
Example training script for HyPE Agent System.

This script demonstrates the complete training workflow:
1. Initialize HyPE system with configuration
2. Execute tasks in online mode to collect trajectories
3. Run offline evolution to improve models
4. Save trained models and checkpoints

Usage:
    python examples/train_hype_system.py --config config.yaml --num-tasks 100
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hype.core.config import HyPEConfig
from hype.system import HyPESystem
from hype.core.data_models import State
from hype.environments.adapters import ToolBenchAdapter, APIBankAdapter, ALFWorldAdapter
from hype.utils.logging_config import setup_logging
from hype.utils.checkpointing import CheckpointManager


# Mock environment for demonstration
class MockEnvironment:
    """Mock environment for demonstration purposes."""
    
    def __init__(self, env_type: str = "toolbench"):
        self.env_type = env_type
        self.step_count = 0
        self.max_steps = 10
    
    def reset(self):
        """Reset environment."""
        self.step_count = 0
        return {
            "observation": f"Starting {self.env_type} task",
            "available_actions": ["action1", "action2", "action3"]
        }
    
    def step(self, action):
        """Execute action in environment."""
        self.step_count += 1
        
        # Simulate environment response
        reward = 0.5 if self.step_count < self.max_steps else 1.0
        done = self.step_count >= self.max_steps
        
        observation = {
            "observation": f"Step {self.step_count} result",
            "available_actions": ["action1", "action2", "action3"]
        }
        
        return observation, reward, done, {}


def create_initial_state(env_type: str) -> State:
    """
    Create initial state for task.
    
    Args:
        env_type: Type of environment
        
    Returns:
        Initial state
    """
    return State(
        observation={
            "observation": f"Starting {env_type} task",
            "available_actions": ["action1", "action2", "action3"]
        },
        history=[],
        timestamp=datetime.now()
    )


def get_environment_adapter(env_type: str):
    """
    Get environment adapter for environment type.
    
    Args:
        env_type: Type of environment
        
    Returns:
        Environment adapter instance
    """
    if env_type == "toolbench":
        return ToolBenchAdapter()
    elif env_type == "api_bank":
        return APIBankAdapter()
    elif env_type == "alfworld":
        return ALFWorldAdapter()
    else:
        raise ValueError(f"Unknown environment type: {env_type}")


def run_training(
    config_path: str,
    num_tasks: int = 100,
    env_type: str = "toolbench",
    evolution_interval: int = 20,
    checkpoint_interval: int = 50,
    output_dir: str = "./training_output"
):
    """
    Run complete training workflow.
    
    Args:
        config_path: Path to configuration file
        num_tasks: Number of tasks to execute
        env_type: Type of environment (toolbench, api_bank, alfworld)
        evolution_interval: Run evolution every N tasks
        checkpoint_interval: Save checkpoint every N tasks
        output_dir: Directory for outputs
    """
    # Setup logging
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(
        log_dir=str(log_dir),
        log_level="INFO",
        log_to_file=True
    )
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("HyPE Agent System Training")
    logger.info("=" * 80)
    logger.info(f"Configuration: {config_path}")
    logger.info(f"Number of tasks: {num_tasks}")
    logger.info(f"Environment type: {env_type}")
    logger.info(f"Evolution interval: {evolution_interval}")
    logger.info(f"Checkpoint interval: {checkpoint_interval}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 80)
    
    # Load configuration
    logger.info("Loading configuration...")
    config = HyPEConfig.from_yaml(config_path)
    logger.info("Configuration loaded successfully")
    
    # Create output directories
    checkpoint_dir = Path(output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    trajectories_dir = Path(output_dir) / "trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(str(checkpoint_dir))
    
    # Initialize HyPE system
    logger.info("Initializing HyPE system...")
    system = HyPESystem(
        config=config.model,
        memory_config=config.principle_memory,
        hmcts_budget=config.hmcts.search_budget,
        hmcts_exploration_constant=config.hmcts.exploration_constant,
        hmcts_max_depth=config.hmcts.max_depth,
        alignment_threshold=config.sr_adapt.alignment_threshold,
        max_steps_per_task=20,
        # Offline evolution parameters
        replay_buffer_size=config.dpvd.replay_buffer_size,
        training_trigger_threshold=config.dpvd.training_trigger_threshold,
        min_trajectories_for_training=10,
        dpvd_beta=config.dpvd.principle_weight,
        dpvd_learning_rate=config.model.learning_rate,
        dpvd_batch_size=config.model.batch_size,
        dpvd_epochs=config.dpvd.value_training_epochs,
        dpo_beta=config.dpo.beta,
        dpo_learning_rate=config.dpo.learning_rate,
        dpo_batch_size=config.dpo.batch_size,
        dpo_epochs=config.dpo.num_epochs,
        principle_success_threshold=0.5
    )
    
    # Initialize system with evolution enabled
    system.initialize(enable_evolution=True)
    logger.info("HyPE system initialized")
    
    # Get environment adapter
    adapter = get_environment_adapter(env_type)
    
    # Training loop
    logger.info(f"Starting training loop for {num_tasks} tasks...")
    
    successful_tasks = 0
    failed_tasks = 0
    
    try:
        for task_num in range(1, num_tasks + 1):
            logger.info(f"\n{'=' * 80}")
            logger.info(f"Task {task_num}/{num_tasks}")
            logger.info(f"{'=' * 80}")
            
            # Create mock environment
            environment = MockEnvironment(env_type)
            environment.reset()
            
            # Create task description
            task_description = f"Complete {env_type} task {task_num}"
            
            # Create initial state
            initial_state = create_initial_state(env_type)
            
            # Execute task
            try:
                trajectory, success = system.execute_task(
                    task=task_description,
                    initial_state=initial_state,
                    environment=environment,
                    environment_adapter=adapter,
                    environment_type=env_type,
                    collect_trajectory=True
                )
                
                if success:
                    successful_tasks += 1
                    logger.info(f"✓ Task {task_num} completed successfully")
                else:
                    failed_tasks += 1
                    logger.info(f"✗ Task {task_num} failed")
                
                # Log statistics
                stats = system.get_trajectory_statistics()
                logger.info(
                    f"Statistics: {stats['successful']}/{stats['total']} successful "
                    f"({stats['success_rate']:.1%}), "
                    f"avg_reward={stats['avg_reward']:.3f}"
                )
                
            except Exception as e:
                logger.error(f"Error executing task {task_num}: {e}")
                failed_tasks += 1
                continue
            
            # Run offline evolution at intervals
            if task_num % evolution_interval == 0:
                logger.info(f"\n{'=' * 80}")
                logger.info(f"Running offline evolution (after {task_num} tasks)")
                logger.info(f"{'=' * 80}")
                
                try:
                    system.run_offline_evolution()
                    logger.info("Offline evolution completed successfully")
                except Exception as e:
                    logger.error(f"Error during offline evolution: {e}")
            
            # Save checkpoint at intervals
            if task_num % checkpoint_interval == 0:
                logger.info(f"\n{'=' * 80}")
                logger.info(f"Saving checkpoint (after {task_num} tasks)")
                logger.info(f"{'=' * 80}")
                
                try:
                    checkpoint_path = checkpoint_manager.save_checkpoint(
                        system=system,
                        metadata={
                            "task_num": task_num,
                            "successful_tasks": successful_tasks,
                            "failed_tasks": failed_tasks,
                            "timestamp": datetime.now().isoformat()
                        }
                    )
                    logger.info(f"Checkpoint saved: {checkpoint_path}")
                    
                    # Export trajectories
                    traj_path = trajectories_dir / f"trajectories_task_{task_num}.json"
                    system.export_trajectories(str(traj_path))
                    logger.info(f"Trajectories exported: {traj_path}")
                    
                except Exception as e:
                    logger.error(f"Error saving checkpoint: {e}")
        
        # Final statistics
        logger.info(f"\n{'=' * 80}")
        logger.info("Training Complete")
        logger.info(f"{'=' * 80}")
        logger.info(f"Total tasks: {num_tasks}")
        logger.info(f"Successful: {successful_tasks} ({successful_tasks/num_tasks:.1%})")
        logger.info(f"Failed: {failed_tasks} ({failed_tasks/num_tasks:.1%})")
        
        final_stats = system.get_trajectory_statistics()
        logger.info(f"\nFinal Statistics:")
        logger.info(f"  Total trajectories: {final_stats['total']}")
        logger.info(f"  Success rate: {final_stats['success_rate']:.1%}")
        logger.info(f"  Average steps: {final_stats['avg_steps']:.1f}")
        logger.info(f"  Average reward: {final_stats['avg_reward']:.3f}")
        
        # Save final checkpoint
        logger.info(f"\nSaving final checkpoint...")
        final_checkpoint = checkpoint_manager.save_checkpoint(
            system=system,
            metadata={
                "task_num": num_tasks,
                "successful_tasks": successful_tasks,
                "failed_tasks": failed_tasks,
                "final": True,
                "timestamp": datetime.now().isoformat()
            }
        )
        logger.info(f"Final checkpoint saved: {final_checkpoint}")
        
        # Export final trajectories
        final_traj_path = trajectories_dir / "trajectories_final.json"
        system.export_trajectories(str(final_traj_path))
        logger.info(f"Final trajectories exported: {final_traj_path}")
        
        logger.info(f"\n{'=' * 80}")
        logger.info("Training completed successfully!")
        logger.info(f"{'=' * 80}")
        
    except KeyboardInterrupt:
        logger.info("\n\nTraining interrupted by user")
        logger.info("Saving checkpoint before exit...")
        
        try:
            interrupt_checkpoint = checkpoint_manager.save_checkpoint(
                system=system,
                metadata={
                    "interrupted": True,
                    "task_num": task_num,
                    "timestamp": datetime.now().isoformat()
                }
            )
            logger.info(f"Checkpoint saved: {interrupt_checkpoint}")
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
    
    finally:
        # Cleanup
        logger.info("Shutting down system...")
        system.shutdown()
        logger.info("System shutdown complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train HyPE Agent System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=100,
        help="Number of tasks to execute"
    )
    
    parser.add_argument(
        "--env-type",
        type=str,
        choices=["toolbench", "api_bank", "alfworld"],
        default="toolbench",
        help="Type of environment"
    )
    
    parser.add_argument(
        "--evolution-interval",
        type=int,
        default=20,
        help="Run offline evolution every N tasks"
    )
    
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=50,
        help="Save checkpoint every N tasks"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./training_output",
        help="Directory for training outputs"
    )
    
    args = parser.parse_args()
    
    # Run training
    run_training(
        config_path=args.config,
        num_tasks=args.num_tasks,
        env_type=args.env_type,
        evolution_interval=args.evolution_interval,
        checkpoint_interval=args.checkpoint_interval,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
