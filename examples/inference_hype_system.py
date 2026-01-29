#!/usr/bin/env python3
"""
Example inference script for HyPE Agent System.

This script demonstrates how to:
1. Load a trained HyPE system from checkpoint
2. Execute tasks in online inference mode
3. Visualize and analyze results

Usage:
    python examples/inference_hype_system.py --checkpoint ./checkpoints/checkpoint_latest.pt --task "Your task here"
"""

import argparse
import logging
import sys
import json
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


def create_initial_state(env_type: str, task_description: str) -> State:
    """
    Create initial state for task.
    
    Args:
        env_type: Type of environment
        task_description: Task description
        
    Returns:
        Initial state
    """
    return State(
        observation={
            "observation": f"Starting {env_type} task: {task_description}",
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


def print_trajectory_summary(trajectory, success: bool):
    """
    Print a summary of the trajectory.
    
    Args:
        trajectory: Trajectory object
        success: Whether task was successful
    """
    print("\n" + "=" * 80)
    print("TRAJECTORY SUMMARY")
    print("=" * 80)
    print(f"Task: {trajectory.task}")
    print(f"Success: {'✓ Yes' if success else '✗ No'}")
    print(f"Total Steps: {len(trajectory.steps)}")
    print(f"Final Reward: {trajectory.final_reward:.3f}")
    print(f"Trajectory ID: {trajectory.id}")
    
    print("\n" + "-" * 80)
    print("STEP-BY-STEP BREAKDOWN")
    print("-" * 80)
    
    for i, step in enumerate(trajectory.steps, 1):
        print(f"\nStep {i}:")
        print(f"  Hypothesis: {step.hypothesis[:100]}...")
        print(f"  Action: {step.action.description[:100]}...")
        print(f"  Reward: {step.reward:.3f}")
        print(f"  Done: {step.done}")
        
        # Show principles used
        if i <= len(trajectory.principles_used):
            principles = trajectory.principles_used[i-1]
            if principles:
                print(f"  Principles Used ({len(principles)}):")
                for j, p in enumerate(principles[:3], 1):  # Show top 3
                    print(f"    {j}. {p.text[:80]}... (credit: {p.credit_score:.3f})")
    
    print("\n" + "=" * 80)


def run_inference(
    checkpoint_path: str = None,
    config_path: str = "config.yaml",
    task: str = None,
    env_type: str = "toolbench",
    interactive: bool = False,
    output_file: str = None,
    verbose: bool = True
):
    """
    Run inference with HyPE system.
    
    Args:
        checkpoint_path: Path to checkpoint file (optional)
        config_path: Path to configuration file
        task: Task description (required if not interactive)
        env_type: Type of environment
        interactive: Whether to run in interactive mode
        output_file: Path to save trajectory (optional)
        verbose: Whether to print detailed output
    """
    # Setup logging
    log_level = "INFO" if verbose else "WARNING"
    setup_logging(log_level=log_level, log_to_file=False)
    logger = logging.getLogger(__name__)
    
    print("=" * 80)
    print("HyPE Agent System - Inference Mode")
    print("=" * 80)
    
    # Load configuration
    logger.info("Loading configuration...")
    config = HyPEConfig.from_yaml(config_path)
    print(f"✓ Configuration loaded from: {config_path}")
    
    # Initialize HyPE system
    logger.info("Initializing HyPE system...")
    system = HyPESystem(
        config=config.model,
        memory_config=config.principle_memory,
        hmcts_budget=config.hmcts.search_budget,
        hmcts_exploration_constant=config.hmcts.exploration_constant,
        hmcts_max_depth=config.hmcts.max_depth,
        alignment_threshold=config.sr_adapt.alignment_threshold,
        max_steps_per_task=20
    )
    
    # Initialize system (without evolution components for inference)
    system.initialize(enable_evolution=False)
    print("✓ HyPE system initialized")
    
    # Load checkpoint if provided
    if checkpoint_path:
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint_manager = CheckpointManager(str(Path(checkpoint_path).parent))
        checkpoint_manager.load_checkpoint(system, checkpoint_path)
        print(f"✓ Checkpoint loaded from: {checkpoint_path}")
    
    # Get system statistics
    stats = system.get_system_stats()
    print(f"\nSystem Statistics:")
    print(f"  Principles in memory: {stats['num_principles']}")
    print(f"  HMCTS budget: {stats['hmcts_budget']}")
    print(f"  Alignment threshold: {stats['alignment_threshold']}")
    print(f"  Max steps per task: {stats['max_steps_per_task']}")
    
    # Get environment adapter
    adapter = get_environment_adapter(env_type)
    
    try:
        if interactive:
            # Interactive mode
            print("\n" + "=" * 80)
            print("INTERACTIVE MODE")
            print("=" * 80)
            print("Enter tasks to execute (type 'quit' to exit)")
            print("-" * 80)
            
            while True:
                task_input = input("\nTask: ").strip()
                
                if task_input.lower() in ['quit', 'exit', 'q']:
                    print("Exiting interactive mode...")
                    break
                
                if not task_input:
                    print("Please enter a task description")
                    continue
                
                # Execute task
                print(f"\nExecuting task: {task_input}")
                print("-" * 80)
                
                environment = MockEnvironment(env_type)
                environment.reset()
                initial_state = create_initial_state(env_type, task_input)
                
                trajectory, success = system.execute_task(
                    task=task_input,
                    initial_state=initial_state,
                    environment=environment,
                    environment_adapter=adapter,
                    environment_type=env_type,
                    collect_trajectory=False
                )
                
                # Print summary
                print_trajectory_summary(trajectory, success)
        
        else:
            # Single task mode
            if not task:
                print("Error: Task description required in non-interactive mode")
                print("Use --task 'your task' or --interactive")
                return
            
            print(f"\n{'=' * 80}")
            print(f"EXECUTING TASK")
            print(f"{'=' * 80}")
            print(f"Task: {task}")
            print(f"Environment: {env_type}")
            print("-" * 80)
            
            # Create environment
            environment = MockEnvironment(env_type)
            environment.reset()
            initial_state = create_initial_state(env_type, task)
            
            # Execute task
            trajectory, success = system.execute_task(
                task=task,
                initial_state=initial_state,
                environment=environment,
                environment_adapter=adapter,
                environment_type=env_type,
                collect_trajectory=False
            )
            
            # Print summary
            print_trajectory_summary(trajectory, success)
            
            # Save trajectory if requested
            if output_file:
                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Convert trajectory to JSON
                traj_data = {
                    "task": trajectory.task,
                    "success": success,
                    "final_reward": trajectory.final_reward,
                    "num_steps": len(trajectory.steps),
                    "steps": [
                        {
                            "hypothesis": step.hypothesis,
                            "action": {
                                "type": step.action.type,
                                "description": step.action.description,
                                "parameters": step.action.parameters
                            },
                            "reward": step.reward,
                            "done": step.done
                        }
                        for step in trajectory.steps
                    ]
                }
                
                with open(output_file, 'w') as f:
                    json.dump(traj_data, f, indent=2)
                
                print(f"\n✓ Trajectory saved to: {output_file}")
    
    except KeyboardInterrupt:
        print("\n\nInference interrupted by user")
    
    finally:
        # Cleanup
        logger.info("Shutting down system...")
        system.shutdown()
        print("\n✓ System shutdown complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run inference with HyPE Agent System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file (optional)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task description to execute"
    )
    
    parser.add_argument(
        "--env-type",
        type=str,
        choices=["toolbench", "api_bank", "alfworld"],
        default="toolbench",
        help="Type of environment"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save trajectory JSON"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Run inference
    run_inference(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        task=args.task,
        env_type=args.env_type,
        interactive=args.interactive,
        output_file=args.output,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
