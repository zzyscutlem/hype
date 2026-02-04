#!/usr/bin/env python3
"""
Cold boot HyPE system on API-Bank training set to accumulate principles.

This script:
1. Loads API-Bank training data
2. Initializes HyPE system with empty principle memory
3. Executes tasks from training set
4. Extracts principles from successful trajectories
5. Builds up principle memory through experience
6. Saves checkpoints periodically

This implements the "cold boot" phase where the system learns from scratch.
"""

import os
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from tqdm import tqdm

from hype.system import HyPESystem
from hype.core.config import HyPEConfig, ModelConfig, PrincipleMemoryConfig
from hype.core.data_models import State
from hype.environments.adapters import APIBankAdapter
from hype.utils.logging_config import setup_logging
from hype.utils.checkpointing import CheckpointManager


# Mock API-Bank environment for cold boot
class MockAPIBankEnvironment:
    """
    Mock environment for API-Bank cold boot.
    
    Since we don't have the actual API implementations during cold boot,
    we use the ground truth from the dataset to simulate execution.
    """
    
    def __init__(self, ground_truth: Dict[str, Any]):
        """
        Initialize mock environment with ground truth.
        
        Args:
            ground_truth: Ground truth data from dataset
        """
        self.ground_truth = ground_truth
        self.current_step = 0
        
        # Parse expected API calls from output field
        # API-Bank format: "API-Request: [ApiName(param='value', ...)]"
        self.expected_api_calls = self._parse_expected_calls(ground_truth.get("output", ""))
        self.max_steps = max(1, len(self.expected_api_calls))
    
    def _parse_expected_calls(self, output: str) -> List[Dict[str, Any]]:
        """
        Parse expected API calls from output string.
        
        Args:
            output: Output string like "API-Request: [ApiName(param='value')]"
            
        Returns:
            List of parsed API calls
        """
        import re
        import logging
        logger = logging.getLogger(__name__)
        
        # Extract API call from output
        # Format: "API-Request: [ApiName(param='value', ...)]"
        match = re.search(r'API-Request:\s*\[([^\]]+)\]', output)
        if not match:
            logger.warning(f"Could not parse API call from output: {output[:100]}")
            return []
        
        api_call_str = match.group(1)
        
        # Extract API name (everything before the first '(')
        api_name_match = re.match(r'([A-Za-z_][A-Za-z0-9_]*)', api_call_str)
        if not api_name_match:
            logger.warning(f"Could not extract API name from: {api_call_str}")
            return []
        
        api_name = api_name_match.group(1)
        
        # For now, just store the API name
        # We can add parameter parsing later if needed
        return [{
            "api_name": api_name,
            "raw_call": api_call_str
        }]
    
    def execute(self, action: Any) -> tuple:
        """
        Execute action using ground truth.
        
        Returns:
            (next_state, reward, done) - 3 values to match SR-Adapt expectations
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Check if we've reached the end
        if self.current_step >= self.max_steps:
            logger.info(f"Task complete: reached max_steps={self.max_steps}")
            return (
                {"observation": "Task complete"},
                0.0,
                True
            )
        
        # Get ground truth for this step (if available)
        if self.expected_api_calls and self.current_step < len(self.expected_api_calls):
            gt_call = self.expected_api_calls[self.current_step]
            logger.info(f"Ground truth API call: {gt_call.get('api_name', 'unknown')}")
            logger.info(f"Raw call: {gt_call.get('raw_call', '')}")
        else:
            gt_call = None
            logger.info("No ground truth available for this step")
        
        # Calculate reward
        if gt_call:
            matches = self._matches_ground_truth(action, gt_call)
            reward = 1.0 if matches else 0.5
            logger.info(f"Action match: {matches}, reward: {reward}")
        else:
            # No ground truth - give partial reward if action is well-formed
            reward = 0.6 if self._is_well_formed(action) else 0.3
            logger.info(f"No ground truth - well-formed check, reward: {reward}")
        
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        # Create next state
        next_state = {
            "observation": gt_call.get("result", "Action executed") if gt_call else "Action executed",
            "step": self.current_step,
            "remaining_steps": self.max_steps - self.current_step
        }
        
        logger.info(f"Step {self.current_step}/{self.max_steps}, reward={reward}, done={done}")
        
        return next_state, reward, done
    
    def _matches_ground_truth(self, action: Any, gt_call: Dict) -> bool:
        """Check if action matches ground truth."""
        import logging
        logger = logging.getLogger(__name__)
        
        if not hasattr(action, 'parameters'):
            logger.warning("Action has no parameters attribute")
            return False
        
        # Check api_name (updated parameter name)
        api_name = action.parameters.get('api_name', '')
        gt_api = gt_call.get('api_name', '')
        
        logger.info(f"Comparing: action api_name='{api_name}' vs ground truth='{gt_api}'")
        
        if not api_name or not gt_api:
            logger.warning(f"Missing API name: action='{api_name}', gt='{gt_api}'")
            return False
        
        # Flexible matching
        match = (
            api_name.lower() in gt_api.lower() or 
            gt_api.lower() in api_name.lower() or
            api_name.lower() == gt_api.lower()
        )
        
        logger.info(f"Match result: {match}")
        return match
    
    def _is_well_formed(self, action: Any) -> bool:
        """Check if action is well-formed (has required fields)."""
        if not hasattr(action, 'parameters'):
            return False
        
        # Check if it has the basic API-Bank structure
        has_api_name = 'api_name' in action.parameters
        has_method = 'method' in action.parameters
        has_params = 'parameters' in action.parameters
        
        return has_api_name and has_method and has_params
    
    def reset(self):
        """Reset environment."""
        self.current_step = 0


def load_training_data(data_path: str) -> List[Dict[str, Any]]:
    """
    Load API-Bank training data.
    
    Args:
        data_path: Path to training data JSON file
        
    Returns:
        List of training examples
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def cold_boot_on_apibank(
    data_path: str,
    config_path: str,
    output_dir: str,
    num_tasks: int = None,
    evolution_interval: int = 20,
    checkpoint_interval: int = 50,
    max_steps_per_task: int = 10
):
    """
    Run cold boot on API-Bank training set.
    
    Args:
        data_path: Path to API-Bank training data
        config_path: Path to HyPE configuration file
        output_dir: Directory for outputs (checkpoints, logs, etc.)
        num_tasks: Number of tasks to process (None for all)
        evolution_interval: Run offline evolution every N tasks
        checkpoint_interval: Save checkpoint every N tasks
        max_steps_per_task: Maximum steps per task
    """
    # Setup output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_dir = output_path / "logs"
    log_dir.mkdir(exist_ok=True)
    setup_logging(
        log_dir=str(log_dir),
        level="INFO"
    )
    logger = logging.getLogger(__name__)
    
    print("\n" + "=" * 70)
    print("🚀 HyPE Cold Boot on API-Bank Training Set")
    print("=" * 70)
    print(f"Data path: {data_path}")
    print(f"Config path: {config_path}")
    print(f"Output directory: {output_path.absolute()}")
    print(f"Evolution interval: {evolution_interval} tasks")
    print(f"Checkpoint interval: {checkpoint_interval} tasks")
    print("=" * 70 + "\n")
    
    # Load training data
    logger.info("Loading API-Bank training data...")
    print("📥 Loading training data...")
    training_data = load_training_data(data_path)
    
    if num_tasks is not None:
        training_data = training_data[:num_tasks]
    
    print(f"   ✅ Loaded {len(training_data)} training examples\n")
    logger.info(f"Loaded {len(training_data)} training examples")
    
    # Load configuration
    logger.info("Loading configuration...")
    print("⚙️  Loading configuration...")
    config = HyPEConfig.from_yaml(config_path)
    print("   ✅ Configuration loaded\n")
    
    # Initialize HyPE system
    logger.info("Initializing HyPE system...")
    print("🔧 Initializing HyPE system...")
    system = HyPESystem(
        config=config.model,
        memory_config=config.principle_memory,
        hmcts_budget=config.hmcts.search_budget,
        max_steps_per_task=max_steps_per_task,
        replay_buffer_size=10000,
        training_trigger_threshold=100,
        min_trajectories_for_training=3  # 🔥 从 10 改为 3，允许周期性训练触发
    )
    
    # Initialize with evolution enabled
    system.initialize(enable_evolution=True)
    print("   ✅ System initialized\n")
    
    # Initialize checkpoint manager
    checkpoint_dir = output_path / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_manager = CheckpointManager(str(checkpoint_dir))
    
    # Initialize API-Bank adapter
    adapter = APIBankAdapter()
    
    # Statistics
    stats = {
        "total_tasks": len(training_data),
        "completed_tasks": 0,
        "successful_tasks": 0,
        "failed_tasks": 0,
        "total_steps": 0,
        "total_reward": 0.0,
        "principles_extracted": 0,
        "evolution_runs": 0
    }
    
    # Cold boot loop
    logger.info("Starting cold boot loop...")
    print("🔄 Starting cold boot loop...\n")
    print("=" * 70)
    
    try:
        for task_idx, example in enumerate(tqdm(training_data, desc="Cold Boot Progress")):
            print(f"\n{'='*70}")
            print(f"📋 Task {task_idx + 1}/{len(training_data)}")
            print(f"{'='*70}")
            
            logger.info(f"Task {task_idx + 1}/{len(training_data)}")
            
            # Extract task information
            task_description = example.get("instruction", example.get("query", ""))
            if not task_description:
                logger.warning(f"Skipping task {task_idx}: no instruction found")
                print(f"⚠️  Skipping: no instruction found\n")
                continue
            
            # Print task details
            print(f"📝 Task: {task_description[:100]}...")
            
            # Parse and display expected API call
            expected_output = example.get("output", "")
            print(f"🎯 Expected output: {expected_output[:150]}...")
            
            # Create mock environment to parse expected calls
            temp_env = MockAPIBankEnvironment(example)
            if temp_env.expected_api_calls:
                print(f"📋 Parsed API calls: {len(temp_env.expected_api_calls)}")
                for i, call in enumerate(temp_env.expected_api_calls):
                    print(f"   {i+1}. {call.get('api_name', 'unknown')}")
            else:
                print(f"⚠️  Could not parse expected API calls")
            print()
            
            # Create initial state
            initial_state = State(
                observation={
                    "task": task_description,
                    "available_apis": example.get("available_apis", [])
                },
                history=[],
                timestamp=datetime.now()
            )
            
            # Create mock environment with ground truth
            environment = MockAPIBankEnvironment(example)
            
            try:
                # Execute task
                print(f"🚀 Executing task...")
                trajectory, success = system.execute_task(
                    task=task_description,
                    initial_state=initial_state,
                    environment=environment,
                    environment_adapter=adapter,
                    environment_type="api_bank",
                    collect_trajectory=True
                )
                
                # Update statistics
                stats["completed_tasks"] += 1
                if success:
                    stats["successful_tasks"] += 1
                    print(f"✅ Task SUCCEEDED")
                else:
                    stats["failed_tasks"] += 1
                    print(f"❌ Task FAILED")
                    
                stats["total_steps"] += len(trajectory.steps)
                stats["total_reward"] += trajectory.final_reward
                
                # Print detailed results
                print(f"📊 Results:")
                print(f"   Steps: {len(trajectory.steps)}")
                print(f"   Reward: {trajectory.final_reward:.3f}")
                print(f"   Success: {success}")
                
                # Print action details
                if trajectory.steps:
                    last_step = trajectory.steps[-1]
                    print(f"   Action type: {last_step.action.type}")
                    print(f"   Action params: {list(last_step.action.parameters.keys())}")
                
                logger.info(
                    f"Task {task_idx + 1} complete: "
                    f"success={success}, "
                    f"steps={len(trajectory.steps)}, "
                    f"reward={trajectory.final_reward:.3f}"
                )
                
            except Exception as e:
                logger.error(f"Error executing task {task_idx + 1}: {e}")
                print(f"❌ Error: {e}")
                stats["failed_tasks"] += 1
                continue
            
            # Run offline evolution periodically
            if (task_idx + 1) % evolution_interval == 0:
                logger.info(f"Running offline evolution at task {task_idx + 1}...")
                print(f"\n🧬 Running offline evolution (task {task_idx + 1})...")
                
                try:
                    evolution_results = system.run_offline_evolution(
                        extract_principles=True,
                        train_value_model=True,
                        train_policy_model=True,
                        add_to_buffer=True
                    )
                    
                    stats["principles_extracted"] += evolution_results.get("principles_extracted", 0)
                    stats["evolution_runs"] += 1
                    
                    logger.info(
                        f"Evolution complete: "
                        f"{evolution_results.get('principles_extracted', 0)} principles extracted"
                    )
                    print(f"   ✅ Extracted {evolution_results.get('principles_extracted', 0)} principles\n")
                    
                except Exception as e:
                    logger.error(f"Error during evolution: {e}")
                    print(f"   ⚠️  Evolution failed: {e}\n")
            
            # Save checkpoint periodically
            if (task_idx + 1) % checkpoint_interval == 0:
                logger.info(f"Saving checkpoint at task {task_idx + 1}...")
                print(f"\n💾 Saving checkpoint (task {task_idx + 1})...")
                
                try:
                    checkpoint_manager.save_checkpoint(
                        system=system,
                        metadata={
                            "task_idx": task_idx + 1,
                            "stats": stats,
                            "timestamp": datetime.now().isoformat()
                        }
                    )
                    print(f"   ✅ Checkpoint saved\n")
                    
                except Exception as e:
                    logger.error(f"Error saving checkpoint: {e}")
                    print(f"   ⚠️  Checkpoint failed: {e}\n")
        
        # Final evolution run
        logger.info("Running final offline evolution...")
        print("\n🧬 Running final offline evolution...")
        
        try:
            evolution_results = system.run_offline_evolution(
                extract_principles=True,
                train_value_model=True,
                train_policy_model=True,
                add_to_buffer=True
            )
            
            stats["principles_extracted"] += evolution_results.get("principles_extracted", 0)
            stats["evolution_runs"] += 1
            
            print(f"   ✅ Final evolution complete\n")
            
        except Exception as e:
            logger.error(f"Error during final evolution: {e}")
            print(f"   ⚠️  Final evolution failed: {e}\n")
        
        # Save final checkpoint
        logger.info("Saving final checkpoint...")
        print("💾 Saving final checkpoint...")
        
        try:
            checkpoint_manager.save_checkpoint(
                system=system,
                metadata={
                    "task_idx": len(training_data),
                    "stats": stats,
                    "timestamp": datetime.now().isoformat(),
                    "final": True
                },
                checkpoint_name="checkpoint_final.pt"
            )
            print("   ✅ Final checkpoint saved\n")
            
        except Exception as e:
            logger.error(f"Error saving final checkpoint: {e}")
            print(f"   ⚠️  Final checkpoint failed: {e}\n")
        
        # Save statistics
        stats_file = output_path / "cold_boot_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Print final statistics
        print("=" * 70)
        print("✅ Cold Boot Complete!")
        print("=" * 70)
        print(f"Total tasks: {stats['total_tasks']}")
        print(f"Completed: {stats['completed_tasks']}")
        print(f"Successful: {stats['successful_tasks']} ({stats['successful_tasks']/max(stats['completed_tasks'], 1)*100:.1f}%)")
        print(f"Failed: {stats['failed_tasks']}")
        print(f"Total steps: {stats['total_steps']}")
        print(f"Average steps: {stats['total_steps']/max(stats['completed_tasks'], 1):.1f}")
        print(f"Total reward: {stats['total_reward']:.2f}")
        print(f"Average reward: {stats['total_reward']/max(stats['completed_tasks'], 1):.3f}")
        print(f"Principles extracted: {stats['principles_extracted']}")
        print(f"Evolution runs: {stats['evolution_runs']}")
        print("=" * 70)
        print(f"\nResults saved to: {output_path.absolute()}")
        print(f"Statistics: {stats_file}")
        print(f"Checkpoints: {checkpoint_dir}")
        print(f"Logs: {log_dir}\n")
        
        logger.info("Cold boot complete")
        logger.info(f"Statistics: {stats}")
        
    finally:
        # Cleanup
        logger.info("Shutting down system...")
        system.shutdown()
        print("🔒 System shutdown complete\n")


def main():
    parser = argparse.ArgumentParser(
        description="Cold boot HyPE system on API-Bank training set"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to API-Bank training data JSON file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to HyPE configuration file (default: config.yaml)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output/cold_boot",
        help="Output directory for checkpoints and logs (default: ./output/cold_boot)"
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="Number of tasks to process (default: all)"
    )
    parser.add_argument(
        "--evolution-interval",
        type=int,
        default=20,
        help="Run offline evolution every N tasks (default: 20)"
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=50,
        help="Save checkpoint every N tasks (default: 50)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="Maximum steps per task (default: 10)"
    )
    
    args = parser.parse_args()
    
    cold_boot_on_apibank(
        data_path=args.data_path,
        config_path=args.config,
        output_dir=args.output_dir,
        num_tasks=args.num_tasks,
        evolution_interval=args.evolution_interval,
        checkpoint_interval=args.checkpoint_interval,
        max_steps_per_task=args.max_steps
    )


if __name__ == "__main__":
    main()
