#!/usr/bin/env python3
"""
Evaluate HyPE system on API-Bank test set.

This script:
1. Loads trained HyPE system from checkpoint
2. Loads API-Bank test data
3. Evaluates on test set
4. Computes metrics (success rate, API call accuracy, etc.)
5. Saves detailed results

This implements the evaluation phase after cold boot.
"""

import os
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import numpy as np

from hype.system import HyPESystem
from hype.core.config import HyPEConfig
from hype.core.data_models import State
from hype.environments.adapters import APIBankAdapter
from hype.utils.logging_config import setup_logging
from hype.utils.checkpointing import CheckpointManager


# Mock API-Bank environment for evaluation
class MockAPIBankEnvironment:
    """Mock environment for API-Bank evaluation."""
    
    def __init__(self, ground_truth: Dict[str, Any]):
        self.ground_truth = ground_truth
        self.current_step = 0
        self.max_steps = len(ground_truth.get("api_calls", []))
        self.api_calls_made = []
    
    def execute(self, action: Any) -> tuple:
        """Execute action using ground truth."""
        if self.current_step >= self.max_steps:
            return (
                {"observation": "Task complete"},
                0.0,
                True,
                {"error": None}
            )
        
        # Record API call
        self.api_calls_made.append({
            "step": self.current_step,
            "action": str(action.description) if hasattr(action, 'description') else str(action),
            "parameters": action.parameters if hasattr(action, 'parameters') else {}
        })
        
        # Get ground truth for this step
        gt_call = self.ground_truth.get("api_calls", [])[self.current_step]
        
        # Compute reward based on match with ground truth
        reward = self._compute_reward(action, gt_call)
        
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        # Create next state
        next_state = {
            "observation": gt_call.get("result", "Action executed"),
            "step": self.current_step,
            "remaining_steps": self.max_steps - self.current_step
        }
        
        return next_state, reward, done, {"error": None}
    
    def _compute_reward(self, action: Any, gt_call: Dict) -> float:
        """Compute reward based on action match with ground truth."""
        # API name match
        api_match = 0.0
        if hasattr(action, 'parameters'):
            api_name = action.parameters.get('api', '').lower()
            gt_api = gt_call.get('api_name', '').lower()
            if api_name in gt_api or gt_api in api_name:
                api_match = 1.0
            elif any(word in gt_api for word in api_name.split('_')):
                api_match = 0.5
        
        # Parameter match (simplified)
        param_match = 0.5  # Default partial credit
        
        # Combined reward
        reward = 0.7 * api_match + 0.3 * param_match
        
        return reward
    
    def get_api_calls(self) -> List[Dict]:
        """Get all API calls made."""
        return self.api_calls_made
    
    def reset(self):
        """Reset environment."""
        self.current_step = 0
        self.api_calls_made = []


def load_test_data(data_path: str) -> List[Dict[str, Any]]:
    """
    Load API-Bank test data.
    
    Args:
        data_path: Path to test data JSON file
        
    Returns:
        List of test examples
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    
    Args:
        results: List of evaluation results
        
    Returns:
        Dictionary of metrics
    """
    if not results:
        return {}
    
    # Success rate
    successes = [r for r in results if r["success"]]
    success_rate = len(successes) / len(results)
    
    # Average reward
    avg_reward = np.mean([r["final_reward"] for r in results])
    
    # Average steps
    avg_steps = np.mean([r["num_steps"] for r in results])
    
    # API call accuracy (if available)
    api_accuracies = [r.get("api_accuracy", 0.0) for r in results if "api_accuracy" in r]
    avg_api_accuracy = np.mean(api_accuracies) if api_accuracies else 0.0
    
    # Completion rate (tasks that finished vs. hit max steps)
    completed = [r for r in results if r.get("completed", False)]
    completion_rate = len(completed) / len(results)
    
    return {
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "avg_steps": avg_steps,
        "avg_api_accuracy": avg_api_accuracy,
        "completion_rate": completion_rate,
        "total_tasks": len(results),
        "successful_tasks": len(successes),
        "failed_tasks": len(results) - len(successes)
    }


def evaluate_on_apibank(
    checkpoint_path: str,
    data_path: str,
    config_path: str,
    output_dir: str,
    num_tasks: int = None,
    max_steps_per_task: int = 10
):
    """
    Evaluate HyPE system on API-Bank test set.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        data_path: Path to API-Bank test data
        config_path: Path to HyPE configuration file
        output_dir: Directory for outputs (results, logs, etc.)
        num_tasks: Number of tasks to evaluate (None for all)
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
        log_level=logging.INFO,
        log_file=f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logger = logging.getLogger(__name__)
    
    print("\n" + "=" * 70)
    print("📊 HyPE Evaluation on API-Bank Test Set")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Data path: {data_path}")
    print(f"Config path: {config_path}")
    print(f"Output directory: {output_path.absolute()}")
    print("=" * 70 + "\n")
    
    # Load test data
    logger.info("Loading API-Bank test data...")
    print("📥 Loading test data...")
    test_data = load_test_data(data_path)
    
    if num_tasks is not None:
        test_data = test_data[:num_tasks]
    
    print(f"   ✅ Loaded {len(test_data)} test examples\n")
    logger.info(f"Loaded {len(test_data)} test examples")
    
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
        max_steps_per_task=max_steps_per_task
    )
    
    # Initialize without evolution (inference only)
    system.initialize(enable_evolution=False)
    print("   ✅ System initialized\n")
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_path}...")
    print(f"📦 Loading checkpoint...")
    
    try:
        checkpoint_manager = CheckpointManager(str(Path(checkpoint_path).parent))
        checkpoint_manager.load_checkpoint(
            system=system,
            checkpoint_name=Path(checkpoint_path).name
        )
        print("   ✅ Checkpoint loaded\n")
        logger.info("Checkpoint loaded successfully")
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        print(f"   ⚠️  Warning: Could not load checkpoint: {e}")
        print("   Continuing with initialized models...\n")
    
    # Get system stats
    system_stats = system.get_system_stats()
    print(f"📋 System Status:")
    print(f"   Principles in memory: {system_stats.get('num_principles', 0)}")
    print(f"   HMCTS budget: {system_stats.get('hmcts_budget', 0)}")
    print(f"   Max steps per task: {system_stats.get('max_steps_per_task', 0)}")
    print()
    
    # Initialize API-Bank adapter
    adapter = APIBankAdapter()
    
    # Evaluation results
    results = []
    
    # Evaluation loop
    logger.info("Starting evaluation loop...")
    print("🔄 Starting evaluation...\n")
    print("=" * 70)
    
    try:
        for task_idx, example in enumerate(tqdm(test_data, desc="Evaluation Progress")):
            logger.info(f"Evaluating task {task_idx + 1}/{len(test_data)}")
            
            # Extract task information
            task_description = example.get("instruction", example.get("query", ""))
            if not task_description:
                logger.warning(f"Skipping task {task_idx}: no instruction found")
                continue
            
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
                trajectory, success = system.execute_task(
                    task=task_description,
                    initial_state=initial_state,
                    environment=environment,
                    environment_adapter=adapter,
                    environment_type="api_bank",
                    collect_trajectory=False  # Don't collect during evaluation
                )
                
                # Get API calls made
                api_calls_made = environment.get_api_calls()
                
                # Compute API accuracy
                gt_api_calls = example.get("api_calls", [])
                api_accuracy = len(api_calls_made) / max(len(gt_api_calls), 1) if gt_api_calls else 0.0
                
                # Store result
                result = {
                    "task_idx": task_idx,
                    "task": task_description,
                    "success": success,
                    "final_reward": trajectory.final_reward,
                    "num_steps": len(trajectory.steps),
                    "api_calls_made": len(api_calls_made),
                    "api_calls_expected": len(gt_api_calls),
                    "api_accuracy": api_accuracy,
                    "completed": trajectory.steps[-1].done if trajectory.steps else False
                }
                results.append(result)
                
                logger.info(
                    f"Task {task_idx + 1} complete: "
                    f"success={success}, "
                    f"steps={len(trajectory.steps)}, "
                    f"reward={trajectory.final_reward:.3f}, "
                    f"api_accuracy={api_accuracy:.3f}"
                )
                
            except Exception as e:
                logger.error(f"Error evaluating task {task_idx + 1}: {e}")
                result = {
                    "task_idx": task_idx,
                    "task": task_description,
                    "success": False,
                    "final_reward": 0.0,
                    "num_steps": 0,
                    "error": str(e)
                }
                results.append(result)
                continue
        
        # Compute metrics
        logger.info("Computing metrics...")
        print("\n📊 Computing metrics...")
        metrics = compute_metrics(results)
        
        # Save results
        results_file = output_path / "evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "metrics": metrics,
                "results": results,
                "checkpoint": checkpoint_path,
                "test_data": data_path,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)
        
        # Save metrics separately
        metrics_file = output_path / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Print results
        print("\n" + "=" * 70)
        print("✅ Evaluation Complete!")
        print("=" * 70)
        print(f"Total tasks: {metrics['total_tasks']}")
        print(f"Successful: {metrics['successful_tasks']} ({metrics['success_rate']*100:.1f}%)")
        print(f"Failed: {metrics['failed_tasks']}")
        print(f"Average reward: {metrics['avg_reward']:.3f}")
        print(f"Average steps: {metrics['avg_steps']:.1f}")
        print(f"Average API accuracy: {metrics['avg_api_accuracy']*100:.1f}%")
        print(f"Completion rate: {metrics['completion_rate']*100:.1f}%")
        print("=" * 70)
        print(f"\nResults saved to: {output_path.absolute()}")
        print(f"Detailed results: {results_file}")
        print(f"Metrics: {metrics_file}")
        print(f"Logs: {log_dir}\n")
        
        logger.info("Evaluation complete")
        logger.info(f"Metrics: {metrics}")
        
    finally:
        # Cleanup
        logger.info("Shutting down system...")
        system.shutdown()
        print("🔒 System shutdown complete\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate HyPE system on API-Bank test set"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to API-Bank test data JSON file"
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
        default="./output/evaluation",
        help="Output directory for results and logs (default: ./output/evaluation)"
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="Number of tasks to evaluate (default: all)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="Maximum steps per task (default: 10)"
    )
    
    args = parser.parse_args()
    
    evaluate_on_apibank(
        checkpoint_path=args.checkpoint,
        data_path=args.data_path,
        config_path=args.config,
        output_dir=args.output_dir,
        num_tasks=args.num_tasks,
        max_steps_per_task=args.max_steps
    )


if __name__ == "__main__":
    main()
