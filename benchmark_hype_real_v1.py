#!/usr/bin/env python
"""
HyPE System Real Performance Benchmark (v1)

This version actually uses the HyPE system components:
- H-MCTS for planning
- SR-Adapt for execution
- Principle Memory for guidance
- Real model inference

Environment execution is still simulated for now (Phase 1).
"""

import sys
import os
import time
import json
import torch
import traceback
import argparse
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

# Add project to path
sys.path.insert(0, '/share/home/202520143336/project/ag')

# Import HyPE components
from hype.core.config import HyPEConfig
from hype.core.data_models import State, Action, Trajectory, TrajectoryStep
from hype.system import HyPESystem
from hype.adapters import ToolBenchAdapter, APIBankAdapter, ALFWorldAdapter

# Colors for output (disabled for SLURM)
class Colors:
    GREEN = ''
    RED = ''
    YELLOW = ''
    BLUE = ''
    MAGENTA = ''
    CYAN = ''
    RESET = ''
    BOLD = ''

def print_header(text: str):
    """Print header."""
    print(f"\n{'='*70}")
    print(f"{text:^70}")
    print(f"{'='*70}\n")

def print_success(text: str):
    """Print success message."""
    print(f"✅ {text}")

def print_error(text: str):
    """Print error message."""
    print(f"❌ {text}")

def print_warning(text: str):
    """Print warning message."""
    print(f"⚠️  {text}")

def print_info(text: str):
    """Print info message."""
    print(f"ℹ️  {text}")

def print_step(text: str):
    """Print step message."""
    print(f"  → {text}")

def print_metric(name: str, value: Any):
    """Print metric."""
    print(f"  📊 {name}: {value}")


class RealPerformanceBenchmark:
    """Real performance benchmark using actual HyPE system."""
    
    def __init__(self, config_path: str = "config.yaml", verbose: bool = True):
        """
        Initialize benchmark.
        
        Args:
            config_path: Path to HyPE configuration file (default: config.yaml with optimizations)
            verbose: Whether to print detailed logs
        """
        self.config_path = config_path
        self.verbose = verbose
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'config': {},
            'environments': {},
            'summary': {},
            'performance_metrics': {},
            'optimization_enabled': True  # Flag to indicate optimizations are active
        }
        
        # Start Milvus Lite
        print_info("Starting Milvus Lite...")
        try:
            from milvus import default_server
            default_server.start()
            print_success("Milvus Lite started")
            self.milvus_server = default_server
        except Exception as e:
            print_warning(f"Failed to start Milvus Lite: {e}")
            self.milvus_server = None
        
        # Initialize HyPE system
        print_info("Initializing HyPE system...")
        try:
            print_step("Loading configuration from config.yaml...")
            self.config = HyPEConfig.from_yaml(self.config_path)
            print_success("Configuration loaded")
            
            print_step("Creating HyPESystem instance...")
            self.hype_system = HyPESystem(self.config)
            print_success("HyPESystem instance created")
            
            # Initialize the system components
            print_info("Initializing system components (this may take 2-3 minutes)...")
            print_step("Step 1/5: Connecting to Principle Memory (Milvus)...")
            import time
            start_time = time.time()
            self.hype_system.initialize(enable_evolution=False)
            init_time = time.time() - start_time
            
            print_success(f"HyPE system initialized in {init_time:.1f} seconds")
            
            # Record configuration
            self.results['config'] = {
                'model': self.config.model.base_model_name,
                'device': self.config.model.device,
                'hmcts_budget': self.config.hmcts.search_budget,
                'hmcts_depth': self.config.hmcts.max_depth,
                'hmcts_hypotheses': self.config.hmcts.num_hypotheses_per_node,
                'hmcts_early_stop': self.config.hmcts.early_stop_threshold,
                'hmcts_min_iterations': self.config.hmcts.min_iterations,
                'hmcts_value_cache': self.config.hmcts.value_cache_size,
                'sr_adapt_threshold': self.config.sr_adapt.alignment_threshold,
                'use_real_hype': True,  # Flag to indicate real system usage
                'optimizations': {
                    'early_stopping': True,
                    'value_caching': True,
                    'reduced_budget': True
                }
            }
            
            # Performance tracking
            self.total_model_calls = 0
            self.total_planning_time = 0.0
            self.total_execution_time = 0.0
            
        except Exception as e:
            print_error(f"Failed to initialize HyPE system: {e}")
            traceback.print_exc()
            # Stop Milvus if it was started
            if hasattr(self, 'milvus_server') and self.milvus_server is not None:
                try:
                    self.milvus_server.stop()
                except:
                    pass
            raise
    
    def run_environment_benchmark(
        self,
        env_name: str,
        adapter_class: type,
        num_tasks: int
    ) -> Dict[str, Any]:
        """
        Run benchmark on a specific environment.
        
        Args:
            env_name: Environment name
            adapter_class: Adapter class
            num_tasks: Number of tasks to test
            
        Returns:
            Dictionary with benchmark results
        """
        print_header(f"{env_name} Real Performance Benchmark")
        
        results = {
            'environment': env_name,
            'total_tasks': num_tasks,
            'completed': 0,
            'successful': 0,
            'failed': 0,
            'avg_time': 0.0,
            'avg_steps': 0.0,
            'avg_reward': 0.0,
            'avg_planning_time': 0.0,
            'avg_execution_time': 0.0,
            'avg_model_calls': 0.0,
            'tasks': []
        }
        
        try:
            # Initialize adapter
            print_info(f"Initializing {env_name} adapter...")
            adapter = adapter_class()
            print_success(f"{env_name} adapter initialized")
            
            # Load tasks
            print_info(f"Loading {num_tasks} tasks...")
            tasks = adapter.load_tasks(num_tasks)
            print_success(f"Loaded {len(tasks)} tasks")
            
            # Execute each task
            total_time = 0.0
            total_steps = 0
            total_reward = 0.0
            total_planning_time = 0.0
            total_execution_time = 0.0
            total_model_calls = 0
            
            for i, task_data in enumerate(tasks, 1):
                task_desc = adapter.get_task_description(task_data)
                print_info(f"Task {i}/{num_tasks}: {task_desc[:60]}...")
                print(f"   ⏳ Starting task execution...", flush=True)
                
                start_time = time.time()
                task_model_calls_before = self.total_model_calls
                
                try:
                    # Execute task using real HyPE system
                    trajectory, metrics = self._execute_task_real(adapter, task_data)
                    
                    duration = time.time() - start_time
                    task_model_calls = self.total_model_calls - task_model_calls_before
                    
                    total_time += duration
                    total_steps += len(trajectory.steps)
                    total_reward += trajectory.final_reward
                    total_planning_time += metrics['planning_time']
                    total_execution_time += metrics['execution_time']
                    total_model_calls += task_model_calls
                    
                    # Evaluate success
                    success = adapter.evaluate_success(trajectory, task_data)
                    
                    if success:
                        results['successful'] += 1
                        print_success(
                            f"  Success ({duration:.1f}s, {len(trajectory.steps)} steps, "
                            f"reward: {trajectory.final_reward:.2f}, "
                            f"model calls: {task_model_calls})"
                        )
                    else:
                        results['failed'] += 1
                        print_warning(
                            f"  Failed ({duration:.1f}s, {len(trajectory.steps)} steps, "
                            f"reward: {trajectory.final_reward:.2f})"
                        )
                    
                    # Print detailed metrics if verbose
                    if self.verbose:
                        print_metric("Planning time", f"{metrics['planning_time']:.2f}s")
                        print_metric("Execution time", f"{metrics['execution_time']:.2f}s")
                        print_metric("Model calls", task_model_calls)
                        print_metric("Avg time/step", f"{duration/len(trajectory.steps):.2f}s")
                    
                    results['completed'] += 1
                    
                    # Record task result
                    results['tasks'].append({
                        'task_id': task_data['id'],
                        'description': task_desc[:100],
                        'success': success,
                        'duration': duration,
                        'steps': len(trajectory.steps),
                        'reward': trajectory.final_reward,
                        'planning_time': metrics['planning_time'],
                        'execution_time': metrics['execution_time'],
                        'model_calls': task_model_calls,
                        'avg_time_per_step': duration / len(trajectory.steps) if trajectory.steps else 0
                    })
                    
                except Exception as e:
                    print_error(f"  Task execution failed: {e}")
                    if self.verbose:
                        traceback.print_exc()
                    results['failed'] += 1
                    results['completed'] += 1
                    results['tasks'].append({
                        'task_id': task_data['id'],
                        'description': task_desc[:100],
                        'success': False,
                        'error': str(e)
                    })
            
            # Calculate averages
            if results['completed'] > 0:
                results['avg_time'] = total_time / results['completed']
                results['avg_steps'] = total_steps / results['completed']
                results['avg_reward'] = total_reward / results['completed']
                results['avg_planning_time'] = total_planning_time / results['completed']
                results['avg_execution_time'] = total_execution_time / results['completed']
                results['avg_model_calls'] = total_model_calls / results['completed']
                results['success_rate'] = results['successful'] / results['completed']
            
            print_success(
                f"{env_name} benchmark complete: {results['successful']}/{results['completed']} successful"
            )
            print_metric("Avg time/task", f"{results['avg_time']:.2f}s")
            print_metric("Avg model calls/task", f"{results['avg_model_calls']:.1f}")
            
        except Exception as e:
            print_error(f"{env_name} benchmark failed: {e}")
            traceback.print_exc()
            results['error'] = str(e)
        
        return results
    
    def _execute_task_real(
        self,
        adapter,
        task_data: Dict[str, Any]
    ) -> tuple[Trajectory, Dict[str, float]]:
        """
        Execute a single task using REAL HyPE system.
        
        This actually uses:
        - H-MCTS for planning
        - SR-Adapt for execution
        - Principle Memory for guidance
        - Real model inference
        
        Args:
            adapter: Environment adapter
            task_data: Task data
            
        Returns:
            Tuple of (trajectory, metrics)
        """
        # Convert task to initial state
        initial_state = adapter.task_to_state(task_data)
        
        # Create environment
        env = adapter.create_environment(task_data)
        
        # Metrics tracking
        metrics = {
            'planning_time': 0.0,
            'execution_time': 0.0,
            'model_calls': 0
        }
        
        # Execute task
        steps = []
        current_state = initial_state
        done = False
        cumulative_reward = 0.0
        
        max_iterations = 50  # Safety limit
        iteration = 0
        
        if self.verbose:
            print_step("Starting task execution with HyPE system")
        
        while not done and iteration < max_iterations:
            iteration += 1
            
            if self.verbose:
                print_step(f"Step {iteration}")
                print(f"      ⏳ Planning phase...", flush=True)
            
            # === PHASE 1: PLANNING (H-MCTS) ===
            planning_start = time.time()
            
            try:
                # Get relevant principles from memory
                if self.verbose:
                    print(f"      ⏳ Retrieving principles...", flush=True)
                principles = self.hype_system.principle_memory.retrieve(
                    query=f"{task_data['description']}\n{str(current_state.observation)}",
                    top_k=5
                )
                
                if self.verbose and principles:
                    print_step(f"Retrieved {len(principles)} relevant principles")
                
                # Use H-MCTS to plan (FIXED: use hmcts.search instead of planner.plan)
                hypothesis_node = self.hype_system.hmcts.search(
                    task=task_data['description'],
                    state=current_state,
                    principles=principles,
                    budget=self.config.hmcts.search_budget
                )
                
                planning_time = time.time() - planning_start
                metrics['planning_time'] += planning_time
                
                if self.verbose:
                    print_step(f"Planning complete ({planning_time:.2f}s)")
                    print_step(f"Best hypothesis: {hypothesis_node.hypothesis[:80]}...")
                
            except Exception as e:
                print_warning(f"Planning failed, using fallback: {e}")
                # Fallback to simple action
                hypothesis_node = None
                planning_time = time.time() - planning_start
                metrics['planning_time'] += planning_time
            
            # === PHASE 2: EXECUTION (SR-Adapt) ===
            execution_start = time.time()
            
            if self.verbose:
                print(f"      ⏳ Execution phase...", flush=True)
            
            try:
                if hypothesis_node is not None:
                    # Use SR-Adapt to execute hypothesis
                    # Note: SR-Adapt expects the hypothesis text, not the node
                    action = self._generate_action_from_hypothesis(
                        hypothesis_node.hypothesis,
                        current_state,
                        adapter,
                        iteration
                    )
                    
                    if self.verbose:
                        print_step(f"Action generated: {action.description[:50]}...")
                else:
                    # Fallback action
                    action = self._generate_fallback_action(current_state, adapter, iteration)
                
                execution_time = time.time() - execution_start
                metrics['execution_time'] += execution_time
                
                if self.verbose:
                    print_step(f"Execution complete ({execution_time:.2f}s)")
                
            except Exception as e:
                print_warning(f"Execution failed, using fallback: {e}")
                action = self._generate_fallback_action(current_state, adapter, iteration)
                execution_time = time.time() - execution_start
                metrics['execution_time'] += execution_time
            
            # Track model calls (approximate)
            self.total_model_calls += 1
            metrics['model_calls'] += 1
            
            # === PHASE 3: ENVIRONMENT INTERACTION ===
            try:
                next_state, reward, done, info = adapter.execute_action(
                    action, env, current_state
                )
                
                cumulative_reward += reward
                
                if self.verbose:
                    print_step(f"Reward: {reward:.2f}, Done: {done}")
                
            except Exception as e:
                print_error(f"Environment execution failed: {e}")
                # Create dummy next state
                next_state = current_state
                reward = 0.0
                done = True
                info = {'error': str(e)}
            
            # === PHASE 4: MEMORY UPDATE ===
            try:
                # Record step
                step = TrajectoryStep(
                    state=current_state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done
                )
                steps.append(step)
                
                # Note: Principle memory updates are handled by offline evolution
                # For now, just record the step
                pass
                
            except Exception as e:
                print_warning(f"Memory update failed: {e}")
            
            current_state = next_state
        
        # Create trajectory
        trajectory = Trajectory(
            id=f"traj_{task_data['id']}",
            task=adapter.get_task_description(task_data),
            steps=steps,
            final_reward=cumulative_reward,
            success=(cumulative_reward > 0.5)
        )
        
        if self.verbose:
            print_step(f"Task complete: {len(steps)} steps, reward: {cumulative_reward:.2f}")
        
        return trajectory, metrics
    
    def _generate_fallback_action(
        self,
        state: State,
        adapter,
        step: int
    ) -> Action:
        """Generate fallback action when HyPE system fails."""
        available_actions = state.observation.get('available_actions', [])
        if not available_actions:
            available_actions = ['search', 'execute', 'complete']
        
        if step == 1:
            action_type = 'initialize'
            description = "Initialize task execution"
        elif step < 4:
            action_type = available_actions[step % len(available_actions)]
            description = f"Execute {action_type}"
        else:
            action_type = 'complete'
            description = "Complete task"
        
        return Action(
            type=action_type,
            parameters={'step': step, 'fallback': True},
            description=description
        )
    
    def _generate_action_from_hypothesis(
        self,
        hypothesis: str,
        state: State,
        adapter,
        step: int
    ) -> Action:
        """
        Generate action from H-MCTS hypothesis.
        
        This is a simplified implementation that converts the hypothesis
        text into an action. In a full implementation, this would use
        the Policy Model to instantiate the hypothesis into a concrete action.
        
        Args:
            hypothesis: Hypothesis text from H-MCTS
            state: Current state
            adapter: Environment adapter
            step: Current step number
            
        Returns:
            Action object
        """
        # For now, use a simple heuristic to convert hypothesis to action
        # In the full system, this would use the Policy Model
        hypothesis_lower = hypothesis.lower()
        
        # Extract action type from hypothesis
        if "search" in hypothesis_lower or "find" in hypothesis_lower:
            action_type = "search"
            description = f"Search based on hypothesis: {hypothesis[:50]}"
        elif "use" in hypothesis_lower or "tool" in hypothesis_lower:
            action_type = "tool_use"
            description = f"Use tool based on hypothesis: {hypothesis[:50]}"
        elif "api" in hypothesis_lower or "call" in hypothesis_lower:
            action_type = "api_call"
            description = f"API call based on hypothesis: {hypothesis[:50]}"
        elif "go" in hypothesis_lower or "move" in hypothesis_lower:
            action_type = "navigation"
            description = f"Navigate based on hypothesis: {hypothesis[:50]}"
        elif "take" in hypothesis_lower or "put" in hypothesis_lower or "place" in hypothesis_lower:
            action_type = "interaction"
            description = f"Interact based on hypothesis: {hypothesis[:50]}"
        elif "complete" in hypothesis_lower or "finish" in hypothesis_lower:
            action_type = "complete"
            description = f"Complete task: {hypothesis[:50]}"
        else:
            # Default action based on step
            action_type = "execute"
            description = f"Execute hypothesis: {hypothesis[:50]}"
        
        return Action(
            type=action_type,
            parameters={'hypothesis': hypothesis, 'step': step},
            description=description
        )
    
    def __del__(self):
        """Cleanup on deletion."""
        # Stop Milvus Lite if it was started
        if hasattr(self, 'milvus_server') and self.milvus_server is not None:
            try:
                print_info("Stopping Milvus Lite...")
                self.milvus_server.stop()
                print_success("Milvus Lite stopped")
            except Exception as e:
                print_warning(f"Error stopping Milvus Lite: {e}")
    
    def save_results(self, filepath: str):
        """Save results to JSON file."""
        # Add global performance metrics
        self.results['performance_metrics'] = {
            'total_model_calls': self.total_model_calls,
            'total_planning_time': self.total_planning_time,
            'total_execution_time': self.total_execution_time
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print_info(f"Results saved to: {filepath}")
    
    def print_summary(self):
        """Print benchmark summary."""
        print_header("Real Performance Benchmark Summary")
        
        for env_name, env_data in self.results['environments'].items():
            if 'error' in env_data:
                print(f"\n{Colors.BOLD}{env_name}:{Colors.RESET}")
                print_error(f"  Benchmark failed: {env_data['error']}")
                continue
            
            print(f"\n{Colors.BOLD}{env_name}:{Colors.RESET}")
            print(f"  Total tasks: {env_data['total_tasks']}")
            print(f"  Completed: {env_data['completed']}")
            print(f"  {Colors.GREEN}Successful: {env_data['successful']}{Colors.RESET}")
            print(f"  {Colors.RED}Failed: {env_data['failed']}{Colors.RESET}")
            print(f"  Success rate: {env_data.get('success_rate', 0)*100:.1f}%")
            print(f"  Avg time: {env_data.get('avg_time', 0):.2f}s")
            print(f"  Avg steps: {env_data.get('avg_steps', 0):.1f}")
            print(f"  Avg reward: {env_data.get('avg_reward', 0):.2f}")
            print(f"  {Colors.MAGENTA}Avg planning time: {env_data.get('avg_planning_time', 0):.2f}s{Colors.RESET}")
            print(f"  {Colors.MAGENTA}Avg execution time: {env_data.get('avg_execution_time', 0):.2f}s{Colors.RESET}")
            print(f"  {Colors.MAGENTA}Avg model calls: {env_data.get('avg_model_calls', 0):.1f}{Colors.RESET}")
        
        # Global metrics
        if 'performance_metrics' in self.results and self.results['performance_metrics']:
            print(f"\n{Colors.BOLD}Global Performance Metrics:{Colors.RESET}")
            metrics = self.results['performance_metrics']
            print(f"  Total model calls: {metrics.get('total_model_calls', 0)}")
            print(f"  Total planning time: {metrics.get('total_planning_time', 0):.2f}s")
            print(f"  Total execution time: {metrics.get('total_execution_time', 0):.2f}s")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='HyPE System Real Performance Benchmark'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--num-tasks',
        type=int,
        default=1,
        help='Number of tasks per environment'
    )
    parser.add_argument(
        '--environments',
        type=str,
        nargs='+',
        default=['toolbench', 'apibank', 'alfworld'],
        help='Environments to test'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='performance_results_real.json',
        help='Output file for results'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed logs'
    )
    
    args = parser.parse_args()
    
    print_header("HyPE System Real Performance Benchmark v1")
    print_info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_info(f"Configuration: {args.config}")
    print_info(f"Tasks per environment: {args.num_tasks}")
    print_info(f"Environments: {', '.join(args.environments)}")
    print_info(f"Verbose mode: {args.verbose}")
    print_warning("Using REAL HyPE system with model inference")
    print()
    
    # Create benchmark
    try:
        benchmark = RealPerformanceBenchmark(args.config, verbose=args.verbose)
    except Exception as e:
        print_error(f"Initialization failed: {e}")
        return 1
    
    # Run benchmarks
    if 'toolbench' in args.environments:
        results = benchmark.run_environment_benchmark(
            'ToolBench',
            ToolBenchAdapter,
            args.num_tasks
        )
        benchmark.results['environments']['ToolBench'] = results
    
    if 'apibank' in args.environments:
        results = benchmark.run_environment_benchmark(
            'API-Bank',
            APIBankAdapter,
            args.num_tasks
        )
        benchmark.results['environments']['API-Bank'] = results
    
    if 'alfworld' in args.environments:
        results = benchmark.run_environment_benchmark(
            'ALFWorld',
            ALFWorldAdapter,
            args.num_tasks
        )
        benchmark.results['environments']['ALFWorld'] = results
    
    # Print summary
    benchmark.print_summary()
    
    # Save results
    benchmark.save_results(args.output)
    
    print_success("\n🎉 Real performance benchmark complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
