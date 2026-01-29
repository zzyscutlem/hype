"""
Logging and monitoring utilities for HyPE system.

This module provides structured logging, metrics tracking, and performance profiling.
"""

import logging
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
import sys


@dataclass
class MetricsTracker:
    """Track system metrics during execution."""
    
    # Retrieval metrics
    retrieval_count: int = 0
    retrieval_latencies: List[float] = field(default_factory=list)
    retrieval_accuracies: List[float] = field(default_factory=list)
    
    # Alignment metrics
    alignment_scores: List[float] = field(default_factory=list)
    correction_attempts: int = 0
    correction_successes: int = 0
    
    # Training metrics
    value_training_losses: List[float] = field(default_factory=list)
    policy_training_losses: List[float] = field(default_factory=list)
    training_epochs_completed: int = 0
    
    # Planning metrics
    hmcts_search_times: List[float] = field(default_factory=list)
    hypothesis_counts: List[int] = field(default_factory=list)
    
    # Execution metrics
    action_execution_count: int = 0
    action_success_count: int = 0
    action_failure_count: int = 0
    
    # Principle metrics
    principle_insertions: int = 0
    principle_merges: int = 0
    principle_retrievals: int = 0
    
    def record_retrieval(self, latency: float, accuracy: Optional[float] = None):
        """Record principle retrieval metrics."""
        self.retrieval_count += 1
        self.retrieval_latencies.append(latency)
        if accuracy is not None:
            self.retrieval_accuracies.append(accuracy)
    
    def record_alignment(self, score: float):
        """Record semantic alignment score."""
        self.alignment_scores.append(score)
    
    def record_correction(self, success: bool):
        """Record LoRA correction attempt."""
        self.correction_attempts += 1
        if success:
            self.correction_successes += 1
    
    def record_value_loss(self, loss: float):
        """Record Value Model training loss."""
        self.value_training_losses.append(loss)
    
    def record_policy_loss(self, loss: float):
        """Record Policy Model training loss."""
        self.policy_training_losses.append(loss)
    
    def record_hmcts_search(self, duration: float, num_hypotheses: int):
        """Record H-MCTS search metrics."""
        self.hmcts_search_times.append(duration)
        self.hypothesis_counts.append(num_hypotheses)
    
    def record_action_execution(self, success: bool):
        """Record action execution result."""
        self.action_execution_count += 1
        if success:
            self.action_success_count += 1
        else:
            self.action_failure_count += 1
    
    def record_principle_operation(self, operation: str):
        """Record principle memory operation."""
        if operation == "insert":
            self.principle_insertions += 1
        elif operation == "merge":
            self.principle_merges += 1
        elif operation == "retrieve":
            self.principle_retrievals += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        summary = {
            "retrieval": {
                "count": self.retrieval_count,
                "avg_latency_ms": sum(self.retrieval_latencies) / len(self.retrieval_latencies) * 1000 if self.retrieval_latencies else 0,
                "avg_accuracy": sum(self.retrieval_accuracies) / len(self.retrieval_accuracies) if self.retrieval_accuracies else 0,
            },
            "alignment": {
                "avg_score": sum(self.alignment_scores) / len(self.alignment_scores) if self.alignment_scores else 0,
                "min_score": min(self.alignment_scores) if self.alignment_scores else 0,
                "max_score": max(self.alignment_scores) if self.alignment_scores else 0,
            },
            "correction": {
                "attempts": self.correction_attempts,
                "successes": self.correction_successes,
                "success_rate": self.correction_successes / self.correction_attempts if self.correction_attempts > 0 else 0,
            },
            "training": {
                "value_avg_loss": sum(self.value_training_losses) / len(self.value_training_losses) if self.value_training_losses else 0,
                "policy_avg_loss": sum(self.policy_training_losses) / len(self.policy_training_losses) if self.policy_training_losses else 0,
                "epochs_completed": self.training_epochs_completed,
            },
            "planning": {
                "avg_search_time_s": sum(self.hmcts_search_times) / len(self.hmcts_search_times) if self.hmcts_search_times else 0,
                "avg_hypotheses": sum(self.hypothesis_counts) / len(self.hypothesis_counts) if self.hypothesis_counts else 0,
            },
            "execution": {
                "total_actions": self.action_execution_count,
                "successes": self.action_success_count,
                "failures": self.action_failure_count,
                "success_rate": self.action_success_count / self.action_execution_count if self.action_execution_count > 0 else 0,
            },
            "principles": {
                "insertions": self.principle_insertions,
                "merges": self.principle_merges,
                "retrievals": self.principle_retrievals,
            }
        }
        return summary
    
    def save_to_file(self, path: str):
        """Save metrics to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.get_summary(), f, indent=2)
    
    def reset(self):
        """Reset all metrics."""
        self.__init__()


class StructuredLogger:
    """Structured logger with JSON output support."""
    
    def __init__(self, name: str, log_dir: str = "./logs", level: str = "INFO"):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name
            log_dir: Directory for log files
            level: Logging level
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level))
        
        # Remove existing handlers
        self.logger.handlers = []
        
        # Console handler with formatted output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level))
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler with JSON output
        log_file = self.log_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level))
        self.logger.addHandler(file_handler)
        
        self.file_handler = file_handler
    
    def _log_structured(self, level: str, message: str, **kwargs):
        """Log structured message with metadata."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "logger": self.name,
            "message": message,
            **kwargs
        }
        
        # Write JSON to file
        self.file_handler.stream.write(json.dumps(log_entry) + "\n")
        self.file_handler.stream.flush()
        
        # Also log to console
        getattr(self.logger, level.lower())(message)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log_structured("DEBUG", message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log_structured("INFO", message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log_structured("WARNING", message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self._log_structured("ERROR", message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self._log_structured("CRITICAL", message, **kwargs)


class PerformanceProfiler:
    """Profile performance of system components."""
    
    def __init__(self):
        """Initialize profiler."""
        self.timings: Dict[str, List[float]] = {}
        self.active_timers: Dict[str, float] = {}
    
    @contextmanager
    def profile(self, operation: str):
        """
        Context manager for profiling an operation.
        
        Args:
            operation: Name of operation to profile
            
        Example:
            with profiler.profile("principle_retrieval"):
                principles = memory.retrieve(query)
        """
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            if operation not in self.timings:
                self.timings[operation] = []
            self.timings[operation].append(duration)
    
    def start_timer(self, operation: str):
        """Start timing an operation."""
        self.active_timers[operation] = time.time()
    
    def stop_timer(self, operation: str):
        """Stop timing an operation and record duration."""
        if operation in self.active_timers:
            duration = time.time() - self.active_timers[operation]
            if operation not in self.timings:
                self.timings[operation] = []
            self.timings[operation].append(duration)
            del self.active_timers[operation]
    
    def get_stats(self, operation: str) -> Dict[str, float]:
        """
        Get statistics for an operation.
        
        Args:
            operation: Operation name
            
        Returns:
            Dictionary with min, max, mean, total times
        """
        if operation not in self.timings or not self.timings[operation]:
            return {"count": 0, "min": 0, "max": 0, "mean": 0, "total": 0}
        
        times = self.timings[operation]
        return {
            "count": len(times),
            "min": min(times),
            "max": max(times),
            "mean": sum(times) / len(times),
            "total": sum(times)
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all operations."""
        return {op: self.get_stats(op) for op in self.timings.keys()}
    
    def print_summary(self):
        """Print performance summary."""
        print("\n" + "="*60)
        print("Performance Profile Summary")
        print("="*60)
        
        for operation, stats in self.get_all_stats().items():
            print(f"\n{operation}:")
            print(f"  Count: {stats['count']}")
            print(f"  Mean:  {stats['mean']*1000:.2f} ms")
            print(f"  Min:   {stats['min']*1000:.2f} ms")
            print(f"  Max:   {stats['max']*1000:.2f} ms")
            print(f"  Total: {stats['total']:.2f} s")
        
        print("="*60 + "\n")
    
    def save_to_file(self, path: str):
        """Save profiling data to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.get_all_stats(), f, indent=2)
    
    def reset(self):
        """Reset all profiling data."""
        self.timings = {}
        self.active_timers = {}


def setup_logging(log_dir: str = "./logs", level: str = "INFO") -> Dict[str, StructuredLogger]:
    """
    Setup logging for all HyPE components.
    
    Args:
        log_dir: Directory for log files
        level: Logging level
        
    Returns:
        Dictionary of loggers for each component
    """
    components = [
        "hype.system",
        "hype.planner",
        "hype.executor",
        "hype.memory",
        "hype.learner",
        "hype.models",
    ]
    
    loggers = {}
    for component in components:
        loggers[component] = StructuredLogger(component, log_dir, level)
    
    return loggers


# Global instances
_metrics_tracker = MetricsTracker()
_profiler = PerformanceProfiler()


def get_metrics_tracker() -> MetricsTracker:
    """Get global metrics tracker instance."""
    return _metrics_tracker


def get_profiler() -> PerformanceProfiler:
    """Get global profiler instance."""
    return _profiler
