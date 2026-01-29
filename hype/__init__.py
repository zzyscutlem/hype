"""
HyPE: Hypothesis-Driven Planning and Semantic Adaptation 
with Evolutionary Principle-Value Distillation

A principle-driven self-evolving intelligent agent system.
"""

__version__ = "0.1.0"
__author__ = "HyPE Team"

from hype.core.data_models import (
    State,
    Action,
    Principle,
    Trajectory,
    TrajectoryStep,
    HypothesisNode,
    TrainingExample,
)
from hype.system import HyPESystem

__all__ = [
    "State",
    "Action",
    "Principle",
    "Trajectory",
    "TrajectoryStep",
    "HypothesisNode",
    "TrainingExample",
    "HyPESystem",
]
