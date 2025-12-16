"""
Experiment configuration and batch running utilities.
"""

from .config import ExperimentConfig, create_configs_from_encodings
from .runner import run_batch_experiments, ExperimentResult, run_linear_baseline_experiment

__all__ = [
    'ExperimentConfig',
    'create_configs_from_encodings',
    'run_batch_experiments',
    'ExperimentResult',
    'run_linear_baseline_experiment',
]
