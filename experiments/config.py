"""
Experiment configuration system for quantum reservoir computing experiments.
"""

from dataclasses import dataclass
from typing import Callable, Optional, List


@dataclass
class ExperimentConfig:
    """
    Configuration for a single experiment run.
    
    Attributes:
        name: Unique identifier for this experiment
        task: Task instance (e.g., TwoU(), TwoSpirals())
        encoding_fn: Encoding function (e.g., bare_encoding_2d)
        feature_fn: Feature extraction function (e.g., rho_to_features)
        reservoir_type: Type of reservoir ('quantum' or 'esn')
        n_train: Number of training samples
        n_test: Number of test samples
        n_run: Number of independent runs (for statistics)
        lam: Ridge regression regularization parameter
        seed_offset: Base seed offset for all runs
        n_jobs: Number of parallel jobs for feature extraction (quantum only)
        esn_params: Dictionary of ESN parameters (only used if reservoir_type='esn')
    """
    name: str
    task: any  # Task instance
    encoding_fn: Optional[Callable] = None
    feature_fn: Optional[Callable] = None
    reservoir_type: str = 'quantum'  # 'quantum' or 'esn'
    n_train: int = 1024
    n_test: int = 128
    n_run: int = 5
    lam: float = 1e-4
    seed_offset: int = 0
    n_jobs: int = 1  # Parallel processing for quantum reservoir
    esn_params: Optional[dict] = None  # For ESN: n_hidden, feature_dim, input_scale, etc.
    
    def __post_init__(self):
        """Validate configuration."""
        if self.reservoir_type not in ['quantum', 'esn']:
            raise ValueError(f"reservoir_type must be 'quantum' or 'esn', got '{self.reservoir_type}'")
        
        if self.reservoir_type == 'quantum' and self.encoding_fn is None:
            raise ValueError("encoding_fn is required for quantum reservoir")
        
        if self.reservoir_type == 'esn' and self.esn_params is None:
            raise ValueError("esn_params is required for ESN reservoir")


def create_configs_from_encodings(
    task,
    encoding_fns: dict,  # {name: encoding_function}
    feature_fn,
    n_train: int = 1024,
    n_test: int = 128,
    n_run: int = 5,
    lam: float = 1e-4,
    seed_offset: int = 0,
    n_jobs: int = 1,
) -> List[ExperimentConfig]:
    """
    Helper function to create multiple experiment configs from a dictionary of encodings.
    
    Args:
        task: Task instance
        encoding_fns: Dictionary mapping encoding names to encoding functions
        feature_fn: Feature extraction function
        n_train, n_test, n_run, lam, seed_offset, n_jobs: Common parameters
        
    Returns:
        List of ExperimentConfig objects
    """
    configs = []
    for enc_name, enc_fn in encoding_fns.items():
        configs.append(ExperimentConfig(
            name=f"qrc_steady_{enc_name}",
            task=task,
            encoding_fn=enc_fn,
            feature_fn=feature_fn,
            reservoir_type='quantum',
            n_train=n_train,
            n_test=n_test,
            n_run=n_run,
            lam=lam,
            seed_offset=seed_offset,
            n_jobs=n_jobs,
        ))
    return configs
