"""
Batch experiment runner for quantum reservoir computing experiments.
"""

import warnings

# Suppress NumPy longdouble warnings (harmless, related to platform compatibility)
warnings.filterwarnings('ignore', category=UserWarning, module='numpy._core.getlimits')

import numpy as np
from typing import List, Dict, Any, Union
from dataclasses import dataclass
from .config import ExperimentConfig
from .core import run, run_linear_baseline
from reservoirs.quantum_reservoir import QuantumSteadyStateReservoir
from reservoirs.esn import ESNReservoir


@dataclass
class ExperimentResult:
    """
    Results from a single experiment configuration.
    """
    config_name: str
    train_accuracies: np.ndarray  # Shape: (n_run,)
    test_accuracies: np.ndarray   # Shape: (n_run,)
    train_mean: float
    train_std: float
    test_mean: float
    test_std: float
    
    def __str__(self):
        return (
            f"{self.config_name}:\n"
            f"  Train: {self.train_mean:.6f} ± {self.train_std:.6f}\n"
            f"  Test:  {self.test_mean:.6f} ± {self.test_std:.6f}"
        )


def run_single_experiment(config: ExperimentConfig, verbose: bool = True) -> ExperimentResult:
    """
    Run a single experiment configuration multiple times (n_run) and aggregate results.
    
    Args:
        config: Experiment configuration
        verbose: Whether to print progress during execution
        
    Returns:
        ExperimentResult with aggregated statistics
    """
    train_accs = np.zeros(config.n_run)
    test_accs = np.zeros(config.n_run)
    
    # Create reservoir (reused across runs, but recreated for each seed)
    for seed_idx in range(config.n_run):
        if verbose:
            print(f"Seed : {seed_idx}")
        
        seed = config.seed_offset + seed_idx
        
        # Create reservoir based on type
        if config.reservoir_type == 'quantum':
            reservoir = QuantumSteadyStateReservoir(
                input_dim=config.task.input_dim,
                encoding_fn=config.encoding_fn,
                feature_fn=config.feature_fn,
                name=config.name,
                n_jobs=config.n_jobs,
            )
        elif config.reservoir_type == 'esn':
            if config.esn_params is None:
                raise ValueError("esn_params required for ESN reservoir")
            reservoir = ESNReservoir(
                input_dim=config.task.input_dim,
                name=config.name,
                **config.esn_params
            )
        else:
            raise ValueError(f"Unknown reservoir_type: {config.reservoir_type}")
        
        # Run experiment
        train_acc, test_acc = run(
            task=config.task,
            reservoir=reservoir,
            n_train=config.n_train,
            n_test=config.n_test,
            lam=config.lam,
            seed=seed,
        )
        
        train_accs[seed_idx] = train_acc
        test_accs[seed_idx] = test_acc
    
    # Compute statistics
    result = ExperimentResult(
        config_name=config.name,
        train_accuracies=train_accs,
        test_accuracies=test_accs,
        train_mean=np.mean(train_accs),
        train_std=np.std(train_accs),
        test_mean=np.mean(test_accs),
        test_std=np.std(test_accs),
    )
    
    if verbose:
        print(f"Train Mean and standard deviation : {result.train_mean}  +-  {result.train_std}")
        print(f"Test Mean and standard deviation : {result.test_mean}  +-  {result.test_std}")
        print()  # Blank line between experiments
    
    return result


def run_batch_experiments(
    configs: List[ExperimentConfig],
    verbose: bool = True,
    return_dict: bool = False,
) -> Union[List[ExperimentResult], Dict[str, ExperimentResult]]:
    """
    Run multiple experiment configurations and return aggregated results.
    
    Args:
        configs: List of experiment configurations to run
        verbose: Whether to print progress during execution
        return_dict: If True, return dict mapping config names to results;
                     if False, return list of results
        
    Returns:
        List of ExperimentResult objects, or dict if return_dict=True
    """
    results = []
    
    for i, config in enumerate(configs):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Experiment {i+1}/{len(configs)}: {config.name}")
            print(f"{'='*60}")
        
        result = run_single_experiment(config, verbose=verbose)
        results.append(result)
    
    if return_dict:
        return {r.config_name: r for r in results}
    return results


def run_linear_baseline_experiment(
    task,
    n_train: int = 1024,
    n_test: int = 128,
    n_run: int = 5,
    lam: float = 1e-4,
    seed_offset: int = 0,
    verbose: bool = True,
) -> ExperimentResult:
    """
    Run linear baseline experiment (no reservoir, direct classification on raw input).
    
    Args:
        task: Task instance
        n_train, n_test, n_run, lam, seed_offset: Experiment parameters
        verbose: Whether to print progress
        
    Returns:
        ExperimentResult with aggregated statistics
    """
    train_accs = np.zeros(n_run)
    test_accs = np.zeros(n_run)
    
    for seed_idx in range(n_run):
        if verbose:
            print(f"Seed : {seed_idx}")
        
        seed = seed_offset + seed_idx
        train_acc, test_acc = run_linear_baseline(
            task=task,
            n_train=n_train,
            n_test=n_test,
            lam=lam,
            seed=seed,
        )
        
        train_accs[seed_idx] = train_acc
        test_accs[seed_idx] = test_acc
    
    result = ExperimentResult(
        config_name="linear_baseline",
        train_accuracies=train_accs,
        test_accuracies=test_accs,
        train_mean=np.mean(train_accs),
        train_std=np.std(train_accs),
        test_mean=np.mean(test_accs),
        test_std=np.std(test_accs),
    )
    
    if verbose:
        print(f"Train Mean and standard deviation : {result.train_mean}  +-  {result.train_std}")
        print(f"Test Mean and standard deviation : {result.test_mean}  +-  {result.test_std}")
        print()
    
    return result
