#!/usr/bin/env python3
"""
Command-line script to run quantum reservoir computing experiments.

Usage:
    python run_experiments.py --task two_u --encodings bare,drive,gamma_loss
    python run_experiments.py --task two_u --all-encodings --n-jobs -1
    python run_experiments.py --task two_spirals --n-train 512 --n-run 10
"""

import os
import argparse
import sys
import warnings

# Suppress NumPy longdouble warnings (harmless, related to platform compatibility)
# This happens during NumPy import when it tries to detect available float types
warnings.filterwarnings('ignore', category=UserWarning, module='numpy._core.getlimits')

# Alternative: Tell NumPy to skip longdouble detection (may cause other issues)
# os.environ['NUMPY_EXPERIMENTAL_DTYPE_API'] = '1'  # Uncomment if needed
from tasks.spirals import TwoSpirals
from tasks.gaussians import TwoGaussians
from tasks.circles import TwoCircles
from tasks.two_u import TwoU
from quantum.encodings import (
    bare_encoding_2d,
    simple_bare_gamma_encoding_2d,
    drive_encoding_2d,
    drive2_encoding_2d,
    drive_gamma_encoding_2d,
    couplings_encoding_2d,
    gamma_encoding_2d,
    gamma1_encoding_2d,
    gamma2_encoding_2d,
    dephase_encoding_2d,
    rich1_encoding_2d,
    rich2_encoding_2d,
    rich3_encoding_2d,
    rich4_encoding_2d,
    bare_drive_encoding_2d,
)
from quantum.features import (
    rho_to_features,
    rho_to_populations,
    rho_to_pauli_features,
)
from experiments import (
    ExperimentConfig,
    create_configs_from_encodings,
    run_batch_experiments,
    run_linear_baseline_experiment,
)


# Mapping of encoding names to functions
ENCODING_MAP = {
    'bare': bare_encoding_2d,
    'simple_bare_gamma': simple_bare_gamma_encoding_2d,
    'drive': drive_encoding_2d,
    'drive2': drive2_encoding_2d,
    'drive_gamma': drive_gamma_encoding_2d,
    'couplings': couplings_encoding_2d,
    'gamma_loss': gamma_encoding_2d,
    'gamma1': gamma1_encoding_2d,
    'gamma2': gamma2_encoding_2d,
    'dephase': dephase_encoding_2d,
    'rich1': rich1_encoding_2d,
    'rich2': rich2_encoding_2d,
    'rich3': rich3_encoding_2d,
    'rich4': rich4_encoding_2d,
    'bare_drive': bare_drive_encoding_2d,
}

# Mapping of task names to classes
TASK_MAP = {
    'two_u': TwoU,
    'two_spirals': TwoSpirals,
    'two_circles': TwoCircles,
    'two_gaussians': TwoGaussians,
}

# Mapping of feature names to functions
FEATURE_MAP = {
    'features': rho_to_features,
    'populations': rho_to_populations,
    'pauli': rho_to_pauli_features,
}


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Run quantum reservoir computing experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run specific encodings on TwoU task
  python run_experiments.py --task two_u --encodings bare,drive,gamma_loss

  # Run all encodings with parallel processing
  python run_experiments.py --task two_u --all-encodings --n-jobs -1

  # Run with custom parameters
  python run_experiments.py --task two_spirals --n-train 512 --n-test 128 --n-run 10

  # Include baseline and ESN
  python run_experiments.py --task two_u --encodings bare --baseline --esn
  
  # Run with custom ESN parameters
  python run_experiments.py --task two_u --encodings bare --esn --esn-hidden 20 --esn-feat 20 --esn-input-scale 1.0
        """
    )
    
    # Task selection
    parser.add_argument(
        '--task',
        type=str,
        choices=list(TASK_MAP.keys()),
        default='two_u',
        help='Task to run (default: two_u)'
    )
    
    # Encoding selection
    encoding_group = parser.add_mutually_exclusive_group(required=True)
    encoding_group.add_argument(
        '--encodings',
        type=str,
        help='Comma-separated list of encodings to test (e.g., bare,drive,gamma_loss)'
    )
    encoding_group.add_argument(
        '--all-encodings',
        action='store_true',
        help='Run all available encodings'
    )
    
    # Feature map
    parser.add_argument(
        '--feature-map',
        type=str,
        choices=list(FEATURE_MAP.keys()),
        default='features',
        help='Feature extraction method (default: features)'
    )
    
    # Experiment parameters
    parser.add_argument(
        '--n-train',
        type=int,
        default=1024,
        help='Number of training samples (default: 1024)'
    )
    parser.add_argument(
        '--n-test',
        type=int,
        default=128,
        help='Number of test samples (default: 128)'
    )
    parser.add_argument(
        '--n-run',
        type=int,
        default=5,
        help='Number of independent runs (default: 5)'
    )
    parser.add_argument(
        '--lam',
        type=float,
        default=1e-4,
        help='Ridge regression regularization (default: 1e-4)'
    )
    parser.add_argument(
        '--seed-offset',
        type=int,
        default=0,
        help='Seed offset for random number generation (default: 0)'
    )
    
    # Parallel processing
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=1,
        help='Number of parallel jobs (-1 for all cores, default: 1)'
    )
    
    # Additional experiments
    parser.add_argument(
        '--baseline',
        action='store_true',
        help='Also run linear baseline experiment'
    )
    parser.add_argument(
        '--esn',
        action='store_true',
        help='Also run ESN reservoir experiment'
    )
    
    # ESN parameters
    parser.add_argument(
        '--esn-hidden',
        type=int,
        default=15,
        help='Number of hidden units in ESN (default: 15)'
    )
    parser.add_argument(
        '--esn-feat',
        type=int,
        default=15,
        help='Number of feature units in ESN (default: 15)'
    )
    parser.add_argument(
        '--esn-input-scale',
        type=float,
        default=0.5,
        help='Input scaling for ESN (default: 0.5)'
    )
    parser.add_argument(
        '--esn-seed',
        type=int,
        default=42,
        help='Random seed for ESN (default: 42)'
    )
    
    # Output
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce output verbosity'
    )
    
    return parser.parse_args()


def get_task_instance(task_name):
    """Get task instance from name."""
    task_class = TASK_MAP[task_name]
    if task_name == 'two_circles':
        # TwoCircles needs parameters: two circles with specified radii, centers, and noise
        # These defaults can be made configurable via CLI in the future if needed
        return task_class(r0=1, center0=(-0.5, 0), r1=1, center1=(0.5, 0), noise=0.005)
    else:
        return task_class()


def get_encodings(encodings_arg, all_encodings):
    """Get encoding functions from arguments."""
    if all_encodings:
        return ENCODING_MAP
    else:
        encoding_names = [e.strip() for e in encodings_arg.split(',')]
        invalid = [e for e in encoding_names if e not in ENCODING_MAP]
        if invalid:
            print(f"Error: Unknown encodings: {invalid}", file=sys.stderr)
            print(f"Available encodings: {', '.join(ENCODING_MAP.keys())}", file=sys.stderr)
            sys.exit(1)
        return {name: ENCODING_MAP[name] for name in encoding_names}


def main():
    """Main entry point."""
    args = parse_args()
    
    # Get task instance
    task = get_task_instance(args.task)
    print(f"Task: {task.name}")
    
    # Get encodings
    encoding_fns = get_encodings(args.encodings, args.all_encodings)
    print(f"Encodings: {', '.join(encoding_fns.keys())}")
    
    # Get feature map
    feature_fn = FEATURE_MAP[args.feature_map]
    print(f"Feature map: {args.feature_map}")
    
    # Create experiment configurations
    experiment_configs = create_configs_from_encodings(
        task=task,
        encoding_fns=encoding_fns,
        feature_fn=feature_fn,
        n_train=args.n_train,
        n_test=args.n_test,
        n_run=args.n_run,
        lam=args.lam,
        seed_offset=args.seed_offset,
        n_jobs=args.n_jobs,
    )
    
    # Add ESN if requested
    if args.esn:
        experiment_configs.append(ExperimentConfig(
            name="esn",
            task=task,
            reservoir_type='esn',
            n_train=args.n_train,
            n_test=args.n_test,
            n_run=args.n_run,
            lam=args.lam,
            seed_offset=args.seed_offset,
            esn_params={
                'n_hidden': args.esn_hidden,
                'feature_dim': args.esn_feat,
                'input_scale': args.esn_input_scale,
                'seed': args.esn_seed,
            }
        ))
    
    print(f"\nRunning {len(experiment_configs)} experiments...")
    print(f"Parameters: n_train={args.n_train}, n_test={args.n_test}, n_run={args.n_run}, n_jobs={args.n_jobs}")
    print("=" * 60)
    
    # Run experiments
    results = run_batch_experiments(
        experiment_configs,
        verbose=not args.quiet,
    )
    
    # Run baseline if requested
    baseline_result = None
    if args.baseline:
        print("\n" + "=" * 60)
        print("Running linear baseline experiment...")
        print("=" * 60)
        baseline_result = run_linear_baseline_experiment(
            task=task,
            n_train=args.n_train,
            n_test=args.n_test,
            n_run=args.n_run,
            lam=args.lam,
            seed_offset=args.seed_offset,
            verbose=not args.quiet,
        )
    
    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    for result in results:
        print(f"\n{result.config_name}:")
        print(f"  Train: {result.train_mean:.6f} ± {result.train_std:.6f}")
        print(f"  Test:  {result.test_mean:.6f} ± {result.test_std:.6f}")
    
    if baseline_result:
        print(f"\nLinear Baseline:")
        print(f"  Train: {baseline_result.train_mean:.6f} ± {baseline_result.train_std:.6f}")
        print(f"  Test:  {baseline_result.test_mean:.6f} ± {baseline_result.test_std:.6f}")
    
    # Find best result
    if results:
        best = max(results, key=lambda r: r.test_mean)
        print(f"\n{'='*60}")
        print(f"Best result: {best.config_name}")
        print(f"  Test accuracy: {best.test_mean:.6f} ± {best.test_std:.6f}")
        print(f"{'='*60}")


if __name__ == '__main__':
    main()
