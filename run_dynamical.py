#!/usr/bin/env python3
"""
Run a simple dynamical reservoir experiment on the Sine/Square task.

The task builds sequences by randomly concatenating 8-point sine or square
waveforms and labels each timestep as sine (0) or square (1). We feed the
sequence into a dynamical reservoir (QRC or ESN) and train a linear readout
(Ridge regression) to classify the labels.

Example usage:
    # QRC: python run_dynamical.py --reservoir qrc --encoding bare --feature-map features --dt 0.1
    # ESN: python run_dynamical.py --reservoir esn --n-reservoir 300 --spectral-radius 0.9 --leak-rate 0.3 --washout 200
"""

import argparse
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed, cpu_count

from tasks.sinesquare import SineSquare
from reservoirs.quantum_dynamical import QuantumDynamicalReservoir
from reservoirs.esn_dynamical import ESNDynamicalReservoir
from quantum.dynamical_encodings import bare_encoding_1d, gamma_encoding_1d, dephase_encoding_1d, rich_encoding_1d
from quantum.features import rho_to_features, rho_to_populations, rho_to_pauli_features
from quantum.quantum_system import rho_list, rho_random


FEATURE_MAP = {
    "features": rho_to_features,         # 15-dim full density features
    "populations": rho_to_populations,   # 3-dim populations
    "pauli": rho_to_pauli_features,      # 6-dim simple observables
}

ENCODING_MAP = {
    "bare": bare_encoding_1d,
    "gamma": gamma_encoding_1d,
    "dephase": dephase_encoding_1d,
    "rich": rich_encoding_1d,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run dynamical reservoir (QRC or ESN) on sine/square task")
    parser.add_argument("--reservoir", type=str, choices=["qrc", "esn"], default="qrc", help="Reservoir type: qrc (quantum) or esn (echo state network)")
    parser.add_argument("--n-train", type=int, default=200, help="Number of training steps (timestep samples)")
    parser.add_argument("--n-test", type=int, default=100, help="Number of test steps (timestep samples)")
    parser.add_argument("--lam", type=float, default=1e-4, help="Ridge regularization")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (base seed, used with seed-offset)")
    parser.add_argument("--seed-offset", type=int, default=0, help="Seed offset for random number generation (default: 0)")
    parser.add_argument("--n-run", type=int, default=1, help="Number of independent runs (default: 1)")
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of parallel jobs (-1 for all cores, default: 1)")
    parser.add_argument("--washout", type=int, default=0, help="Number of initial timesteps to discard (washout period)")
    
    # QRC-specific arguments
    parser.add_argument("--dt", type=float, default=0.1, help="Time step for evolution (QRC)")
    parser.add_argument("--encoding", type=str, choices=list(ENCODING_MAP.keys()), default="bare", help="Encoding to use (QRC)")
    parser.add_argument("--feature-map", type=str, choices=list(FEATURE_MAP.keys()), default="features", help="Feature map (QRC)")
    parser.add_argument(
        "--init-state",
        type=str,
        choices=list(rho_list.keys()) + ["random"],
        default=None,
        help="Optional initial state: one of predefined rho_list keys or 'random' (QRC)",
    )
    parser.add_argument(
        "--init-from-ss",
        action="store_true",
        help="Start from steady state at u0 (overrides init-state if provided) (QRC)",
    )
    parser.add_argument(
        "--ss-u0",
        type=float,
        default=0.0,
        help="Input value u0 used to compute steady state when --init-from-ss is set (QRC)",
    )
    
    # ESN-specific arguments
    parser.add_argument("--n-reservoir", type=int, default=300, help="Number of reservoir units (ESN)")
    parser.add_argument("--spectral-radius", type=float, default=0.9, help="Spectral radius of reservoir weight matrix (ESN)")
    parser.add_argument("--sparsity", type=float, default=0.1, help="Sparsity of reservoir connections (ESN)")
    parser.add_argument("--input-scale", type=float, default=1.0, help="Scaling factor for input weights (ESN)")
    parser.add_argument("--leak-rate", type=float, default=0.3, help="Leaky integration rate (ESN)")
    parser.add_argument("--noise-std", type=float, default=0.0, help="Standard deviation of noise added to state update (ESN)")
    
    parser.add_argument("--verbose", action="store_true", help="Print extra info")
    return parser.parse_args()


def _resolve_initial_state(init_state_name: str | None, seed: int):
    """Resolve initial state specification to a density matrix (or None)."""
    if init_state_name is None:
        return None
    if init_state_name == "random":
        # Make random state reproducible w.r.t. provided seed
        np.random.seed(seed)
        return rho_random()
    return rho_list[init_state_name]


def run_single_experiment(
    reservoir_type: str,
    n_train: int,
    n_test: int,
    lam: float,
    seed: int,
    washout: int,
    # QRC parameters
    dt: float = 0.1,
    encoding: str = "bare",
    feature_map: str = "features",
    init_state_name: str | None = None,
    init_from_ss: bool = False,
    ss_u0: float = 0.0,
    # ESN parameters
    n_reservoir: int = 300,
    spectral_radius: float = 0.9,
    sparsity: float = 0.1,
    input_scale: float = 1.0,
    leak_rate: float = 0.3,
    noise_std: float = 0.0,
    verbose: bool = False,
):
    """
    Run a single dynamical reservoir experiment (QRC or ESN) with given seed.
    
    Returns:
        tuple: (train_acc, test_acc)
    """
    # Task: sine vs square per timestep classification
    task = SineSquare()  # use task's internal sequence_length

    # Generate train/test sequences (multiple sequences)
    seqs_train, labels_train = task.make_sequence_dataset(
        n_sequences=n_train, seed=seed
    )
    seqs_test, labels_test = task.make_sequence_dataset(
        n_sequences=n_test, seed=seed + 1
    )

    # Concatenate all sequences into a single timeline (per-timestep samples)
    X_train = np.concatenate(seqs_train).reshape(-1, 1)
    y_train = np.concatenate(labels_train)
    X_test = np.concatenate(seqs_test).reshape(-1, 1)
    y_test = np.concatenate(labels_test)

    # Instantiate reservoir based on type
    if reservoir_type == "qrc":
        encoding_fn = ENCODING_MAP[encoding]
        feature_fn = FEATURE_MAP[feature_map]
        initial_state = None if init_from_ss else _resolve_initial_state(init_state_name, seed)
        reservoir = QuantumDynamicalReservoir(
            input_dim=1,
            encoding_fn=encoding_fn,
            feature_fn=feature_fn,
            dt=dt,
            initial_state=initial_state,
            init_from_ss=init_from_ss,
            ss_u0=ss_u0,
            name=f"qrc_dyn_{encoding}",
        )
    elif reservoir_type == "esn":
        reservoir = ESNDynamicalReservoir(
            input_dim=1,
            n_reservoir=n_reservoir,
            spectral_radius=spectral_radius,
            sparsity=sparsity,
            input_scale=input_scale,
            leak_rate=leak_rate,
            noise_std=noise_std,
            seed=seed,
            name="esn_dynamical",
        )
    else:
        raise ValueError(f"Unknown reservoir type: {reservoir_type}")

    # Process sequences to features (per timestep)
    feats_train = reservoir.process_sequence(X_train[:, 0])
    reservoir.reset()
    feats_test = reservoir.process_sequence(X_test[:, 0])

    # Apply washout: discard first 'washout' timesteps
    if washout > 0:
        if washout >= len(feats_train):
            raise ValueError(f"Washout ({washout}) must be less than training sequence length ({len(feats_train)})")
        if washout >= len(feats_test):
            raise ValueError(f"Washout ({washout}) must be less than test sequence length ({len(feats_test)})")
        
        feats_train = feats_train[washout:]
        y_train = y_train[washout:]
        feats_test = feats_test[washout:]
        y_test = y_test[washout:]

    # Train linear readout (Ridge regression); classify with threshold 0.5
    clf = Ridge(alpha=lam)
    clf.fit(feats_train, y_train)
    y_pred_train = clf.predict(feats_train)
    y_pred_test = clf.predict(feats_test)

    train_acc = accuracy_score(y_train, (y_pred_train >= 0.5).astype(int))
    test_acc = accuracy_score(y_test, (y_pred_test >= 0.5).astype(int))

    if verbose:
        print(f"Train features shape: {feats_train.shape}")
        print(f"Test  features shape: {feats_test.shape}")
        if washout > 0:
            print(f"Washout applied: {washout} timesteps discarded")

    return train_acc, test_acc


def run(args):
    """Run multiple independent experiments and aggregate results."""
    # Determine number of parallel jobs
    if args.n_jobs == -1:
        n_jobs = cpu_count()
    elif args.n_jobs > cpu_count():
        import warnings
        warnings.warn(f"n_jobs={args.n_jobs} exceeds available CPUs ({cpu_count()}). Using {cpu_count()} instead.")
        n_jobs = cpu_count()
    else:
        n_jobs = max(1, args.n_jobs)  # At least 1

    # Prepare arguments for parallel execution
    if args.n_run == 1:
        # Single run - no need for parallelization
        seed = args.seed + args.seed_offset
        train_acc, test_acc = run_single_experiment(
            reservoir_type=args.reservoir,
            n_train=args.n_train,
            n_test=args.n_test,
            lam=args.lam,
            seed=seed,
            washout=args.washout,
            dt=args.dt,
            encoding=args.encoding,
            feature_map=args.feature_map,
            init_state_name=args.init_state,
            init_from_ss=args.init_from_ss,
            ss_u0=args.ss_u0,
            n_reservoir=args.n_reservoir,
            spectral_radius=args.spectral_radius,
            sparsity=args.sparsity,
            input_scale=args.input_scale,
            leak_rate=args.leak_rate,
            noise_std=args.noise_std,
            verbose=args.verbose,
        )
        train_accs = np.array([train_acc])
        test_accs = np.array([test_acc])
    else:
        # Multiple runs - use parallelization if n_jobs > 1
        seeds = [args.seed + args.seed_offset + i for i in range(args.n_run)]
        
        if n_jobs == 1:
            # Sequential execution
            results = [
                run_single_experiment(
                    reservoir_type=args.reservoir,
                    n_train=args.n_train,
                    n_test=args.n_test,
                    lam=args.lam,
                    seed=seed,
                    washout=args.washout,
                    dt=args.dt,
                    encoding=args.encoding,
                    feature_map=args.feature_map,
                    init_state_name=args.init_state,
                    init_from_ss=args.init_from_ss,
                    ss_u0=args.ss_u0,
                    n_reservoir=args.n_reservoir,
                    spectral_radius=args.spectral_radius,
                    sparsity=args.sparsity,
                    input_scale=args.input_scale,
                    leak_rate=args.leak_rate,
                    noise_std=args.noise_std,
                    verbose=False,  # Don't print verbose info for each run
                )
                for seed in seeds
            ]
        else:
            # Parallel execution
            results = Parallel(n_jobs=n_jobs)(
                delayed(run_single_experiment)(
                    reservoir_type=args.reservoir,
                    n_train=args.n_train,
                    n_test=args.n_test,
                    lam=args.lam,
                    seed=seed,
                    washout=args.washout,
                    dt=args.dt,
                    encoding=args.encoding,
                    feature_map=args.feature_map,
                    init_state_name=args.init_state,
                    init_from_ss=args.init_from_ss,
                    ss_u0=args.ss_u0,
                    n_reservoir=args.n_reservoir,
                    spectral_radius=args.spectral_radius,
                    sparsity=args.sparsity,
                    input_scale=args.input_scale,
                    leak_rate=args.leak_rate,
                    noise_std=args.noise_std,
                    verbose=False,  # Don't print verbose info for each run
                )
                for seed in seeds
            )
        
        train_accs = np.array([r[0] for r in results])
        test_accs = np.array([r[1] for r in results])

    # Compute statistics
    train_mean = np.mean(train_accs)
    train_std = np.std(train_accs)
    test_mean = np.mean(test_accs)
    test_std = np.std(test_accs)

    # Print summary
    reservoir_label = "QRC" if args.reservoir == "qrc" else "ESN"
    print(f"=== Dynamical {reservoir_label}: Sine/Square ===")
    print(f"Reservoir type: {args.reservoir}")
    print(f"n_train/n_test: {args.n_train}/{args.n_test}")
    print(f"lambda (ridge): {args.lam}")
    print(f"Washout       : {args.washout}")
    print(f"n_run         : {args.n_run}")
    print(f"n_jobs        : {n_jobs}")
    print(f"Seed offset   : {args.seed_offset}")
    
    if args.reservoir == "qrc":
        print(f"Encoding      : {args.encoding}")
        print(f"Feature map   : {args.feature_map}")
        print(f"dt            : {args.dt}")
        if args.init_from_ss:
            print(f"Init state    : steady state at u0={args.ss_u0}")
        elif args.init_state:
            print(f"Init state    : {args.init_state}")
        else:
            print(f"Init state    : |00><00|")
    else:  # ESN
        print(f"n_reservoir   : {args.n_reservoir}")
        print(f"Spectral radius: {args.spectral_radius}")
        print(f"Sparsity      : {args.sparsity}")
        print(f"Input scale   : {args.input_scale}")
        print(f"Leak rate     : {args.leak_rate}")
        print(f"Noise std     : {args.noise_std}")
    
    if args.n_run > 1:
        print(f"Train acc     : {train_mean:.4f} ± {train_std:.4f}")
        print(f"Test  acc     : {test_mean:.4f} ± {test_std:.4f}")
    else:
        print(f"Seed          : {args.seed + args.seed_offset}")
        print(f"Train acc     : {train_mean:.4f}")
        print(f"Test  acc     : {test_mean:.4f}")


if __name__ == "__main__":
    run(parse_args())

