"""
Core experiment execution functions for quantum reservoir computing.

This module provides the low-level functions for running single experiments
with quantum or classical reservoirs.
"""

import warnings

# Suppress NumPy longdouble warnings (harmless, related to platform compatibility)
warnings.filterwarnings('ignore', category=UserWarning, module='numpy._core.getlimits')

from tasks.spirals import TwoSpirals
from reservoirs.quantum_reservoir import QuantumSteadyStateReservoir
from reservoirs.esn import ESNReservoir
from quantum.encodings import bare_encoding_2d
from readout.linear import train_ridge_classifier, predict_labels, accuracy

def run(task, reservoir, n_train=512, n_test=128, lam=1e-4, seed=0):
    """
    Run a single experiment with a given task and reservoir.
    
    Args:
        task: Task instance (e.g., TwoU(), TwoSpirals())
        reservoir: Reservoir instance (quantum or ESN)
        n_train: Number of training samples
        n_test: Number of test samples
        lam: Ridge regression regularization parameter
        seed: Random seed for data generation
        
    Returns:
        Tuple of (train_accuracy, test_accuracy)
    """
    # Generate data
    X_train, y_train = task.make_dataset(n_train, seed=seed)
    X_test,  y_test  = task.make_dataset(n_test,  seed=seed+1)

    # Extract features
    Phi_train = reservoir.features(X_train)
    Phi_test  = reservoir.features(X_test)

    # Train linear readout
    W = train_ridge_classifier(Phi_train, y_train, lam=lam)

    # Evaluate
    y_pred_train = predict_labels(Phi_train, W)
    y_pred_test  = predict_labels(Phi_test,  W)

    print(f"Task: {task.name}, reservoir: {reservoir.name}")
    print("Train accuracy:", accuracy(y_train, y_pred_train))
    print("Test  accuracy:", accuracy(y_test,  y_pred_test))
    return accuracy(y_train, y_pred_train), accuracy(y_test,  y_pred_test)


def run_linear_baseline(task, n_train=512, n_test=256, lam=1e-4, seed=0):
    """
    Train a linear classifier directly on raw input (x,y) without a reservoir.
    
    Args:
        task: Task instance
        n_train: Number of training samples
        n_test: Number of test samples
        lam: Ridge regression regularization parameter
        seed: Random seed for data generation
        
    Returns:
        Tuple of (train_accuracy, test_accuracy)
    """
    # Get raw data
    X_train, y_train = task.make_dataset(n_train, seed=seed)
    X_test,  y_test  = task.make_dataset(n_test,  seed=seed+1)

    # Use your existing ridge training
    from readout.linear import train_ridge_classifier, predict_labels, accuracy
    
    W = train_ridge_classifier(X_train, y_train, lam=lam)

    y_pred_train = predict_labels(X_train, W)
    y_pred_test  = predict_labels(X_test,  W)

    train_acc = accuracy(y_train, y_pred_train)
    test_acc  = accuracy(y_test,  y_pred_test)

    print(f"[Baseline linear] Task: {task.name}")
    print("Train accuracy:", train_acc)
    print("Test  accuracy:", test_acc)
    return train_acc, test_acc
