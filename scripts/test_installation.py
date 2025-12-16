#!/usr/bin/env python3
"""
Test script to verify that the quantum reservoir computing installation is working correctly.

This script performs basic checks:
1. Imports all required modules
2. Runs a minimal experiment to verify functionality
3. Reports any issues found
"""

import sys
import warnings

# Suppress NumPy longdouble warnings (harmless, related to platform compatibility)
warnings.filterwarnings('ignore', category=UserWarning, module='numpy._core.getlimits')

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import numpy as np
        print(f"  ✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"  ✗ NumPy import failed: {e}")
        return False
    
    try:
        import qutip
        print(f"  ✓ QuTiP {qutip.__version__}")
    except ImportError as e:
        print(f"  ✗ QuTiP import failed: {e}")
        return False
    
    try:
        import joblib
        print(f"  ✓ Joblib {joblib.__version__}")
    except ImportError as e:
        print(f"  ✗ Joblib import failed: {e}")
        return False
    
    try:
        import matplotlib
        print(f"  ✓ Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"  ✗ Matplotlib import failed: {e}")
        return False
    
    # Test project modules
    try:
        from tasks.two_u import TwoU
        print("  ✓ Tasks module")
    except ImportError as e:
        print(f"  ✗ Tasks module import failed: {e}")
        return False
    
    try:
        from quantum.encodings import bare_encoding_2d
        print("  ✓ Quantum encodings module")
    except ImportError as e:
        print(f"  ✗ Quantum encodings import failed: {e}")
        return False
    
    try:
        from reservoirs.quantum_reservoir import QuantumSteadyStateReservoir
        print("  ✓ Quantum reservoir module")
    except ImportError as e:
        print(f"  ✗ Quantum reservoir import failed: {e}")
        return False
    
    try:
        from readout.linear import train_ridge_classifier, predict_labels, accuracy
        print("  ✓ Readout module")
    except ImportError as e:
        print(f"  ✗ Readout module import failed: {e}")
        return False
    
    try:
        from experiments.config import ExperimentConfig
        print("  ✓ Experiments module")
    except ImportError as e:
        print(f"  ✗ Experiments module import failed: {e}")
        return False
    
    return True


def test_minimal_experiment():
    """Run a minimal experiment to verify functionality."""
    print("\nRunning minimal experiment...")
    
    try:
        import numpy as np
        from tasks.two_u import TwoU
        from reservoirs.quantum_reservoir import QuantumSteadyStateReservoir
        from quantum.encodings import bare_encoding_2d
        from quantum.features import rho_to_features
        from readout.linear import train_ridge_classifier, predict_labels, accuracy
        
        # Create a simple task
        task = TwoU()
        print("  ✓ Task created")
        
        # Generate a small dataset
        X_train, y_train = task.make_dataset(n_train=10, seed=42)
        X_test, y_test = task.make_dataset(n_test=5, seed=43)
        print(f"  ✓ Dataset generated (train: {len(X_train)}, test: {len(X_test)})")
        
        # Create reservoir
        reservoir = QuantumSteadyStateReservoir(
            input_dim=2,
            encoding_fn=bare_encoding_2d,
            feature_fn=rho_to_features,
            name="test_reservoir",
            n_jobs=1,
        )
        print("  ✓ Reservoir created")
        
        # Extract features
        Phi_train = reservoir.features(X_train)
        Phi_test = reservoir.features(X_test)
        print(f"  ✓ Features extracted (dim: {Phi_train.shape[1]})")
        
        # Train classifier
        W = train_ridge_classifier(Phi_train, y_train, lam=1e-4)
        print("  ✓ Classifier trained")
        
        # Evaluate
        y_pred_train = predict_labels(Phi_train, W)
        y_pred_test = predict_labels(Phi_test, W)
        
        train_acc = accuracy(y_train, y_pred_train)
        test_acc = accuracy(y_test, y_pred_test)
        
        print(f"  ✓ Experiment completed")
        print(f"    Train accuracy: {train_acc:.4f}")
        print(f"    Test accuracy: {test_acc:.4f}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Quantum Reservoir Computing - Installation Test")
    print("=" * 60)
    
    # Test imports
    if not test_imports():
        print("\n✗ Import tests failed. Please check your installation.")
        sys.exit(1)
    
    # Test minimal experiment
    if not test_minimal_experiment():
        print("\n✗ Experiment test failed. Please check your installation.")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✓ All tests passed! Installation is working correctly.")
    print("=" * 60)
    sys.exit(0)


if __name__ == "__main__":
    main()
