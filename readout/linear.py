# Suppress NumPy warnings (must be before NumPy import)
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='numpy._core.getlimits')
warnings.filterwarnings('ignore', message='.*longdouble.*', category=UserWarning)

import numpy as np

def train_ridge_classifier(X: np.ndarray, y: np.ndarray, lam: float = 1e-6) -> np.ndarray:
    """Train linear classifier with ridge regression on targets in {0,1}."""
    N, D = X.shape
    Xb = np.hstack([X, np.ones((N, 1))])
    t = 2*y - 1  # map 0 -> -1, 1 -> +1

    A = Xb.T @ Xb + lam * np.eye(D + 1)
    b = Xb.T @ t
    W = np.linalg.solve(A, b)
    return W

def predict_labels(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    N, D = X.shape
    Xb = np.hstack([X, np.ones((N, 1))])
    scores = Xb @ W
    return (scores > 0).astype(int)

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))
