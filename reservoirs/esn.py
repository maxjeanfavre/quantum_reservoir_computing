import numpy as np
from .base import Reservoir

def relu(x):
    return np.maximum(0, x)

class ESNReservoir(Reservoir):
    """
    Random-feature reservoir:
    h = ReLU(Win @ u + b)
    features = Wout @ h
    """
    def __init__(self, input_dim: int,
                 n_hidden: int = 300,
                 feature_dim: int = 300,   # typically set = n_hidden
                 input_scale: float = 1.0,
                 seed: int | None = None,
                 name: str = "esn_relu"):
        super().__init__(input_dim=input_dim,
                         feature_dim=feature_dim,
                         name=name)
        rng = np.random.default_rng(seed)

        self.n_hidden = n_hidden

        # Input → hidden weights
        self.Win = (rng.random((n_hidden, input_dim)) * 2 - 1) * input_scale
        
        # Biases (ESSENTIAL for non-radial nonlinearities)
        self.b = rng.normal(0, 0.5, size=n_hidden)

        # Hidden → feature weights (linear)
        self.Wout = rng.normal(0, 1, size=(feature_dim, n_hidden))

    def _feature_single(self, u: np.ndarray) -> np.ndarray:
        h = relu(self.Win @ u + self.b)   # (n_hidden,)
        return self.Wout @ h              # (feature_dim,)

    def features(self, X: np.ndarray) -> np.ndarray:
        return np.vstack([self._feature_single(X[i]) for i in range(X.shape[0])])
