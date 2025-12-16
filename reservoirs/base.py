
import numpy as np

class Reservoir:
    """Abstract base class for feature-generating reservoirs."""
    def __init__(self, input_dim: int, feature_dim: int, name: str):
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.name = name

    def features(self, X: np.ndarray) -> np.ndarray:
        """Map inputs X (N, input_dim) to feature matrix Phi (N, feature_dim)."""
        raise NotImplementedError
