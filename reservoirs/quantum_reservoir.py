# Suppress NumPy warnings (must be before NumPy import for worker processes)
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='numpy._core.getlimits')
warnings.filterwarnings('ignore', message='.*longdouble.*', category=UserWarning)

import numpy as np
from joblib import Parallel, delayed, cpu_count
from .base import Reservoir
from quantum.quantum_system import steady_state_from_params
from quantum.features import rho_to_features

class QuantumSteadyStateReservoir(Reservoir):
    """2-qubit quantum reservoir using steady-state density matrix as feature source."""
    def __init__(
        self,
        input_dim: int,
        encoding_fn,
        feature_fn=rho_to_features,
        name: str = "qrc_steady",
        n_jobs: int = 1,
    ):
        super().__init__(input_dim=input_dim, feature_dim=15, name=name)
        self.encoding_fn = encoding_fn
        self.feature_fn = feature_fn
        # Clamp n_jobs to reasonable value
        if n_jobs == -1:
            self.n_jobs = cpu_count()
        elif n_jobs > cpu_count():
            # Don't use more jobs than available CPUs
            import warnings
            warnings.warn(f"n_jobs={n_jobs} exceeds available CPUs ({cpu_count()}). Using {cpu_count()} instead.")
            self.n_jobs = cpu_count()
        else:
            self.n_jobs = max(1, n_jobs)  # At least 1

    def _sample_to_features(self, u: np.ndarray) -> np.ndarray:
        params = self.encoding_fn(u)
        rho_ss = steady_state_from_params(params)
        return self.feature_fn(rho_ss)  # <- use the chosen map

    def features(self, X: np.ndarray) -> np.ndarray:
        assert X.shape[1] == self.input_dim
        n_samples = X.shape[0]
        
        # Use parallel processing if n_jobs > 1, otherwise sequential
        if self.n_jobs == 1:
            feats = [self._sample_to_features(X[i]) for i in range(n_samples)]
        else:
            feats = Parallel(n_jobs=self.n_jobs)(
                delayed(self._sample_to_features)(X[i]) 
                for i in range(n_samples)
            )
        
        return np.vstack(feats)
