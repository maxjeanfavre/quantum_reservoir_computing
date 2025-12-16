# Suppress NumPy warnings (must be before NumPy import for worker processes)
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='numpy._core.getlimits')
warnings.filterwarnings('ignore', message='.*longdouble.*', category=UserWarning)

import numpy as np
from qutip import ket2dm
from .base import Reservoir
from quantum.quantum_system import (
    build_hamiltonian, build_c_ops, evolve_step, rho_list, rho_iss, rho_random,
)
from quantum.features import rho_to_features

class QuantumDynamicalReservoir(Reservoir):
    """
    2-qubit quantum reservoir using time-evolved density matrix as feature source.
    
    The reservoir maintains an internal quantum state that evolves according to
    the Lindblad master equation. At each time step, an input u_t is encoded into
    system parameters, and the state evolves for duration dt.
    """
    
    def __init__(
        self,
        input_dim: int = 1,
        encoding_fn=None,
        feature_fn=rho_to_features,
        dt: float = 0.1,
        initial_state=None,
        init_from_ss: bool = False,
        ss_u0: float = 0.0,
        name: str = "qrc_dynamical",
    ):
        """
        Parameters:
        -----------
        input_dim : int
            Dimension of input (typically 1 for time-series)
        encoding_fn : callable
            Function mapping scalar input u_t -> params dict
        feature_fn : callable
            Function mapping density matrix -> feature vector
        dt : float
            Time step duration for evolution
        initial_state : Qobj or ket, optional
            Initial state. If None and init_from_ss is False, uses |00⟩⟨00|.
        init_from_ss : bool
            If True and initial_state is None, set the initial state to the
            steady state at input ss_u0 computed with the provided encoding_fn.
        ss_u0 : float
            Input value used to compute the steady state when init_from_ss is True.
        method : str
            Solver method for mesolve ("adams", "bdf", etc.)
        name : str
            Name identifier for this reservoir
        """
        # Feature dimension depends on feature_fn, default is 15 for rho_to_features
        feature_dim = 15 if feature_fn == rho_to_features else None
        if feature_dim is None:
            # Try to infer from feature_fn by calling with a dummy state
            try:
                dummy_rho = rho_list["mixed"]
                dummy_feat = feature_fn(dummy_rho)
                feature_dim = len(dummy_feat)
            except:
                raise ValueError("Could not determine feature_dim. Please specify explicitly or use rho_to_features.")
        
        super().__init__(input_dim=input_dim, feature_dim=feature_dim, name=name)
        self.encoding_fn = encoding_fn
        self.feature_fn = feature_fn
        self.dt = dt
        
        # Set initial state
        if init_from_ss:
            if encoding_fn is None:
                raise ValueError("encoding_fn must be provided when init_from_ss=True")
            self.initial_state = rho_iss(u0=ss_u0, encoding_fn=encoding_fn)
        else:
            self.initial_state = self._normalize_initial_state(initial_state)
        
        # Current state (will be reset to initial_state)
        self.rho = self.initial_state.copy()
    
    @staticmethod
    def _normalize_initial_state(initial_state):
        """
        Normalize an initial state specification to a density matrix Qobj.
        
        Accepts:
            - None: returns |00><00|
            - str in rho_list keys: uses that predefined density
            - "random": draws a random valid density matrix
            - Qobj ket: converts to density matrix
            - Qobj density matrix: returned as-is
        """
        if initial_state is None:
            return rho_list["00"]
        
        if isinstance(initial_state, str):
            if initial_state == "random":
                return rho_random()
            if initial_state not in rho_list:
                raise ValueError(f"Unknown initial_state '{initial_state}'. Available: {list(rho_list.keys()) + ['random']}")
            return rho_list[initial_state]
        
        # If a ket is provided, convert to density matrix
        if hasattr(initial_state, "isket") and initial_state.isket:
            return ket2dm(initial_state)
        
        return initial_state
    
    def reset(self):
        """Reset the reservoir state to the initial state."""
        self.rho = self.initial_state.copy()
    
    def step(self, u_t) -> np.ndarray:
        """
        Process one input and evolve the state for one time step.
        
        Parameters:
        -----------
        u_t : float or np.ndarray
            Input at time t (scalar for input_dim=1, array for input_dim>1)
        
        Returns:
        --------
        np.ndarray
            Feature vector extracted from the evolved state
        """
        if self.encoding_fn is None:
            raise ValueError("encoding_fn must be provided")
        
        # Encode input into parameters
        params = self.encoding_fn(u_t)
        
        # Build Hamiltonian and collapse operators
        H = build_hamiltonian(params)
        c_ops = build_c_ops(params)
        
        # Evolve state for one time step
        self.rho = evolve_step(H, c_ops, self.rho, self.dt)
        
        # Extract features
        return self.feature_fn(self.rho)
    
    def process_sequence(self, u_seq: np.ndarray) -> np.ndarray:
        """
        Process a full input sequence and return feature matrix.
        
        Parameters:
        -----------
        u_seq : np.ndarray
            Input sequence of shape (T,) for scalar inputs
        
        Returns:
        --------
        np.ndarray
            Feature matrix of shape (T, feature_dim)
        """
        self.reset()
        T = len(u_seq)
        features = []
        
        for t in range(T):
            feat = self.step(u_seq[t])
            features.append(feat)
        
        return np.vstack(features)
    
    def features(self, X: np.ndarray) -> np.ndarray:
        """
        Process inputs and return features.
        
        Note: For dynamical reservoirs, prefer using process_sequence() for
        time-series data. This method maintains compatibility with the base
        Reservoir interface.
        
        Parameters:
        -----------
        X : np.ndarray
            - If 1D: treated as a single sequence (T,)
            - If 2D with shape (N, 1): treated as N separate single-step inputs
              (reservoir is reset between each)
        
        Returns:
        --------
        np.ndarray
            - If 1D input: feature array of shape (T, feature_dim)
            - If 2D input: feature array of shape (N, feature_dim)
        """
        X = np.asarray(X)
        
        # Handle single sequence case (1D array)
        if X.ndim == 1:
            return self.process_sequence(X)
        
        # Handle 2D case: (N, input_dim)
        assert X.ndim == 2, f"Expected 1D or 2D input, got {X.ndim}D"
        assert X.shape[1] == self.input_dim, f"Expected input_dim={self.input_dim}, got {X.shape[1]}"
        
        # Treat each row as a separate input, resetting between each
        # This maintains compatibility but process_sequence() is preferred for sequences
        results = []
        for i in range(X.shape[0]):
            self.reset()
            # Extract input value(s)
            if self.input_dim == 1:
                u_t = float(X[i, 0])
            else:
                u_t = X[i]  # Multi-dimensional input
            feat = self.step(u_t)
            results.append(feat)
        
        return np.vstack(results)

