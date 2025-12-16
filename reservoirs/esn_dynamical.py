import numpy as np
from .base import Reservoir


class ESNDynamicalReservoir(Reservoir):
    """
    Echo State Network (ESN) with recurrent dynamics for time-series processing.
    
    The reservoir maintains an internal state that evolves according to:
        x_{t+1} = (1 - leak_rate) * x_t + leak_rate * tanh(W @ x_t + Win @ u_t + b + noise)
    
    Features are extracted from the reservoir state at each timestep.
    """
    
    def __init__(
        self,
        input_dim: int = 1,
        n_reservoir: int = 300,
        spectral_radius: float = 0.9,
        sparsity: float = 0.1,
        input_scale: float = 1.0,
        leak_rate: float = 0.3,
        noise_std: float = 0.0,
        seed: int | None = None,
        name: str = "esn_dynamical",
    ):
        """
        Parameters:
        -----------
        input_dim : int
            Dimension of input (typically 1 for time-series)
        n_reservoir : int
            Number of reservoir units (hidden state dimension)
        spectral_radius : float
            Largest absolute eigenvalue of the reservoir weight matrix W.
            Controls the memory capacity and stability (typically 0.7-1.0)
        sparsity : float
            Fraction of non-zero connections in W (0.0 = dense, 1.0 = no connections)
        input_scale : float
            Scaling factor for input-to-reservoir weights
        leak_rate : float
            Leaky integration rate (0.0 = no memory, 1.0 = no leaky integration)
            Controls the speed of state updates (typically 0.1-0.5)
        noise_std : float
            Standard deviation of noise added to state update (for regularization)
        seed : int, optional
            Random seed for weight initialization
        name : str
            Name identifier for this reservoir
        """
        # Feature dimension equals reservoir size (state is the feature)
        feature_dim = n_reservoir
        
        super().__init__(input_dim=input_dim, feature_dim=feature_dim, name=name)
        
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.input_scale = input_scale
        self.leak_rate = leak_rate
        self.noise_std = noise_std
        
        rng = np.random.default_rng(seed)
        
        # Initialize reservoir weight matrix W (sparse, random)
        W = rng.random((n_reservoir, n_reservoir)) * 2 - 1  # Uniform [-1, 1]
        
        # Apply sparsity: randomly set connections to zero
        if sparsity > 0.0:
            mask = rng.random((n_reservoir, n_reservoir)) > sparsity
            W = W * mask
        
        # Normalize to desired spectral radius
        # Compute largest eigenvalue and scale
        eigenvals = np.linalg.eigvals(W)
        max_eigenval = np.max(np.abs(eigenvals))
        if max_eigenval > 0:
            W = W * (spectral_radius / max_eigenval)
        
        self.W = W
        
        # Input-to-reservoir weights
        self.Win = (rng.random((n_reservoir, input_dim)) * 2 - 1) * input_scale
        
        # Bias term
        self.b = rng.normal(0, 0.1, size=n_reservoir)
        
        # Initial state (will be reset to this)
        self.initial_state = np.zeros(n_reservoir)
        
        # Current state
        self.x = self.initial_state.copy()
    
    def reset(self):
        """Reset the reservoir state to the initial state (zeros)."""
        self.x = self.initial_state.copy()
    
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
            Feature vector (current reservoir state) of shape (n_reservoir,)
        """
        # Ensure u_t is a numpy array
        if np.isscalar(u_t):
            u_t = np.array([u_t])
        else:
            u_t = np.asarray(u_t)
        
        # Flatten if needed
        if u_t.ndim > 1:
            u_t = u_t.flatten()
        
        # Compute pre-activation
        pre_activation = self.W @ self.x + self.Win @ u_t + self.b
        
        # Add noise if specified
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, size=self.n_reservoir)
            pre_activation += noise
        
        # Leaky integration update
        new_state = np.tanh(pre_activation)
        self.x = (1 - self.leak_rate) * self.x + self.leak_rate * new_state
        
        # Return current state as features
        return self.x.copy()
    
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
            - If 2D with shape (N, input_dim): treated as N separate single-step inputs
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

