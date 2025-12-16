import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from .base import Task

class MemoryCapacity(Task):
    """
    Memory capacity task for evaluating reservoir computing systems.
    
    The task measures how well a reservoir can remember past inputs by
    predicting delayed versions of the input sequence.
    
    Memory capacity is defined as: MC = Σ_k R²(k), where R²(k) is the
    coefficient of determination for predicting u(t-k) from reservoir
    features at time t.
    """
    
    def __init__(
        self,
        sequence_length: int = 2000,
        input_range: tuple[float, float] = (-1.0, 1.0),
        washout: int = 100,
        name: str = "memory_capacity"
    ):
        """
        Parameters:
        -----------
        sequence_length : int
            Length of input sequence to generate
        input_range : tuple[float, float]
            Range for random input values
        washout : int
            Number of initial time steps to discard (transient)
        name : str
            Task name identifier
        """
        super().__init__(name=name, input_dim=1)
        self.sequence_length = sequence_length
        self.input_range = input_range
        self.washout = washout
    
    def make_dataset(self, n_samples: int, seed: int | None = None, delay: int = 0):
        """
        Generate dataset for predicting input delayed by 'delay' steps.
        
        Parameters:
        -----------
        n_samples : int
            Length of sequence to generate (overrides sequence_length)
        seed : int, optional
            Random seed
        delay : int
            Delay value k (predict u(t-k))
        
        Returns:
        --------
        X : np.ndarray
            Input sequence of shape (n_samples, 1)
        y : np.ndarray
            Target sequence (delayed input) of shape (n_samples,)
        """
        rng = np.random.default_rng(seed)
        
        # Generate random input sequence
        u_seq = rng.uniform(
            self.input_range[0],
            self.input_range[1],
            size=n_samples + delay
        )
        
        # Create delayed target
        if delay == 0:
            y = u_seq.copy()
        else:
            y = u_seq[delay:]
            u_seq = u_seq[:-delay]
        
        # Reshape for consistency with Task interface
        X = u_seq.reshape(-1, 1)
        y = y[:len(X)]  # Ensure same length
        
        return X, y
    
    def generate_sequence(self, seed: int | None = None) -> np.ndarray:
        """
        Generate a random input sequence for memory capacity evaluation.
        
        Parameters:
        -----------
        seed : int, optional
            Random seed
        
        Returns:
        --------
        np.ndarray
            Input sequence of shape (sequence_length,)
        """
        rng = np.random.default_rng(seed)
        return rng.uniform(
            self.input_range[0],
            self.input_range[1],
            size=self.sequence_length
        )
    
    def compute_memory_capacity(
        self,
        reservoir,
        max_delay: int = 50,
        train_length: int | None = None,
        test_length: int | None = None,
        lam: float = 1e-6,
        seed: int | None = None
    ) -> tuple[float, np.ndarray]:
        """
        Compute memory capacity for a given reservoir.
        
        Parameters:
        -----------
        reservoir : Reservoir
            Reservoir instance with process_sequence or features method
        max_delay : int
            Maximum delay to evaluate
        train_length : int, optional
            Length of training sequence (default: sequence_length - washout)
        test_length : int, optional
            Length of test sequence (default: same as train_length)
        lam : float
            Regularization parameter for Ridge regression
        seed : int, optional
            Random seed
        
        Returns:
        --------
        mc : float
            Total memory capacity (sum of R² scores)
        r2_scores : np.ndarray
            Array of R² scores for each delay k in [0, max_delay]
        """
        if train_length is None:
            train_length = self.sequence_length - self.washout
        if test_length is None:
            test_length = train_length
        
        # Generate input sequences
        # Need extra length to handle max_delay: process train_length + washout inputs,
        # but need inputs going back max_delay steps for delayed targets
        rng = np.random.default_rng(seed)
        total_train_len = train_length + self.washout + max_delay
        total_test_len = test_length + self.washout + max_delay
        
        u_train = rng.uniform(
            self.input_range[0],
            self.input_range[1],
            size=total_train_len
        )
        u_test = rng.uniform(
            self.input_range[0],
            self.input_range[1],
            size=total_test_len
        )
        
        # Process sequences through reservoir
        # Process full sequences to get features
        if hasattr(reservoir, 'reset'):
            reservoir.reset()
        if hasattr(reservoir, 'process_sequence'):
            # Use process_sequence if available (more efficient)
            feat_train_full = reservoir.process_sequence(u_train[:train_length + self.washout])
            reservoir.reset()
            feat_test_full = reservoir.process_sequence(u_test[:test_length + self.washout])
        else:
            # Fall back to features method
            X_train = u_train[:train_length + self.washout].reshape(-1, 1)
            X_test = u_test[:test_length + self.washout].reshape(-1, 1)
            feat_train_full = reservoir.features(X_train)
            feat_test_full = reservoir.features(X_test)
        
        # Discard washout period from features
        feat_train = feat_train_full[self.washout:]
        feat_test = feat_test_full[self.washout:]
        
        r2_scores = np.zeros(max_delay + 1)
        
        # Evaluate for each delay
        for k in range(max_delay + 1):
            # Create targets: predict u(t-k) where t is the time index of features
            # Features start at time washout, so feature at index i corresponds to time washout + i
            # Target for feature at time washout + i should be input at time washout + i - k
            # Input sequence indices: [0, 1, ..., train_length + washout + max_delay - 1]
            # For k=0: targets are inputs at [washout, washout+1, ..., washout+train_length-1]
            # For k>0: targets are inputs at [washout-k, washout-k+1, ..., washout-k+train_length-1]
            start_idx_train = self.washout - k
            start_idx_test = self.washout - k
            
            y_train = u_train[start_idx_train:start_idx_train + train_length]
            y_test = u_test[start_idx_test:start_idx_test + test_length]
            
            # Ensure lengths match
            min_len = min(len(feat_train), len(y_train))
            feat_train_k = feat_train[:min_len]
            y_train_k = y_train[:min_len]
            
            min_len_test = min(len(feat_test), len(y_test))
            feat_test_k = feat_test[:min_len_test]
            y_test_k = y_test[:min_len_test]
            
            # Train linear readout
            if len(feat_train_k) > 0 and len(feat_test_k) > 0:
                model = Ridge(alpha=lam)
                model.fit(feat_train_k, y_train_k)
                
                # Predict and compute R²
                y_pred = model.predict(feat_test_k)
                r2 = r2_score(y_test_k, y_pred)
                r2_scores[k] = max(0.0, r2)  # R² can be negative, but MC uses max(0, R²)
        
        mc = np.sum(r2_scores)
        return mc, r2_scores

