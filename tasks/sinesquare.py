import numpy as np
from .base import Task

class SineSquare(Task):
    """
    A waveform classification task for evaluating reservoir computing systems.
    
    The task generates sequences by randomly concatenating 8-point sine or square
    waveforms. Each time step is labeled according to which pattern it belongs to.
    
    Sine waveform: [0, 0.7071, 1, 0.7071, 0, -0.7071, -1, -0.7071]
    Square waveform: [1, 1, 1, 1, -1, -1, -1, -1]
    
    Labels: 0 for sine, 1 for square
    """
    
    # Define the 8-point patterns
    SINE_PATTERN = np.array([0, 0.7071, 1, 0.7071, 0, -0.7071, -1, -0.7071])
    SQUARE_PATTERN = np.array([1, 1, 1, 1, -1, -1, -1, -1])
    PATTERN_LENGTH = 8
    
    def __init__(
        self,
        sequence_length: int = 97,
        sine_prob: float = 0.5,
        name: str = "sine_square"
    ):
        """
        Parameters:
        -----------
        sequence_length : int
            Total length of sequence to generate (will be rounded to multiple of 8)
        sine_prob : float
            Probability of choosing sine pattern (default: 0.5, balanced)
        name : str
            Task name identifier
        """
        super().__init__(name=name, input_dim=1)
        self.sequence_length = sequence_length
        self.sine_prob = sine_prob
    
    def generate_sequence(self, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate a sequence by randomly concatenating sine and square patterns.
        
        Parameters:
        -----------
        seed : int, optional
            Random seed
        
        Returns:
        --------
        sequence : np.ndarray
            Input sequence of shape (sequence_length,)
        labels : np.ndarray
            Labels indicating pattern type: 0 for sine, 1 for square, shape (sequence_length,)
        """
        rng = np.random.default_rng(seed)
        
        # Calculate number of complete patterns
        n_patterns = (self.sequence_length + self.PATTERN_LENGTH - 1) // self.PATTERN_LENGTH
        
        sequence = []
        labels = []
        
        for _ in range(n_patterns):
            # Randomly choose sine or square
            is_sine = rng.random() < self.sine_prob
            
            if is_sine:
                pattern = self.SINE_PATTERN.copy()
                label = 0
            else:
                pattern = self.SQUARE_PATTERN.copy()
                label = 1
            
            sequence.extend(pattern)
            labels.extend([label] * self.PATTERN_LENGTH)
        
        # Trim to exact sequence_length
        sequence = np.array(sequence[:self.sequence_length])
        labels = np.array(labels[:self.sequence_length])
        
        return sequence, labels
    
    def make_dataset(self, n_samples: int, seed: int | None = None):
        """
        Generate dataset for classification.
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate. Each sample is a single time step.
            The actual sequence length will be n_samples.
        seed : int, optional
            Random seed
        
        Returns:
        --------
        X : np.ndarray
            Input sequence of shape (n_samples, 1)
        y : np.ndarray
            Labels of shape (n_samples,) with values in {0, 1}
            0 = sine, 1 = square
        """
        # Generate a single long sequence
        sequence, labels = self.generate_sequence(seed=seed)
        
        # If n_samples differs from sequence_length, adjust
        if n_samples != len(sequence):
            # Generate enough patterns to cover n_samples
            self.sequence_length = n_samples
            sequence, labels = self.generate_sequence(seed=seed)
        
        # Reshape for Task interface: (n_samples, input_dim=1)
        X = sequence.reshape(-1, 1)
        y = labels
        
        return X, y
    
    def make_sequence_dataset(self, n_sequences: int = 1, seed: int | None = None):
        """
        Generate multiple sequences for time-series processing.
        
        Parameters:
        -----------
        n_sequences : int
            Number of sequences to generate
        seed : int, optional
            Random seed
        
        Returns:
        --------
        sequences : list of np.ndarray
            List of sequences, each of shape (sequence_length,)
        labels_list : list of np.ndarray
            List of label arrays, each of shape (sequence_length,)
        """
        rng = np.random.default_rng(seed)
        sequences = []
        labels_list = []
        
        for i in range(n_sequences):
            seq, labels = self.generate_sequence(seed=rng.integers(0, 2**31))
            sequences.append(seq)
            labels_list.append(labels)

        #print(labels_list)
        
        return sequences, labels_list
