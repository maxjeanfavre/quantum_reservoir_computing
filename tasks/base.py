
import numpy as np

class Task:
    def __init__(self, name: str, input_dim: int):
        self.name = name
        self.input_dim = input_dim

    def make_dataset(self, n_samples: int, seed: int | None = None):
        """Return (X, y) with:
           X : (N, input_dim)
           y : (N,) labels (e.g. {0,1})
        """
        raise NotImplementedError
    