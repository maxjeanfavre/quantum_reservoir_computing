# tasks/gaussians.py

import numpy as np
from .base import Task

class TwoGaussians(Task):
    """
    Binary classification task: two 2D Gaussian blobs.

    Class 0 ~ N(mean0, cov * I) + N(mean1, cov * I)
    Class 1 ~ N(mean2, cov * I) + N(mean3, cov * I)
    """

    def __init__(self,
                 mean0=(-0.7, -0.7),
                 mean1=(0.7, 0.7),
                 mean2=(-0.7, 0.7),
                 mean3=(0.7, -0.7),
                 cov=0.05):
        super().__init__(name="two_gaussians", input_dim=2)
        self.mean0 = np.array(mean0, dtype=float)
        self.mean1 = np.array(mean1, dtype=float)
        self.mean2 = np.array(mean2, dtype=float)
        self.mean3 = np.array(mean3, dtype=float)
        self.cov = float(cov)

    def make_dataset(self, n_samples: int, seed: int | None = None):
        """
        Returns:
            X : (N, 2)
            y : (N,) in {0,1}
        """
        rng = np.random.default_rng(seed)

        N_tot_0 = n_samples // 2
        N0 = N_tot_0 // 2
        N1 = N_tot_0 - N0

        N_tot_1 = n_samples - N_tot_0
        N2 = N_tot_1 // 2
        N3 = N_tot_1 - N2

        x0 = rng.multivariate_normal(self.mean0, self.cov * np.eye(2), N0)
        x1 = rng.multivariate_normal(self.mean1, self.cov * np.eye(2), N1)
        x2 = rng.multivariate_normal(self.mean2, self.cov * np.eye(2), N2)
        x3 = rng.multivariate_normal(self.mean3, self.cov * np.eye(2), N3)

        X = np.vstack([x0, x1, x2, x3])
        y = np.concatenate([
            np.zeros(N_tot_0, dtype=int),  # label 0
            np.ones(N_tot_1, dtype=int)    # label 1
        ])

        # Shuffle (X, y) together
        idx = rng.permutation(n_samples)
        return X[idx], y[idx]
