# tasks/circles.py

import numpy as np
from .base import Task

class TwoCircles(Task):
    """
    Binary classification task:
    Points sampled from two circles, each with its own radius and center.

    Class 0 -> circle centered at center0, radius r0
    Class 1 -> circle centered at center1, radius r1
    """

    def __init__(self,
                 r0: float = 0.5,
                 r1: float = 1.5,
                 noise: float = 0.05,
                 center0: tuple[float, float] = (0.0, 0.0),
                 center1: tuple[float, float] = (0.0, 0.0)):
        super().__init__(name="two_circles", input_dim=2)
        self.r0 = float(r0)
        self.r1 = float(r1)
        self.noise = float(noise)
        self.cx0, self.cy0 = map(float, center0)
        self.cx1, self.cy1 = map(float, center1)

    def make_dataset(self, n_samples: int, seed: int | None = None):
        """
        Returns:
            X : (N, 2)
            y : (N,) labels in {0,1}
        """
        rng = np.random.default_rng(seed)

        N0 = n_samples // 2
        N1 = n_samples - N0

        # ------- Class 0 circle -------
        theta0 = rng.uniform(0, 2*np.pi, N0)
        rvals0 = self.r0 + self.noise * rng.standard_normal(N0)
        x0 = np.stack([rvals0 * np.cos(theta0),
                       rvals0 * np.sin(theta0)], axis=1)
        x0[:, 0] += self.cx0
        x0[:, 1] += self.cy0
        y0 = np.zeros(N0, dtype=int)

        # ------- Class 1 circle -------
        theta1 = rng.uniform(0, 2*np.pi, N1)
        rvals1 = self.r1 + self.noise * rng.standard_normal(N1)
        x1 = np.stack([rvals1 * np.cos(theta1),
                       rvals1 * np.sin(theta1)], axis=1)
        x1[:, 0] += self.cx1
        x1[:, 1] += self.cy1
        y1 = np.ones(N1, dtype=int)

        # ------- Combine & shuffle -------
        X = np.vstack([x0, x1])
        y = np.concatenate([y0, y1])
        idx = rng.permutation(n_samples)
        return X[idx], y[idx]
