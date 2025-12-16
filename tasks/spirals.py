# tasks/spirals.py
import numpy as np
from .base import Task

def make_spiral_point(phi, lab,
                      a=1/(6*np.pi), b=0.15,
                      spiral_type="linear"):
    if spiral_type == "linear":
        r = a * phi
    elif spiral_type == "log":
        r = a * np.exp(b * phi)
    else:
        raise ValueError(f"Unknown spiral_type: {spiral_type}")

    if lab == 0:
        x = r * np.cos(phi)
        y = r * np.sin(phi)
    else:
        x = -r * np.cos(phi)
        y = -r * np.sin(phi)
    return x, y


class TwoSpirals(Task):
    def __init__(self, a=1/(2*np.pi), b=0.15, spiral_type="linear"):
        super().__init__(name=f"two_spirals_{spiral_type}", input_dim=2)
        self.a = a
        self.b = b
        self.spiral_type = spiral_type

    def make_dataset(self, n_samples: int, seed: int | None = None):
        rng = np.random.default_rng(seed)
        phis = rng.uniform(0, 2*np.pi, n_samples)
        labs = rng.integers(0, 2, n_samples)

        X = np.empty((n_samples, 2), dtype=float)
        for i in range(n_samples):
            X[i, 0], X[i, 1] = make_spiral_point(
                phis[i], labs[i],
                a=self.a, b=self.b,
                spiral_type=self.spiral_type
            )
        y = labs.astype(int)
        return X, y
