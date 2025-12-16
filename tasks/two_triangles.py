# tasks/two_u.py
import numpy as np
from .base import Task


def _sample_polyline(
    rng: np.random.Generator,
    vertices: np.ndarray,
    n_samples: int,
    noise: float = 0.03,
) -> np.ndarray:
    """
    Sample points along a polyline defined by `vertices`,
    approximately uniformly in arc length, and add isotropic Gaussian noise.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.
    vertices : np.ndarray
        Array of shape (n_vertices, 2) with the polyline vertices.
    n_samples : int
        Number of points to sample along the polyline.
    noise : float, default 0.03
        Standard deviation of the Gaussian noise added to each point.

    Returns
    -------
    pts : np.ndarray
        Array of shape (n_samples, 2) with sampled points.
    """
    vertices = np.asarray(vertices, dtype=float)
    seg_vecs = vertices[1:] - vertices[:-1]          # (n_segments, 2)
    seg_lens = np.linalg.norm(seg_vecs, axis=1)      # (n_segments,)
    cum_lens = np.cumsum(seg_lens)
    total_len = cum_lens[-1]

    # Sample u uniformly in [0, total_len] and map to a segment index
    u = rng.uniform(0.0, total_len, size=n_samples)
    seg_idx = np.searchsorted(cum_lens, u)           # which segment each point falls into

    seg_start = vertices[seg_idx]
    seg_vec = seg_vecs[seg_idx]
    prev_cum = np.concatenate([[0.0], cum_lens[:-1]])

    # Relative position within each segment in [0, 1]
    t = (u - prev_cum[seg_idx]) / seg_lens[seg_idx]
    pts = seg_start + t[:, None] * seg_vec

    if noise > 0.0:
        pts += rng.normal(scale=noise, size=pts.shape)

    return pts


class TwoTriangles(Task):
    """
    Two interlocked open open triangles.
    The dataset consists of noisy samples along these polylines.
    """

    def __init__(self, noise: float = 0.01):
        super().__init__(name="two_triangles", input_dim=2)
        self.noise = noise

    def make_dataset(self, n_samples: int, seed: int | None = None):
        """
        Generate a dataset of interlocked U-shapes.

        Parameters
        ----------
        n_samples : int
            Total number of samples (both classes combined).
        seed : int or None, default None
            Seed for the random number generator.

        Returns
        -------
        X : np.ndarray
            Array of shape (n_samples, 2) with 2D input points.
        y : np.ndarray
            Array of shape (n_samples,) with integer class labels (0 or 1).
        """
        rng = np.random.default_rng(seed)

        # Roughly balance samples between both classes
        n0 = n_samples // 2
        n1 = n_samples - n0

        # Class 0:
        verts0 = np.array([
            [-1.0, -1.0],
            [ 0.0,  0.6],
            [ 1.0, -1.0],
        ])

        # Class 1:
        verts1 = np.array([
            [ -1.0, -0.6],
            [ 0.0,  1.0],
            [ 1.0, -0.6],
        ])

        X0 = _sample_polyline(rng, verts0, n0, noise=self.noise)
        X1 = _sample_polyline(rng, verts1, n1, noise=self.noise)

        X = np.vstack([X0, X1])
        y = np.concatenate([
            np.zeros(n0, dtype=int),
            np.ones(n1, dtype=int),
        ])

        # Randomly shuffle the dataset while keeping X–y alignment
        idx = rng.permutation(n_samples)
        return X[idx], y[idx]
