"""Random perturbation helpers for synthetic document rendering."""

from __future__ import annotations

import numpy as np


def pos(val: float, rng: np.random.Generator, sigma: float = 0.001) -> float:
    """Add Gaussian jitter to a normalised position coordinate."""
    return val + rng.normal(0.0, sigma)


def size(base: float, rng: np.random.Generator, sigma: float = 0.4) -> float:
    """Return *base* font size with added Gaussian noise (minimum 4 pt)."""
    return max(4.0, base + rng.normal(0.0, sigma))


def rotation(rng: np.random.Generator, sigma: float = 0.4) -> float:
    """Return a small random rotation angle in degrees."""
    return float(rng.normal(0.0, sigma))


def linewidth(base: float, rng: np.random.Generator, sigma: float = 0.08) -> float:
    """Return *base* line width with added noise (minimum 0.2)."""
    return max(0.2, base + rng.normal(0.0, sigma))


def gray(base: float, rng: np.random.Generator, sigma: float = 0.04) -> float:
    """Return a slightly jittered gray value clamped to [0, 1]."""
    return float(np.clip(base + rng.normal(0.0, sigma), 0.0, 1.0))
