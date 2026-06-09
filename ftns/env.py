"""Stochastic environmental state sequence."""

from __future__ import annotations

import numpy as np


def env_state(T: int, para, rng: np.random.Generator | None = None) -> np.ndarray:
    """Generate S_{1:T} in {-1, +1} with switching probability epsilon.

    Initial state is +1 or -1 with equal probability.
    """
    if rng is None:
        rng = np.random.default_rng()
    eps = para.epsilon
    S = np.empty(T, dtype=int)
    s = 1 if rng.random() > 0.5 else -1
    S[0] = s
    for t in range(1, T):
        if rng.random() < eps:
            s = -s
        S[t] = s
    return S
