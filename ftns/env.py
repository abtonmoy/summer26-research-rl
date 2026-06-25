"""Stochastic environmental state sequence."""

from __future__ import annotations

import numpy as np

from .config import eps_pm, g_star


def env_state(T: int, para, rng: np.random.Generator | None = None) -> np.ndarray:
    """Generate S_{1:T} in {-1, +1} with directional switching rates.

    Leaving the high state (+1) has probability eps_minus, leaving the low
    state (-1) has probability eps_plus. The initial state is drawn from the
    stationary distribution: +1 with probability g* = eps_p / (eps_p + eps_m).
    Symmetric rates recover equal switching and a 1/2 initial split.
    """
    if rng is None:
        rng = np.random.default_rng()
    eps_p, eps_m = eps_pm(para)
    gs = g_star(para)
    S = np.empty(T, dtype=int)
    s = 1 if rng.random() < gs else -1
    S[0] = s
    for t in range(1, T):
        rate = eps_m if s == 1 else eps_p
        if rng.random() < rate:
            s = -s
        S[t] = s
    return S
