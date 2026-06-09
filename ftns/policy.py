"""Convert between belief-grid policies and threshold distributions."""

from __future__ import annotations

import numpy as np


def policy2dist(policy: np.ndarray, gD: np.ndarray, num_para) -> np.ndarray:
    """Belief-grid step policy -> sorted threshold vector of length N.

    For each grid step where the policy increases, emit (n_step) copies of the
    belief at that step. Matches MATLAB `ftn_policy2dist`.
    """
    policy = np.asarray(policy, dtype=int).ravel()
    gN = num_para.gN
    dns = policy[1:] - policy[:-1]
    out = []
    for k in range(gN - 1):
        if dns[k] > 0:
            out.append(np.full(int(dns[k]), gD[k + 1]))
    if not out:
        return np.zeros(0)
    return np.concatenate(out)


def dist2policy(dist: np.ndarray, num_para) -> np.ndarray:
    """Threshold distribution(s) -> step policy n*(g) on belief grid.

    Accepts a 1-D vector (length N) or a 2-D matrix (M x N) of distributions.
    Returns shape (gN,) or (M, gN).
    Matches MATLAB `ftn_dist2policy`: policy(j) = #{i: theta_i <= gD(j)}.
    """
    gN = num_para.gN
    gD = np.linspace(0.0, 1.0, gN)
    dist = np.asarray(dist, dtype=float)
    if dist.ndim == 1:
        return np.sum(dist[:, None] <= gD[None, :], axis=0).astype(int)
    # 2-D: M x N
    # broadcast to M x N x gN
    cmp = dist[:, :, None] <= gD[None, None, :]
    return cmp.sum(axis=1).astype(int)
