"""Heterogeneity metric: std-dev of threshold locations within an interval."""

from __future__ import annotations

import numpy as np


def hetero(policy: np.ndarray, I: tuple[float, float], gD: np.ndarray) -> float:
    """Std-dev of threshold locations within interval I = (lo, hi).

    Mirrors `ftn_hetero.m`: thresholds are the belief locations where the
    policy steps up; weights are the step sizes. Returns 0 if total step is 0
    (degenerate / no thresholds in interval).
    """
    policy = np.asarray(policy, dtype=float).ravel()
    gD = np.asarray(gD, dtype=float)

    # MATLAB: [~, n_lo] = max(gD >= I(1));      -> first index where gD >= lo
    #         [~, n_hi] = max(gD >= I(2)); n_hi = n_hi - 1;
    n_lo = int(np.argmax(gD >= I[0]))
    n_hi = int(np.argmax(gD >= I[1])) - 1

    # dns = [policy(n_lo), diff(policy(n_lo:n_hi))]  (MATLAB 1-based)
    # In python 0-based: dns[0] = policy[n_lo],
    # dns[1..n_hi-n_lo] = policy[n_lo+1:n_hi+1] - policy[n_lo:n_hi]
    head = np.array([policy[n_lo]])
    tail = policy[n_lo + 1 : n_hi + 1] - policy[n_lo : n_hi]
    dns = np.concatenate([head, tail])

    mask = dns > 0
    th_count = dns[mask]
    th_val = gD[: len(dns)][mask] if False else None  # noqa: F841
    # MATLAB indexes gD by the same range (positions of dns entries). The
    # entries of `dns` align with belief grid positions [n_lo, n_lo+1, ..., n_hi]
    th_val = gD[n_lo : n_hi + 1][mask]
    nn = th_count.sum()
    if nn == 0:
        return 0.0
    avg = (th_count * th_val).sum() / nn
    var = (th_count * (th_val - avg) ** 2).sum() / nn
    return float(np.sqrt(var))
