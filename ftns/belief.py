"""Belief-update functions."""

from __future__ import annotations

import numpy as np
from scipy.stats import binom, hypergeom

from .config import eps_pm


def update_belief(g: float, x: int, a: int, para) -> float:
    """One-step Bayesian + drift update assuming full observation of (a, x).

    Drift uses directional switching rates: the posterior weight on the high
    state keeps (1 - eps_minus) and the low-state weight leaks up at eps_plus.
    Symmetric (eps_plus == eps_minus) recovers the original update.
    """
    gp, gm = para.gamma_p, para.gamma_m
    eps_p, eps_m = eps_pm(para)
    bgp = g * (gp**x) * (1 - gp) ** (a - x)
    bgm = (1 - g) * (gm**x) * (1 - gm) ** (a - x)
    return ((1 - eps_m) * bgp + eps_p * bgm) / (bgp + bgm)


def update_partialcomm(z: int, y: int, m: int, a: int, g: float, para) -> float:
    """Bayesian update with partial communication + optional private obs.

    Inputs match `ftn_update_partialcomm.m`:
      z in {-1, 0, 1}: -1 = no private obs, 1 = private success, 0 = private failure
      y = # successes among m encountered
      m = # encountered (sample size)
      a = total committed agents
      g = current belief
    """
    eps_p, eps_m = eps_pm(para)
    gp, gm = para.gamma_p, para.gamma_m

    # Degenerate case: no committed agents -> no observation, only noise update.
    # MATLAB's hygepdf(0,0,0,0) returns 1 (degenerate hypergeometric), but
    # scipy's hypergeom.pmf(0,0,0,0) returns NaN. Handle explicitly.
    if a == 0:
        return (1 - eps_m) * g + eps_p * (1 - g)

    x_vec = np.arange(a + 1)
    # Hypergeometric likelihood p(y|x,m) with population a, successes x_vec, sample m.
    # scipy: hypergeom.pmf(k, M, n, N) -> k=y, M=a, n=x_vec, N=m
    lik = hypergeom.pmf(y, a, x_vec, m)
    Bp = binom.pmf(x_vec, a, gp)
    Bm = binom.pmf(x_vec, a, gm)

    Fp = np.sum(lik * Bp)
    Fm = np.sum(lik * Bm)

    if z == -1:
        U = Fp * g / (Fm * (1 - g) + Fp * g)
    elif z == 1:
        U = gp * Fp * g / (gp * Fp * g + gm * Fm * (1 - g))
    elif z == 0:
        U = (1 - gp) * Fp * g / ((1 - gp) * Fp * g + (1 - gm) * Fm * (1 - g))
    else:
        raise ValueError(f"z must be in {{-1, 0, 1}}, got {z}")

    return (1 - eps_m) * U + eps_p * (1 - U)
