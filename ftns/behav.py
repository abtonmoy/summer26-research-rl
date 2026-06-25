"""Trajectory simulation under a fixed policy."""

from __future__ import annotations

import numpy as np

from .belief import update_belief
from .config import g_star
from .env import env_state


def _act(g: float, policy: np.ndarray, dg: float) -> int:
    gi = int(round(g / dg))
    return int(policy[gi])


def behav_dyn(policy, st, para, num_para, rng: np.random.Generator | None = None) -> dict:
    """Deterministic state sequence -> belief / commitment / success trajectory.

    `st` is in {0, 1} (NOT {-1, 1}) — matches `ftn_behav_dyn.m`.
    Returns dict with keys gt, gt_tru, at, xt, rt.
    """
    if rng is None:
        rng = np.random.default_rng()
    T = len(st)
    dg = 1.0 / (num_para.gN - 1)
    policy = np.asarray(policy, dtype=int).ravel()

    gt = np.zeros(T)
    at = np.zeros(T, dtype=int)
    xt = np.zeros(T, dtype=int)
    rt = np.zeros(T)

    g = g_star(para)
    for t in range(T):
        gt[t] = g
        a = _act(g, policy, dg)
        at[t] = a
        gamma = para.gamma_p if st[t] == 1 else para.gamma_m
        x = int(rng.binomial(a, gamma)) if a > 0 else 0
        xt[t] = x
        rt[t] = x - para.lambda1 * a
        g = update_belief(g, x, a, para)

    return {"gt": gt, "gt_tru": np.asarray(st), "at": at, "xt": xt, "rt": rt}


def behav_dyn_stoch(policy, para, num_para, rng: np.random.Generator | None = None) -> dict:
    """Stochastic state sequence -> trajectory with per-agent commitment/success.

    Matches `ftn_behav_dyn_stoch.m`. Returns dict with:
      st : (T,) in {-1, 1}
      gt : (T,) belief trajectory
      at : (N, T) commitment indicator
      xt : (N, T) success indicator
    """
    if rng is None:
        rng = np.random.default_rng()
    N = para.N
    T = num_para.T
    dg = 1.0 / (num_para.gN - 1)
    policy = np.asarray(policy, dtype=int).ravel()

    gt = np.zeros(T)
    at = np.zeros((N, T), dtype=np.int8)
    xt = np.zeros((N, T), dtype=np.int8)

    st = env_state(T, para, rng=rng)  # {-1, 1}

    g = g_star(para)
    for t in range(T):
        gt[t] = g
        n1 = _act(g, policy, dg)
        gamma = para.gamma_p if st[t] == 1 else para.gamma_m
        at[:n1, t] = 1
        if n1 > 0:
            xt[:n1, t] = (rng.random(n1) < gamma).astype(np.int8)
        x = int(xt[:, t].sum())
        g = update_belief(g, x, n1, para)

    return {"st": st, "gt": gt, "at": at, "xt": xt}
