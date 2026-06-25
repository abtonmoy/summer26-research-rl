"""Bellman value-iteration core.

Discretizes belief g over a uniform grid of size `gN` on [0, 1]. For each
possible action n in {0, ..., N} we precompute:
  - r_bar[n, g] : immediate expected return when n agents commit at belief g
  - K_n         : linear operator on the value function V mapping V -> E[V(g')|n,g]
The transition operator uses the binomial likelihood of x successes among n
committed agents and the Bayesian + drift belief update. Bilinear interpolation
puts g' back on the discretized grid.
"""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed

from .config import eps_pm


def _build_K_for_n(
    n: int,
    gD: np.ndarray,
    dg: float,
    gN: int,
    gamma_p: float,
    gamma_m: float,
    eps_p: float,
    eps_m: float,
) -> np.ndarray:
    """Build the (gN x gN) interpolation operator K for action n.

    Column ng holds the distribution over future-belief grid indices given
    g0 = gD[ng] and action n, weighted by E[g'|x,n,g] x P(x|n,g).
    """
    K = np.zeros((gN, gN))

    r1 = gamma_p  # success prob in state +
    r0 = gamma_m  # success prob in state -

    for ng in range(gN):
        g0 = gD[ng]
        # x ranges 0..n
        x = np.arange(n + 1)
        # P(x|n) = g*Binom(n,r1) + (1-g)*Binom(n,r0) — total event probability
        from scipy.stats import binom  # local import to keep top clean
        Ev = binom.pmf(x, n, r1) * g0 + binom.pmf(x, n, r0) * (1 - g0)

        # Posterior g' = ((1-eps_m)*g*P(x|+) + eps_p*(1-g)*P(x|-)) /
        #                (g*P(x|+) + (1-g)*P(x|-))
        # Directional drift: high-state weight keeps (1-eps_minus), low-state
        # weight leaks up at eps_plus. Symmetric rates recover the original.
        Bp = (r1**x) * ((1 - r1) ** (n - x))
        Bm = (r0**x) * ((1 - r0) ** (n - x))
        num = (1 - eps_m) * g0 * Bp + eps_p * (1 - g0) * Bm
        den = g0 * Bp + (1 - g0) * Bm
        # avoid div-by-zero (only happens if Bp==Bm==0, which requires non-physical input)
        with np.errstate(divide="ignore", invalid="ignore"):
            Gv = np.where(den > 0, num / den, 0.0)

        # linear interpolation onto belief grid
        gv_scaled = Gv / dg
        i = np.floor(gv_scaled).astype(int)
        s = gv_scaled - i

        for x1 in range(n + 1):
            ii = i[x1]
            ss = s[x1]
            ee = Ev[x1]
            if ii < gN - 1:
                K[ii, ng] += (1 - ss) * ee
                K[ii + 1, ng] += ss * ee
            else:
                K[gN - 1, ng] += ee
    return K


def bellman_rhs_component(para, num_para, n_jobs: int = -1) -> dict:
    """Precompute (r_bar, V_bracket) for the Bellman backup.

    Returns a dict with:
      r_bar     : (N+1, gN) array of immediate returns
      V_bracket : list of (gN, gN) ndarrays, one per action n=0..N
    """
    N = para.N
    gN = num_para.gN
    gD = np.linspace(0.0, 1.0, gN)
    dg = gD[1]

    # immediate expected reward by commitment: r(g) = gamma_p*g + gamma_m*(1-g)
    r_g = para.gamma_p * gD + para.gamma_m * (1 - gD)
    r1 = r_g - para.lambda1  # net per-committer return
    r0 = -para.lambda0       # idling cost (paid by N-n idlers)

    n_vec = np.arange(N + 1).reshape(-1, 1)             # column
    n_idle = np.arange(N, -1, -1).reshape(-1, 1)        # column reversed
    r_bar = n_vec * r1 + n_idle * r0                     # (N+1, gN)

    # Parallel build of K_n
    eps_p, eps_m = eps_pm(para)
    Kcell = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_build_K_for_n)(
            n, gD, dg, gN, para.gamma_p, para.gamma_m, eps_p, eps_m
        )
        for n in range(N + 1)
    )

    return {"r_bar": r_bar, "V_bracket": Kcell}


def bellman_rhs(V: np.ndarray, comp: dict) -> np.ndarray:
    """Compute right-hand side V1a[n, g] = r_bar[n, g] + (V @ K_n)[g]."""
    r_bar = comp["r_bar"]
    V_bracket = comp["V_bracket"]
    rows = [V @ K for K in V_bracket]   # each is (gN,)
    EV = np.vstack(rows)                # (N+1, gN)
    return r_bar + EV


def bellman_sol(comp: dict, num_para, tol: float = 1e-10, max_iter: int = 100_000) -> tuple[np.ndarray, float]:
    """Relative value iteration.

    Returns (policy_opt, rho_opt) where policy_opt is the action n*(g) on the
    belief grid and rho_opt is the long-run average return.
    """
    gN = num_para.gN
    V = np.zeros(gN)
    err = tol + 1.0
    rho = 0.0
    V1a = None
    it = 0
    while err > tol and it < max_iter:
        V1a = bellman_rhs(V, comp)        # (N+1, gN)
        V1 = V1a.max(axis=0)              # (gN,)
        rho = V1[0] - V[0]
        V1 = V1 - rho
        err = float(np.abs(V1 - V).sum())
        V = V1
        it += 1

    policy_opt = V1a.argmax(axis=0).astype(int)
    return policy_opt, float(rho)


def avg_return(policy: np.ndarray, comp: dict, para, num_para,
               tol: float = 1e-6, max_iter: int = 100_000) -> float:
    """Average return when *following* a fixed policy (no max over actions).

    Mirrors `ftn_avg_return.m`.
    """
    gN = num_para.gN
    policy = np.asarray(policy, dtype=int).ravel()

    V = np.zeros(gN)
    err = tol + 1.0
    rho = 0.0
    it = 0
    while err > tol and it < max_iter:
        V1a = bellman_rhs(V, comp)
        # follow policy: pick row policy[g] at column g
        V1 = V1a[policy, np.arange(gN)]
        rho = V1[0] - V[0]
        V1 = V1 - rho
        err = float(np.abs(V1 - V).sum())
        V = V1
        it += 1
    return float(rho)


def avg_return_batch(policies, comp: dict, num_para,
                     tol: float = 1e-6, max_iter: int = 100_000) -> np.ndarray:
    """Vectorized `avg_return` for a whole *batch* of fixed policies.

    Mathematically identical to calling `avg_return` on each row of `policies`,
    but solves the entire population simultaneously with numpy. Two wins over the
    per-policy path:

      1. For policy m we only need its own transition K_{policy[m,g]} at each
         state g, not `bellman_rhs`'s full (N+1, gN) stack over every action.
      2. The population dimension M moves from a Python/joblib loop into a single
         batched value iteration (one einsum per step).

    `policies` is (M, gN) of integer actions on the belief grid. Returns rho of
    shape (M,). Deterministic, so it agrees with `avg_return` to iteration tol.
    """
    gN = num_para.gN
    policies = np.asarray(policies, dtype=int)
    squeeze = policies.ndim == 1
    if squeeze:
        policies = policies[None, :]
    M = policies.shape[0]

    r_bar = comp["r_bar"]                       # (N+1, gN)
    K_stack = np.asarray(comp["V_bracket"])     # (N+1, gN, gN)

    cols = np.arange(gN)
    # Pt[m, g, i] = K_{policy[m,g]}[i, g], so EV[m,g] = sum_i Pt[m,g,i] * V[m,i]
    # matches the scalar (V @ K_n)[g] = sum_i V[i] K_n[i,g].
    Pt = K_stack[policies, :, cols[None, :]]    # (M, gN, gN)
    r_m = r_bar[policies, cols[None, :]]        # (M, gN)

    V = np.zeros((M, gN))
    rho = np.zeros(M)
    for _ in range(max_iter):
        EV = np.einsum("mgi,mi->mg", Pt, V, optimize=True)
        V1 = r_m + EV
        rho = V1[:, 0] - V[:, 0]
        V1 = V1 - rho[:, None]
        err = np.abs(V1 - V).sum(axis=1)
        V = V1
        if np.all(err <= tol):
            break

    return float(rho[0]) if squeeze else rho


def crit_belif(para) -> float:
    """Closed-form critical belief: r(g) - lambda1 = 0."""
    return (para.lambda1 - para.gamma_m) / (para.gamma_p - para.gamma_m)
