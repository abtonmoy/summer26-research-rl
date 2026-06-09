"""Monte-Carlo evaluation of policies."""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed

from .behav import behav_dyn_stoch
from .belief import update_partialcomm
from .bellman import crit_belif


# -------------------------------------------------------------------------
# Performance MC: rho, conditional accuracy, unconditional accuracy.
# Matches ftn_mc_performance.m.
# -------------------------------------------------------------------------
def _one_perf_rep(policy, para, num_para, seed):
    rng = np.random.default_rng(seed)
    behav = behav_dyn_stoch(policy, para, num_para, rng=rng)
    xt, at = behav["xt"], behav["at"]
    Ea = at.mean()
    Ex = xt.mean()
    rho = Ex - para.lambda1 * Ea
    st01 = (behav["st"] + 1) // 2     # {-1,1} -> {0,1}
    # conditional accuracy: among rounds where state == +, fraction committed
    denom = para.N * st01.sum()
    acc = float(((at == st01[None, :]) * st01[None, :]).sum()) / denom if denom > 0 else 0.0
    # unconditional accuracy: fraction of agent-rounds with at == st
    acc_uncond = float((at == st01[None, :]).mean())
    return rho, acc, acc_uncond


def mc_performance(policy, MC, para, num_para, n_jobs: int = -1, seed: int = 0):
    """Returns (rho, acc, acc_uncond) averaged over MC replicates."""
    seeds = np.random.SeedSequence(seed).spawn(MC)
    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_one_perf_rep)(policy, para, num_para, s) for s in seeds
    )
    arr = np.array(results)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean()), float(arr[:, 2].mean())


# -------------------------------------------------------------------------
# Response-time MC. Matches ftn_mc_response.m.
# -------------------------------------------------------------------------
def _s_est(g, s, thc):
    """Estimated state indicator: -1 if disagreement with truth, +1 if agreement."""
    return -((-1) ** int(((g - thc) * s) > 0))


def _update_belief(g, x, a, para):
    gp, gm, eps = para.gamma_p, para.gamma_m, para.epsilon
    bgp = g * (gp**x) * (1 - gp) ** (a - x)
    bgm = (1 - g) * (gm**x) * (1 - gm) ** (a - x)
    return ((1 - eps) * bgp + eps * bgm) / (bgp + bgm)


def _response_time(Ti, g, s, p, thc, dg, policy, para, rng):
    """Return (g, z) where z = [response_time, success_flag].

    Iterates Ti+1 update steps; "success" = belief moves to agree with the new
    state. S(1,1) in MATLAB is the s_est BEFORE any update, so if the initial
    belief already agrees with state s, response time is 0.
    """
    success_t = -1
    # MATLAB S(1,1) = ftn_s_est(g, s, thc) - initial check before updates
    if _s_est(g, s, thc) == 1:
        success_t = 0
    for t in range(Ti + 1):
        a = int(policy[int(round(g / dg))])
        x = int(rng.binomial(a, p)) if a > 0 else 0
        g = _update_belief(g, x, a, para)
        if success_t < 0 and _s_est(g, s, thc) == 1:
            success_t = t + 1
    if success_t >= 0:
        return g, np.array([success_t, 1])
    return g, np.array([Ti, 0])


def mc_response(policy, para, num_para, MC: int, seed: int = 0, n_jobs: int = -1):
    """Returns (T_lohi, T_hilo) each a (2, MC) array: row 0 = time, row 1 = success."""
    dg = 1.0 / (num_para.gN - 1)
    thc = crit_belif(para)

    seeds = np.random.SeedSequence(seed).spawn(MC)

    def one(s):
        rng = np.random.default_rng(s)
        g = 0.5
        # lo->hi: state +1, success rate gamma_p
        Ti = int(rng.geometric(para.epsilon) - 1)
        g, z_lohi = _response_time(Ti, g, 1, para.gamma_p, thc, dg, policy, para, rng)
        # hi->lo: state -1, success rate gamma_m
        Ti = int(rng.geometric(para.epsilon) - 1)
        g, z_hilo = _response_time(Ti, g, -1, para.gamma_m, thc, dg, policy, para, rng)
        return z_lohi, z_hilo

    results = Parallel(n_jobs=n_jobs, prefer="processes")(delayed(one)(s) for s in seeds)
    T_lohi = np.column_stack([r[0] for r in results])
    T_hilo = np.column_stack([r[1] for r in results])
    return T_lohi, T_hilo


# -------------------------------------------------------------------------
# Partial-communication MC. Matches ftn_mc_partialcomm.m.
# -------------------------------------------------------------------------
def _one_partialcomm_rep(theta, T, para, seed):
    rng = np.random.default_rng(seed)
    N = para.N
    kappa = para.kappa
    eps = para.epsilon

    g = 0.5 * np.ones(N)
    r_total = 0.0
    a_plus = 0
    t_plus = 0
    S = 1 if rng.random() < 0.5 else -1

    for _ in range(T):
        gamma = para.gamma_p if S == 1 else para.gamma_m
        C = np.where(g > theta)[0]
        a = len(C)
        if S == 1:
            a_plus += a
            t_plus += 1

        z_vec = -np.ones(N, dtype=int)
        if a > 0:
            z_obs = (rng.random(a) < gamma).astype(int)
            z_vec[C] = z_obs
            x = int(z_obs.sum())
        else:
            x = 0
        r_total += x - para.lambda1 * a

        g_new = np.empty(N)
        for i in range(N):
            zi = int(z_vec[i])
            if zi == -1:
                if a == 0:
                    m_i, y_i = 0, 0
                else:
                    m_i = int(rng.binomial(a, kappa))
                    y_i = int(rng.hypergeometric(x, a - x, m_i)) if (a > 0 and m_i > 0) else 0
            elif zi == 1:
                if a == 1:
                    m_i, y_i = 0, 0
                else:
                    m_i = int(rng.binomial(a - 1, kappa))
                    pop = a - 1
                    ngood = x - 1
                    nbad = pop - ngood
                    y_i = int(rng.hypergeometric(ngood, nbad, m_i)) if (pop > 0 and m_i > 0) else 0
            else:  # zi == 0
                if a == 1:
                    m_i, y_i = 0, 0
                else:
                    m_i = int(rng.binomial(a - 1, kappa))
                    pop = a - 1
                    ngood = x
                    nbad = pop - ngood
                    y_i = int(rng.hypergeometric(ngood, nbad, m_i)) if (pop > 0 and m_i > 0) else 0
            g_new[i] = update_partialcomm(zi, y_i, m_i, a, g[i], para)
        g = g_new

        if rng.random() < eps:
            S = -S

    rho_mc = r_total / T
    alpha_mc = a_plus / (N * max(t_plus, 1))
    return rho_mc, alpha_mc


def mc_partialcomm(theta, T: int, MC: int, para, n_jobs: int = -1, seed: int = 0):
    """Returns (rho_mc, alpha_mc) arrays of shape (MC,)."""
    theta = np.asarray(theta, dtype=float).ravel()
    seeds = np.random.SeedSequence(seed).spawn(MC)
    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_one_partialcomm_rep)(theta, T, para, s) for s in seeds
    )
    arr = np.array(results)
    return arr[:, 0], arr[:, 1]
