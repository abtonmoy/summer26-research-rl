"""Monte-Carlo evaluation of policies."""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import binom

from .behav import behav_dyn_stoch
from .belief import update_partialcomm
from .bellman import crit_belif
from .config import eps_pm, g_star


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


def mc_performance_batch(policy, MC, para, num_para, n_jobs=None, seed: int = 0):
    """Vectorized `mc_performance`: all MC replicates stepped together in numpy.

    `n_jobs` is accepted and ignored (single-process numpy) so this is a drop-in
    replacement for `mc_performance`'s call signature.

    Same statistics (rho, conditional accuracy, unconditional accuracy) as the
    joblib path, but instead of N-by-T per-agent arrays per replicate it tracks
    only the per-step committed count n1 and successes x (drawn as one binomial
    per replicate) — the three statistics depend on nothing finer. RNG draw
    structure differs from the scalar path, so results match within MC noise
    (~1/sqrt(MC)), not bit-for-bit. Returns (rho, acc, acc_uncond).
    """
    rng = np.random.default_rng(seed)
    N = int(para.N)
    T = int(num_para.T)
    gN = int(num_para.gN)
    dg = 1.0 / (gN - 1)
    policy = np.asarray(policy, dtype=int).ravel()
    gp, gm = para.gamma_p, para.gamma_m
    eps_p, eps_m = eps_pm(para)
    gs = g_star(para)
    lam = para.lambda1

    # Per-replicate environment sequences with directional switching.
    s = np.where(rng.random(MC) < gs, 1, -1).astype(np.int64)
    S = np.empty((MC, T), dtype=np.int64)
    S[:, 0] = s
    for t in range(1, T):
        rate = np.where(s == 1, eps_m, eps_p)
        s = np.where(rng.random(MC) < rate, -s, s)
        S[:, t] = s

    g = np.full(MC, gs)
    sum_xn = np.zeros(MC)      # sum_t (x - lam*n1)
    num_acc = np.zeros(MC)     # sum over +state rounds of n1 (committed-correct)
    nplus = np.zeros(MC)       # count of +state rounds
    matches = np.zeros(MC)     # sum_t agent-rounds with action == state
    for t in range(T):
        gi = np.clip(np.rint(g / dg).astype(int), 0, gN - 1)
        n1 = policy[gi]                                   # committed per replicate
        st = S[:, t]
        gamma = np.where(st == 1, gp, gm)
        x = rng.binomial(n1, gamma)                       # successes per replicate
        plus = st == 1
        sum_xn += x - lam * n1
        nplus += plus
        num_acc += np.where(plus, n1, 0)
        matches += np.where(plus, n1, N - n1)
        # full-observation belief update (vectorized, asymmetric drift)
        bgp = g * (gp**x) * (1 - gp) ** (n1 - x)
        bgm = (1 - g) * (gm**x) * (1 - gm) ** (n1 - x)
        g = ((1 - eps_m) * bgp + eps_p * bgm) / (bgp + bgm)

    rho = sum_xn / (N * T)
    acc = np.where(nplus > 0, num_acc / (N * np.maximum(nplus, 1)), 0.0)
    acc_uncond = matches / (N * T)
    return float(rho.mean()), float(acc.mean()), float(acc_uncond.mean())


# -------------------------------------------------------------------------
# Response-time MC. Matches ftn_mc_response.m.
# -------------------------------------------------------------------------
def _s_est(g, s, thc):
    """Estimated state indicator: -1 if disagreement with truth, +1 if agreement."""
    return -((-1) ** int(((g - thc) * s) > 0))


def _update_belief(g, x, a, para):
    gp, gm = para.gamma_p, para.gamma_m
    eps_p, eps_m = eps_pm(para)
    bgp = g * (gp**x) * (1 - gp) ** (a - x)
    bgm = (1 - g) * (gm**x) * (1 - gm) ** (a - x)
    return ((1 - eps_m) * bgp + eps_p * bgm) / (bgp + bgm)


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

    eps_p, eps_m = eps_pm(para)
    gs = g_star(para)

    def one(s):
        rng = np.random.default_rng(s)
        g = gs
        # lo->hi: new state +1 (high); dwell ~ geometric(rate of leaving high = eps_m)
        Ti = int(rng.geometric(eps_m) - 1)
        g, z_lohi = _response_time(Ti, g, 1, para.gamma_p, thc, dg, policy, para, rng)
        # hi->lo: new state -1 (low); dwell ~ geometric(rate of leaving low = eps_p)
        Ti = int(rng.geometric(eps_p) - 1)
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
    eps_p, eps_m = eps_pm(para)
    gs = g_star(para)

    g = gs * np.ones(N)
    r_total = 0.0
    a_plus = 0
    t_plus = 0
    S = 1 if rng.random() < gs else -1

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

        rate = eps_m if S == 1 else eps_p
        if rng.random() < rate:
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


def partialcomm_run_batch(theta, S, para, rng):
    """Vectorized partial-communication episodes over a batch of replicates.

    `S` is (MC, T) of environment states in {-1, +1} (one row per replicate;
    may be shared / common-random-numbers across callers). Steps all MC episodes
    together in numpy — the per-agent communication loop is vectorized over N and
    the replicate loop over MC; only the belief recursion over T stays serial.

    Returns (r_total, a_plus, t_plus), each shape (MC,):
      r_total = sum_t (x_t - lambda1 * a_t)
      a_plus  = sum over +state rounds of committed count
      t_plus  = number of +state rounds

    Equivalence to the scalar `_one_partialcomm_rep` uses the identity
        sum_x Hypergeom(y; a, x, m) * Binom(x; a, p) = Binom(y; m, p),
    so the peer-evidence marginal likelihood is just `binom.pmf(y, m, p)` — no
    per-replicate variable-length sum over the hidden success count. Draw order
    differs from the scalar path, so results match within MC noise.
    """
    theta = np.asarray(theta, dtype=float).ravel()
    N = int(para.N)
    MC, T = S.shape
    kappa = para.kappa
    gp, gm = para.gamma_p, para.gamma_m
    eps_p, eps_m = eps_pm(para)
    gs = g_star(para)
    lam = para.lambda1

    G = np.full((MC, N), gs)
    r_total = np.zeros(MC)
    a_plus = np.zeros(MC)
    t_plus = np.zeros(MC)

    for t in range(T):
        St = S[:, t]
        gamma = np.where(St == 1, gp, gm)               # (MC,)
        C = G > theta[None, :]                           # (MC, N) committed
        a = C.sum(1)                                     # (MC,)
        succ = (rng.random((MC, N)) < gamma[:, None]) & C
        z = np.where(C, np.where(succ, 1, 0), -1)        # {-1 idle, 0 fail, 1 success}
        x = succ.sum(1)                                  # (MC,) successes among committed
        r_total += x - lam * a
        plus = St == 1
        a_plus += np.where(plus, a, 0)
        t_plus += plus

        ab = a[:, None]
        xb = x[:, None]
        # sample size m_i ~ Binomial(pop, kappa); idle agents see all `a` committed,
        # committed agents see the other `a-1`.
        pop = np.clip(np.where(z == -1, ab, ab - 1), 0, None)
        m = rng.binomial(pop, kappa)
        # observed successes y_i ~ Hypergeometric(ngood, nbad, m) from the realized x.
        ngood = np.where(z == 1, xb - 1, xb)
        nbad = np.where(z == 0, ab - 1 - xb, ab - xb)
        valid = m > 0
        ng_s = np.where(valid, ngood, 1)
        nb_s = np.where(valid, nbad, 0)
        m_s = np.where(valid, m, 1)
        y = np.where(valid, rng.hypergeometric(ng_s, nb_s, m_s), 0)

        # Bayesian update with private obs z and peer evidence (y of m).
        Fp = binom.pmf(y, m, gp)                         # = marginal P(y | m, gamma_+)
        Fm = binom.pmf(y, m, gm)
        cp = np.where(z == 1, gp, np.where(z == 0, 1.0 - gp, 1.0))
        cm = np.where(z == 1, gm, np.where(z == 0, 1.0 - gm, 1.0))
        num = cp * Fp * G
        den = num + cm * Fm * (1.0 - G)
        U = num / den
        G = (1.0 - eps_m) * U + eps_p * (1.0 - U)

    return r_total, a_plus, t_plus


def mc_partialcomm_batch(theta, T: int, MC: int, para, n_jobs=None, seed: int = 0):
    """Vectorized drop-in for `mc_partialcomm`. Returns (rho_mc, alpha_mc), (MC,).

    `n_jobs` accepted and ignored. Generates one (MC, T) environment block with
    directional switching, then runs all replicates with `partialcomm_run_batch`.
    Matches the scalar path within MC noise (~1/sqrt(MC)).
    """
    rng = np.random.default_rng(seed)
    N = int(para.N)
    eps_p, eps_m = eps_pm(para)
    gs = g_star(para)

    s = np.where(rng.random(MC) < gs, 1, -1).astype(np.int64)
    S = np.empty((MC, T), dtype=np.int64)
    S[:, 0] = s
    for t in range(1, T):
        rate = np.where(s == 1, eps_m, eps_p)
        s = np.where(rng.random(MC) < rate, -s, s)
        S[:, t] = s

    r_total, a_plus, t_plus = partialcomm_run_batch(theta, S, para, rng)
    rho = r_total / T
    alpha = a_plus / (N * np.maximum(t_plus, 1))
    return rho, alpha
