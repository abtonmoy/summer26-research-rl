"""figS7_partialcomm.py

Figure S7: partial-communication results.
  A. Belief dynamics: individual + mean traces for kappa=0.7; mean for kappa=1.0
  B. Mean committed agents over time for kappa = 0.5, 0.7, 0.9, 1.0
  C. Normalized return vs kappa
  D. Accuracy vs kappa

kappa (kappa) is the *information-sharing probability*: the probability that
each agent observes each returning forager's outcome on a given round. kappa=1
is perfect sharing (every agent sees every committed agent's result — the
paper's headline assumption); kappa<1 means each outcome is seen by only a
random fraction of the group. See figS7b for the scout-count-vs-kappa test of
the paper's prediction that imperfect sharing needs a larger scout cohort.

Mirrors `figS7_partialcomm.m`.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from figures._common import OUTPUT, OKABE_ITO, savefig
from ftns import (
    NumPara,
    Para,
    bellman_rhs_component,
    bellman_sol,
    load_mat,
    policy2dist,
    update_partialcomm,
)
from ftns.config import EPSILON, GAMMA_MINUS, GAMMA_PLUS, GC, LAMBDA_COST


def _simulate_belief_dynamics(theta, kappa, para, S, seed=0):
    rng = np.random.default_rng(seed)
    N = int(para.N)
    T = len(S)
    g = 0.1 * np.ones(N)
    G = np.zeros((N, T + 1))
    G[:, 0] = g
    a_history = np.zeros(T, dtype=int)
    for t in range(T):
        gamma = para.gamma_p if S[t] == 1 else para.gamma_m
        C = np.where(g > theta)[0]
        a = len(C)
        a_history[t] = a
        z_vec = -np.ones(N, dtype=int)
        if a > 0:
            z_obs = (rng.random(a) < gamma).astype(int)
            z_vec[C] = z_obs
            x = int(z_obs.sum())
        else:
            x = 0
        g_new = np.empty(N)
        for i in range(N):
            zi = int(z_vec[i])
            if zi == -1:
                if a == 0:
                    m_i, y_i = 0, 0
                else:
                    m_i = int(rng.binomial(a, kappa))
                    y_i = int(rng.hypergeometric(x, a - x, m_i)) if m_i > 0 else 0
            elif zi == 1:
                if a == 1:
                    m_i, y_i = 0, 0
                else:
                    m_i = int(rng.binomial(a - 1, kappa))
                    pop = a - 1
                    ngood = x - 1
                    y_i = int(rng.hypergeometric(ngood, pop - ngood, m_i)) if m_i > 0 else 0
            else:
                if a == 1:
                    m_i, y_i = 0, 0
                else:
                    m_i = int(rng.binomial(a - 1, kappa))
                    pop = a - 1
                    ngood = x
                    y_i = int(rng.hypergeometric(ngood, pop - ngood, m_i)) if m_i > 0 else 0
            g_new[i] = update_partialcomm(zi, y_i, m_i, a, g[i], para)
        g = g_new
        G[:, t + 1] = g
    return G, a_history


def main():
    para = Para()
    para.N = 100
    para.epsilon = EPSILON
    para.gamma_p = GAMMA_PLUS
    para.gamma_m = GAMMA_MINUS
    para.lambda1 = LAMBDA_COST
    para.lambda0 = 0.0
    num_para = NumPara()
    num_para.gN = 201

    N = int(para.N)
    S = np.concatenate([-np.ones(10), np.ones(10), -np.ones(10)]).astype(int)
    T = len(S)

    comp = bellman_rhs_component(para, num_para)
    policy_opt, _ = bellman_sol(comp, num_para)
    theta = policy2dist(policy_opt, np.linspace(0, 1, num_para.gN), num_para).astype(float)

    kappa_all = [0.5, 0.7, 0.9, 1.0]
    G_all, a_all = {}, {}
    for ci, kap in enumerate(kappa_all):
        para.kappa = kap
        G, a = _simulate_belief_dynamics(theta, kap, para, S, seed=ci)
        G_all[kap] = G
        a_all[kap] = a

    perf = load_mat(os.path.join(OUTPUT, "fig_partialcomm_performance.mat"))
    rho_mc_all = np.array(perf["rho_mc_all"])
    alpha_mc_all = np.array(perf["alpha_mc_all"])
    kappa_vals = np.array(perf["kappa_vals"]).ravel()
    MC = int(perf["MC"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Panel A: belief
    axA = axes[0, 0]
    col_lo = OKABE_ITO["green"]
    col_hi = OKABE_ITO["black"]
    axA.plot(np.arange(T + 1), G_all[1.0].mean(axis=0), "-",
             color=col_hi, lw=4, label=r"$\kappa = 1.0$")
    for row in G_all[0.7]:
        axA.plot(np.arange(T + 1), row, color=col_lo, lw=0.5, alpha=0.15)
    axA.plot(np.arange(T + 1), G_all[0.7].mean(axis=0), "--",
             color=col_lo, lw=4, label=r"$\kappa = 0.7$, mean")
    axA.axvline(10, color="k", ls="--", lw=1)
    axA.axvline(20, color="k", ls="--", lw=1)
    axA.set_xlim(0, T)
    axA.set_ylim(0, 1)
    axA.set_xlabel("time, t")
    axA.set_ylabel(r"belief, $g_i$")
    axA.legend(loc="upper left")

    # Panel B: committed agents
    axB = axes[0, 1]
    col_map = {0.5: OKABE_ITO["purple"], 0.7: OKABE_ITO["green"],
               0.9: OKABE_ITO["vermillion"], 1.0: OKABE_ITO["black"]}
    ls_map = {1.0: "-", 0.9: "--", 0.7: "-.", 0.5: ":"}
    for kap in [1.0, 0.9, 0.7, 0.5]:
        axB.plot(np.arange(1, T + 1), a_all[kap], ls_map[kap],
                 color=col_map[kap], lw=4, label=rf"$\kappa = {kap}$")
    axB.axvline(10, color="k", ls="--", lw=1)
    axB.axvline(20, color="k", ls="--", lw=1)
    axB.set_xlim(1, T)
    axB.set_ylim(0, N)
    axB.set_xlabel("time, t")
    axB.set_ylabel("committed agents, a")
    axB.legend(loc="upper right")

    # Panel C: rho/rho_max vs kappa
    axC = axes[1, 0]
    n_kappa = len(kappa_vals)
    alpha_pt = min(1.0, 50 / max(MC, 1))
    size_pt = max(5, 1000 / np.sqrt(max(MC, 1)))
    for idx in range(n_kappa):
        x_jit = (idx + 2) + 0.05 * np.random.default_rng(idx).standard_normal(MC)
        axC.scatter(x_jit, rho_mc_all[:, idx], s=size_pt,
                    color=OKABE_ITO["sky"], alpha=alpha_pt)
        axC.errorbar(idx + 2, rho_mc_all[:, idx].mean(),
                     yerr=rho_mc_all[:, idx].std(), fmt="ko",
                     markersize=6, capsize=5)
    axC.plot(np.arange(2, n_kappa + 2), rho_mc_all.mean(axis=0),
             "k-", lw=2)
    axC.set_xticks(np.arange(2, n_kappa + 2))
    axC.set_xticklabels([str(k) for k in kappa_vals])
    axC.set_xlabel(r"$\kappa$")
    axC.set_ylabel(r"$\rho / \rho_{max}$")

    # Panel D: accuracy vs kappa
    axD = axes[1, 1]
    for idx in range(n_kappa):
        x_jit = (idx + 2) + 0.05 * np.random.default_rng(idx + 50).standard_normal(MC)
        axD.scatter(x_jit, alpha_mc_all[:, idx], s=size_pt,
                    color=OKABE_ITO["sky"], alpha=alpha_pt)
        axD.errorbar(idx + 2, alpha_mc_all[:, idx].mean(),
                     yerr=alpha_mc_all[:, idx].std(), fmt="ko",
                     markersize=6, capsize=5)
    axD.plot(np.arange(2, n_kappa + 2), alpha_mc_all.mean(axis=0),
             "k-", lw=2)
    axD.set_xticks(np.arange(2, n_kappa + 2))
    axD.set_xticklabels([str(k) for k in kappa_vals])
    axD.set_xlabel(r"$\kappa$")
    axD.set_ylabel(r"$\alpha$")

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.text(0.5, 0.01,
             r"$\kappa$ = probability each agent observes each returning "
             r"forager's outcome ($\kappa=1$: perfect sharing).",
             ha="center", va="bottom", fontsize=11)
    savefig(fig, "figS7_partialcomm")


if __name__ == "__main__":
    main()
