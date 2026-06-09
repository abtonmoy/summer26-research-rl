"""fig2_normative_comp.py

Figure 2:
  A. Threshold distribution at N = 100 + optimal policy
  C. Sublinear scaling of scout count with N
  D. Commitment dynamics under a prescribed state sequence

Mirrors `fig2_normative_comp.m`.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from figures._common import OUTPUT, savefig
from ftns import (
    NumPara,
    Para,
    behav_dyn,
    bellman_rhs_component,
    bellman_sol,
    load_mat,
)


def main():
    para_shared = load_mat(os.path.join(OUTPUT, "fig_normative_comp_para.mat"))
    nswipe = load_mat(os.path.join(OUTPUT, "fig_normative_comp_Nswipe.mat"))

    gc = float(para_shared["gc"])
    N_cases = np.array(nswipe["N_cases"]).astype(int).tolist()
    threshold_set = nswipe["threshold_set"]
    policy_set = nswipe["policy_set"]
    scout_set = np.array(nswipe["scout_set"]).astype(int)
    gD = np.array(nswipe["gD"])

    para = Para(**nswipe["para"])
    num_para = NumPara(**nswipe["num_para"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel A
    N_ref = 100
    ki = N_cases.index(N_ref)
    theta_dist = np.array(threshold_set[ki])
    policy_ref = np.array(policy_set[ki])

    axA = axes[0]
    axA.hist(theta_dist, bins=np.arange(0, 1.05, 0.05), color="C0", alpha=0.7)
    axA.set_xlim(0, 1)
    axA.set_ylim(0, N_ref)
    axA.set_xlabel(r"Threshold $\theta$", fontsize=14)
    axA.set_ylabel("Agents", fontsize=14)
    axA2 = axA.twinx()
    axA2.plot(gD, policy_ref, "k-", lw=2)
    axA2.set_ylim(0, N_ref)
    axA2.set_ylabel(r"Committed agents $n^*(g)$", fontsize=14)
    axA.set_title("A: Threshold distribution (N=100)", fontsize=12)

    # Panel C
    axC = axes[1]
    axC.plot(N_cases, scout_set, "-o", lw=2, label=r"Scouts ($\theta < g_c$)")
    axC.plot(N_cases, N_cases, "--", lw=2, label="N")
    axC.set_xscale("log")
    axC.set_xlabel("Group size N", fontsize=14)
    axC.set_ylabel("Scout count", fontsize=14)
    axC.set_xlim(1, 1000)
    axC.set_ylim(0, 50)
    axC.legend(loc="upper left")
    axC.set_title("C: Scout count vs N", fontsize=12)

    # Panel D: Commitment dynamics under prescribed sequence
    para.N = 100
    comp = bellman_rhs_component(para, num_para)
    policy_opt, _ = bellman_sol(comp, num_para)
    st = np.zeros(31, dtype=int)
    st[11:21] = 1
    traj = behav_dyn(policy_opt, st, para, num_para,
                     rng=np.random.default_rng(0))

    axD = axes[2]
    t_axis = np.arange(31)
    gt_with_init = np.concatenate([[0.5], traj["gt"][:-1]])
    axD.plot(t_axis, gt_with_init, "-", lw=2)
    axD.scatter(t_axis, gt_with_init, s=40)
    axD.axhline(gc, color="k", ls="--", lw=1)
    axD.axvline(10, color="k", ls="--", lw=1)
    axD.axvline(20, color="k", ls="--", lw=1)
    axD.set_xlabel("Time t", fontsize=14)
    axD.set_ylabel("Belief g", fontsize=14)
    axD.set_ylim(0, 1)
    axD2 = axD.twinx()
    at_with_init = np.concatenate([[0], traj["at"][:-1]])
    axD2.scatter(t_axis, at_with_init, s=40, color="C1")
    axD2.set_ylim(-10, 110)
    axD2.set_ylabel("Committed agents (a)", fontsize=14)
    axD.set_title("D: Commitment dynamics", fontsize=12)

    fig.tight_layout()
    savefig(fig, "fig2_normative_comp")


if __name__ == "__main__":
    main()
