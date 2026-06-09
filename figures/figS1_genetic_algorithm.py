"""figS1_genetic_algorithm.py

Figure S1: GA validation.
  1. Policy comparison for N = 100 (optimal vs best-GA vs avg-GA)
  2. Convergence trajectory for all N
  3. Scouts vs group size

Mirrors `figS1_genetic_algorithm.m`.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from figures._common import OUTPUT, OKABE_ITO, savefig
from ftns import NumPara, dist2policy, load_mat, policy2dist
from ftns.config import SCOUT_CUTOFF, THETA_C


def main():
    N_cases_default = [10, 25, 50, 100, 200]
    # Use whichever are available on disk
    N_cases = []
    data = []
    for N in N_cases_default:
        path = os.path.join(OUTPUT, f"fig_genetic_algorithm_{N}.mat")
        if os.path.exists(path):
            N_cases.append(N)
            data.append(load_mat(path))
    if not data:
        raise FileNotFoundError(
            "No fig_genetic_algorithm_*.mat files found. Run simS1 first.")

    gD = np.array(data[0]["gD"])
    num_para = NumPara(**data[0]["num_para"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: policies for N closest to 100
    target = 100
    idx = min(range(len(N_cases)), key=lambda i: abs(N_cases[i] - target))
    d = data[idx]
    best_thresh = np.array(d["best_genes"])[-1, :]
    ga_best_policy = dist2policy(best_thresh, num_para)
    ga_avg_policy = np.array(d["avg_policy"])

    axes[0].plot(gD, np.array(d["policy_opt"]), "-", lw=4,
                 color=OKABE_ITO["black"], label="Optimal")
    axes[0].plot(gD, ga_best_policy, "-", lw=2,
                 color=OKABE_ITO["blue"], label="Best, GA")
    axes[0].plot(gD, ga_avg_policy, "--", lw=2,
                 color=OKABE_ITO["sky"], label="Average, GA")
    axes[0].set_xlabel("belief (g)")
    axes[0].set_ylabel("committed individuals")
    axes[0].legend(loc="upper left")
    axes[0].set_title(f"N = {N_cases[idx]}")

    # Panel 2: convergence
    conv_colors = [OKABE_ITO["orange"], OKABE_ITO["sky"], OKABE_ITO["green"],
                   OKABE_ITO["blue"], OKABE_ITO["vermillion"]]
    for k, d in enumerate(data):
        ct = np.array(d["conv_traj"])
        ct = ct[ct > 0]  # mask uninitialized tail
        axes[1].plot(np.arange(1, len(ct) + 1), ct,
                     color=conv_colors[k % len(conv_colors)], lw=3,
                     label=f"N={N_cases[k]}")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("generation")
    axes[1].set_ylabel("relative error")
    axes[1].legend(loc="upper right")

    # Panel 3: scout count vs N across the full N = 10^1..10^3 range.
    # The optimal scaling comes from sim2's normative sweep (which spans the
    # whole range cheaply); GA best-evolved counts are overlaid wherever a GA
    # run is available. A c*ln(N) reference makes the logarithmic scaling
    # visually verifiable.
    nsw = load_mat(os.path.join(OUTPUT, "fig_normative_comp_Nswipe.mat"))
    N_opt = np.array(nsw["N_cases"]).astype(int)
    gD_opt = np.array(nsw["gD"])
    # optimal scout count = # thresholds below theta_c, recomputed from the
    # saved policies so the cutoff is explicit and matches the baseline.
    opt_count_full = []
    for k in range(len(N_opt)):
        th = policy2dist(np.array(nsw["policy_set"][k]), gD_opt, num_para)
        opt_count_full.append(int(np.sum(th < THETA_C)))
    opt_count_full = np.array(opt_count_full)
    rng_mask = (N_opt >= 10) & (N_opt <= 1000)
    N_plot = N_opt[rng_mask]
    opt_plot = opt_count_full[rng_mask]

    # GA best-evolved scout count (theta < 1/2) at every available GA N.
    ga_count = [int(np.sum(np.array(d["best_genes"])[-1, :] < SCOUT_CUTOFF))
                for d in data]

    # c*ln(N) reference: least-squares fit through the origin to the optimal
    # counts over the plotted range.
    lnN = np.log(N_plot.astype(float))
    c = float(np.sum(opt_plot * lnN) / np.sum(lnN ** 2))
    N_ref = np.logspace(1, 3, 100)
    axes[2].plot(N_ref, c * np.log(N_ref), ":", lw=2,
                 color=OKABE_ITO["gray"], label=rf"$c\,\ln N$ (c={c:.2f})")
    axes[2].plot(N_plot, opt_plot, "o-", lw=2,
                 color=OKABE_ITO["black"], label=r"Optimal ($\theta < \theta_c$)")
    axes[2].plot(N_cases, ga_count, "s--", lw=2,
                 color=OKABE_ITO["blue"], label=r"Best GA ($\theta < 1/2$)")
    axes[2].set_xscale("log")
    axes[2].set_xlim(10, 1000)
    axes[2].set_xlabel("N")
    axes[2].set_ylabel("Individuals")
    axes[2].legend(loc="upper left")
    axes[2].set_title("Scouts vs group size")

    fig.tight_layout()
    savefig(fig, "figS1_genetic_algorithm")


if __name__ == "__main__":
    main()
