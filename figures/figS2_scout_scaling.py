"""figS2_scout_scaling.py — Mirrors `figS2_scout_scaling.m`."""

import os

import numpy as np
import matplotlib.pyplot as plt

from figures._common import OUTPUT, OKABE_ITO, savefig
from ftns import load_mat


def main():
    d = load_mat(os.path.join(OUTPUT, "fig_scout_scaling.mat"))
    N_cases = np.array(d["N_cases"])

    OI = [OKABE_ITO["orange"], OKABE_ITO["green"], OKABE_ITO["blue"], OKABE_ITO["vermillion"]]
    ls_map = ["-", "--", "-", ":"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel A: lambda
    sl = np.array(d["scout_lambda"])
    for m, lam in enumerate(np.array(d["lambda_cases"])):
        axes[0].plot(N_cases, sl[:, m], ls_map[m], color=OI[m], lw=3,
                     label=rf"$\lambda$ = {lam:.1f}")
    axes[0].plot(N_cases, N_cases, "--k", lw=2)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Group size N")
    axes[0].set_ylabel("Scout count")
    axes[0].set_xlim(1, 1000)
    axes[0].set_ylim(0, 50)
    axes[0].legend(loc="upper left")

    # Panel B: delta_gamma
    sd = np.array(d["scout_dgamma"])
    for m, dg in enumerate(np.array(d["dgamma_cases"])):
        axes[1].plot(N_cases, sd[:, m], ls_map[m], color=OI[m], lw=3,
                     label=rf"$\Delta\gamma$ = {dg:.1f}")
    axes[1].plot(N_cases, N_cases, "--k", lw=2)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Group size N")
    axes[1].set_ylabel("Scout count")
    axes[1].set_xlim(1, 1000)
    axes[1].set_ylim(0, 100)
    axes[1].legend(loc="upper left")

    # Panel C: epsilon
    se = np.array(d["scout_epsilon"])
    for m, eps in enumerate(np.array(d["epsilon_cases"])):
        axes[2].plot(N_cases, se[:, m], ls_map[m], color=OI[m], lw=3,
                     label=rf"$\epsilon$ = {eps:.2f}")
    axes[2].plot(N_cases, N_cases, "--k", lw=2)
    axes[2].set_xscale("log")
    axes[2].set_xlabel("Group size N")
    axes[2].set_ylabel("Scout count")
    axes[2].set_xlim(1, 1000)
    axes[2].set_ylim(0, 50)
    axes[2].legend(loc="upper left")

    fig.tight_layout()
    savefig(fig, "figS2_scout_scaling")


if __name__ == "__main__":
    main()
