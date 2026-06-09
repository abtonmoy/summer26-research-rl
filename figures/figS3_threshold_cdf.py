"""figS3_threshold_cdf.py — Mirrors `figS3_threshold_cdf.m`."""

import os

import numpy as np
import matplotlib.pyplot as plt

from figures._common import OUTPUT, OKABE_ITO, savefig
from ftns import load_mat
from ftns.config import THETA_C


def main():
    d = load_mat(os.path.join(OUTPUT, "fig_threshold_cdf.mat"))
    N_cases = list(np.array(d["N_cases"]).astype(int))
    policy_set_N = d["policy_set_N"]
    gD = np.array(d["gD"])
    nN = len(N_cases)
    base = np.array(OKABE_ITO["blue"])

    fig, ax = plt.subplots(figsize=(7, 6))
    for k, N in enumerate(N_cases):
        brightness = 0.7 * (1 - k / nN)
        color = base + brightness * (1 - base)
        ax.plot(gD, np.array(policy_set_N[k]) / N, "-", lw=3,
                color=color, label=f"N = {N}")
    # Baseline critical threshold: the CDF should step up at theta_c = 0.70.
    ax.axvline(THETA_C, color="k", ls="--", lw=1.5,
               label=rf"$\theta_c$ = {THETA_C:.2f}")
    ax.set_xlabel("Belief g")
    ax.set_ylabel("CDF")
    ax.legend(loc="upper left")
    fig.tight_layout()
    savefig(fig, "figS3_threshold_cdf")


if __name__ == "__main__":
    main()
