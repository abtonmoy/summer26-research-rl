"""fig4_heterogeneity.py

Figure 4: heterogeneity vs lambda, delta_gamma, epsilon, and 2D heatmap.
Mirrors `fig4_heterogeneity.m`.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from figures._common import OUTPUT, OKABE_ITO, savefig
from ftns import load_mat


def main():
    d1 = load_mat(os.path.join(OUTPUT, "fig_heterogeneity_1d.mat"))
    dh = load_mat(os.path.join(OUTPUT, "fig_heterogeneity_heatmap.mat"))

    lam_fix = np.array(d1["lam_fix"]).ravel()
    OI_lam = [OKABE_ITO["blue"], OKABE_ITO["vermillion"]]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Panel A
    axes[0].plot(d1["lambda_vec"], d1["h_lambda"], "k-", lw=4)
    axes[0].set_xlabel(r"$\lambda$", fontsize=14)
    axes[0].set_ylabel(r"$\sigma$", fontsize=14)
    axes[0].set_title("A")

    # Panel B
    h_dg = np.array(d1["h_dg"])
    for ki in range(h_dg.shape[1]):
        axes[1].plot(d1["dg_vec"], h_dg[:, ki], color=OI_lam[ki], lw=4,
                     label=rf"$\lambda$ = {lam_fix[ki]:.1f}")
    axes[1].set_xlabel(r"$\Delta\gamma$", fontsize=14)
    axes[1].set_ylabel(r"$\sigma$", fontsize=14)
    axes[1].legend(loc="upper left")
    axes[1].set_title("B")

    # Panel C
    h_eps = np.array(d1["h_eps"])
    for ki in range(h_eps.shape[1]):
        axes[2].plot(d1["eps_vec"], h_eps[:, ki], color=OI_lam[ki], lw=4,
                     label=rf"$\lambda$ = {lam_fix[ki]:.1f}")
    axes[2].set_xlabel(r"$\epsilon$", fontsize=14)
    axes[2].set_ylabel(r"$\sigma$", fontsize=14)
    axes[2].legend(loc="upper left")
    axes[2].set_title("C")

    # Panel D: heatmap. Orientation matches the paper and figS6 — lambda on the
    # x-axis, delta_gamma on the y-axis — so the high-heterogeneity wedge points
    # down to its apex near lambda ~ 0.5 at low delta_gamma. X/Y are coordinate
    # meshes (X=delta_gamma, Y=lambda) aligned with Z[lambda_idx, dg_idx], so
    # passing (Y, X, Z) places lambda on x and delta_gamma on y without
    # touching the data.
    X = np.array(dh["X"])  # delta_gamma
    Y = np.array(dh["Y"])  # lambda
    Z = np.array(dh["Z"])  # rows = lambda, cols = delta_gamma
    pc = axes[3].pcolormesh(Y, X, Z, shading="auto", cmap="viridis")
    fig.colorbar(pc, ax=axes[3])
    axes[3].set_xlabel(r"$\lambda$", fontsize=14)
    axes[3].set_ylabel(r"$\Delta\gamma$", fontsize=14)
    axes[3].set_title("D")

    fig.tight_layout()
    savefig(fig, "fig4_heterogeneity")


if __name__ == "__main__":
    main()
