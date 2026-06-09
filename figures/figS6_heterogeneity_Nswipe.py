"""figS6_heterogeneity_Nswipe.py — Mirrors `figS6_heterogeneity_Nswipe.m`."""

import os

import numpy as np
import matplotlib.pyplot as plt

from figures._common import OUTPUT, savefig
from ftns import load_mat


def main():
    threshold = 0.5
    cmin, cmax = 0, 0.16

    files = [
        ("fig_heterogeneity_heatmap_N50.mat", 50),
        ("fig_heterogeneity_heatmap.mat", 100),
        ("fig_heterogeneity_heatmap_N200.mat", 200),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    pc = None
    for p, (fname, N) in enumerate(files):
        path = os.path.join(OUTPUT, fname)
        if not os.path.exists(path):
            axes[p].set_title(f"(missing) {fname}")
            continue
        d = load_mat(path)
        Z = np.array(d["Z"])
        Z = np.where(Z > threshold, 0, Z)
        # Y = lambda, X = delta_gamma. plot Y on x-axis, X on y-axis (matches MATLAB).
        Y = np.array(d["Y"])
        X = np.array(d["X"])
        pc = axes[p].pcolormesh(Y, X, Z, shading="auto",
                                vmin=cmin, vmax=cmax, cmap="viridis")
        axes[p].set_xlabel(r"$\lambda$", fontsize=14)
        axes[p].set_ylabel(r"$\Delta\gamma$", fontsize=14)
        axes[p].set_title(f"N = {N}", fontsize=14)
    if pc is not None:
        fig.colorbar(pc, ax=axes[-1], label=r"$\sigma$")
    fig.tight_layout()
    savefig(fig, "figS6_heterogeneity_Nswipe")


if __name__ == "__main__":
    main()
