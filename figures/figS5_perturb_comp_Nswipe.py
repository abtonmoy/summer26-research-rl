"""figS5_perturb_comp_Nswipe.py — Mirrors `figS5_perturb_comp_Nswipe.m`."""

import os

import numpy as np
import matplotlib.pyplot as plt

from figures._common import OUTPUT, OKABE_ITO, savefig
from ftns import load_mat


def main():
    OI = [OKABE_ITO["black"], OKABE_ITO["blue"], OKABE_ITO["green"], OKABE_ITO["vermillion"]]
    brightness = [0.6, 0.3, 0.0]
    N_labels = ["N = 50", "N = 100", "N = 200"]

    data = []
    # N = 50
    data.append(load_mat(os.path.join(OUTPUT, "fig_perturb_comp_N50.mat")))
    # N = 100: merge perform + interp
    d100 = {}
    d100.update(load_mat(os.path.join(OUTPUT, "fig_perturb_comp_perform.mat")))
    d100.update(load_mat(os.path.join(OUTPUT, "fig_perturb_comp_interp.mat")))
    data.append(d100)
    # N = 200
    data.append(load_mat(os.path.join(OUTPUT, "fig_perturb_comp_N200.mat")))

    fig, ax = plt.subplots(figsize=(7, 6))
    for pi, d in enumerate(data):
        br = brightness[pi]
        rho_max = float(d["rho_max"])
        rho_interp = np.array(d["rho_interp"])
        acc_interp = np.array(d["acc_interp"])
        rho_set = np.array(d["rho_set"])
        acc_set = np.array(d["acc_set"])

        for k in range(3):
            base = np.array(OI[k + 1])
            color = base + br * (1 - base)
            ax.plot(acc_interp[:, k], rho_interp[:, k] / rho_max,
                    "-", color=color, lw=5)
        for k in range(4):
            base = np.array(OI[k])
            color = base + br * (1 - base)
            label = N_labels[pi] if k == 0 else None
            ax.scatter(acc_set[k], rho_set[k] / rho_max, s=200,
                       color=color, label=label)
    ax.set_xlabel(r"Accuracy $\alpha$", fontsize=14)
    ax.set_ylabel(r"$\rho / \rho_{max}$", fontsize=14)
    ax.legend(loc="lower right")
    fig.tight_layout()
    savefig(fig, "figS5_perturb_comp_Nswipe")


if __name__ == "__main__":
    main()
