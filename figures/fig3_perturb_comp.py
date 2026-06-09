"""fig3_perturb_comp.py

Figure 3: threshold histograms for four policies + accuracy-vs-return scatter.
Mirrors `fig3_perturb_comp.m`.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from figures._common import OUTPUT, OKABE_ITO, savefig
from ftns import NumPara, Para, load_mat, policy2dist


def main():
    para_data = load_mat(os.path.join(OUTPUT, "fig_perturb_comp_para.mat"))
    perf = load_mat(os.path.join(OUTPUT, "fig_perturb_comp_perform.mat"))
    interp = load_mat(os.path.join(OUTPUT, "fig_perturb_comp_interp.mat"))

    policy_set = para_data["policy_set"]
    gD = np.array(para_data["gD"])
    para = Para(**para_data["para"])
    num_para = NumPara(**para_data["num_para"])

    OI = [OKABE_ITO["black"], OKABE_ITO["blue"], OKABE_ITO["green"], OKABE_ITO["vermillion"]]
    # Paper convention: name perturbations by *risk tolerance*, the opposite of
    # threshold height. The series with explorer mass near theta~0 (was
    # "Low-threshold") is the bold, "High risk-tolerant" cohort; the one with
    # mass pushed up near theta_c (was "High-threshold") is "Low risk-tolerant".
    labels = ["Optimal", "Desync", "High risk-tolerant", "Low risk-tolerant"]

    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 4)

    # Threshold histograms
    panel_pos = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for k in range(4):
        ax = fig.add_subplot(gs[panel_pos[k]])
        theta = policy2dist(np.array(policy_set[k]), gD, num_para)
        ax.hist(theta, bins=np.arange(0, 1.05, 0.05), color=OI[k])
        ax.set_xlim(0, 1)
        ax.set_yscale("log")
        ax.set_ylim(0.5, int(para.N))
        ax.set_xlabel(r"Threshold $\theta$", fontsize=12)
        ax.set_ylabel("Agents", fontsize=12)
        ax.set_title(labels[k], fontsize=12)

    # Accuracy vs return
    ax = fig.add_subplot(gs[:, 2:])
    rho_set = np.array(perf["rho_set"])
    acc_set = np.array(perf["acc_set"])
    rho_max = float(perf["rho_max"])
    rho_interp = np.array(interp["rho_interp"])
    acc_interp = np.array(interp["acc_interp"])

    # "Collapse" reference: if every threshold sits above 1/2 the collective
    # never explores and return drops to 0. Dashed diagonal from the origin
    # toward the optimum makes this trap explicit (the "Low risk-tolerant"
    # curve terminates at the origin).
    acc_opt = float(acc_set[0])
    rho_opt_norm = float(rho_set[0] / rho_max)
    ax.plot([0.0, acc_opt], [0.0, rho_opt_norm], "--", color="0.5", lw=2,
            zorder=0, label=r"collapse if all thresholds $> 1/2$")

    for k in range(3):
        ax.plot(acc_interp[:, k], rho_interp[:, k] / rho_max,
                "-", color=OI[k + 1], lw=4)
    for k in range(4):
        ax.scatter(acc_set[k], rho_set[k] / rho_max, s=200,
                   color=OI[k], label=labels[k])
    ax.set_xlabel(r"Accuracy $\alpha$", fontsize=14)
    ax.set_ylabel(r"$\rho / \rho_{max}$", fontsize=14)
    ax.legend(loc="lower right")

    fig.tight_layout()
    savefig(fig, "fig3_perturb_comp")


if __name__ == "__main__":
    main()
