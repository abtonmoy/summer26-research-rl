"""figS4_response_time.py — Mirrors `figS4_response_time.m`."""

import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from figures._common import OUTPUT, savefig
from ftns import load_mat


def _smooth(Z, sigma=2):
    M = np.where(np.isnan(Z), 0, Z)
    W = np.where(np.isnan(Z), 0, 1.0)
    Mf = gaussian_filter(M, sigma=sigma)
    Wf = gaussian_filter(W, sigma=sigma)
    out = np.where(Wf > 1e-6, Mf / Wf, np.nan)
    return out


def _process(Z, Lam, beta_min=-1, beta_max=3):
    Z = Z.copy()
    Z[np.isnan(Z) & (Lam < 0.5)] = beta_min
    Z[np.isnan(Z) & (Lam > 0.5)] = beta_max
    Z[Z > beta_max] = beta_max
    Z[Z < beta_min] = beta_min
    return Z


def main():
    beta_min, beta_max = -1, 3
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: lambda x delta_gamma
    d = load_mat(os.path.join(OUTPUT, "fig_response_time_ldg.mat"))
    tau_mp = np.array(d["tau_mp_ldg"])
    tau_pm = np.array(d["tau_pm_ldg"])
    Lambda = np.array(d["Lambda_ldg"])
    Dgamma = np.array(d["Dgamma"])

    with np.errstate(divide="ignore", invalid="ignore"):
        Z = np.log(tau_mp / tau_pm)
    Z = _process(Z, Lambda, beta_min, beta_max)
    Z_smooth = _smooth(Z)
    pc = axes[0].pcolormesh(Lambda, Dgamma, Z, shading="auto",
                            vmin=beta_min, vmax=beta_max, cmap="RdBu_r")
    axes[0].contour(Lambda, Dgamma, Z_smooth, levels=[0],
                    colors="gray", linestyles=":")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel(r"Cost $\lambda$", fontsize=14)
    axes[0].set_ylabel(r"Reward diff. $\Delta\gamma$", fontsize=14)
    fig.colorbar(pc, ax=axes[0])

    # Panel 2: lambda x epsilon
    d = load_mat(os.path.join(OUTPUT, "fig_response_time_leps.mat"))
    tau_mp = np.array(d["tau_mp_leps"])
    tau_pm = np.array(d["tau_pm_leps"])
    Lambda = np.array(d["Lambda_leps"])
    Epsilon = np.array(d["Epsilon"])
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = np.log(tau_mp / tau_pm)
    Z = _process(Z, Lambda, beta_min, beta_max)
    Z_smooth = _smooth(Z)
    pc = axes[1].pcolormesh(Lambda, Epsilon, Z, shading="auto",
                            vmin=beta_min, vmax=beta_max, cmap="RdBu_r")
    axes[1].contour(Lambda, Epsilon, Z_smooth, levels=[0],
                    colors="gray", linestyles=":")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 0.5)
    axes[1].set_xlabel(r"Cost $\lambda$", fontsize=14)
    axes[1].set_ylabel(r"Switch rate $\epsilon$", fontsize=14)
    cb = fig.colorbar(pc, ax=axes[1])
    cb.set_label(r"$\log(\tau_{-,+} / \tau_{+,-})$", fontsize=14)

    fig.tight_layout()
    savefig(fig, "figS4_response_time")


if __name__ == "__main__":
    main()
