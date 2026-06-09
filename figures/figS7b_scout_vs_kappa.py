"""figS7b_scout_vs_kappa.py

Companion to figS7. Plots the optimal scout count as a function of the
information-sharing probability kappa, testing the paper's prediction that
imperfect sharing (smaller kappa) requires a larger exploratory cohort.

Left panel : s*(kappa) (upper edge of the statistically-tied return plateau)
             with the tied plateau shaded, and the full-observation optimal
             scout count as a horizontal reference. The plateau exists because
             "mild" scouts (0.5 < theta < theta_c) are behaviourally close to
             deliberators, so a range of s yields statistically equal return.
Right panel: the partial-communication return landscape rho(s) per kappa.

Reads:
  output/fig_scout_vs_kappa.mat   (run simS7b_scout_vs_kappa first)
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from figures._common import OUTPUT, OKABE_ITO, savefig
from ftns import load_mat


def main():
    d = load_mat(os.path.join(OUTPUT, "fig_scout_vs_kappa.mat"))
    kappa_vals = np.array(d["kappa_vals"]).ravel().astype(float)
    scout_grid = np.array(d["scout_grid"]).ravel().astype(int)
    rho_grid = np.array(d["rho_grid"])          # (n_kappa, n_s)
    best_scout = np.array(d["best_scout"]).ravel().astype(int)
    opt_full = int(np.array(d["opt_scout_fullcomm"]).ravel()[0])
    s_star_full = int(np.array(d.get("s_star_full", best_scout[-1])).ravel()[0])
    plateau_lo = np.array(d.get("plateau_lo", best_scout)).ravel().astype(int)
    plateau_hi = np.array(d.get("plateau_hi", best_scout)).ravel().astype(int)
    n_bold = int(np.array(d.get("n_bold", 0)).ravel()[0])

    # x ascending 0.5 -> 1.0, consistent with the main figS7 panels.
    order = np.argsort(kappa_vals)
    kappa_vals = kappa_vals[order]
    rho_grid = rho_grid[order, :]
    best_scout = best_scout[order]
    plateau_lo = plateau_lo[order]
    plateau_hi = plateau_hi[order]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: optimal scout count vs kappa, with the tied plateau shaded.
    axL = axes[0]
    axL.fill_between(kappa_vals, plateau_lo, plateau_hi, color=OKABE_ITO["sky"],
                     alpha=0.30, label="return-tied plateau")
    axL.plot(kappa_vals, best_scout, "o-", lw=2.5, color=OKABE_ITO["vermillion"],
             label=r"$s^*(\kappa)$ = plateau upper edge")
    axL.axhline(opt_full, ls="--", lw=2, color=OKABE_ITO["black"],
                label=rf"full-obs optimal ($\theta<\theta_c$) = {opt_full}")
    if n_bold:
        axL.axhline(n_bold, ls=":", lw=1.5, color=OKABE_ITO["gray"],
                    label=rf"functionally bold ($\theta<1/2$) = {n_bold}")
    axL.set_xlabel(r"information-sharing probability $\kappa$", fontsize=13)
    axL.set_ylabel("scout count", fontsize=13)
    axL.set_ylim(bottom=0)
    axL.legend(loc="best", fontsize=9)

    # Data-driven verdict on the optimal BAND (kappa ascending: index 0 = lowest
    # kappa). The optimum is a wide return-tied plateau, so we report how its
    # upper edge moves rather than a noisy single-point trend.
    band_shift = int(plateau_hi[0] - plateau_hi[-1])  # upper edge: low-kappa - high-kappa
    if band_shift > 2:
        verdict = "optimal band extends to MORE scouts as\nsharing degrades (supports prediction)"
    elif band_shift < -2:
        verdict = "optimal band extends to FEWER scouts as\nsharing degrades (opposite to prediction)"
    else:
        verdict = "optimal band ~unchanged across $\\kappa$\n(no robust support for the prediction)"
    contained = plateau_lo[-1] <= opt_full <= plateau_hi[-1]
    gate = (rf"gate: full-obs optimum {opt_full} lies in $\kappa{{=}}1$ "
            rf"plateau [{plateau_lo[-1]},{plateau_hi[-1]}] "
            + (r"$\checkmark$" if contained else r"$\times$"))
    axL.set_title(verdict + "\n" + gate, fontsize=11)

    # Right: return landscape rho(s) per kappa
    axR = axes[1]
    n_kappa = len(kappa_vals)
    base = np.array(OKABE_ITO["sky"])
    for ki in range(n_kappa):
        shade = 0.75 * (ki / max(n_kappa - 1, 1))
        color = base * (1 - shade)  # darker = higher kappa
        axR.plot(scout_grid, rho_grid[ki, :], "-o", lw=2, ms=4, color=color,
                 label=rf"$\kappa$ = {kappa_vals[ki]:.1f}")
        si = int(np.where(scout_grid == best_scout[ki])[0][0])
        axR.scatter([scout_grid[si]], [rho_grid[ki, si]], s=120,
                    facecolors="none", edgecolors="k", linewidths=1.5, zorder=5)
    axR.axvline(opt_full, ls="--", lw=1.5, color=OKABE_ITO["black"], alpha=0.6)
    axR.set_xlabel("scout count $s$", fontsize=13)
    axR.set_ylabel(r"partial-comm. return $\rho$", fontsize=13)
    # Zoom to the plateau so its (flat) shape and the gentle post-23 decline are
    # legible; the s=0 degenerate point (rho=0) sits below the view by design.
    pos = rho_grid[:, scout_grid > 0]
    lo, hi = float(pos.min()), float(pos.max())
    pad = 0.15 * (hi - lo + 1e-9)
    axR.set_ylim(lo - pad, hi + pad)
    axR.legend(loc="lower center", fontsize=9, ncol=2)
    axR.set_title(r"return landscape (flat plateau; circled = $s^*$, dashed = full-obs opt)",
                  fontsize=11)

    fig.tight_layout()
    savefig(fig, "figS7b_scout_vs_kappa")


if __name__ == "__main__":
    main()
