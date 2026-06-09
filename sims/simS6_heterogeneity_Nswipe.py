"""simS6_heterogeneity_Nswipe.py

Heterogeneity heatmaps over lambda x delta_gamma for N = 50 and N = 200.

Writes:
  output/fig_heterogeneity_heatmap_N50.mat
  output/fig_heterogeneity_heatmap_N200.mat

Mirrors `simS6_heterogeneity_Nswipe.m`.
"""

import os
import time

import numpy as np

from sims._common import OUTPUT, parse_args
from ftns import (
    NumPara,
    Para,
    bellman_rhs_component,
    bellman_sol,
    hetero,
    load_mat,
    save_mat,
)


def is_valid(lmd, dg, eps):
    return (lmd >= (eps - 0.5) * dg + 0.5) and (lmd <= (0.5 - eps) * dg + 0.5)


def main():
    args = parse_args()
    shared = load_mat(os.path.join(OUTPUT, "fig_normative_comp_para.mat"))
    para = Para(**shared["para"])
    num_para = NumPara(**shared["num_para"])
    para.lambda0 = 0.0
    num_para.gN = 201
    gD = np.linspace(0, 1, num_para.gN)

    M = 11 if args.quick else 51
    lambda_vec = np.linspace(0, 1, M)
    dg_vec = np.linspace(0, 1, M)
    eps_fix = 0.1

    N_cases = [50, 200]
    for N in N_cases:
        para.N = N
        para.epsilon = eps_fix
        print(f"\n=== N = {N} ===")

        X, Y = np.meshgrid(dg_vec, lambda_vec)
        Z = np.zeros_like(X)

        t0 = time.time()
        for i in range(M):
            dg = float(dg_vec[i])
            para.gamma_p = 0.5 * (1 + dg)
            para.gamma_m = 0.5 * (1 - dg)
            for j in range(M):
                lmd = float(lambda_vec[j])
                para.lambda1 = lmd
                if is_valid(lmd, dg, eps_fix):
                    comp = bellman_rhs_component(para, num_para, n_jobs=args.n_jobs)
                    policy_opt, _ = bellman_sol(comp, num_para)
                    Z[j, i] = hetero(policy_opt, (0, 1), gD)
            print(f"  N={N} | dg={dg:.2f} | col {i+1}/{M} | elapsed {time.time()-t0:.0f}s")

        fname = os.path.join(OUTPUT, f"fig_heterogeneity_heatmap_N{N}.mat")
        save_mat(
            fname,
            {
                "X": X,
                "Y": Y,
                "Z": Z,
                "dg_vec": dg_vec,
                "lambda_vec": lambda_vec,
                "eps_fix": eps_fix,
                "para": dict(para),
                "num_para": dict(num_para),
            },
        )
        print(f"Saved: {fname}")


if __name__ == "__main__":
    main()
