"""simS7_partialcomm_performance.py

Monte-Carlo simulation of foraging under partial communication. Sweeps kappa
and evaluates per-replicate return and accuracy.

Writes:
  output/fig_partialcomm_performance.mat

Mirrors `simS7_partialcomm_performance.m`.
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
    mc_partialcomm_batch as mc_partialcomm,
    policy2dist,
    save_mat,
)
from ftns.config import EPSILON, GAMMA_MINUS, GAMMA_PLUS, LAMBDA_COST, N_DEFAULT


def main():
    args = parse_args()

    para = Para()
    para.N = N_DEFAULT
    para.epsilon = EPSILON
    para.gamma_p = GAMMA_PLUS
    para.gamma_m = GAMMA_MINUS
    para.lambda1 = LAMBDA_COST
    para.lambda0 = 0.0

    num_para = NumPara()
    num_para.gN = 201

    T = 200 if args.quick else 1000
    MC = 50 if args.quick else 1000

    comp = bellman_rhs_component(para, num_para, n_jobs=args.n_jobs)
    policy_opt, _ = bellman_sol(comp, num_para)
    theta = policy2dist(policy_opt, np.linspace(0, 1, num_para.gN), num_para)
    theta = theta.astype(float)

    kappa_vals = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0] if not args.quick else [0.7, 1.0]
    n_kappa = len(kappa_vals)

    rho_mc_all = np.zeros((MC, n_kappa))
    alpha_mc_all = np.zeros((MC, n_kappa))

    for idx, kappa in enumerate(kappa_vals):
        para.kappa = kappa
        t0 = time.time()
        print(f"Running kappa = {kappa}")
        rho_mc_all[:, idx], alpha_mc_all[:, idx] = mc_partialcomm(
            theta, T, MC, para, n_jobs=args.n_jobs, seed=args.seed + idx
        )
        print(f"  kappa={kappa} done | {time.time()-t0:.1f}s")

    rho_max = para.N * (para.gamma_p - para.lambda1) / 2.0
    rho_mc_all = rho_mc_all / rho_max

    save_mat(
        os.path.join(OUTPUT, "fig_partialcomm_performance.mat"),
        {
            "rho_mc_all": rho_mc_all,
            "alpha_mc_all": alpha_mc_all,
            "kappa_vals": np.array(kappa_vals),
            "para": dict(para),
            "T": T,
            "MC": MC,
            "rho_max": rho_max,
        },
    )
    print(f"Saved: {OUTPUT}/fig_partialcomm_performance.mat")


if __name__ == "__main__":
    main()
