"""sim4_heterogeneity.py

Sweeps over lambda, delta_gamma, epsilon (1-D) and lambda x delta_gamma (2-D)
to compute heterogeneity of the normative threshold distribution.

Writes:
  output/fig_heterogeneity_para.mat
  output/fig_heterogeneity_1d.mat
  output/fig_heterogeneity_heatmap.mat

Mirrors `sim4_heterogeneity.m`.
"""

import os
import time

import numpy as np

from sims._common import OUTPUT, parse_args
from ftns import (
    bellman_rhs_component,
    bellman_sol,
    hetero,
    load_mat,
    save_mat,
    Para,
    NumPara,
)


def is_valid(lmd, dg, eps):
    return (lmd >= (eps - 0.5) * dg + 0.5) and (lmd <= (0.5 - eps) * dg + 0.5)


def main():
    args = parse_args()

    shared = load_mat(os.path.join(OUTPUT, "fig_normative_comp_para.mat"))
    para = Para(**shared["para"])
    num_para = NumPara(**shared["num_para"])

    para.N = 100
    num_para.gN = 201
    gD = np.linspace(0, 1, num_para.gN)
    M = 11 if args.quick else 51

    dg_fix = 0.6
    eps_fix = 0.1
    lam_fix = [0.4, 0.6]

    # Panel A: sweep lambda
    print("Panel A: sweeping lambda ...")
    lambda_vec = np.linspace(0, 1, M)
    h_lambda = np.zeros(M)
    para.gamma_p = 0.5 * (1 + dg_fix)
    para.gamma_m = 0.5 * (1 - dg_fix)
    para.epsilon = eps_fix
    for i in range(M):
        para.lambda1 = float(lambda_vec[i])
        if is_valid(para.lambda1, dg_fix, eps_fix):
            comp = bellman_rhs_component(para, num_para, n_jobs=args.n_jobs)
            policy_opt, _ = bellman_sol(comp, num_para)
            h_lambda[i] = hetero(policy_opt, (0, 1), gD)
        print(f"  lambda = {para.lambda1:.2f} | {i+1}/{M} | h = {h_lambda[i]:.4f}")

    # Panel B: sweep delta_gamma at two lambda values
    print("Panel B: sweeping delta_gamma ...")
    dg_vec = np.linspace(0, 1, M)
    h_dg = np.zeros((M, len(lam_fix)))
    para.epsilon = eps_fix
    for ki, lam in enumerate(lam_fix):
        para.lambda1 = lam
        for i in range(M):
            dg = float(dg_vec[i])
            para.gamma_p = 0.5 * (1 + dg)
            para.gamma_m = 0.5 * (1 - dg)
            if is_valid(lam, dg, eps_fix):
                comp = bellman_rhs_component(para, num_para, n_jobs=args.n_jobs)
                policy_opt, _ = bellman_sol(comp, num_para)
                h_dg[i, ki] = hetero(policy_opt, (0, 1), gD)
            print(f"  lambda = {lam:.1f} | dg = {dg:.2f} | h = {h_dg[i, ki]:.4f}")

    # Panel C: sweep epsilon at two lambda values
    print("Panel C: sweeping epsilon ...")
    eps_vec_full = np.linspace(0, 0.5, M)
    eps_vec = eps_vec_full[1:]  # exclude 0
    h_eps = np.zeros((len(eps_vec), len(lam_fix)))
    para.gamma_p = 0.5 * (1 + dg_fix)
    para.gamma_m = 0.5 * (1 - dg_fix)
    for ki, lam in enumerate(lam_fix):
        para.lambda1 = lam
        for i in range(len(eps_vec)):
            para.epsilon = float(eps_vec[i])
            if is_valid(lam, dg_fix, eps_vec[i]):
                comp = bellman_rhs_component(para, num_para, n_jobs=args.n_jobs)
                policy_opt, _ = bellman_sol(comp, num_para)
                h_eps[i, ki] = hetero(policy_opt, (0, 1), gD)
            print(f"  lambda = {lam:.1f} | eps = {eps_vec[i]:.2f} | h = {h_eps[i, ki]:.4f}")

    save_mat(
        os.path.join(OUTPUT, "fig_heterogeneity_1d.mat"),
        {
            "lambda_vec": lambda_vec,
            "h_lambda": h_lambda,
            "dg_vec": dg_vec,
            "h_dg": h_dg,
            "eps_vec": eps_vec,
            "h_eps": h_eps,
            "dg_fix": dg_fix,
            "eps_fix": eps_fix,
            "lam_fix": np.array(lam_fix),
        },
    )
    print(f"Saved: {OUTPUT}/fig_heterogeneity_1d.mat")

    # Panel D: heatmap
    print("Panel D: heatmap ...")
    X, Y = np.meshgrid(dg_vec, lambda_vec)
    Z = np.zeros_like(X)
    para.epsilon = eps_fix
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
        print(f"  dg = {dg:.2f} | col {i+1}/{M} | elapsed {time.time()-t0:.0f}s")

    save_mat(
        os.path.join(OUTPUT, "fig_heterogeneity_heatmap.mat"),
        {
            "X": X,
            "Y": Y,
            "Z": Z,
            "dg_vec": dg_vec,
            "lambda_vec": lambda_vec,
            "eps_fix": eps_fix,
        },
    )
    print(f"Saved: {OUTPUT}/fig_heterogeneity_heatmap.mat")

    save_mat(
        os.path.join(OUTPUT, "fig_heterogeneity_para.mat"),
        {
            "para": dict(para),
            "num_para": dict(num_para),
            "gD": gD,
            "M": M,
            "dg_fix": dg_fix,
            "eps_fix": eps_fix,
            "lam_fix": np.array(lam_fix),
        },
    )
    print(f"Saved: {OUTPUT}/fig_heterogeneity_para.mat")


if __name__ == "__main__":
    main()
