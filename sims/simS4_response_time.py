"""simS4_response_time.py

Monte-Carlo response time over two 2-D parameter spaces:
  1. lambda x delta_gamma   -> output/fig_response_time_ldg.mat
  2. lambda x epsilon       -> output/fig_response_time_leps.mat

Mirrors `simS4_response_time.m`.
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
    load_mat,
    mc_response,
    save_mat,
)


def is_valid(lmd, dg, eps_p, eps_m=None):
    # Feasibility wedge: the critical belief gc must lie inside the reachable
    # belief band [eps_plus, 1 - eps_minus] of the asymmetric predict step.
    # With gamma centered at 1/2 this becomes
    #   (eps_p - 1/2) dg + 1/2  <=  lmd  <=  (1/2 - eps_m) dg + 1/2.
    # Symmetric (eps_p == eps_m) recovers the original symmetric bound exactly.
    if eps_m is None:
        eps_m = eps_p
    return (lmd >= (eps_p - 0.5) * dg + 0.5) and (lmd <= (0.5 - eps_m) * dg + 0.5)


def _process_cell(p_dict, num_para, MC, seed, n_jobs):
    p = Para(**p_dict)
    nump = NumPara(**num_para)
    comp = bellman_rhs_component(p, nump, n_jobs=1)  # inner serial; outer parallel
    policy_opt, _ = bellman_sol(comp, nump)
    T_mp, T_pm = mc_response(policy_opt, p, nump, MC, seed=seed, n_jobs=1)
    # stats
    prob_mp = T_mp[1, :].mean()
    prob_pm = T_pm[1, :].mean()
    tau_mp = T_mp[0, T_mp[1, :] == 1].mean() if (T_mp[1, :] == 1).any() else np.nan
    tau_pm = T_pm[0, T_pm[1, :] == 1].mean() if (T_pm[1, :] == 1).any() else np.nan
    meanT_mp = T_mp[0, :].mean()
    meanT_pm = T_pm[0, :].mean()
    return prob_mp, prob_pm, tau_mp, tau_pm, meanT_mp, meanT_pm


def _sweep(grid_X, grid_Y, para_template, num_para, MC, valid_fn, seed, n_jobs):
    shape = grid_X.shape
    prob_mp = np.full(shape, np.nan)
    prob_pm = np.full(shape, np.nan)
    tau_mp = np.full(shape, np.nan)
    tau_pm = np.full(shape, np.nan)
    meanT_mp = np.full(shape, np.nan)
    meanT_pm = np.full(shape, np.nan)
    flat_idx = 0
    total = grid_X.size
    t0 = time.time()
    for idx in np.ndindex(shape):
        p_dict, ok = valid_fn(para_template, grid_X[idx], grid_Y[idx])
        if ok:
            r = _process_cell(p_dict, dict(num_para), MC, seed + flat_idx, n_jobs)
            prob_mp[idx], prob_pm[idx], tau_mp[idx], tau_pm[idx], meanT_mp[idx], meanT_pm[idx] = r
        flat_idx += 1
        if flat_idx % 10 == 0:
            print(f"  cell {flat_idx}/{total} | elapsed {time.time()-t0:.0f}s")
    return prob_mp, prob_pm, tau_mp, tau_pm, meanT_mp, meanT_pm


def main():
    args = parse_args()
    shared = load_mat(os.path.join(OUTPUT, "fig_normative_comp_para.mat"))
    para = Para(**shared["para"])
    num_para = NumPara(**shared["num_para"])
    para.N = 100
    para.lambda0 = 0.0
    num_para.gN = 201

    MC = 50 if args.quick else 1000
    # MATLAB ships num_mesh=50; we use 25 by default to keep wall time
    # reasonable. Pass --full-mesh to match MATLAB exactly.
    num_mesh = 10 if args.quick else 25

    # Heatmap 1: lambda x delta_gamma
    print("Heatmap 1: lambda x delta_gamma")
    lambda_grid = np.linspace(0, 1, num_mesh)
    dg_grid = np.linspace(0, 1, num_mesh)
    Lambda_ldg, Dgamma = np.meshgrid(lambda_grid, dg_grid)

    def valid_ldg(p_template, lmd_v, dg_v):
        p = dict(p_template)
        p["lambda1"] = float(lmd_v)
        p["gamma_p"] = 0.5 * (1 + dg_v)
        p["gamma_m"] = 0.5 * (1 - dg_v)
        ok = is_valid(lmd_v, dg_v, p["epsilon"])
        return p, ok

    res = _sweep(Lambda_ldg, Dgamma, dict(para), num_para, MC, valid_ldg,
                 args.seed, n_jobs=args.n_jobs)
    prob_mp_ldg, prob_pm_ldg, tau_mp_ldg, tau_pm_ldg, meanT_mp_ldg, meanT_pm_ldg = res
    save_mat(
        os.path.join(OUTPUT, "fig_response_time_ldg.mat"),
        {
            "Lambda_ldg": Lambda_ldg,
            "Dgamma": Dgamma,
            "prob_mp_ldg": prob_mp_ldg,
            "prob_pm_ldg": prob_pm_ldg,
            "tau_mp_ldg": tau_mp_ldg,
            "tau_pm_ldg": tau_pm_ldg,
            "meanT_mp_ldg": meanT_mp_ldg,
            "meanT_pm_ldg": meanT_pm_ldg,
            "lambda_grid": lambda_grid,
            "dg_grid": dg_grid,
            "para": dict(para),
            "num_para": dict(num_para),
            "MC": MC,
        },
    )
    print(f"Saved: {OUTPUT}/fig_response_time_ldg.mat")

    # Heatmap 2: lambda x epsilon
    print("Heatmap 2: lambda x epsilon")
    eps_full = np.linspace(0, 0.5, num_mesh + 2)
    eps_grid = eps_full[1:-1]
    Lambda_leps, Epsilon = np.meshgrid(lambda_grid, eps_grid)
    dg_fix = para.gamma_p - para.gamma_m

    def valid_leps(p_template, lmd_v, eps_v):
        p = dict(p_template)
        p["lambda1"] = float(lmd_v)
        p["epsilon"] = float(eps_v)
        ok = is_valid(lmd_v, dg_fix, eps_v)
        return p, ok

    res = _sweep(Lambda_leps, Epsilon, dict(para), num_para, MC, valid_leps,
                 args.seed + 10_000, n_jobs=args.n_jobs)
    prob_mp_leps, prob_pm_leps, tau_mp_leps, tau_pm_leps, meanT_mp_leps, meanT_pm_leps = res
    save_mat(
        os.path.join(OUTPUT, "fig_response_time_leps.mat"),
        {
            "Lambda_leps": Lambda_leps,
            "Epsilon": Epsilon,
            "prob_mp_leps": prob_mp_leps,
            "prob_pm_leps": prob_pm_leps,
            "tau_mp_leps": tau_mp_leps,
            "tau_pm_leps": tau_pm_leps,
            "meanT_mp_leps": meanT_mp_leps,
            "meanT_pm_leps": meanT_pm_leps,
            "lambda_grid": lambda_grid,
            "eps_grid": eps_grid,
            "para": dict(para),
            "num_para": dict(num_para),
            "MC": MC,
        },
    )
    print(f"Saved: {OUTPUT}/fig_response_time_leps.mat")


if __name__ == "__main__":
    main()
