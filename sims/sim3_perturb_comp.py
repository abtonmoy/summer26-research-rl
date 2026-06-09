"""sim3_perturb_comp.py

Builds four allocation policies (optimal, desync, low-threshold, high-threshold)
at N = 100, MC-evaluates each, and interpolates between optimal and each of the
three perturbed extremes.

Writes:
  output/fig_perturb_comp_para.mat
  output/fig_perturb_comp_perform.mat
  output/fig_perturb_comp_interp.mat

Mirrors `sim3_perturb_comp.m`.
"""

import os
import time

import numpy as np

from sims._common import OUTPUT, parse_args
from ftns import (
    bellman_rhs_component,
    bellman_sol,
    mc_performance,
    save_mat,
    load_mat,
    Para,
    NumPara,
)


def main():
    args = parse_args()

    shared = load_mat(os.path.join(OUTPUT, "fig_normative_comp_para.mat"))
    para = Para(**shared["para"])
    num_para = NumPara(**shared["num_para"])
    gc = float(shared["gc"])

    para.N = 100
    num_para.T = 2_000 if args.quick else 20_000
    MC = 50 if args.quick else 1000

    gN = int(num_para.gN)
    gD = np.linspace(0.0, 1.0, gN)
    gc_idx = int(np.argmax(gD >= gc))  # 0-based first index where gD >= gc

    # Optimal
    comp = bellman_rhs_component(para, num_para, n_jobs=args.n_jobs)
    policy_opt, rho_opt = bellman_sol(comp, num_para)
    Ns = int(policy_opt[gc_idx - 1])     # MATLAB policy_opt(gc_idx - 1)
    N = int(para.N)

    # Perturbed policies
    # desynchronized: uniform ramp above gc, scouts unchanged below
    gDhi = gD[gc_idx:]
    U = np.floor((N - Ns) * (gDhi - gc) / (1.0 - gc)).astype(int)
    policy_desync = np.concatenate([policy_opt[:gc_idx], Ns + U]).astype(int)

    # low-threshold: all scouts commit below gc, all deliberate above
    policy_lo = (Ns * (gD < gc) + N * (gD >= gc)).astype(int)
    policy_lo[0] = 0

    # high-threshold: no scouts below gc, all commit above
    policy_hi = (N * (gD >= gc)).astype(int)
    policy_hi[gc_idx - 1] = Ns

    policy_set = [policy_opt, policy_desync, policy_lo, policy_hi]

    save_mat(
        os.path.join(OUTPUT, "fig_perturb_comp_para.mat"),
        {
            "policy_set": np.array(policy_set, dtype=object),
            "gD": gD,
            "gc_idx": gc_idx + 1,
            "Ns": Ns,
            "MC": MC,
            "para": dict(para),
            "num_para": dict(num_para),
            "gc": gc,
        },
    )
    print(f"Saved: {OUTPUT}/fig_perturb_comp_para.mat")

    rho_set = np.zeros(4)
    acc_set = np.zeros(4)
    acc_uncond_set = np.zeros(4)
    for k in range(4):
        t0 = time.time()
        rho_set[k], acc_set[k], acc_uncond_set[k] = mc_performance(
            policy_set[k], MC, para, num_para, n_jobs=args.n_jobs, seed=args.seed + k
        )
        print(f"  policy {k+1}/4 | rho={rho_set[k]:.4f} | acc={acc_set[k]:.4f} | {time.time()-t0:.1f}s")

    rho_max = 0.5 * (para.gamma_p - para.lambda1)

    save_mat(
        os.path.join(OUTPUT, "fig_perturb_comp_perform.mat"),
        {
            "rho_set": rho_set,
            "acc_set": acc_set,
            "acc_uncond_set": acc_uncond_set,
            "rho_max": rho_max,
        },
    )
    print(f"Saved: {OUTPUT}/fig_perturb_comp_perform.mat")

    # Interpolation
    Npsi = 6 if args.quick else 11
    Psi = np.linspace(0.0, 1.0, Npsi)
    rho_interp = np.zeros((Npsi, 3))
    acc_interp = np.zeros((Npsi, 3))
    acc_uncond_interp = np.zeros((Npsi, 3))

    for k in range(1, 4):
        rho_interp[0, k - 1] = rho_set[0]
        rho_interp[-1, k - 1] = rho_set[k]
        acc_interp[0, k - 1] = acc_set[0]
        acc_interp[-1, k - 1] = acc_set[k]
        acc_uncond_interp[0, k - 1] = acc_uncond_set[0]
        acc_uncond_interp[-1, k - 1] = acc_uncond_set[k]

        for np_ in range(1, Npsi - 1):
            psi = Psi[np_]
            mix = np.floor(psi * policy_set[k] + (1 - psi) * policy_opt).astype(int)
            t0 = time.time()
            rho_interp[np_, k - 1], acc_interp[np_, k - 1], acc_uncond_interp[np_, k - 1] = (
                mc_performance(mix, MC, para, num_para, n_jobs=args.n_jobs,
                               seed=args.seed + 100 * k + np_)
            )
            print(f"  policy {k+1} | psi={psi:.2f} | {time.time()-t0:.1f}s")

    save_mat(
        os.path.join(OUTPUT, "fig_perturb_comp_interp.mat"),
        {
            "Psi": Psi,
            "rho_interp": rho_interp,
            "acc_interp": acc_interp,
            "acc_uncond_interp": acc_uncond_interp,
            "rho_max": rho_max,
        },
    )
    print(f"Saved: {OUTPUT}/fig_perturb_comp_interp.mat")


if __name__ == "__main__":
    main()
