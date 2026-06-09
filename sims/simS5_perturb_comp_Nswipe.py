"""simS5_perturb_comp_Nswipe.py

Replicates sim3 for N = 50 and N = 200.

Writes:
  output/fig_perturb_comp_N50.mat
  output/fig_perturb_comp_N200.mat

Mirrors `simS5_perturb_comp_Nswipe.m`.
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
    mc_performance,
    save_mat,
)


def main():
    args = parse_args()
    shared = load_mat(os.path.join(OUTPUT, "fig_normative_comp_para.mat"))
    para = Para(**shared["para"])
    num_para = NumPara(**shared["num_para"])
    gc = float(shared["gc"])

    para.gamma_p = 0.8
    para.gamma_m = 0.2
    para.epsilon = 0.1
    para.lambda1 = para.gamma_p * gc + para.gamma_m * (1 - gc)
    para.lambda0 = 0.0
    num_para.gN = 201
    num_para.T = 2_000 if args.quick else 20_000
    MC = 50 if args.quick else 1000

    gD = np.linspace(0, 1, num_para.gN)
    gc_idx = int(np.argmax(gD >= gc))

    N_cases = [50, 200]

    for N in N_cases:
        para.N = N
        print(f"\n=== N = {N} ===")
        comp = bellman_rhs_component(para, num_para, n_jobs=args.n_jobs)
        policy_opt, rho_opt = bellman_sol(comp, num_para)

        Ns = int(policy_opt[gc_idx - 1])

        gDhi = gD[gc_idx:]
        U = np.floor((N - Ns) * (gDhi - gc) / (1.0 - gc)).astype(int)
        policy_desync = np.concatenate([policy_opt[:gc_idx], Ns + U]).astype(int)
        policy_lo = (Ns * (gD < gc) + N * (gD >= gc)).astype(int)
        policy_lo[0] = 0
        policy_hi = (N * (gD >= gc)).astype(int)
        policy_hi[gc_idx - 1] = Ns

        policy_set = [policy_opt, policy_desync, policy_lo, policy_hi]

        rho_set = np.zeros(4)
        acc_set = np.zeros(4)
        acc_uncond_set = np.zeros(4)
        for k in range(4):
            t0 = time.time()
            rho_set[k], acc_set[k], acc_uncond_set[k] = mc_performance(
                policy_set[k], MC, para, num_para, n_jobs=args.n_jobs,
                seed=args.seed + 1000 * N + k,
            )
            print(f"  policy {k+1}/4 | rho={rho_set[k]:.4f} | {time.time()-t0:.1f}s")

        rho_max = 0.5 * (para.gamma_p - para.lambda1)

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
                                   seed=args.seed + 1000 * N + 100 * k + np_)
                )
                print(f"  policy {k+1} | psi={psi:.2f} | {time.time()-t0:.1f}s")

        fname = os.path.join(OUTPUT, f"fig_perturb_comp_N{N}.mat")
        save_mat(
            fname,
            {
                "policy_set": np.array(policy_set, dtype=object),
                "gD": gD,
                "gc_idx": gc_idx + 1,
                "Ns": Ns,
                "MC": MC,
                "para": dict(para),
                "num_para": dict(num_para),
                "gc": gc,
                "rho_set": rho_set,
                "acc_set": acc_set,
                "acc_uncond_set": acc_uncond_set,
                "rho_max": rho_max,
                "Psi": Psi,
                "rho_interp": rho_interp,
                "acc_interp": acc_interp,
                "acc_uncond_interp": acc_uncond_interp,
            },
        )
        print(f"Saved: {fname}")


if __name__ == "__main__":
    main()
