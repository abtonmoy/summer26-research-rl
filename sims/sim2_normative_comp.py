"""sim2_normative_comp.py

Sweeps group sizes N and computes the Bellman-optimal threshold composition
for each. Writes:
  output/fig_normative_comp_para.mat    - shared parameters
  output/fig_normative_comp_Nswipe.mat  - threshold distributions & rho per N

Mirrors `sim2_normative_comp.m`.
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
    policy2dist,
    save_mat,
)
from ftns.config import EPSILON, GAMMA_MINUS, GAMMA_PLUS, GC, LAMBDA_COST


def main():
    args = parse_args()

    para = Para()
    para.gamma_p = GAMMA_PLUS
    para.gamma_m = GAMMA_MINUS
    para.epsilon = EPSILON
    gc = GC
    para.lambda1 = LAMBDA_COST
    para.lambda0 = 0.0

    num_para = NumPara()
    num_para.gN = 201

    save_mat(
        os.path.join(OUTPUT, "fig_normative_comp_para.mat"),
        {"para": dict(para), "num_para": dict(num_para), "gc": gc},
    )
    print(f"Saved: {OUTPUT}/fig_normative_comp_para.mat")

    if args.quick:
        N_cases = [1, 2, 10, 100]
    else:
        N_cases = [1, 2, 5, 10, 20, 40, 80, 100, 200, 400, 800, 1000]

    nN = len(N_cases)
    gD = np.linspace(0, 1, num_para.gN)
    gc_idx = int(np.argmax(gD >= gc))

    policy_set = [None] * nN
    threshold_set = [None] * nN
    rho_set = np.zeros(nN)
    scout_set = np.zeros(nN, dtype=int)

    for ki, N in enumerate(N_cases):
        para.N = N
        t0 = time.time()
        comp = bellman_rhs_component(para, num_para, n_jobs=args.n_jobs)
        policy_opt, rho_opt = bellman_sol(comp, num_para)
        theta = policy2dist(policy_opt, gD, num_para)
        policy_set[ki] = policy_opt
        threshold_set[ki] = theta
        rho_set[ki] = rho_opt
        scout_set[ki] = int(np.sum(theta < gc))
        dt = time.time() - t0
        print(f"N = {N:4d} | rho = {rho_opt:.4f} | scouts = {scout_set[ki]:3d} | {dt:.1f}s")

    save_mat(
        os.path.join(OUTPUT, "fig_normative_comp_Nswipe.mat"),
        {
            "N_cases": np.array(N_cases),
            "policy_set": np.array(policy_set, dtype=object),
            "threshold_set": np.array(threshold_set, dtype=object),
            "rho_set": rho_set,
            "scout_set": scout_set,
            "gD": gD,
            "gc_idx": gc_idx + 1,    # write 1-based for parity with MATLAB
            "para": dict(para),
            "num_para": dict(num_para),
            "gc": gc,
        },
    )
    print(f"Saved: {OUTPUT}/fig_normative_comp_Nswipe.mat")


if __name__ == "__main__":
    main()
