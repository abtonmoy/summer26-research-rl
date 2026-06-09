"""simS2_scout_scaling.py

Sweeps group size N for varying lambda, delta_gamma, and epsilon to compute
the normative scout count at each combination.

Writes:
  output/fig_scout_scaling.mat

Mirrors `simS2_scout_scaling.m`.
"""

import os

import numpy as np

from sims._common import OUTPUT, parse_args
from ftns import (
    NumPara,
    Para,
    bellman_rhs_component,
    bellman_sol,
    crit_belif,
    load_mat,
    save_mat,
)


def main():
    args = parse_args()

    shared = load_mat(os.path.join(OUTPUT, "fig_normative_comp_para.mat"))
    para = Para(**shared["para"])
    num_para = NumPara(**shared["num_para"])
    num_para.gN = 201
    gD = np.linspace(0, 1, num_para.gN)

    if args.quick:
        N_cases = [10, 100, 1000]
    else:
        N_cases = [1, 2, 5, 10, 20, 40, 80, 100, 200, 400, 800, 1000]
    nN = len(N_cases)

    # Sweep A: vary lambda
    print("Sweep A: lambda")
    lambda_cases = [0.4, 0.5, 0.6, 0.7]
    scout_lambda = np.zeros((nN, len(lambda_cases)))
    para.gamma_p = 0.8
    para.gamma_m = 0.2
    para.epsilon = 0.1
    for m, lam in enumerate(lambda_cases):
        para.lambda1 = lam
        gc_m = crit_belif(para)
        gc_idx = int(np.argmax(gD >= gc_m)) - 1  # MATLAB find(gD < gc, 1, 'last')
        for n, N in enumerate(N_cases):
            para.N = N
            comp = bellman_rhs_component(para, num_para, n_jobs=args.n_jobs)
            policy_opt, _ = bellman_sol(comp, num_para)
            scout_lambda[n, m] = policy_opt[gc_idx]
            print(f"  lam={lam:.1f} | N={N:4d} | scouts={int(scout_lambda[n, m])}")

    # Sweep B: vary delta_gamma
    print("Sweep B: delta_gamma")
    dgamma_cases = [0.4, 0.5, 0.6, 0.7]
    scout_dgamma = np.zeros((nN, len(dgamma_cases)))
    para.lambda1 = 0.6
    para.epsilon = 0.1
    for m, dg in enumerate(dgamma_cases):
        para.gamma_p = 0.5 + dg / 2
        para.gamma_m = 0.5 - dg / 2
        gc_m = crit_belif(para)
        gc_idx = int(np.argmax(gD >= gc_m)) - 1
        for n, N in enumerate(N_cases):
            para.N = N
            comp = bellman_rhs_component(para, num_para, n_jobs=args.n_jobs)
            policy_opt, _ = bellman_sol(comp, num_para)
            scout_dgamma[n, m] = policy_opt[gc_idx]
            print(f"  dg={dg:.1f} | N={N:4d} | scouts={int(scout_dgamma[n, m])}")

    # Sweep C: vary epsilon
    print("Sweep C: epsilon")
    epsilon_cases = [0.05, 0.1, 0.2, 0.3]
    scout_epsilon = np.zeros((nN, len(epsilon_cases)))
    para.gamma_p = 0.8
    para.gamma_m = 0.2
    para.lambda1 = 0.6
    for m, eps in enumerate(epsilon_cases):
        para.epsilon = eps
        gc_m = crit_belif(para)
        gc_idx = int(np.argmax(gD >= gc_m)) - 1
        for n, N in enumerate(N_cases):
            para.N = N
            comp = bellman_rhs_component(para, num_para, n_jobs=args.n_jobs)
            policy_opt, _ = bellman_sol(comp, num_para)
            scout_epsilon[n, m] = policy_opt[gc_idx]
            print(f"  eps={eps:.2f} | N={N:4d} | scouts={int(scout_epsilon[n, m])}")

    save_mat(
        os.path.join(OUTPUT, "fig_scout_scaling.mat"),
        {
            "N_cases": np.array(N_cases),
            "lambda_cases": np.array(lambda_cases),
            "scout_lambda": scout_lambda,
            "dgamma_cases": np.array(dgamma_cases),
            "scout_dgamma": scout_dgamma,
            "epsilon_cases": np.array(epsilon_cases),
            "scout_epsilon": scout_epsilon,
            "para": dict(para),
            "num_para": dict(num_para),
            "gD": gD,
        },
    )
    print(f"Saved: {OUTPUT}/fig_scout_scaling.mat")


if __name__ == "__main__":
    main()
