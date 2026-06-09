"""simS3_threshold_cdf.py

Computes optimal policies for several N to plot the threshold CDF.

Writes:
  output/fig_threshold_cdf.mat

Mirrors `simS3_threshold_cdf.m`.
"""

import os

import numpy as np

from sims._common import OUTPUT, parse_args
from ftns import NumPara, Para, bellman_rhs_component, bellman_sol, load_mat, save_mat
from ftns.config import EPSILON, GAMMA_MINUS, GAMMA_PLUS, LAMBDA_COST


def main():
    args = parse_args()
    shared = load_mat(os.path.join(OUTPUT, "fig_normative_comp_para.mat"))
    para = Para(**shared["para"])
    num_para = NumPara(**shared["num_para"])

    # Use the shared baseline (lambda = 0.62 -> theta_c = 0.70). The previous
    # value (0.6) put the CDF step at theta_c = 0.667, off the baseline.
    para.gamma_p = GAMMA_PLUS
    para.gamma_m = GAMMA_MINUS
    para.epsilon = EPSILON
    para.lambda1 = LAMBDA_COST
    para.lambda0 = 0.0
    num_para.gN = 201
    gD = np.linspace(0, 1, num_para.gN)

    N_cases = [40, 100, 400] if args.quick else [40, 80, 100, 200, 400, 800, 1000]

    policy_set_N = []
    for N in N_cases:
        para.N = N
        comp = bellman_rhs_component(para, num_para, n_jobs=args.n_jobs)
        policy_opt, _ = bellman_sol(comp, num_para)
        policy_set_N.append(policy_opt)
        print(f"  N = {N} done")

    save_mat(
        os.path.join(OUTPUT, "fig_threshold_cdf.mat"),
        {
            "N_cases": np.array(N_cases),
            "policy_set_N": np.array(policy_set_N, dtype=object),
            "gD": gD,
            "para": dict(para),
            "num_para": dict(num_para),
        },
    )
    print(f"Saved: {OUTPUT}/fig_threshold_cdf.mat")


if __name__ == "__main__":
    main()
