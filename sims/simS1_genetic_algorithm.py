"""simS1_genetic_algorithm.py

Runs the genetic algorithm for several group sizes N. Validates that
evolutionary search converges to the Bellman-optimal threshold composition.

Writes:
  output/fig_genetic_algorithm_<N>.mat for each N

Mirrors `simS1_genetic_algorithm.m`.
"""

import os
import time

import numpy as np

from sims._common import OUTPUT, parse_args
from ftns import (
    NumPara,
    Para,
    avg_return_batch,
    bellman_rhs_component,
    bellman_sol,
    crit_belif,
    dist2policy,
    save_mat,
    update_ga,
)
from ftns.config import EPSILON, GAMMA_MINUS, GAMMA_PLUS, LAMBDA_COST


def _score_pop(policies, comp, para, num_para, n_jobs):
    # Vectorized batch value-iteration over the whole population (~100x faster
    # than looping avg_return per policy). n_jobs kept for signature parity.
    return np.asarray(avg_return_batch(policies, comp, num_para))


def main():
    args = parse_args()

    para = Para()
    para.gamma_p = GAMMA_PLUS
    para.gamma_m = GAMMA_MINUS
    para.epsilon = EPSILON
    para.lambda1 = LAMBDA_COST
    para.lambda0 = 0.0

    num_para = NumPara()
    num_para.gN = 201
    num_para.crossover_rate = 0.8
    num_para.mutation_rate = 0.05
    num_para.sigma = 0.05
    num_para.elite_count = 20

    pop_size = 40 if args.quick else 200
    max_gen = 30 if args.quick else 300
    # Default to a smaller N sweep than the MATLAB output for tractable wall
    # time. The full MATLAB sweep covers [10, 25, 50, 100, 200, 400, 800]; we
    # validate the convergence trend on a representative subset.
    N_cases = [10, 100] if args.quick else [10, 50, 100, 200]
    gD = np.linspace(0, 1, num_para.gN)

    for N in N_cases:
        para.N = N
        print(f"\n=== GA for N = {N} ===")
        gc = crit_belif(para)

        comp = bellman_rhs_component(para, num_para, n_jobs=args.n_jobs)
        policy_opt, rho_opt = bellman_sol(comp, num_para)

        rng = np.random.default_rng(args.seed + N)
        genes = gc * np.ones((pop_size, N))

        policies = dist2policy(genes, num_para)
        scores = _score_pop(policies, comp, para, num_para, args.n_jobs)

        best_genes = np.zeros((max_gen, N))
        conv_traj = np.zeros(max_gen)

        for gen in range(max_gen):
            t0 = time.time()
            genes = update_ga(scores, genes, num_para, rng=rng)
            policies = dist2policy(genes, num_para)
            scores = np.asarray(avg_return_batch(policies, comp, num_para))

            best_idx = int(scores.argmax())
            best_score = float(scores[best_idx])
            best_genes[gen, :] = genes[best_idx, :]
            conv_traj[gen] = (rho_opt - best_score) / rho_opt
            dt = time.time() - t0
            print(f"  N={N} | gen {gen+1:3d}/{max_gen} | best={best_score:.4f} "
                  f"| conv={conv_traj[gen]:.4e} | {dt:.1f}s")

        avg_policy = policies.mean(axis=0)
        save_mat(
            os.path.join(OUTPUT, f"fig_genetic_algorithm_{N}.mat"),
            {
                "best_genes": best_genes,
                "conv_traj": conv_traj,
                "para": dict(para),
                "num_para": dict(num_para),
                "genes": genes,
                "policies": policies,
                "avg_policy": avg_policy,
                "policy_opt": policy_opt,
                "rho_opt": rho_opt,
                "pop_size": pop_size,
                "max_gen": max_gen,
                "gD": gD,
            },
        )
        print(f"Saved: {OUTPUT}/fig_genetic_algorithm_{N}.mat")


if __name__ == "__main__":
    main()
