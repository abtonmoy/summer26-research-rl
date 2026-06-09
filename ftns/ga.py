"""Genetic-algorithm operators for evolving threshold distributions.

Mirrors `ftn_update_ga.m`: elitism + roulette-wheel selection + single-point
crossover + Gaussian mutation. `score_multipolicy` evaluates a batch of
policies in parallel against a shared environmental state sequence.
"""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed

from .env import env_state
from .utils import stoch_return


def _roulette_selection(genes: np.ndarray, scores: np.ndarray, n: int,
                        rng: np.random.Generator) -> np.ndarray:
    s = scores - scores.min() + 1e-6
    prob = s / s.sum()
    cum = np.cumsum(prob)
    r = rng.random(n)
    idx = np.searchsorted(cum, r)
    idx = np.clip(idx, 0, len(scores) - 1)
    return genes[idx, :].copy()


def _crossover(parents: np.ndarray, rate: float, gene_length: int,
               rng: np.random.Generator) -> np.ndarray:
    offspring = parents.copy()
    n = parents.shape[0]
    for i in range(0, n - 1, 2):
        if rng.random() < rate:
            pt = rng.integers(1, gene_length)  # crossover point in [1, gene_length-1]
            offspring[i, pt:] = parents[i + 1, pt:]
            offspring[i + 1, pt:] = parents[i, pt:]
    return offspring


def _mutate(genes: np.ndarray, rate: float, sigma: float,
            rng: np.random.Generator) -> np.ndarray:
    mask = rng.random(genes.shape) < rate
    genes = genes + mask * (sigma * rng.standard_normal(genes.shape))
    return np.clip(genes, 0.0, 1.0)


def update_ga(scores: np.ndarray, genes: np.ndarray, num_para,
              rng: np.random.Generator | None = None) -> np.ndarray:
    """One-generation GA update. Returns new gene population."""
    if rng is None:
        rng = np.random.default_rng()

    crossover_rate = num_para.crossover_rate
    mutation_rate = num_para.mutation_rate
    elite_count = num_para.elite_count
    sigma = num_para.sigma

    pop_size, gene_length = genes.shape
    sorted_idx = np.argsort(-scores)
    elites = genes[sorted_idx[:elite_count], :].copy()

    parents = _roulette_selection(genes, scores, pop_size - elite_count, rng)
    offspring = _crossover(parents, crossover_rate, gene_length, rng)
    offspring = _mutate(offspring, mutation_rate, sigma, rng)

    return np.vstack([elites, offspring])


def score_multipolicy(policies: np.ndarray, M: int, T: int, para, num_para,
                      n_jobs: int = -1, seed: int = 0):
    """Parallel scoring of M policies under a single shared state sequence.

    Returns (scores, best_idx).
    """
    rng = np.random.default_rng(seed)
    S = env_state(T, para, rng=rng)
    seeds = np.random.SeedSequence(seed + 1).spawn(M)

    def _one(pol, s):
        sub_rng = np.random.default_rng(s)
        return stoch_return(pol, S, para, num_para, rng=sub_rng)

    scores = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_one)(policies[m, :], seeds[m]) for m in range(M)
    )
    scores = np.array(scores)
    return scores, int(scores.argmax())
