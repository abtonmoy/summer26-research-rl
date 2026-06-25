# Collective Foraging and Division of Labor — Python Reproduction

Python port of the MATLAB codebase for:

> **Daring few, patient many: division of labor in decentralized foraging collectives**
> Hyunjoong Kim, Zachary Kilpatrick, Krešimir Josić

The original MATLAB source is in `../../code_matlab/division-of-labor-abt/`. This
port preserves the algorithmic structure (Bellman value iteration on a
discretized belief grid, Monte-Carlo evaluation, GA optimization, partial-
communication MC) and the on-disk layout (`output/` for `.mat`/`.npz` data,
`figs/` for figures).

## Requirements

```
python >= 3.10
numpy
scipy
matplotlib
joblib            # parfor replacement
tqdm              # progress
```

Install with `pip install -r requirements.txt`, or with [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.13
uv pip install -r requirements.txt
uv run python -m sims.sim2_normative_comp   # run anything via `uv run`
```

## Layout

```
repo/
├── ftns/         reusable functions (bellman, policy, mc, ga, ...)
├── sims/         simulation scripts (sim2, sim3, ..., simS1-S7)
├── figures/      figure-generation scripts (fig2, fig3, ..., figS1-S7)
├── output/       .mat/.npz outputs from sims
├── figs/         .png/.pdf figure outputs
└── tests/        validation against MATLAB reference outputs
```

## Function mapping (MATLAB → Python)

| MATLAB                              | Python                                              |
|-------------------------------------|-----------------------------------------------------|
| `ftn_bellman_rhs_component.m`       | `ftns.bellman.bellman_rhs_component`                |
| `ftn_bellman_rhs.m`                 | `ftns.bellman.bellman_rhs`                          |
| `ftn_bellman_sol.m`                 | `ftns.bellman.bellman_sol`                          |
| `ftn_avg_return.m`                  | `ftns.bellman.avg_return`                           |
| `ftn_crit_belif.m`                  | `ftns.bellman.crit_belif`                           |
| `ftn_policy2dist.m`                 | `ftns.policy.policy2dist`                           |
| `ftn_dist2policy.m`                 | `ftns.policy.dist2policy`                           |
| `ftn_env_state.m`                   | `ftns.env.env_state`                                |
| `ftn_behav_dyn.m`                   | `ftns.behav.behav_dyn`                              |
| `ftn_behav_dyn_stoch.m`             | `ftns.behav.behav_dyn_stoch`                        |
| `ftn_mc_performance.m`              | `ftns.mc.mc_performance`                            |
| `ftn_mc_response.m`                 | `ftns.mc.mc_response`                               |
| `ftn_mc_partialcomm.m`              | `ftns.mc.mc_partialcomm`                            |
| `ftn_update_partialcomm.m`          | `ftns.belief.update_partialcomm`                    |
| `ftn_update_ga.m`                   | `ftns.ga.update_ga`                                 |
| `ftn_score_multipolicy.m`           | `ftns.ga.score_multipolicy`                         |
| `ftn_stoch_return.m`                | `ftns.utils.stoch_return`                           |
| `ftn_hetero.m`                      | `ftns.hetero.hetero`                                |
| `ftn_indiv.m`                       | `ftns.utils.indiv`                                  |

## How to run

Always run `sim2` first — it produces the shared parameter file that the other
scripts load.

```bash
cd repo
python -m sims.sim2_normative_comp
python -m figures.fig2_normative_comp

python -m sims.sim3_perturb_comp
python -m figures.fig3_perturb_comp

python -m sims.sim4_heterogeneity
python -m figures.fig4_heterogeneity

# Supplementary
python -m sims.simS1_genetic_algorithm
python -m figures.figS1_genetic_algorithm
# ... etc
```

Each sim writes a `.mat` file (scipy.io format) into `output/` so the data is
interoperable with the original MATLAB pipeline.

## Notes on the port

- `parfor` is replaced with `joblib.Parallel`. Set `JOBLIB_N_JOBS=1` in the env
  to disable.
- MATLAB random distributions map as follows:
  - `binopdf` → `scipy.stats.binom.pmf`
  - `hygepdf(y, a, x, m)` → `scipy.stats.hypergeom.pmf(y, a, x, m)`
  - `binornd(n, p)` → `np.random.binomial(n, p)`
  - `geornd(p)` → `np.random.geometric(p) - 1` (MATLAB returns # failures
    before first success; NumPy returns # of trials)
  - `random('hyge', a, x, m)` → `np.random.hypergeometric(x, a-x, m)`
- MATLAB's `ga` solver inside `ftn_update_optindiv` is not used by any
  shipped sim; if you call `update_optindiv` it uses
  `scipy.optimize.differential_evolution` as the replacement.
- All sims accept `--quick` to shrink expensive sweeps for smoke testing.

## Performance — vectorized fast paths

The Monte-Carlo and population sweeps were the wall-clock bottleneck (the full
suite ran ~10-12 h). Each now has a **numpy-batched** drop-in replacement that
moves the replicate/population dimension out of Python (and joblib) loops into
vectorized array ops; only the serial time / belief recursion is kept as a loop.
The original scalar functions are preserved as references and for validation.

| Scalar (reference)    | Batched fast path                                  | Used by        | Speedup                | Equivalence                       |
|-----------------------|----------------------------------------------------|----------------|------------------------|-----------------------------------|
| `bellman.avg_return`  | `bellman.avg_return_batch`                         | simS1 (GA)     | ~40-100x (grows with N)| deterministic — matches to ~1e-8  |
| `mc.mc_performance`   | `mc.mc_performance_batch`                          | sim3, simS5    | ~44x                   | within MC noise (~1/sqrt(MC))     |
| `mc.mc_partialcomm`   | `mc.mc_partialcomm_batch` / `mc.partialcomm_run_batch` | simS7, simS7b | ~57-70x             | within MC noise                   |

How they work:

- **`avg_return_batch`** solves the entire GA population's policy-evaluation
  value iteration at once with a single batched `einsum`, and evaluates only each
  policy's own action per state (the scalar path computed all `N+1` actions via
  `bellman_rhs`). It is deterministic, so it reproduces `avg_return` to iteration
  tolerance (`max |Δρ| ≈ 1e-8`). This took simS1 from ~10 h to ~15 min.
- **`mc_performance_batch`** steps all MC replicates together, tracking only the
  per-step committed count and successes (drawn as one binomial per replicate) —
  the `rho`/accuracy statistics need nothing finer, so no `N×T` per-agent arrays.
- **`mc_partialcomm_batch` / `partialcomm_run_batch`** vectorize the partial-comm
  MC over replicates *and* the per-agent communication loop over `N`. The key
  enabler is the identity
  `Σ_x Hypergeom(y; a, x, m)·Binom(x; a, p) = Binom(y; m, p)`,
  which collapses the peer-evidence marginal likelihood to a single binomial pmf
  and removes the dominant `scipy.stats` per-call overhead. simS7 (full) dropped
  from ~2.5 h to ~2 min.

Because the MC fast paths draw random numbers in a different order than the
scalar loops, they agree with the references **within MC noise**, not bit-for-bit
(the deterministic `avg_return_batch` does match bit-for-bit). All of this is
checked by `python tests/test_vs_matlab.py`: deterministic outputs (sim2, simS2,
simS3, simS6) stay bit-exact, and the MC-driven outputs stay within tolerance.

Still scalar (not yet batched), and now the longest sims: the deterministic
Bellman heatmaps `sim4` and `simS6`, and `mc_response` (used by `simS4`). These
would need a batched `bellman_sol` (with the max-over-actions step) to speed up.
