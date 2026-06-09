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

Install with `pip install -r requirements.txt`.

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
