# Symmetric vs. Asymmetric Environmental Switching

This document records the extension of the reproduction from the paper's
**symmetric** two-state switching model to a **directional (asymmetric)** one,
the exact code changes it required, and the downstream implications of each
change. It accompanies the theory note `docs/asymetric_switch.pdf`
(*Asymmetric Environmental Switching in Decentralized Foraging Collectives*).

---

## 1. The model change in one line

The hidden environment is a two-state Markov chain (high `+`, low `-`). The
original model uses a **single** switching rate `ε` for both directions:

```
Pr(flip) = ε        (symmetric: equal dwell times in + and -)
```

The asymmetric model replaces it with **two directional rates**:

| Symbol  | Meaning                                            |
|---------|----------------------------------------------------|
| `ε₋`    | probability of **leaving the high** state (`+ → -`) |
| `ε₊`    | probability of **leaving the low** state (`- → +`)  |

The symmetric model is recovered exactly when `ε₊ = ε₋ = ε`.

### The one consequence everything else follows from

With no observations, belief relaxes toward the **stationary high-state
probability**

```
g* = ε₊ / (ε₊ + ε₋)
```

instead of `½`. (Solve `g = (1−ε₋)·g + ε₊·(1−g)`.) In the symmetric case
`g* = ½`, so nothing changes; with unequal rates the whole inference cycle is
re-anchored at `g*`.

---

## 2. Where the change enters: the belief-propagation step

The inference cycle is **observe → update (Bayes) → predict (drift)**. The
switching rate enters at exactly **one** point — the predict step:

```
symmetric:   g' = (1 − ε)·U   + ε·(1 − U)
asymmetric:  g' = (1 − ε₋)·U  + ε₊·(1 − U)
```

where `U` is the post-observation posterior on the high state. The "stay high"
weight becomes `1 − ε₋` and the "flip up from low" weight becomes `ε₊`. The
likelihood (binomial) and the Bayes measurement step are **untouched** — they
describe outcomes *within* a fixed state and never see the switching rate.

---

## 3. Code changes

All changes are backward compatible: directional rates default to the symmetric
`ε`, so existing code and results are unaffected unless you explicitly set
`para.eps_p` / `para.eps_m`.

### 3.0 Parameter plumbing — `ftns/config.py`

New constants and two helpers used everywhere:

```python
EPS_PLUS  = EPSILON          # default symmetric
EPS_MINUS = EPSILON
G_STAR    = EPS_PLUS / (EPS_PLUS + EPS_MINUS)   # = 1/2 symmetric

def eps_pm(para):   # -> (eps_plus, eps_minus), falls back to para.epsilon
def g_star(para):   # -> eps_plus / (eps_plus + eps_minus), falls back to 1/2
```

`eps_pm` returns `para.eps_p` / `para.eps_m` when set, otherwise the symmetric
`para.epsilon`. Every other change reads the rates through these helpers, so
there is a single source of truth.

### 3.1 Belief-propagation drift (the one conceptual change)

The drift term `(1−ε)·hi + ε·lo → (1−ε₋)·hi + ε₊·lo` was applied everywhere a
belief is propagated:

| File | Function | What changed |
|------|----------|--------------|
| `ftns/belief.py` | `update_belief` | `((1-eps_m)*bgp + eps_p*bgm)/(bgp+bgm)` |
| `ftns/belief.py` | `update_partialcomm` (no-obs and main) | `(1-eps_m)*U + eps_p*(1-U)` |
| `ftns/bellman.py` | `_build_K_for_n` | drift in the transition operator (signature now takes `eps_p, eps_m`) |
| `ftns/mc.py` | `_update_belief` | same drift |
| `ftns/utils.py` | `_update_belief` | same drift |

Here `bgp = g·P(x\|+)` is the unnormalised posterior weight on the high state and
`bgm = (1−g)·P(x\|−)` the weight on the low state, so `bgp` carries `(1−ε₋)` and
`bgm` carries `ε₊`.

### 3.2 The environment Markov chain — state-dependent flips

The true-state sequence now flips at a rate that depends on the current state,
and the **initial state is drawn from the stationary distribution** (`+` with
probability `g*`):

```python
rate = eps_m if s == 1 else eps_p     # leave-high vs leave-low
if rng.random() < rate:
    s = -s
```

Applied in `ftns/env.py` (`env_state`), `ftns/mc.py`
(`_one_partialcomm_rep`, `partialcomm_run_batch`, `mc_partialcomm_batch`), and
the inlined environment loop in `sims/simS7b_scout_vs_kappa.py`.

### 3.3 The no-information anchor `½ → g*`

Every place that initialises belief to `0.5` now uses `g_star(para)`
(identical when symmetric): `ftns/behav.py` (both trajectory functions),
`ftns/mc.py` (`mc_response`, partial-comm reps), `ftns/utils.py`
(`stoch_return`), and `sims/simS7b`.

### 3.4 Response-time dwell distributions — `ftns/mc.py::mc_response`

The two measurement windows now use **different** geometric dwell times:

```python
Ti = geometric(eps_m)    # lo -> hi : window = time spent in the HIGH state
Ti = geometric(eps_p)    # hi -> lo : window = time spent in the LOW state
```

(Symmetric used a single `geometric(ε)` for both.)

### 3.5 The feasibility wedge — `sims/{sim4,simS4,simS6}.py::is_valid`

This is the only change that is a **re-derivation**, not a substitution. The
critical belief `gc` must lie inside the reachable belief band of the predict
step, which under asymmetry is `[ε₊, 1 − ε₋]` (not `[ε, 1 − ε]`). With γ
centred at ½ this gives

```python
def is_valid(lmd, dg, eps_p, eps_m=None):
    # gc ∈ [eps_plus, 1 - eps_minus]
    return (lmd >= (eps_p - 0.5)*dg + 0.5) and (lmd <= (0.5 - eps_m)*dg + 0.5)
```

Setting `eps_p = eps_m` recovers the original symmetric bound exactly.

---

## 4. Downstream implications

### What **shifts** (everything anchored at the relaxation point)

| Quantity | Symmetric | Asymmetric |
|----------|-----------|------------|
| No-info relaxation belief | `½` | `g* = ε₊/(ε₊+ε₋)` |
| Feasibility / starvation-trap band for `gc` | `[ε, 1−ε]` | `[ε₊, 1−ε₋]` |
| Heterogeneity wedge (`is_valid`) | symmetric around `½` | asymmetric around `g*` |
| Response-time bias (`+→−` vs `−→+`) | equal dwell `geometric(ε)` | `geometric(ε₋)` vs `geometric(ε₊)` |
| Stationary fraction of time in `+` | `½` | `g*` |

Intuition: when `ε₊ > ε₋` the world spends more time high, beliefs relax upward,
the collective commits more readily, and the trap/wedge/response asymmetry all
tilt accordingly (and vice versa).

### What is **preserved** (no switching rate in their content)

| Quantity | Why it is invariant |
|----------|---------------------|
| Reward `r(g) = γ₊·g + γ₋·(1−g)` | pure reward structure, no `ε` |
| Critical threshold `θ_c = (λ−γ₋)/(γ₊−γ₋)` | from the reward, no `ε` (`bellman.crit_belif`) |
| Binomial observation likelihood `B(x\|a,γ)` | describes within-state outcomes only |
| Value-iteration / Bellman machinery | unchanged solver; only the transition operator's drift differs |
| **`O(log N)` explorer-scaling theorem** | lives in the reward binomial's large-deviation rate function, which never sees the switching rate |

So the paper's qualitative conclusions (daring few + patient many, sublinear
scout scaling) survive the generalization; only the symmetric algebra breaks,
with `g*` replacing `½` as the natural anchor.

---

## 5. How to use it

```python
from ftns import Para
para = Para()
para.gamma_p, para.gamma_m, para.lambda1, para.lambda0 = 0.8, 0.2, 0.62, 0.0
para.N = 100

# Symmetric (default): set only epsilon
para.epsilon = 0.1                 # eps_pm -> (0.1, 0.1), g_star = 0.5

# Asymmetric: set the directional rates
para.eps_p = 0.05                  # leave-low  (enter high)
para.eps_m = 0.20                  # leave-high
# eps_pm -> (0.05, 0.20),  g_star = 0.05 / 0.25 = 0.20
```

Everything downstream (`bellman_*`, `behav_*`, `mc_*`, the sims) picks the rates
up automatically via `eps_pm` / `g_star`.

---

## 6. Backward compatibility and validation

- **Deterministic paths are bit-exact when `eps_p = eps_m`.** `g_star` returns
  exactly `0.5` (since `x/(x+x)` is exact in IEEE-754) and every drift expression
  reduces to the original term, so `sim2`'s Bellman optimum reproduces the
  reference to ~1e-14.
- **MC paths agree within MC noise.** The only realization-level change in the
  symmetric limit is the environment's initial-state draw (now stationary), which
  affects only the Monte-Carlo sims and only within `~1/√MC`.
- `python tests/test_vs_matlab.py` passes all comparisons with the symmetric
  defaults (deterministic outputs bit-exact, MC outputs within tolerance).

---

## 7. Status

The asymmetric machinery is **implemented and symmetric-validated**. All runs to
date use `eps_p = eps_m` (i.e. they reproduce the symmetric baseline). The
asymmetric regime (`g* ≠ ½`) — how the scout/deliberator split, heterogeneity
wedge, and response-time bias actually move with unequal dwell times — is
implemented but **not yet swept**; that is the natural next experiment.
