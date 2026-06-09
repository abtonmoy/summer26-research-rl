"""Baseline model parameters — single source of truth.

Every sim/figure that needs the paper's baseline parameter set should import
from here instead of hardcoding values, so no script silently uses a different
cost `lambda` (or any other baseline parameter).

Paper baseline:
    N = 100
    gamma_plus  = 0.8
    gamma_minus = 0.2          (i.e. gamma = 0.5 +/- 0.3)
    epsilon     = 0.1
    lambda_cost = 0.62
which gives the critical belief / critical threshold
    theta_c = (lambda - gamma_minus) / (gamma_plus - gamma_minus) = 0.70
"""

N_DEFAULT = 100

GAMMA_PLUS = 0.8
GAMMA_MINUS = 0.2
EPSILON = 0.1

# Critical belief gc: agents commit once their belief exceeds gc. The cost
# lambda is pinned so that the immediate net return r(gc) - lambda = 0 at gc.
GC = 0.7
LAMBDA_COST = GAMMA_PLUS * GC + GAMMA_MINUS * (1 - GC)   # = 0.62

# Critical threshold in theta-space (where the optimal policy steps up).
THETA_C = (LAMBDA_COST - GAMMA_MINUS) / (GAMMA_PLUS - GAMMA_MINUS)   # = 0.70

# Scout cutoff used by the GA figures (a "scout" is bold: threshold < 1/2).
SCOUT_CUTOFF = 0.5
