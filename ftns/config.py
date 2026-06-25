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

# --- Asymmetric environmental switching (extension of the symmetric model) ---
# Two directional rates replace the single epsilon:
#   eps_minus = probability of LEAVING the high state,
#   eps_plus  = probability of LEAVING the low state (entering high).
# Default both to the symmetric EPSILON so all baseline results are unchanged.
# To model unequal dwell times set para.eps_p / para.eps_m on the parameter
# struct; everything else reads them through eps_pm()/g_star() below.
EPS_PLUS = EPSILON
EPS_MINUS = EPSILON

# No-information relaxation point. With no observations the predict step relaxes
# belief toward the stationary high-state probability
#   g* = eps_plus / (eps_plus + eps_minus),
# which equals 1/2 in the symmetric baseline.
G_STAR = EPS_PLUS / (EPS_PLUS + EPS_MINUS)

# Critical belief gc: agents commit once their belief exceeds gc. The cost
# lambda is pinned so that the immediate net return r(gc) - lambda = 0 at gc.
GC = 0.7
LAMBDA_COST = GAMMA_PLUS * GC + GAMMA_MINUS * (1 - GC)   # = 0.62

# Critical threshold in theta-space (where the optimal policy steps up).
THETA_C = (LAMBDA_COST - GAMMA_MINUS) / (GAMMA_PLUS - GAMMA_MINUS)   # = 0.70

# Scout cutoff used by the GA figures (a "scout" is bold: threshold < 1/2).
SCOUT_CUTOFF = 0.5


def eps_pm(para):
    """Return ``(eps_plus, eps_minus)`` for a parameter struct.

    Falls back to the symmetric ``para.epsilon`` when the directional rates
    ``eps_p`` / ``eps_m`` are not set, so symmetric callers are unaffected.
    Accepts a ``Para`` (dict subclass) or a plain ``dict``.
    """
    eps = para["epsilon"] if "epsilon" in para else None
    ep = para["eps_p"] if "eps_p" in para else eps
    em = para["eps_m"] if "eps_m" in para else eps
    return ep, em


def g_star(para):
    """Stationary high-state probability ``g* = eps_plus / (eps_plus + eps_minus)``.

    This is the belief the no-information predict step relaxes toward; it equals
    ``1/2`` in the symmetric case (and is bit-exactly ``0.5`` when
    ``eps_p == eps_m`` because ``x / (x + x)`` is exact in IEEE-754).
    """
    ep, em = eps_pm(para)
    total = ep + em
    return ep / total if total > 0 else 0.5
