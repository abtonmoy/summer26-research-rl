"""Small utilities and MATLAB-struct shims.

Provides:
- `Para`, `NumPara`: simple dataclasses mirroring MATLAB's `para`, `num_para`
  structs. Both accept arbitrary fields via `__init__(**kwargs)` and behave like
  attribute namespaces.
- `save_mat`, `load_mat`: thin wrappers around scipy.io that handle cell-array
  -> object-array round-trips and squeeze MATLAB's trailing singleton dims.
- `stoch_return`, `indiv`: MATLAB helpers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.io import loadmat, savemat

from .config import eps_pm, g_star


class Para(dict):
    """Mutable parameter struct with attribute access."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def copy(self) -> "Para":
        return Para(super().copy())


NumPara = Para  # same shape, distinct name for readability


def save_mat(path: str, data: dict[str, Any]) -> None:
    """Save dict to .mat. Cell-array-style ragged data should be wrapped in
    object arrays before calling."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    savemat(path, data, do_compression=True)


def _mat_to_py(obj):
    """Recursively convert scipy mat_struct -> dict, and np.ndarray of object
    dtype to list, so loaded data is python-native."""
    from scipy.io.matlab._mio5_params import mat_struct

    if isinstance(obj, mat_struct):
        return {f: _mat_to_py(getattr(obj, f)) for f in obj._fieldnames}
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        return [_mat_to_py(x) for x in obj.tolist()]
    return obj


def load_mat(path: str) -> dict[str, Any]:
    """Load .mat and strip scipy's metadata keys. Converts mat_struct -> dict."""
    raw = loadmat(path, squeeze_me=True, struct_as_record=False)
    return {k: _mat_to_py(v) for k, v in raw.items() if not k.startswith("__")}


# -------------------------------------------------------------------------
# Stochastic single-trial return under a fixed policy (mirrors ftn_stoch_return.m)
# -------------------------------------------------------------------------
def _act_policy(g: float, policy: np.ndarray, dg: float) -> int:
    gi = int(round(g / dg))  # MATLAB: round(1 + g/dg), 1-based -> 0-based
    return int(policy[gi])


def _update_belief(g: float, x: int, a: int, para: Para) -> float:
    gp, gm = para.gamma_p, para.gamma_m
    eps_p, eps_m = eps_pm(para)
    bgp = g * (gp**x) * (1 - gp) ** (a - x)
    bgm = (1 - g) * (gm**x) * (1 - gm) ** (a - x)
    return ((1 - eps_m) * bgp + eps_p * bgm) / (bgp + bgm)


def stoch_return(
    policy: np.ndarray,
    S: np.ndarray,
    para: Para,
    num_para: NumPara,
    rng: np.random.Generator | None = None,
) -> float:
    """Sum of x_t - lambda1 * a_t over a prescribed state sequence S."""
    if rng is None:
        rng = np.random.default_rng()
    dg = 1.0 / (num_para.gN - 1)
    T = len(S)
    gamma = np.array([para.gamma_m, para.gamma_p])

    sumR = 0.0
    g = g_star(para)
    for t in range(T):
        a = _act_policy(g, policy, dg)
        gammat = gamma[int((S[t] + 1) // 2)]  # state in {-1,1} -> idx
        x = int(rng.binomial(a, gammat)) if a > 0 else 0
        g = _update_belief(g, x, a, para)
        sumR += x - para.lambda1 * a
    return sumR


def indiv(x: np.ndarray | float, v: np.ndarray, idx: np.ndarray | int) -> np.ndarray:
    """Replace v[idx] with x and return v."""
    v = np.asarray(v).copy()
    v[idx] = x
    return v
