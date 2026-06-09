"""Shared boilerplate for sim scripts."""

import argparse
import os
import sys

# Make repo root importable when run as `python -m sims.simX`
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

OUTPUT = os.path.join(REPO, "output")
FIGS = os.path.join(REPO, "figs")
os.makedirs(OUTPUT, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)


def parse_args(extra: list[tuple[str, dict]] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true",
                   help="Smaller sweep / MC for smoke testing.")
    p.add_argument("--n-jobs", type=int, default=-1,
                   help="joblib parallel jobs (default -1 = all cores).")
    p.add_argument("--seed", type=int, default=0,
                   help="Base seed for RNG streams.")
    if extra:
        for name, kw in extra:
            p.add_argument(name, **kw)
    return p.parse_args()
