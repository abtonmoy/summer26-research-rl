"""Shared style + path helpers for figure scripts."""

import os
import sys

import matplotlib
# Headless / file-save only. Override by setting MPLBACKEND env var before import.
if os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

OUTPUT = os.path.join(REPO, "output")
FIGS = os.path.join(REPO, "figs")
os.makedirs(FIGS, exist_ok=True)

# Okabe-Ito palette: black, orange, sky blue, bluish green, yellow, blue,
# vermillion, reddish purple, gray
OKABE_ITO = {
    "black":  (0.000, 0.000, 0.000),
    "orange": (0.902, 0.624, 0.000),
    "sky":    (0.337, 0.706, 0.914),
    "green":  (0.000, 0.620, 0.451),
    "yellow": (0.941, 0.894, 0.259),
    "blue":   (0.000, 0.447, 0.698),
    "vermillion": (0.835, 0.369, 0.000),
    "purple": (0.800, 0.475, 0.655),
    "gray":   (0.600, 0.600, 0.600),
}


def savefig(fig, name: str):
    """Save figure as both .png and .pdf into figs/."""
    base = os.path.join(FIGS, name)
    fig.savefig(base + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    print(f"Saved: {base}.png")
