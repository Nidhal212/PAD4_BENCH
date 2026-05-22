"""
PAD4-Bench diagnostic utilities (shared by d01–d05).

Conventions
-----------
* Each diagnostic script is independently runnable from anywhere.
* Inputs come from `data/processed/` (read-only).
* Figure outputs go to `manuscript/figures/`.
* JSON summaries go to `results/diagnostics/`.
* All RNG seeds are deterministic.

This module exposes:
* Path resolution that anchors to the repo root (same logic as the
  curation pipeline) so scripts work regardless of cwd.
* Matplotlib defaults tuned for Journal of Cheminformatics: 300 DPI,
  colorblind-safe palette (Okabe-Ito), no top/right spines.
* A `save_summary` helper that produces a flat JSON next to the figure.
"""
from __future__ import annotations
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

RANDOM_SEED = 42

# Okabe-Ito palette: colorblind-safe, 8 colors. Recommended for scientific figures.
# Reference: https://jfly.uni-koeln.de/color/
OKABE_ITO = [
    "#000000",  # 0 black
    "#E69F00",  # 1 orange
    "#56B4E9",  # 2 sky blue
    "#009E73",  # 3 bluish green
    "#F0E442",  # 4 yellow
    "#0072B2",  # 5 blue
    "#D55E00",  # 6 vermilion
    "#CC79A7",  # 7 reddish purple
]


# ── Path resolution ────────────────────────────────────────────────────
def find_repo_root() -> Path:
    """Walk up from this file looking for a directory containing data/raw."""
    here = Path(__file__).resolve().parent
    for parent in [here, *here.parents][:5]:
        if (parent / "data" / "raw").exists():
            return parent
    raise RuntimeError(
        f"Could not find repo root from {here}. Expected a parent dir "
        f"containing data/raw."
    )


def default_input_dir() -> Path:
    return find_repo_root() / "data" / "processed"


def default_figures_dir() -> Path:
    d = find_repo_root() / "manuscript" / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def default_summaries_dir() -> Path:
    d = find_repo_root() / "results" / "diagnostics"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── Matplotlib styling ────────────────────────────────────────────────
def setup_matplotlib():
    """Apply Journal-of-Cheminformatics-appropriate defaults.
    Call this at the top of every diagnostic script."""
    import matplotlib
    matplotlib.use("Agg")  # headless rendering, no display required
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi":          150,    # screen preview
        "savefig.dpi":         300,    # publication
        "savefig.bbox":        "tight",
        "savefig.pad_inches":  0.05,
        "font.family":         "sans-serif",
        "font.sans-serif":     ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size":           10,
        "axes.labelsize":      10,
        "axes.titlesize":      11,
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "axes.grid":           True,
        "grid.color":          "#dddddd",
        "grid.linestyle":      "-",
        "grid.linewidth":      0.5,
        "xtick.labelsize":     9,
        "ytick.labelsize":     9,
        "legend.fontsize":     9,
        "legend.frameon":      False,
    })
    return plt


# ── Logging ───────────────────────────────────────────────────────────
def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    np.random.seed(RANDOM_SEED)


# ── Summary writer ────────────────────────────────────────────────────
def save_summary(
    summary: dict,
    name: str,
    out_dir: Optional[Path] = None,
) -> Path:
    """
    Write a flat JSON summary next to the diagnostic's figure.

    `summary` is a flat dict of headline numbers (the kind a reviewer
    would want to quote). `generated` and `script` keys are added
    automatically.
    """
    out_dir = out_dir or default_summaries_dir()
    summary_full = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "script":    name,
        **summary,
    }
    path = Path(out_dir) / f"{name}.json"
    with open(path, "w") as f:
        json.dump(summary_full, f, indent=2, default=str)
    logging.info(f"  → {path.relative_to(find_repo_root())}")
    return path


# ── Figure writer ─────────────────────────────────────────────────────
def save_figure(fig, name: str, out_dir: Optional[Path] = None) -> Path:
    """Save a matplotlib figure to manuscript/figures/<name>.png."""
    out_dir = out_dir or default_figures_dir()
    path = Path(out_dir) / f"{name}.png"
    fig.savefig(path)
    logging.info(f"  → {path.relative_to(find_repo_root())}")
    return path


# ── Pretty-printed table for stdout ────────────────────────────────────
def log_table(rows, headers, indent: int = 2):
    """Print a table to stdout the same way the curation pipeline does."""
    pad = " " * indent
    widths = [max(len(str(r[i])) for r in [headers] + list(rows))
              for i in range(len(headers))]
    fmt = pad + "  ".join(f"{{:<{w}}}" for w in widths)
    sep = pad + "  ".join("-" * w for w in widths)
    logging.info(fmt.format(*headers))
    logging.info(sep)
    for row in rows:
        logging.info(fmt.format(*[str(v) for v in row]))
