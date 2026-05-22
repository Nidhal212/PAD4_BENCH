#!/usr/bin/env python3
"""
d03 — Weight distribution diagnostic.

Question this answers: does ml_weight contribute meaningful information,
or has it collapsed toward the mean (which would make weighted training
equivalent to unweighted)?

Outputs:
  manuscript/figures/d03_weight_distribution.png — three-panel figure:
    (a) ml_weight histogram with mean/median/std
    (b) label_uncertainty_score_v2 histogram (the main input to ml_weight)
    (c) ml_weight broken down by fidelity_level (boxplot)
  results/diagnostics/d03_weight_distribution.json — headline numbers
"""
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Local import (works whether run as `python scripts/diagnostics/d03_*.py`
# or `python -m scripts.diagnostics.d03_*`)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    OKABE_ITO, default_input_dir, log_table, save_figure, save_summary,
    setup_logging, setup_matplotlib,
)

NAME = "d03_weight_distribution"


def _coefficient_of_variation(s: pd.Series) -> float:
    """CV = std / mean. < 0.1 typically means "essentially constant"."""
    s = s.dropna()
    if len(s) == 0 or s.mean() == 0:
        return float("nan")
    return float(s.std() / s.mean())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None,
                        help="Path to pad_t1_ic50_aggregated.csv")
    args = parser.parse_args(argv)

    setup_logging()
    plt = setup_matplotlib()

    csv = args.input or (default_input_dir() / "pad_t1_ic50_aggregated.csv")
    if not csv.exists():
        logging.error(f"Cannot find {csv}")
        return 1
    logging.info(f"Loading {csv}")
    df = pd.read_csv(csv)
    logging.info(f"  {len(df):,} compounds")

    if "ml_weight" not in df.columns:
        logging.error("ml_weight column missing — re-run curation pipeline")
        return 1

    w   = df["ml_weight"].dropna()
    lu  = df.get("label_uncertainty_score_v2", pd.Series(dtype=float)).dropna()
    fid = df.get("fidelity_level", pd.Series(dtype=str))

    # ── Headline numbers ─────────────────────────────────────────────
    cv      = _coefficient_of_variation(w)
    iqr_w   = float(w.quantile(0.75) - w.quantile(0.25))
    p01, p99 = float(w.quantile(0.01)), float(w.quantile(0.99))
    n_at_min = int((w <= w.min() + 1e-6).sum())
    n_at_max = int((w >= w.max() - 1e-6).sum())

    log_table([
        ("compounds",                f"{len(w):,}"),
        ("ml_weight mean",           f"{w.mean():.4f}"),
        ("ml_weight median",         f"{w.median():.4f}"),
        ("ml_weight std",            f"{w.std():.4f}"),
        ("ml_weight min / max",      f"{w.min():.4f} / {w.max():.4f}"),
        ("ml_weight IQR",            f"{iqr_w:.4f}"),
        ("ml_weight 1%–99% range",   f"{p01:.3f}–{p99:.3f}"),
        ("coefficient of variation", f"{cv:.3f}  "
                                     f"({'OK' if cv >= 0.20 else 'COLLAPSED — weighting may have low effect'})"),
        ("compounds at floor (0.01)", f"{n_at_min:,}"),
        ("compounds at ceiling (1.0)", f"{n_at_max:,}"),
    ], ["Metric", "Value"])

    # ── Figure ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # (a) ml_weight histogram
    ax = axes[0]
    ax.hist(w, bins=50, color=OKABE_ITO[5], edgecolor="white", linewidth=0.4)
    ax.axvline(w.mean(),   color=OKABE_ITO[6], linewidth=1.5,
               label=f"mean = {w.mean():.3f}")
    ax.axvline(w.median(), color=OKABE_ITO[3], linewidth=1.5, linestyle="--",
               label=f"median = {w.median():.3f}")
    ax.set_xlabel("ml_weight")
    ax.set_ylabel("compounds")
    ax.set_title(f"(a) ml_weight distribution\n(CV = {cv:.2f})")
    ax.legend(loc="upper right")
    ax.set_xlim(0, 1.05)

    # (b) label_uncertainty_score_v2 — the dominant input
    ax = axes[1]
    if len(lu) > 0:
        ax.hist(lu, bins=50, color=OKABE_ITO[2], edgecolor="white", linewidth=0.4)
        ax.axvline(lu.mean(), color=OKABE_ITO[6], linewidth=1.5,
                   label=f"mean = {lu.mean():.3f}")
        ax.set_xlabel("label_uncertainty_score_v2")
        ax.set_ylabel("compounds")
        ax.set_title("(b) label uncertainty\n(primary driver of ml_weight)")
        ax.legend(loc="upper right")
    else:
        ax.text(0.5, 0.5, "label_uncertainty_score_v2\nnot available",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("(b) label uncertainty")

    # (c) ml_weight by fidelity tier
    ax = axes[2]
    fid_order = ["T1_high", "T1_confirmed", "T1_standard"]
    by_fid = []
    fid_labels = []
    for level in fid_order:
        mask = (fid == level)
        if mask.any():
            by_fid.append(w[mask].values)
            fid_labels.append(f"{level}\n(n={mask.sum():,})")
    if by_fid:
        # tick_labels (matplotlib >= 3.9) replaced labels; support both.
        try:
            bp = ax.boxplot(by_fid, tick_labels=fid_labels, widths=0.55,
                            patch_artist=True, showfliers=False)
        except TypeError:
            bp = ax.boxplot(by_fid, labels=fid_labels, widths=0.55,
                            patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"],
                                [OKABE_ITO[3], OKABE_ITO[2], OKABE_ITO[1]]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel("ml_weight")
        ax.set_title("(c) ml_weight by fidelity tier")
        ax.set_ylim(0, 1.05)
    else:
        ax.text(0.5, 0.5, "fidelity_level not available",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("(c) ml_weight by fidelity tier")

    fig.tight_layout()
    save_figure(fig, NAME)
    plt.close(fig)

    # ── Summary JSON ──────────────────────────────────────────────────
    summary = {
        "n_compounds": int(len(w)),
        "ml_weight_mean":   round(float(w.mean()),   4),
        "ml_weight_median": round(float(w.median()), 4),
        "ml_weight_std":    round(float(w.std()),    4),
        "ml_weight_min":    round(float(w.min()),    4),
        "ml_weight_max":    round(float(w.max()),    4),
        "ml_weight_iqr":    round(iqr_w, 4),
        "ml_weight_p01":    round(p01, 4),
        "ml_weight_p99":    round(p99, 4),
        "coefficient_of_variation": round(cv, 4),
        "n_at_floor":   n_at_min,
        "n_at_ceiling": n_at_max,
        "label_uncertainty_v2_mean":   round(float(lu.mean()),   4) if len(lu) else None,
        "label_uncertainty_v2_median": round(float(lu.median()), 4) if len(lu) else None,
        "by_fidelity": {
            level: {
                "n": int((fid == level).sum()),
                "ml_weight_mean":   round(float(w[fid == level].mean()), 4)
                                    if (fid == level).any() else None,
                "ml_weight_median": round(float(w[fid == level].median()), 4)
                                    if (fid == level).any() else None,
            }
            for level in ["T1_high", "T1_confirmed", "T1_standard"]
        },
        "interpretation": (
            "OK — weighting contributes meaningful signal (CV >= 0.20)"
            if cv >= 0.20 else
            "WARNING — ml_weight has low variability; weighted training may "
            "produce results similar to unweighted. Re-examine LU computation "
            "or weight aggregation."
        ),
    }
    save_summary(summary, NAME)
    return 0


if __name__ == "__main__":
    sys.exit(main())
