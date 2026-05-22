#!/usr/bin/env python3
"""
d02 — Activity distributions and source drift.

Question this answers: are pIC50 distributions consistent across the
three data sources (BindingDB / ChEMBL / PubChem), or is there hidden
selection bias that should be disclosed in the paper?

Method:
  - Plot the overall T1 pIC50 histogram with KDE.
  - Plot per-source KDEs on the same axes for visual comparison.
  - Run pairwise two-sample Kolmogorov–Smirnov tests across sources.
  - Compute Cohen's d as an effect-size measure (KS only tells you if
    distributions differ — Cohen's d tells you by how much in std-units).

Outputs:
  manuscript/figures/d02_activity_distributions.png — two-panel figure:
    (a) overall pIC50 histogram + KDE + active/inactive cutoff
    (b) per-source KDE overlay
  results/diagnostics/d02_activity_distributions.json — KS p-values,
    Cohen's d, mean/median/std per source.

Note on `source_list` semantics:
  In pad_t1_ic50_aggregated.csv, source_list is a comma-separated string
  (e.g., "BindingDB,PubChem") because a single aggregated compound can
  combine measurements from multiple sources. For per-source comparison
  we use compounds that are FROM EXACTLY ONE source (single-source
  records). This is the cleanest comparison; mixed-source compounds are
  reported separately in the JSON for completeness.
"""
from __future__ import annotations
import argparse
import logging
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    OKABE_ITO, default_input_dir, log_table, save_figure, save_summary,
    setup_logging, setup_matplotlib,
)

NAME = "d02_activity_distributions"

# Map source label → palette index (consistent across all panels)
SOURCE_COLOR = {
    "BindingDB": OKABE_ITO[5],   # blue
    "ChEMBL":    OKABE_ITO[3],   # green
    "PubChem":   OKABE_ITO[6],   # vermilion
}
ACTIVE_PIC50_THRESHOLD = 6.0   # matches PACTIVITY_THRESHOLD in pipeline


def _gaussian_kde(values: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    """Tiny Gaussian KDE; avoids the SciPy dependency."""
    from scipy.stats import gaussian_kde
    return gaussian_kde(values)(x_grid)


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled-SD Cohen's d for two independent samples."""
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    s_pooled = np.sqrt(
        ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
        / (len(a) + len(b) - 2)
    )
    if s_pooled == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / s_pooled)


def _split_by_source(df: pd.DataFrame) -> dict:
    """
    Partition T1 compounds by their source_list value.
    Returns {"single_source": {source: pIC50_array, ...},
             "mixed":         pIC50_array,
             "all":           pIC50_array}
    """
    pic50 = df["pIC50"].dropna()
    source = df.loc[pic50.index, "source_list"].astype(str)

    single = {}
    for src in ("BindingDB", "ChEMBL", "PubChem"):
        mask = (source == src)
        if mask.any():
            single[src] = pic50[mask].values

    mixed_mask = source.str.contains(",", na=False)
    mixed = pic50[mixed_mask].values

    return {
        "single_source": single,
        "mixed":         mixed,
        "all":           pic50.values,
    }


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

    if "pIC50" not in df.columns or "source_list" not in df.columns:
        logging.error("Required columns missing: pIC50 and/or source_list")
        return 1

    parts = _split_by_source(df)
    pic_all = parts["all"]
    pic_mix = parts["mixed"]
    sources = parts["single_source"]

    # ── Headline numbers ─────────────────────────────────────────────
    rows = [
        ("All T1 compounds",
         f"{len(pic_all):,}",
         f"{pic_all.mean():.3f}", f"{np.median(pic_all):.3f}",
         f"{pic_all.std():.3f}",  f"{(pic_all >= ACTIVE_PIC50_THRESHOLD).mean()*100:.1f}%"),
    ]
    for src in ("BindingDB", "ChEMBL", "PubChem"):
        if src in sources:
            v = sources[src]
            rows.append((
                f"  single-source: {src}",
                f"{len(v):,}",
                f"{v.mean():.3f}", f"{np.median(v):.3f}",
                f"{v.std():.3f}",
                f"{(v >= ACTIVE_PIC50_THRESHOLD).mean()*100:.1f}%",
            ))
    if len(pic_mix):
        rows.append((
            "  mixed-source (≥2)",
            f"{len(pic_mix):,}",
            f"{pic_mix.mean():.3f}", f"{np.median(pic_mix):.3f}",
            f"{pic_mix.std():.3f}",
            f"{(pic_mix >= ACTIVE_PIC50_THRESHOLD).mean()*100:.1f}%",
        ))
    log_table(rows, ["Subset", "n", "mean", "median", "std", "%active"])

    # ── Pairwise KS + Cohen's d ──────────────────────────────────────
    from scipy.stats import ks_2samp
    pair_results = []
    rows_pair = []
    src_keys = list(sources.keys())
    MIN_N_PER_GROUP = 30   # below this, Cohen's d is unstable; report n/a
    for a, b in combinations(src_keys, 2):
        va, vb = sources[a], sources[b]
        ks_stat, ks_p = ks_2samp(va, vb)
        d = _cohens_d(va, vb)
        n_min = min(len(va), len(vb))
        if n_min < MIN_N_PER_GROUP:
            interp = f"insufficient (n_min={n_min})"
            d_for_summary = None
        else:
            # Effect-size interpretation per Cohen (1988): |d| < 0.2 = negligible,
            # 0.2–0.5 = small, 0.5–0.8 = medium, ≥ 0.8 = large.
            interp = ("negligible" if abs(d) < 0.2 else
                      "small"      if abs(d) < 0.5 else
                      "medium"     if abs(d) < 0.8 else
                      "large")
            d_for_summary = round(float(d), 4)
        rows_pair.append((
            f"{a} vs {b}",
            f"{len(va):,} / {len(vb):,}",
            f"{ks_stat:.3f}",
            f"{ks_p:.2e}",
            f"{d:+.3f}" if n_min >= MIN_N_PER_GROUP else "n/a",
            interp,
        ))
        pair_results.append({
            "pair": f"{a}_vs_{b}",
            "n_a": int(len(va)), "n_b": int(len(vb)),
            "ks_statistic": round(float(ks_stat), 4),
            "ks_pvalue":    float(ks_p),
            "cohens_d":     d_for_summary,
            "effect_size":  interp,
        })
    if rows_pair:
        logging.info("")
        log_table(rows_pair,
                  ["Pair", "n_a / n_b", "KS stat", "KS p", "Cohen's d", "effect"])

    # ── Figure ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # (a) Overall histogram + KDE
    ax = axes[0]
    bin_edges = np.arange(np.floor(pic_all.min()*2)/2,
                          np.ceil(pic_all.max()*2)/2 + 0.25, 0.25)
    ax.hist(pic_all, bins=bin_edges, color=OKABE_ITO[5],
            edgecolor="white", linewidth=0.4, alpha=0.85)
    # KDE on a secondary axis (density scale)
    x_grid = np.linspace(pic_all.min(), pic_all.max(), 400)
    kde = _gaussian_kde(pic_all, x_grid)
    ax2 = ax.twinx()
    ax2.plot(x_grid, kde, color=OKABE_ITO[6], linewidth=1.5)
    ax2.set_ylabel("density", color=OKABE_ITO[6])
    ax2.tick_params(axis="y", labelcolor=OKABE_ITO[6])
    ax2.spines["top"].set_visible(False)
    ax2.set_ylim(bottom=0)
    ax.axvline(ACTIVE_PIC50_THRESHOLD, color="black", linewidth=1.0,
               linestyle="--", label=f"active cutoff (pIC50 ≥ {ACTIVE_PIC50_THRESHOLD})")
    ax.set_xlabel("pIC50")
    ax.set_ylabel("compounds")
    ax.set_title(f"(a) overall T1 pIC50 distribution\n(n = {len(pic_all):,})")
    ax.legend(loc="upper left")

    # (b) Per-source KDEs overlaid
    ax = axes[1]
    x_grid = np.linspace(2, 12, 400)
    for src, values in sources.items():
        if len(values) < 30:   # KDE unreliable on small samples
            ax.plot([], [], label=f"{src} (n={len(values):,}, too few for KDE)")
            continue
        kde = _gaussian_kde(values, x_grid)
        ax.plot(x_grid, kde, color=SOURCE_COLOR.get(src, "black"),
                linewidth=1.5, label=f"{src} (n={len(values):,})")
    if len(pic_mix) >= 30:
        kde = _gaussian_kde(pic_mix, x_grid)
        ax.plot(x_grid, kde, color="black", linewidth=1.0, linestyle=":",
                label=f"mixed-source (n={len(pic_mix):,})")
    ax.axvline(ACTIVE_PIC50_THRESHOLD, color="black", linewidth=1.0,
               linestyle="--", alpha=0.5)
    ax.set_xlabel("pIC50")
    ax.set_ylabel("density")
    ax.set_title("(b) per-source pIC50 KDEs\n(single-source compounds only)")
    ax.legend(loc="upper left")
    ax.set_xlim(pic_all.min() - 0.2, pic_all.max() + 0.2)

    fig.tight_layout()
    save_figure(fig, NAME)
    plt.close(fig)

    # ── Summary JSON ──────────────────────────────────────────────────
    def _stats(v):
        if len(v) == 0:
            return None
        return {
            "n":                int(len(v)),
            "mean":             round(float(np.mean(v)),   4),
            "median":           round(float(np.median(v)), 4),
            "std":              round(float(np.std(v)),    4),
            "min":              round(float(np.min(v)),    4),
            "max":              round(float(np.max(v)),    4),
            "active_fraction":  round(float((v >= ACTIVE_PIC50_THRESHOLD).mean()), 4),
        }

    summary = {
        "n_compounds":            int(len(pic_all)),
        "active_pic50_threshold": ACTIVE_PIC50_THRESHOLD,
        "all":     _stats(pic_all),
        "by_single_source": {src: _stats(v) for src, v in sources.items()},
        "mixed_source": _stats(pic_mix),
        "pairwise_drift_tests": pair_results,
        "interpretation": _interpret_drift(pair_results),
    }
    save_summary(summary, NAME)
    return 0


def _interpret_drift(pair_results) -> str:
    """Plain-language interpretation for the JSON summary."""
    if not pair_results:
        return "Insufficient data for pairwise drift tests."
    # Only consider pairs with enough samples for a meaningful effect-size estimate.
    powered = [r for r in pair_results
               if not str(r["effect_size"]).startswith("insufficient")]
    if not powered:
        return ("All pair comparisons were under-powered (one source has fewer "
                "than 30 single-source compounds). No reliable drift assessment.")
    has_large  = any(r["effect_size"] == "large"  for r in powered)
    has_medium = any(r["effect_size"] == "medium" for r in powered)
    n_skipped  = len(pair_results) - len(powered)
    note = (f" ({n_skipped} pair(s) skipped due to small sample size.)"
            if n_skipped else "")
    if has_large:
        return (
            "WARNING: large source drift detected (Cohen's |d| >= 0.8 in at "
            "least one well-powered pair). Disclose in §3; consider "
            "source-stratified analyses or restricting to a single source for "
            "benchmarks." + note
        )
    if has_medium:
        return (
            "Moderate source drift (medium Cohen's d in at least one pair). "
            "Worth disclosing; per-source RMSE/R² in benchmarks recommended." + note
        )
    return (
        "OK: no large or medium-magnitude drift between adequately-sampled "
        "sources. Pooling across sources is defensible." + note
    )


if __name__ == "__main__":
    sys.exit(main())