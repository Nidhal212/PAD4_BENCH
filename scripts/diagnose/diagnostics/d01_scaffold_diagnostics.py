#!/usr/bin/env python3
"""
d01 — Scaffold diagnostics.

Question this answers: how scaffold-imbalanced is the T1 dataset? This
informs both the splits (singleton scaffolds can't be used in
scaffold-CV) and the §4 framing (claim "scaffold-balanced benchmark"
needs evidence).

Pipeline already stores:
  stereo_stripped_scaffold (Bemis-Murcko scaffold, stereo stripped)
  scaffold_id              (integer per unique scaffold)
  scaffold_size            (number of compounds with this scaffold)
  is_frequent_scaffold     (>= FREQUENT_SCAFFOLD_MIN, default 10)
  scaffold_diversity_score (1 - relative frequency)

This script:
  - Counts unique scaffolds, singleton scaffolds, and frequent scaffolds.
  - Plots the scaffold-size distribution (log-log).
  - Plots the cumulative compound coverage by scaffold rank
    (the "Lorenz curve" of scaffolds — shows concentration).
  - Tables the top-N most frequent scaffolds with example SMILES.

Outputs:
  manuscript/figures/d01_scaffold_diagnostics.png — three-panel:
    (a) scaffold-size histogram (log-log)
    (b) cumulative coverage by scaffold rank
    (c) compound count by activity bin × top-5 scaffold
  results/diagnostics/d01_scaffold_diagnostics.json
"""
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    OKABE_ITO, default_input_dir, log_table, save_figure, save_summary,
    setup_logging, setup_matplotlib,
)

NAME = "d01_scaffold_diagnostics"

ACTIVE_PIC50_THRESHOLD = 6.0
FREQUENT_SCAFFOLD_MIN  = 10   # matches pipeline default


def _gini(values: np.ndarray) -> float:
    """
    Gini coefficient of an array (concentration measure).
    0 = uniform, 1 = all in one bucket. We use it on scaffold sizes.
    """
    v = np.sort(np.asarray(values, dtype=float))
    n = len(v)
    if n == 0 or v.sum() == 0:
        return float("nan")
    cum = np.cumsum(v)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None,
                        help="Path to pad_t1_ic50_aggregated.csv")
    parser.add_argument("--top-n", type=int, default=10,
                        help="Number of top scaffolds to show in summary")
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

    if "stereo_stripped_scaffold" not in df.columns:
        logging.error("stereo_stripped_scaffold column missing — re-run curation")
        return 1

    # Treat empty scaffolds as "no scaffold" (acyclic compounds)
    scaf = df["stereo_stripped_scaffold"].fillna("").astype(str)
    has_scaffold = scaf != ""

    # Scaffold size distribution: how many compounds per unique scaffold
    scaf_counts = scaf[has_scaffold].value_counts()
    sizes = scaf_counts.values   # array of compound counts per scaffold

    n_compounds   = len(df)
    n_acyclic     = int((~has_scaffold).sum())
    n_unique_scaf = len(scaf_counts)
    n_singletons  = int((sizes == 1).sum())
    n_frequent    = int((sizes >= FREQUENT_SCAFFOLD_MIN).sum())
    n_in_freq     = int(scaf_counts[scaf_counts >= FREQUENT_SCAFFOLD_MIN].sum())
    gini = _gini(sizes)

    # ── Headline numbers ─────────────────────────────────────────────
    log_table([
        ("compounds",                       f"{n_compounds:,}"),
        ("acyclic compounds (no scaffold)", f"{n_acyclic:,}"),
        ("unique scaffolds",                f"{n_unique_scaf:,}"),
        ("singleton scaffolds",
         f"{n_singletons:,} ({n_singletons/max(n_unique_scaf,1)*100:.1f}% of scaffolds)"),
        ("frequent scaffolds (≥10)",
         f"{n_frequent:,} ({n_frequent/max(n_unique_scaf,1)*100:.1f}% of scaffolds)"),
        ("compounds in frequent scaffolds",
         f"{n_in_freq:,} ({n_in_freq/max(n_compounds,1)*100:.1f}% of compounds)"),
        ("scaffolds covering 50% of compounds",
         f"{_min_scaffolds_covering(sizes, 0.50):,}"),
        ("scaffolds covering 80% of compounds",
         f"{_min_scaffolds_covering(sizes, 0.80):,}"),
        ("Gini coefficient of scaffold sizes", f"{gini:.3f}"),
        ("max scaffold size", f"{int(sizes.max()):,}"),
        ("median scaffold size", f"{int(np.median(sizes)):,}"),
    ], ["Metric", "Value"])

    # ── Top-N scaffolds with activity stats ───────────────────────────
    pic50 = df.set_index("inchikey")["pIC50"] if "pIC50" in df.columns else None
    rows_top = []
    top_scaffolds = scaf_counts.head(args.top_n)
    logging.info("")
    logging.info(f"Top {args.top_n} scaffolds by frequency:")
    for rank, (scaf_smi, count) in enumerate(top_scaffolds.items(), 1):
        sub = df[scaf == scaf_smi]
        if pic50 is not None and "pIC50" in sub.columns:
            v = sub["pIC50"].dropna()
            mean_pic = f"{v.mean():.2f}" if len(v) else "—"
            pct_active = f"{(v >= ACTIVE_PIC50_THRESHOLD).mean()*100:.0f}%" if len(v) else "—"
        else:
            mean_pic = pct_active = "—"
        # Truncate long SMILES for display
        smi_disp = scaf_smi if len(scaf_smi) <= 40 else scaf_smi[:37] + "..."
        rows_top.append((
            rank, f"{count:,}", mean_pic, pct_active, smi_disp,
        ))
    log_table(rows_top, ["rank", "n", "mean pIC50", "%active", "scaffold SMILES"])

    # ── Figure ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))

    # (a) Scaffold size distribution (log-log) — the long-tail signature
    ax = axes[0]
    size_hist = pd.Series(sizes).value_counts().sort_index()
    ax.scatter(size_hist.index, size_hist.values, s=12, color=OKABE_ITO[5],
               alpha=0.7, edgecolors="none")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvline(FREQUENT_SCAFFOLD_MIN, color=OKABE_ITO[6], linewidth=1.0,
               linestyle="--", label=f"frequent threshold (n ≥ {FREQUENT_SCAFFOLD_MIN})")
    ax.set_xlabel("compounds per scaffold (log)")
    ax.set_ylabel("number of scaffolds (log)")
    ax.set_title(f"(a) scaffold size distribution\n"
                 f"({n_unique_scaf:,} unique scaffolds, "
                 f"Gini = {gini:.2f})")
    ax.legend(loc="upper right")

    # (b) Cumulative compound coverage by scaffold rank (Lorenz curve)
    ax = axes[1]
    sizes_sorted = np.sort(sizes)[::-1]
    cum_compounds = np.cumsum(sizes_sorted) / sizes_sorted.sum()
    rank_frac = np.arange(1, len(sizes_sorted) + 1) / len(sizes_sorted)
    ax.plot(rank_frac, cum_compounds, color=OKABE_ITO[5], linewidth=1.5,
            label="actual")
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=0.8,
            label="uniform reference")
    # Mark the 50% and 80% points
    for target in [0.5, 0.8]:
        idx = np.argmax(cum_compounds >= target)
        ax.axhline(target, color=OKABE_ITO[1], linewidth=0.5, linestyle=":")
        ax.axvline(rank_frac[idx], color=OKABE_ITO[1], linewidth=0.5, linestyle=":")
    ax.set_xlabel("scaffold rank fraction (most-common first)")
    ax.set_ylabel("cumulative fraction of compounds")
    ax.set_title("(b) cumulative compound coverage\n"
                 "(Lorenz curve)")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)

    # (c) Activity distribution within top scaffolds
    ax = axes[2]
    if pic50 is not None and len(top_scaffolds) > 0:
        top5 = top_scaffolds.head(5)
        positions = []
        labels = []
        data_per_scaf = []
        for rank, (scaf_smi, count) in enumerate(top5.items(), 1):
            sub_pic = df.loc[scaf == scaf_smi, "pIC50"].dropna().values
            if len(sub_pic) == 0:
                continue
            positions.append(rank)
            labels.append(f"#{rank}\n(n={count:,})")
            data_per_scaf.append(sub_pic)
        if data_per_scaf:
            try:
                bp = ax.boxplot(data_per_scaf, positions=positions,
                                tick_labels=labels, widths=0.55,
                                patch_artist=True, showfliers=False)
            except TypeError:
                bp = ax.boxplot(data_per_scaf, positions=positions,
                                labels=labels, widths=0.55,
                                patch_artist=True, showfliers=False)
            for patch in bp["boxes"]:
                patch.set_facecolor(OKABE_ITO[5])
                patch.set_alpha(0.7)
            ax.axhline(ACTIVE_PIC50_THRESHOLD, color="black", linestyle="--",
                       linewidth=0.8, label="active cutoff")
            ax.set_xlabel("scaffold (rank)")
            ax.set_ylabel("pIC50")
            ax.set_title("(c) top-5 scaffolds: pIC50 distribution")
            ax.legend(loc="lower left", fontsize=8)
        else:
            ax.text(0.5, 0.5, "no pIC50 data", ha="center", va="center",
                    transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, "pIC50 unavailable", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("(c) pIC50 by top scaffold")

    fig.tight_layout()
    save_figure(fig, NAME)
    plt.close(fig)

    # ── Summary JSON ──────────────────────────────────────────────────
    summary = {
        "n_compounds":       int(n_compounds),
        "n_acyclic":         int(n_acyclic),
        "n_unique_scaffolds": int(n_unique_scaf),
        "n_singleton_scaffolds": int(n_singletons),
        "singleton_fraction_of_scaffolds": round(n_singletons / max(n_unique_scaf, 1), 4),
        "n_frequent_scaffolds":  int(n_frequent),
        "frequent_threshold":    FREQUENT_SCAFFOLD_MIN,
        "n_compounds_in_frequent_scaffolds": int(n_in_freq),
        "fraction_compounds_in_frequent_scaffolds":
            round(n_in_freq / max(n_compounds, 1), 4),
        "scaffolds_covering_50pct": int(_min_scaffolds_covering(sizes, 0.50)),
        "scaffolds_covering_80pct": int(_min_scaffolds_covering(sizes, 0.80)),
        "scaffold_size_gini":       round(gini, 4),
        "max_scaffold_size":        int(sizes.max()),
        "median_scaffold_size":     int(np.median(sizes)),
        "top_scaffolds": [
            {
                "rank": rank,
                "n_compounds": int(count),
                "scaffold_smiles": scaf_smi,
                "mean_pic50": (
                    round(float(df.loc[scaf == scaf_smi, "pIC50"].dropna().mean()), 3)
                    if pic50 is not None and "pIC50" in df.columns
                    and df.loc[scaf == scaf_smi, "pIC50"].dropna().any()
                    else None
                ),
            }
            for rank, (scaf_smi, count) in enumerate(top_scaffolds.items(), 1)
        ],
        "interpretation": _interpret(n_singletons, n_unique_scaf, gini, n_frequent,
                                      n_compounds, n_in_freq),
    }
    save_summary(summary, NAME)
    return 0


def _min_scaffolds_covering(sizes: np.ndarray, fraction: float) -> int:
    """Smallest number of scaffolds (ranked by size, descending) that cover `fraction` of compounds."""
    if len(sizes) == 0:
        return 0
    sorted_sizes = np.sort(sizes)[::-1]
    cum = np.cumsum(sorted_sizes) / sorted_sizes.sum()
    return int(np.argmax(cum >= fraction)) + 1


def _interpret(n_singletons, n_unique, gini, n_frequent, n_compounds, n_in_freq) -> str:
    parts = []
    sing_frac = n_singletons / max(n_unique, 1)
    if sing_frac > 0.7:
        parts.append(
            f"Singleton-dominated chemistry: {sing_frac*100:.0f}% of unique "
            f"scaffolds are represented by a single compound. Many compounds "
            f"will be unsplittable in scaffold-CV — consider random-CV as a "
            f"complement, or report scaffold-CV only on the non-singleton subset."
        )
    elif sing_frac > 0.5:
        parts.append(
            f"Moderate singleton fraction ({sing_frac*100:.0f}% of scaffolds "
            f"are singletons). Scaffold-CV will be feasible but with reduced "
            f"effective sample size."
        )
    else:
        parts.append(
            f"Singleton fraction ({sing_frac*100:.0f}%) is low; "
            f"scaffold-CV is well-supported."
        )
    if gini > 0.6:
        parts.append(
            f"Scaffold-size distribution is concentrated (Gini = {gini:.2f}); "
            f"a small number of scaffolds dominate. The pipeline's "
            f"scaffold_aware_weight + frequent_scaffold capping is the right "
            f"mitigation."
        )
    if n_in_freq / max(n_compounds, 1) > 0.40:
        parts.append(
            f"{n_in_freq/n_compounds*100:.0f}% of compounds belong to "
            f"frequent scaffolds (≥{FREQUENT_SCAFFOLD_MIN}) — the dataset "
            f"is series-biased and benchmarks should report scaffold-aware "
            f"metrics."
        )
    return " ".join(parts)


if __name__ == "__main__":
    sys.exit(main())
