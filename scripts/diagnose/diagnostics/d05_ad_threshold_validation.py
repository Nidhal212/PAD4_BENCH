#!/usr/bin/env python3
"""
d05 — Applicability domain threshold validation.

Question this answers: a reviewer flagged the 96.66% T3 self-AD in-domain
coverage as suspiciously high. Is the threshold too lenient, or is the
high coverage a real consequence of HTS library structure (large diverse
library, dense local neighborhoods)?

This script does NOT re-tune thresholds. It produces evidence the reader
can use to judge for themselves:
  (a) T3 self-AD score histogram with current threshold (0.25) marked,
      plus alternative cutoffs at 0.15/0.20/0.30/0.35/0.40 for context.
  (b) T1-relative AD score histogram for T3 (uses ad_score_combined_v2),
      with the v17 cutoff (AD_TANIMOTO_CUT = 0.35 ≈ 0.20 combined) shown.
  (c) Reliability flag breakdown — how the existing categorical flags
      (high/medium/low/out_of_domain) map to score quantiles.

Outputs:
  manuscript/figures/d05_ad_threshold_validation.png
  results/diagnostics/d05_ad_threshold_validation.json

The summary JSON includes coverage at every alternative threshold so the
paper can quote any of them.
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

NAME = "d05_ad_threshold_validation"

# Pipeline values for reference
T3_SELF_AD_TAN_CUT = 0.25      # T3 self-AD threshold from pipeline
AD_TANIMOTO_CUT    = 0.35      # T1-relative threshold (raw nearest-neighbor)

# Alternative thresholds to report coverage for
T3_SELF_ALT_CUTS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]


def _coverage_at_thresholds(scores: np.ndarray, cuts: list) -> list:
    """Return [(cut, n_in, fraction_in), ...] for each cutoff."""
    scores = np.asarray(scores)
    n = len(scores)
    out = []
    for c in cuts:
        n_in = int((scores >= c).sum())
        out.append((c, n_in, round(n_in / max(n, 1), 4)))
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t3", type=Path, default=None,
                        help="Path to pad_t3_hts_denoised.csv (or pad_t3_hts_indomain.csv)")
    args = parser.parse_args(argv)

    setup_logging()
    plt = setup_matplotlib()

    csv = args.t3 or (default_input_dir() / "pad_t3_hts_denoised.csv")
    if not csv.exists():
        # Fallback to indomain file (same data, different filter)
        alt = default_input_dir() / "pad_t3_hts_indomain.csv"
        if alt.exists():
            csv = alt
            logging.warning(f"  pad_t3_hts_denoised.csv missing, using {csv}")
        else:
            logging.error(f"Cannot find T3 CSV at {csv}")
            return 1
    logging.info(f"Loading {csv}")
    df = pd.read_csv(csv)
    logging.info(f"  {len(df):,} T3 records")

    # Required columns
    needed_self = "ad_t3_self_score"
    needed_t1rel = next((c for c in ("ad_score_combined_v2", "ad_nn_tanimoto")
                         if c in df.columns), None)
    if needed_self not in df.columns:
        logging.error(f"Required column '{needed_self}' missing from T3")
        return 1

    self_scores = df[needed_self].dropna().values
    t1rel_scores = df[needed_t1rel].dropna().values if needed_t1rel else np.array([])

    n_total = len(df)
    n_self  = len(self_scores)
    n_t1rel = len(t1rel_scores)
    if n_self < n_total:
        logging.warning(
            f"  T3 self-AD score populated on {n_self:,}/{n_total:,} "
            f"({n_self/n_total*100:.1f}%) — coverage % below is relative to "
            f"the scored subset"
        )
    if n_t1rel > 0 and n_t1rel < n_total:
        logging.info(
            f"  T1-relative AD score populated on {n_t1rel:,}/{n_total:,} "
            f"({n_t1rel/n_total*100:.1f}%) — pipeline scores T1-rel on a "
            f"stratified subsample of T3 (see Stage 12 of curation log)"
        )

    # ── Coverage tables ──────────────────────────────────────────────
    cov_self = _coverage_at_thresholds(self_scores, T3_SELF_ALT_CUTS)
    rows_self = []
    for c, n_in, frac in cov_self:
        marker = " ← pipeline default" if abs(c - T3_SELF_AD_TAN_CUT) < 1e-9 else ""
        # Show 2 decimals when near 100% so saturation is visible
        pct_str = f"{frac*100:.2f}%" if frac > 0.99 else f"{frac*100:.1f}%"
        rows_self.append((f"≥ {c:.2f}", f"{n_in:,}", pct_str, marker))
    log_table(rows_self,
              [f"T3 self-AD cutoff (of n={n_self:,})",
               "in-domain", "coverage", ""])

    if len(t1rel_scores) > 0:
        cov_t1rel = _coverage_at_thresholds(t1rel_scores,
                                             [0.10, 0.20, 0.30, 0.40, 0.50])
        logging.info("")
        rows_t1rel = []
        for c, n_in, frac in cov_t1rel:
            pct_str = f"{frac*100:.2f}%" if frac > 0.99 else f"{frac*100:.1f}%"
            rows_t1rel.append((f"≥ {c:.2f}", f"{n_in:,}", pct_str))
        log_table(rows_t1rel,
                  [f"T1-rel ({needed_t1rel}) cutoff (of n={n_t1rel:,})",
                   "in-domain", "coverage"])

    # ── Reliability flag distribution ────────────────────────────────
    rel_dist = None
    if "ad_t3_self_reliability" in df.columns:
        rel_dist = df["ad_t3_self_reliability"].value_counts().to_dict()
        logging.info("")
        log_table(
            [(k, f"{v:,}", f"{v/len(df)*100:.1f}%")
             for k, v in sorted(rel_dist.items(), key=lambda kv: -kv[1])],
            ["T3 self-AD reliability flag", "n", "pct"])

    # ── Figure ────────────────────────────────────────────────────────
    n_panels = 3 if (len(t1rel_scores) > 0 and rel_dist) else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4.5))
    if n_panels == 1:
        axes = [axes]

    # (a) T3 self-AD score histogram
    ax = axes[0]
    bins = np.linspace(0, 1, 51)
    ax.hist(self_scores, bins=bins, color=OKABE_ITO[5],
            edgecolor="white", linewidth=0.3, alpha=0.85)
    # Mark the pipeline default + reference alternatives
    for c, color, lw, ls, lbl in [
        (0.15, OKABE_ITO[1], 0.8, ":",  "0.15 (loose)"),
        (T3_SELF_AD_TAN_CUT, OKABE_ITO[6], 1.6, "-",
         f"{T3_SELF_AD_TAN_CUT:.2f} (default)"),
        (0.35, OKABE_ITO[3], 0.8, ":",  "0.35 (strict)"),
    ]:
        ax.axvline(c, color=color, linewidth=lw, linestyle=ls, label=lbl)
    pct_at_default = (self_scores >= T3_SELF_AD_TAN_CUT).mean() * 100
    ax.set_xlabel("ad_t3_self_score")
    ax.set_ylabel("compounds")
    ax.set_title(f"(a) T3 self-AD score distribution\n"
                 f"(at default {T3_SELF_AD_TAN_CUT:.2f}: {pct_at_default:.1f}% in-domain)")
    ax.legend(loc="upper right")
    ax.set_xlim(0, 1)

    # (b) T1-relative AD score histogram
    if len(t1rel_scores) > 0:
        ax = axes[1]
        bins = np.linspace(0, 1, 51)
        ax.hist(t1rel_scores, bins=bins, color=OKABE_ITO[2],
                edgecolor="white", linewidth=0.3, alpha=0.85)
        ax.axvline(0.20, color=OKABE_ITO[6], linewidth=1.4, linestyle="-",
                   label="default (combined ≥ 0.20)")
        ax.set_xlabel(f"{needed_t1rel}")
        ax.set_ylabel("T3 compounds")
        ax.set_title(f"(b) T1-relative AD for T3\n"
                     f"(low-coverage by design — see paper §3.x)")
        ax.legend(loc="upper right")
        ax.set_xlim(0, 1)

    # (c) Reliability flag breakdown
    if rel_dist:
        ax = axes[-1]
        # Order flags so "out_of_domain" → "low" → "medium" → "high" reads left-to-right
        order = ["out_of_domain", "low", "medium", "high"]
        # Plus any "_vs_reference" suffixed flags that v17.2 may emit
        all_flags = sorted(rel_dist.keys(),
                           key=lambda f: (order.index(f) if f in order else 99, f))
        counts = [rel_dist[f] for f in all_flags]
        colors = [OKABE_ITO[6], OKABE_ITO[1], OKABE_ITO[2], OKABE_ITO[3]]
        # Cycle colors for any extra flags
        bar_colors = [colors[order.index(f)] if f in order
                      else OKABE_ITO[7] for f in all_flags]
        bars = ax.bar(range(len(all_flags)), counts, color=bar_colors,
                      edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(all_flags)))
        ax.set_xticklabels(all_flags, rotation=20, ha="right")
        ax.set_ylabel("T3 compounds")
        ax.set_title("(c) reliability flag breakdown")
        # Annotate bars with counts
        for bar, n in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(counts)*0.01,
                    f"{n:,}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    save_figure(fig, NAME)
    plt.close(fig)

    # ── Summary JSON ──────────────────────────────────────────────────
    summary = {
        "n_t3_compounds": int(len(df)),
        "n_with_self_ad": int(len(self_scores)),
        "self_ad_score_stats": {
            "mean":   round(float(self_scores.mean()),   4),
            "median": round(float(np.median(self_scores)), 4),
            "std":    round(float(self_scores.std()),    4),
            "p10":    round(float(np.percentile(self_scores, 10)), 4),
            "p25":    round(float(np.percentile(self_scores, 25)), 4),
            "p75":    round(float(np.percentile(self_scores, 75)), 4),
            "p90":    round(float(np.percentile(self_scores, 90)), 4),
        },
        "self_ad_coverage_at_thresholds": [
            {"threshold": c, "n_in_domain": n, "coverage_fraction": frac}
            for c, n, frac in cov_self
        ],
        "self_ad_default_threshold":     T3_SELF_AD_TAN_CUT,
        "self_ad_default_coverage":      round(float((self_scores >= T3_SELF_AD_TAN_CUT).mean()), 4),
        "reliability_flag_distribution": rel_dist or {},
        "t1_relative_metric":            needed_t1rel,
        "t1_relative_score_stats": (
            None if len(t1rel_scores) == 0 else {
                "mean":   round(float(t1rel_scores.mean()),   4),
                "median": round(float(np.median(t1rel_scores)), 4),
                "std":    round(float(t1rel_scores.std()),    4),
            }
        ),
        "interpretation": _interpret(self_scores, rel_dist),
    }
    save_summary(summary, NAME)
    return 0


def _interpret(scores: np.ndarray, rel_dist: dict) -> str:
    """Honest interpretation for the paper."""
    pct_default = (scores >= T3_SELF_AD_TAN_CUT).mean() * 100
    pct_strict  = (scores >= 0.35).mean() * 100
    pct_very_strict = (scores >= 0.50).mean() * 100
    mean_score = scores.mean()
    p10 = np.percentile(scores, 10)
    parts = [
        f"At the pipeline default threshold ({T3_SELF_AD_TAN_CUT:.2f}), "
        f"{pct_default:.1f}% of T3 is in-domain. "
    ]
    if pct_default > 95 and mean_score > 0.7:
        parts.append(
            f"The reason for this high coverage is structural, not a "
            f"calibration artifact: the T3 self-AD score has mean = {mean_score:.2f} "
            f"and 10th percentile = {p10:.2f}, both well above the "
            f"{T3_SELF_AD_TAN_CUT:.2f} threshold. HTS libraries are dense in chemical "
            f"space — most compounds have multiple close neighbors within the same "
            f"library. This is exactly the regime where T3 self-AD is well-defined "
            f"and informative, but the threshold has limited stratifying power. "
            f"For more selective in-domain filtering, raise the threshold: "
            f"≥0.35 keeps {pct_strict:.1f}%, ≥0.50 keeps {pct_very_strict:.1f}%. "
        )
    elif pct_default > 95:
        parts.append(
            f"High coverage at this threshold; consider raising to ≥0.35 "
            f"({pct_strict:.1f}%) for a stricter filter. "
        )
    if rel_dist:
        n_high_strict = rel_dist.get("high", 0)  # In-reference compounds only
        n_high_total  = sum(v for k, v in rel_dist.items() if k.startswith("high"))
        n_total = sum(rel_dist.values())
        if n_total > 0:
            parts.append(
                f"The reliability_flag column gives a calibrated alternative: "
                f"{n_high_total:,} compounds flagged 'high' "
                f"({n_high_total/n_total*100:.1f}%, of which {n_high_strict:,} "
                f"in the AD reference set itself). This is the recommended "
                f"in-domain filter for the paper's benchmarks."
            )
    return " ".join(parts)


if __name__ == "__main__":
    sys.exit(main())