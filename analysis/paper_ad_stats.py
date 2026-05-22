#!/usr/bin/env python3
"""
PAD4_BENCH Paper 1 - applicability domain statistics (T14 replacement).

The binned table T14_applicability_domain.csv produced earlier suffers from
heavy bin-size imbalance (most splits have 70-90% of test compounds in the
'very near' Tanimoto bin), making per-bin ECE unreliable.

This script replaces it with three more defensible per-split statistics:

  (1) Correlation between max-train Tanimoto and absolute prediction error,
      with bootstrap 95% CI. Pearson and Spearman both reported.

  (2) Fraction of test compounds with high absolute error (|gap| > 0.5).

  (3) Fraction of those high-error compounds that fall in the 'very near' AD
      bin (max Tanimoto > 0.85). Quantifies how often the standard AD heuristic
      ('only trust predictions on in-domain compounds') would FAIL to flag
      high-error predictions.

Reads:  paper/tables/supp/T14_applicability_domain_per_compound.csv
Writes: paper/tables/supp/T14_applicability_domain_stats.{csv,md,tex}

Usage:
    cd /home/nidhal/PAD4_BENCH
    python paper_ad_stats.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_TBL_SUPP = PROJECT_ROOT / "paper" / "tables" / "supp"
PER_COMPOUND = OUT_TBL_SUPP / "T14_applicability_domain_per_compound.csv"

STRATEGIES = ["random", "scaffold", "confirmed", "lead_opt",
              "similarity", "cliff_aware"]
STRAT_PRETTY = {
    "random": "Random", "scaffold": "Scaffold", "confirmed": "Confirmed",
    "lead_opt": "Lead-Opt", "similarity": "Similarity",
    "cliff_aware": "Cliff-Aware",
}

HIGH_ERROR_THRESHOLD = 0.5
IN_DOMAIN_TANIMOTO   = 0.85
N_BOOTSTRAP = 1000
BOOT_SEED   = 42


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def bootstrap_corr(x: np.ndarray, y: np.ndarray,
                    method: str = "pearson",
                    n_boot: int = N_BOOTSTRAP,
                    seed: int = BOOT_SEED) -> tuple[float, float, float]:
    """Return (point, lo, hi) for the correlation."""
    if len(x) < 5 or np.std(x) == 0 or np.std(y) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    if method == "pearson":
        point = float(pearsonr(x, y).statistic)
    else:
        point = float(spearmanr(x, y).statistic)
    n = len(x)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xs = x[idx]
        ys = y[idx]
        if np.std(xs) == 0 or np.std(ys) == 0:
            continue
        try:
            if method == "pearson":
                boots.append(float(pearsonr(xs, ys).statistic))
            else:
                boots.append(float(spearmanr(xs, ys).statistic))
        except Exception:
            pass
    if not boots:
        return (point, float("nan"), float("nan"))
    boots = np.asarray(boots)
    return (point, float(np.percentile(boots, 2.5)),
            float(np.percentile(boots, 97.5)))


def fmt_corr(point: float, lo: float, hi: float) -> str:
    if np.isnan(point):
        return "--"
    return f"{point:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def save_table(df: pd.DataFrame, out_dir: Path, name: str,
                caption: str = "", label: str = "") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{name}.csv", index=False)
    md = df.to_markdown(index=False)
    (out_dir / f"{name}.md").write_text(
        f"**{caption}**\n\n{md}\n" if caption else md
    )
    cols = df.columns.tolist()
    col_spec = "l" + "r" * (len(cols) - 1)
    tex = [r"\begin{table}[ht]", r"\centering"]
    if caption:
        tex.append(rf"\caption{{{caption}}}")
    if label:
        tex.append(rf"\label{{{label}}}")
    tex.append(rf"\begin{{tabular}}{{{col_spec}}}")
    tex.append(r"\toprule")
    tex.append(" & ".join(str(c) for c in cols) + r" \\")
    tex.append(r"\midrule")
    for _, row in df.iterrows():
        cells = []
        for v in row.values:
            if isinstance(v, float):
                cells.append("--" if np.isnan(v) else f"{v:.4f}")
            else:
                cells.append(str(v).replace("_", r"\_").replace("%", r"\%"))
        tex.append(" & ".join(cells) + r" \\")
    tex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (out_dir / f"{name}.tex").write_text("\n".join(tex))
    print(f"    wrote {(out_dir / name).relative_to(PROJECT_ROOT)}.{{csv,md,tex}}",
          flush=True)


def main() -> int:
    if not PER_COMPOUND.exists():
        print(f"ERROR: {PER_COMPOUND} missing. Run paper_calibration_ad.py first.",
              flush=True)
        return 1

    df = pd.read_csv(PER_COMPOUND)
    log(f"loaded {len(df):,} test-compound records from {PER_COMPOUND.name}")

    rows = []
    for strategy in STRATEGIES:
        sub = df[df["strategy"] == strategy]
        n = len(sub)
        if n == 0:
            continue

        x = sub["max_train_tanimoto"].values.astype(np.float64)
        y = sub["abs_prob_gap"].values.astype(np.float64)

        # Statistic 1: correlations with CI
        p_pt, p_lo, p_hi = bootstrap_corr(x, y, method="pearson")
        s_pt, s_lo, s_hi = bootstrap_corr(x, y, method="spearman")

        # Statistic 2: fraction of high-error compounds
        high_err = (y > HIGH_ERROR_THRESHOLD)
        frac_high = float(high_err.mean())
        n_high = int(high_err.sum())

        # Statistic 3: of those high-error compounds, what fraction are in the
        # 'very near' AD bin (Tanimoto > 0.85)? This is the key AD-failure rate.
        if n_high > 0:
            in_domain_among_high = (x[high_err] > IN_DOMAIN_TANIMOTO)
            frac_indomain_high = float(in_domain_among_high.mean())
        else:
            frac_indomain_high = float("nan")

        # Statistic 4: among 'very near' compounds, what fraction are still
        # high-error? This is the false-confidence rate of AD filtering.
        in_domain_mask = (x > IN_DOMAIN_TANIMOTO)
        n_in_domain = int(in_domain_mask.sum())
        if n_in_domain > 0:
            high_among_indomain = (y[in_domain_mask] > HIGH_ERROR_THRESHOLD)
            frac_high_indomain = float(high_among_indomain.mean())
        else:
            frac_high_indomain = float("nan")

        # Statistic 5: median Tanimoto in this split (context for the
        # correlation - low-spread splits have weaker stats by construction)
        tan_median = float(np.median(x))
        tan_iqr = float(np.percentile(x, 75) - np.percentile(x, 25))

        rows.append({
            "Split": STRAT_PRETTY[strategy],
            "n test": n,
            "Tanimoto median (IQR)":   f"{tan_median:.3f} ({tan_iqr:.3f})",
            "Pearson r [95% CI]":      fmt_corr(p_pt, p_lo, p_hi),
            "Spearman ρ [95% CI]":     fmt_corr(s_pt, s_lo, s_hi),
            "High-error rate (|gap|>0.5)": f"{frac_high*100:.1f}% ({n_high}/{n})",
            "High-error in-domain (Tan>0.85)":
                "--" if np.isnan(frac_indomain_high)
                else f"{frac_indomain_high*100:.1f}%",
            "False-confidence rate":
                "--" if np.isnan(frac_high_indomain)
                else f"{frac_high_indomain*100:.1f}% ({int(high_among_indomain.sum()) if n_in_domain > 0 else 0}/{n_in_domain})",
        })

    out = pd.DataFrame(rows)
    print()
    print(out.to_string(index=False))
    print()

    save_table(out, OUT_TBL_SUPP, "T14_applicability_domain_stats",
               caption=(
                   "Applicability domain (AD) statistics per split. "
                   "Pearson and Spearman correlations between max train-test "
                   "Tanimoto (ECFP4) and absolute prediction error "
                   "(|y\\_true - y\\_proba|), with 1,000-sample bootstrap 95\\% "
                   "CIs. 'High-error rate' is the fraction of test compounds "
                   "with absolute error > 0.5. 'High-error in-domain' is the "
                   "fraction of those high-error compounds that fall in the "
                   "near-train AD bin (Tanimoto > 0.85), and the "
                   "'false-confidence rate' is the fraction of in-domain "
                   "compounds that are nevertheless high-error: an AD-filter "
                   "based on Tanimoto > 0.85 would still retain these "
                   "high-error predictions."),
               label="tab:T14_ad_stats")

    log("DONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
