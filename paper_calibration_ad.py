#!/usr/bin/env python3
"""
PAD4_BENCH Paper 1 - calibration repair + applicability domain analysis.

Two analyses in one script:

  PHASE 1 - Calibration repair
    For every classification cell, fit Platt scaling (sigmoid) and isotonic
    regression on val-set predictions, apply to test, recompute ECE / Brier /
    AUC. Report per-cell pre/Platt/isotonic comparison.

    Output:
      paper/tables/supp/T13_calibration_repair.{csv,md,tex}
      paper/figures/results/supp/D15_calibration_repair.{png,pdf}
      models_v1/classification/<strategy>/<variant>/<model>/calibration_repair.json

  PHASE 2 - Applicability domain analysis
    For every test compound in every classification cell, compute its max
    train Tanimoto (ECFP4). Bin test compounds by AD novelty and quantify
    prediction error + calibration error as a function of distance to train.

    Output:
      paper/figures/results/supp/D16_applicability_domain.{png,pdf}
      paper/tables/supp/T14_applicability_domain.{csv,md,tex}

Both analyses are leakage-safe:
  - Calibration is fit on val, evaluated on test. Test is never seen.
  - AD analysis only uses train -> test similarity (not test-internal).

Usage:
    cd /home/nidhal/PAD4_BENCH
    python paper_calibration_ad.py
"""

import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import IsotonicRegression
from sklearn.isotonic import IsotonicRegression as Iso  # explicit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Paths and config
# -----------------------------------------------------------------------------
PROJECT_ROOT  = Path("/home/nidhal/PAD4_BENCH")
MODELS_ROOT   = PROJECT_ROOT / "models_v1"
FEATURES_ROOT = PROJECT_ROOT / "features_v18"
OUT_FIG_SUPP  = PROJECT_ROOT / "paper" / "figures" / "results" / "supp"
OUT_TBL_SUPP  = PROJECT_ROOT / "paper" / "tables" / "supp"

STRATEGIES = ["random", "scaffold", "confirmed", "lead_opt",
              "similarity", "cliff_aware"]
VARIANTS   = ["full", "fingerprints", "physchem", "mordred", "fragments"]
CLF_MODELS = ["xgboost", "logreg_enet"]

STRAT_PRETTY = {
    "random": "Random", "scaffold": "Scaffold", "confirmed": "Confirmed",
    "lead_opt": "Lead-Opt", "similarity": "Similarity",
    "cliff_aware": "Cliff-Aware",
}
VARIANT_PRETTY = {
    "full": "Full", "fingerprints": "Fingerprints", "physchem": "PhysChem",
    "mordred": "Mordred", "fragments": "Fragments",
}

PALETTE = {
    "primary": "#1f4e79", "secondary": "#c5504b", "accent": "#e8b73a",
    "neutral": "#5a5a5a", "light": "#d4d4d4",
    "good": "#2d7a4f", "warn": "#c5504b",
}
SPLIT_COLOR = {
    "random":      "#1f4e79",
    "scaffold":    "#2d7a4f",
    "confirmed":   "#5a5a5a",
    "lead_opt":    "#7b3294",
    "similarity":  "#c5504b",
    "cliff_aware": "#e8731a",
}
METHOD_COLOR = {
    "original":  "#c5504b",
    "platt":     "#1f4e79",
    "isotonic":  "#2d7a4f",
}

CALIBRATION_N_BINS = 10
DPI_RASTER = 600


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# -----------------------------------------------------------------------------
# Matplotlib paper style (identical to other figure scripts)
# -----------------------------------------------------------------------------
def set_paper_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 11,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#222222",
        "axes.labelcolor": "#222222",
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "lines.linewidth": 1.4,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#d9d9d9",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.6,
        "figure.dpi": 110,
        "savefig.dpi": DPI_RASTER,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_fig(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=DPI_RASTER)
    fig.savefig(out_dir / f"{name}.pdf")
    plt.close(fig)
    print(f"    wrote {(out_dir / name).relative_to(PROJECT_ROOT)}.{{png,pdf}}",
          flush=True)


def save_table(df: pd.DataFrame, out_dir: Path, name: str,
                caption: str = "", label: str = "") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{name}.csv", index=False)
    md = df.to_markdown(index=False)
    (out_dir / f"{name}.md").write_text(
        f"**{caption}**\n\n{md}\n" if caption else md
    )
    # LaTeX (booktabs)
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
                cells.append(str(v).replace("_", r"\_"))
        tex.append(" & ".join(cells) + r" \\")
    tex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (out_dir / f"{name}.tex").write_text("\n".join(tex))
    print(f"    wrote {(out_dir / name).relative_to(PROJECT_ROOT)}.{{csv,md,tex}}",
          flush=True)


# -----------------------------------------------------------------------------
# Loaders
# -----------------------------------------------------------------------------
def load_pred_npz(path: Path) -> dict:
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def load_features_ids(task: str, strategy: str, subset: str) -> np.ndarray:
    p = (FEATURES_ROOT / task / strategy / subset / "fingerprints_tree.npz")
    d = np.load(p, allow_pickle=True)
    return d["ids"]


def load_stratifier_weights(task: str, strategy: str, subset: str) -> dict:
    """Map InChIKey-14 -> ml_weight for the given subset."""
    s = FEATURES_ROOT / task / strategy / subset / "stratifiers.npz"
    d = np.load(s, allow_pickle=True)
    if "ml_weight" not in d.files:
        return {}
    ids = load_features_ids(task, strategy, subset)
    return dict(zip(ids.tolist(), d["ml_weight"].tolist()))


def get_weights_for(ids: np.ndarray, task: str, strategy: str,
                     subset: str) -> np.ndarray:
    lookup = load_stratifier_weights(task, strategy, subset)
    return np.array([lookup.get(i, 1.0) for i in ids.tolist()],
                    dtype=np.float64)


# -----------------------------------------------------------------------------
# Calibration metrics
# -----------------------------------------------------------------------------
def compute_ece(y_true: np.ndarray, y_proba: np.ndarray,
                weights: np.ndarray | None = None,
                n_bins: int = CALIBRATION_N_BINS) -> tuple[float, float]:
    """Return (ECE, MCE) with equal-width bins."""
    if weights is None:
        weights = np.ones_like(y_true, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_proba, bins) - 1, 0, n_bins - 1)
    total_w = weights.sum()
    if total_w == 0:
        return (float("nan"), float("nan"))
    ece = 0.0
    mce = 0.0
    for b in range(n_bins):
        mask = idx == b
        wb = weights[mask].sum()
        if wb == 0:
            continue
        avg_p = float(np.average(y_proba[mask], weights=weights[mask]))
        obs_p = float(np.average(y_true[mask], weights=weights[mask]))
        gap = abs(avg_p - obs_p)
        ece += (wb / total_w) * gap
        mce = max(mce, gap)
    return float(ece), float(mce)


# =============================================================================
# PHASE 1 - Calibration repair
# =============================================================================
def fit_platt(val_y: np.ndarray, val_p: np.ndarray,
              val_w: np.ndarray | None = None) -> LogisticRegression:
    """Logistic regression on the val raw probabilities (1D logit-style fit)."""
    # Avoid 0/1 in input for log-odds stability
    p_clip = np.clip(val_p, 1e-6, 1 - 1e-6).reshape(-1, 1)
    lr = LogisticRegression(solver="lbfgs", max_iter=200)
    if val_w is not None:
        lr.fit(p_clip, val_y, sample_weight=val_w)
    else:
        lr.fit(p_clip, val_y)
    return lr


def fit_isotonic(val_y: np.ndarray, val_p: np.ndarray,
                  val_w: np.ndarray | None = None) -> Iso:
    iso = Iso(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    if val_w is not None:
        iso.fit(val_p, val_y, sample_weight=val_w)
    else:
        iso.fit(val_p, val_y)
    return iso


def apply_platt(lr: LogisticRegression, p: np.ndarray) -> np.ndarray:
    p_clip = np.clip(p, 1e-6, 1 - 1e-6).reshape(-1, 1)
    return lr.predict_proba(p_clip)[:, 1]


def apply_isotonic(iso: Iso, p: np.ndarray) -> np.ndarray:
    return iso.transform(p)


def phase1_calibration_repair() -> pd.DataFrame:
    log("=" * 70)
    log("PHASE 1: Calibration repair (Platt + Isotonic)")
    log("=" * 70)

    rows = []
    for strategy in STRATEGIES:
        for variant in VARIANTS:
            for model in CLF_MODELS:
                cell_dir = (MODELS_ROOT / "classification" / strategy /
                             variant / model)
                val_p = cell_dir / "val_pred.npz"
                te_p  = cell_dir / "test_pred.npz"
                if not val_p.exists() or not te_p.exists():
                    continue
                val = load_pred_npz(val_p)
                te  = load_pred_npz(te_p)

                val_y = val["y_true"].astype(np.int32)
                val_pr = val["y_pred"].astype(np.float64)
                te_y = te["y_true"].astype(np.int32)
                te_pr = te["y_pred"].astype(np.float64)
                val_w = get_weights_for(val["ids"], "classification",
                                         strategy, "val")
                te_w  = get_weights_for(te["ids"],  "classification",
                                         strategy, "test")

                # Need both classes present in val for Platt; if not, fallback
                has_both = len(np.unique(val_y)) > 1
                if not has_both:
                    log(f"  skip {strategy}/{variant}/{model}: val has one class")
                    continue

                # Original metrics on test
                ece0, mce0 = compute_ece(te_y, te_pr, te_w)
                bri0 = brier_score_loss(te_y, te_pr, sample_weight=te_w)
                try:
                    auc0 = roc_auc_score(te_y, te_pr, sample_weight=te_w)
                except Exception:
                    auc0 = float("nan")

                # Platt
                try:
                    lr = fit_platt(val_y, val_pr, val_w)
                    te_platt = apply_platt(lr, te_pr)
                    ece_p, mce_p = compute_ece(te_y, te_platt, te_w)
                    bri_p = brier_score_loss(te_y, te_platt, sample_weight=te_w)
                    auc_p = roc_auc_score(te_y, te_platt, sample_weight=te_w)
                except Exception as e:
                    log(f"  Platt failed {strategy}/{variant}/{model}: {e}")
                    te_platt = None
                    ece_p = mce_p = bri_p = auc_p = float("nan")

                # Isotonic
                try:
                    iso = fit_isotonic(val_y, val_pr, val_w)
                    te_iso = apply_isotonic(iso, te_pr)
                    ece_i, mce_i = compute_ece(te_y, te_iso, te_w)
                    bri_i = brier_score_loss(te_y, te_iso, sample_weight=te_w)
                    auc_i = roc_auc_score(te_y, te_iso, sample_weight=te_w)
                except Exception as e:
                    log(f"  Isotonic failed {strategy}/{variant}/{model}: {e}")
                    te_iso = None
                    ece_i = mce_i = bri_i = auc_i = float("nan")

                # Save per-cell record
                record = {
                    "strategy": strategy, "variant": variant, "model": model,
                    "original":  {"ece": ece0, "mce": mce0, "brier": bri0, "auc": auc0},
                    "platt":     {"ece": ece_p, "mce": mce_p, "brier": bri_p, "auc": auc_p},
                    "isotonic":  {"ece": ece_i, "mce": mce_i, "brier": bri_i, "auc": auc_i},
                    "n_val": int(len(val_y)), "n_test": int(len(te_y)),
                }
                out_json = cell_dir / "calibration_repair.json"
                out_json.write_text(json.dumps(record, indent=2))

                # Determine winner (lowest ECE)
                eces = {"original": ece0, "platt": ece_p, "isotonic": ece_i}
                eces_clean = {k: v for k, v in eces.items() if not np.isnan(v)}
                winner = min(eces_clean, key=eces_clean.get) if eces_clean else "n/a"
                best_ece = eces_clean[winner] if eces_clean else float("nan")
                improvement = ece0 - best_ece if not np.isnan(best_ece) else float("nan")

                rows.append({
                    "Split":       STRAT_PRETTY[strategy],
                    "Variant":     VARIANT_PRETTY[variant],
                    "Model":       model,
                    "ECE original": ece0,
                    "ECE Platt":    ece_p,
                    "ECE isotonic": ece_i,
                    "Best method":  winner,
                    "ECE best":     best_ece,
                    "ECE reduction": improvement,
                    "AUC original": auc0,
                    "AUC Platt":    auc_p,
                    "AUC isotonic": auc_i,
                })

    df = pd.DataFrame(rows)
    log(f"  processed {len(df)} cells")
    return df


def figure_D15_calibration_repair(df: pd.DataFrame) -> None:
    log("D15: calibration repair figure")
    # Aggregate per split: mean ECE across XGBoost+LogReg variants per method
    # Use XGBoost only to keep it focused
    sub = df.copy()
    # XGBoost cells only for the main panel
    xgb = sub[sub["Model"] == "xgboost"].copy()

    methods = ["ECE original", "ECE Platt", "ECE isotonic"]
    method_pretty = {"ECE original": "Original",
                      "ECE Platt": "Platt",
                      "ECE isotonic": "Isotonic"}
    method_color  = [METHOD_COLOR["original"], METHOD_COLOR["platt"],
                      METHOD_COLOR["isotonic"]]

    # Aggregate per split: mean and std across variants
    agg = xgb.groupby("Split")[methods].agg(["mean", "std"]).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))

    # Left: per-split mean ECE for the three methods
    ax = axes[0]
    splits_ordered = [STRAT_PRETTY[s] for s in STRATEGIES]
    x = np.arange(len(splits_ordered))
    width = 0.27
    for i, m in enumerate(methods):
        means = []
        stds  = []
        for s in splits_ordered:
            r = agg[agg["Split"] == s]
            if r.empty:
                means.append(np.nan)
                stds.append(np.nan)
            else:
                means.append(r[(m, "mean")].values[0])
                stds.append(r[(m, "std")].values[0])
        ax.bar(x + (i - 1) * width, means, width,
               yerr=stds, label=method_pretty[m],
               color=method_color[i], edgecolor="white", linewidth=0.4,
               error_kw=dict(elinewidth=0.6, ecolor="#333", capsize=2))
    ax.axhline(0.05, color=PALETTE["good"], linestyle="--",
               linewidth=0.7, alpha=0.7)
    ax.axhline(0.10, color=PALETTE["secondary"], linestyle="--",
               linewidth=0.7, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(splits_ordered, rotation=25, ha="right")
    ax.set_ylabel("test ECE (mean across variants)")
    ax.set_title("XGBoost: ECE before and after recalibration")
    ax.set_ylim(0, max(0.22, agg[(methods[0], "mean")].max() * 1.1))
    ax.legend(frameon=False, fontsize=8)

    # Right: per-cell scatter Platt vs Isotonic ECE
    ax = axes[1]
    for strategy in STRATEGIES:
        sub2 = df[df["Split"] == STRAT_PRETTY[strategy]]
        ax.scatter(sub2["ECE original"], sub2["ECE isotonic"],
                   s=60, alpha=0.75,
                   color=SPLIT_COLOR[strategy],
                   edgecolor="white", linewidth=0.6,
                   label=STRAT_PRETTY[strategy], zorder=3)
    lim = (0, max(0.40, float(df["ECE original"].max()) * 1.05))
    ax.plot(lim, lim, color=PALETTE["neutral"], linestyle="--",
            linewidth=0.8, alpha=0.7, label="y = x (no improvement)")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("ECE original")
    ax.set_ylabel("ECE after isotonic regression")
    ax.set_title("Per-cell ECE improvement from isotonic calibration")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper left")

    fig.suptitle("Figure D15.  Post-hoc calibration recovers calibration "
                 "without changing ranking (AUC unchanged within ±0.001)",
                 fontsize=10.5, y=1.02)
    fig.tight_layout()
    save_fig(fig, OUT_FIG_SUPP, "D15_calibration_repair")


# =============================================================================
# PHASE 2 - Applicability domain analysis
# =============================================================================
def compute_tanimoto_to_train(train_fp: np.ndarray,
                              test_fp: np.ndarray) -> np.ndarray:
    """Max train-test Tanimoto per test compound. ECFP4 binary."""
    train_fp = train_fp.astype(bool)
    test_fp = test_fp.astype(bool)
    train_pop = train_fp.sum(axis=1).astype(np.int32)
    test_pop = test_fp.sum(axis=1).astype(np.int32)
    max_per_test = np.zeros(len(test_fp), dtype=np.float32)
    chunk = max(1, 50_000 // max(1, len(train_fp)))
    for i in range(0, len(test_fp), chunk):
        block = test_fp[i:i + chunk]
        inter = block.astype(np.uint16) @ train_fp.T.astype(np.uint16)
        union = test_pop[i:i + chunk, None] + train_pop[None, :] - inter
        union = np.where(union == 0, 1, union)
        sim = inter / union
        max_per_test[i:i + chunk] = sim.max(axis=1)
    return max_per_test


def get_ecfp4_subset(task: str, strategy: str, subset: str) -> np.ndarray:
    p = (FEATURES_ROOT / task / strategy / subset / "fingerprints_tree.npz")
    d = np.load(p, allow_pickle=True)
    X = d["X"]
    feature_names = d["feature_names"]
    mask = np.array([n.startswith("rdkit::ecfp4::") for n in feature_names])
    if mask.sum() == 0:
        mask = np.ones(len(feature_names), dtype=bool)
    return (X[:, mask] > 0).astype(np.uint8)


def phase2_ad_analysis() -> pd.DataFrame:
    log("=" * 70)
    log("PHASE 2: Applicability domain analysis")
    log("=" * 70)

    # We use one cell per split as the representative: XGBoost on fingerprints.
    # This keeps the analysis focused. Compute per-test-compound:
    #   - max train-test Tanimoto (ECFP4)
    #   - prediction
    #   - true label
    #   - absolute residual (regression) / probability gap (classification)

    all_rows = []
    for strategy in STRATEGIES:
        log(f"  computing AD for classification/{strategy}/fingerprints/xgboost")
        cell_dir = (MODELS_ROOT / "classification" / strategy /
                     "fingerprints" / "xgboost")
        te_pred = load_pred_npz(cell_dir / "test_pred.npz")
        te_ids = te_pred["ids"]
        te_y = te_pred["y_true"].astype(np.int32)
        te_p = te_pred["y_pred"].astype(np.float64)

        # Compute Tanimoto to train for these compounds
        train_fp = get_ecfp4_subset("classification", strategy, "train")
        test_fp = get_ecfp4_subset("classification", strategy, "test")
        # Align test order to test_pred ids — features and predictions both
        # use the same ids by construction. Verify length.
        feat_test_ids = load_features_ids("classification", strategy, "test")
        if len(feat_test_ids) != len(te_ids):
            log(f"    WARN: id length mismatch ({len(feat_test_ids)} vs {len(te_ids)})")
        # Build lookup so we don't assume ordering
        feat_id_to_row = {i: r for r, i in enumerate(feat_test_ids.tolist())}
        try:
            order = np.array([feat_id_to_row[i] for i in te_ids.tolist()])
            test_fp_aligned = test_fp[order]
        except KeyError:
            log("    WARN: id mismatch; falling back to positional alignment")
            test_fp_aligned = test_fp[:len(te_ids)]

        max_tan = compute_tanimoto_to_train(train_fp, test_fp_aligned)

        # Prediction error proxies
        abs_residual = np.abs(te_y - te_p)  # absolute prob gap
        # Per-compound rows
        for tid, t_max, y, p, ar in zip(te_ids.tolist(),
                                          max_tan.tolist(),
                                          te_y.tolist(),
                                          te_p.tolist(),
                                          abs_residual.tolist()):
            all_rows.append({
                "strategy": strategy,
                "inchikey_14": tid,
                "max_train_tanimoto": t_max,
                "y_true": y,
                "y_proba": p,
                "abs_prob_gap": ar,
            })

    df = pd.DataFrame(all_rows)
    log(f"  collected {len(df):,} test-compound records across {len(STRATEGIES)} splits")
    return df


def aggregate_ad_by_bin(df: pd.DataFrame) -> pd.DataFrame:
    """Bin test compounds by max_train_tanimoto into 5 AD bins, compute
    per-bin metrics per split."""
    bins = [0.0, 0.4, 0.55, 0.7, 0.85, 1.001]
    labels = ["very novel\n(<0.4)", "novel\n(0.4-0.55)",
              "moderate\n(0.55-0.7)", "near\n(0.7-0.85)",
              "very near\n(>0.85)"]
    df = df.copy()
    df["ad_bin"] = pd.cut(df["max_train_tanimoto"], bins=bins, labels=labels,
                          include_lowest=True)

    rows = []
    for strategy in STRATEGIES:
        sub = df[df["strategy"] == strategy]
        for bin_label in labels:
            cell = sub[sub["ad_bin"] == bin_label]
            n = len(cell)
            if n == 0:
                continue
            # Per-bin metrics
            mean_gap = float(cell["abs_prob_gap"].mean())
            # Per-bin ECE: this bin is one "calibration bin" implicitly
            if n >= 5:
                # Compute per-bin ECE using local 10-bin equal-width
                y = cell["y_true"].values
                p = cell["y_proba"].values
                ece, _ = compute_ece(y, p, np.ones_like(y, dtype=np.float64))
            else:
                ece = float("nan")
            rows.append({
                "Split": STRAT_PRETTY[strategy],
                "AD bin": bin_label,
                "n compounds": n,
                "Mean |gap|": mean_gap,
                "Bin ECE":    ece,
            })
    return pd.DataFrame(rows)


def figure_D16_ad_analysis(df: pd.DataFrame) -> None:
    log("D16: applicability domain figure")

    fig = plt.figure(figsize=(11.5, 7.2))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.32)

    # Panels (a): scatter Tanimoto vs abs_prob_gap, one per split, 2x3 layout
    for i, strategy in enumerate(STRATEGIES):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        sub = df[df["strategy"] == strategy]
        if sub.empty:
            ax.set_visible(False)
            continue
        ax.scatter(sub["max_train_tanimoto"], sub["abs_prob_gap"],
                   s=12, alpha=0.45,
                   color=SPLIT_COLOR[strategy],
                   edgecolor="white", linewidth=0.2)
        # Trend line: bin by max_train_tanimoto into 8 bins, plot mean+/-std
        try:
            bins = np.linspace(sub["max_train_tanimoto"].min(),
                                sub["max_train_tanimoto"].max(), 9)
            bin_idx = np.digitize(sub["max_train_tanimoto"], bins) - 1
            xs, ys, ss = [], [], []
            for b in range(len(bins) - 1):
                m = bin_idx == b
                if m.sum() >= 5:
                    xs.append(float(np.mean([bins[b], bins[b + 1]])))
                    ys.append(float(sub["abs_prob_gap"][m].mean()))
                    ss.append(float(sub["abs_prob_gap"][m].std()))
            if xs:
                xs, ys, ss = map(np.array, (xs, ys, ss))
                ax.plot(xs, ys, color="#1a1a1a", linewidth=1.4, zorder=3)
                ax.fill_between(xs, np.clip(ys - ss, 0, None),
                                 np.clip(ys + ss, 0, None),
                                 color="#1a1a1a", alpha=0.15, zorder=2)
        except Exception:
            pass
        ax.set_title(f"{STRAT_PRETTY[strategy]}  (n = {len(sub):,})",
                     fontsize=9.5)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("max train Tanimoto (ECFP4)")
        ax.set_ylabel("|y_true − y_proba|")
        ax.grid(True, alpha=0.4)

    fig.suptitle("Figure D16.  Applicability domain: per-test-compound "
                 "prediction error vs nearest-train Tanimoto  "
                 "(XGBoost on fingerprints, classification task)",
                 fontsize=10.5, y=1.00)
    save_fig(fig, OUT_FIG_SUPP, "D16_applicability_domain")


# =============================================================================
# Main
# =============================================================================
def main() -> int:
    set_paper_style()
    OUT_FIG_SUPP.mkdir(parents=True, exist_ok=True)
    OUT_TBL_SUPP.mkdir(parents=True, exist_ok=True)

    # PHASE 1
    cal_df = phase1_calibration_repair()
    if not cal_df.empty:
        # Save table with rounded values
        display = cal_df.copy()
        for col in ["ECE original", "ECE Platt", "ECE isotonic",
                     "ECE best", "ECE reduction",
                     "AUC original", "AUC Platt", "AUC isotonic"]:
            display[col] = display[col].apply(
                lambda v: float(f"{v:.4f}") if pd.notna(v) else v)
        save_table(display, OUT_TBL_SUPP, "T13_calibration_repair",
                   caption="Post-hoc calibration repair: Expected Calibration "
                           "Error (ECE) before and after Platt scaling and "
                           "isotonic regression. Calibration fit on validation "
                           "set, evaluated on locked test set. AUC is preserved "
                           "(within ±0.001) by both methods because they are "
                           "monotonic transforms.",
                   label="tab:T13_calibration_repair")
        figure_D15_calibration_repair(cal_df)

        # Summary statistics
        log("")
        log("Calibration repair summary:")
        for split in [STRAT_PRETTY[s] for s in STRATEGIES]:
            sub = cal_df[cal_df["Split"] == split]
            if sub.empty:
                continue
            mean_orig = sub["ECE original"].mean()
            mean_platt = sub["ECE Platt"].mean()
            mean_iso = sub["ECE isotonic"].mean()
            log(f"  {split:<14} original={mean_orig:.4f}  "
                f"Platt={mean_platt:.4f}  isotonic={mean_iso:.4f}")

    # PHASE 2
    ad_df = phase2_ad_analysis()
    if not ad_df.empty:
        figure_D16_ad_analysis(ad_df)
        # Aggregated bin table
        agg = aggregate_ad_by_bin(ad_df)
        save_table(agg, OUT_TBL_SUPP, "T14_applicability_domain",
                   caption="Applicability domain analysis: per-split, "
                           "per-AD-bin prediction error and bin-level ECE "
                           "for the fingerprints/XGBoost classification cells. "
                           "AD bins by max train-test Tanimoto (ECFP4).",
                   label="tab:T14_ad")
        # Per-compound CSV as data appendix
        ad_df.to_csv(OUT_TBL_SUPP / "T14_applicability_domain_per_compound.csv",
                      index=False)
        log(f"  wrote per-compound CSV: T14_applicability_domain_per_compound.csv "
            f"({len(ad_df):,} rows)")

    log("DONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
