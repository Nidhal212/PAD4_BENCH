#!/usr/bin/env python3
"""
PAD4_BENCH Paper 1 - results figures.

Produces 9 results figures (5 main, 4 supplementary) at 600 DPI raster + vector
PDF. Pulls from models_v1/all_results.csv and per-cell metrics.json files.

Figures:
  Main:
    R1  headline heatmaps          test R² + test AUC across 6×5 grid
    R2  headline with CIs          XGBoost point + 95% CI per cell
    R3  linear vs tree             paired XGB vs ENet/LogReg
    R4  CV-vs-test gap             scatter, color by split
    R5  calibration vs accuracy    scatter, AUC vs ECE per cell

  Supplementary:
    R6  stacking lift              per-cell AUC delta bar
    R7  seed robustness            box plot across 3 seeds
    R8  threshold recalibration    default-vs-tuned MCC per cell
    R9  ece+auc summary by split   per-split bar comparison

All figures saved at ≥600 DPI as PNG + vector PDF in:
  paper/figures/results/main/
  paper/figures/results/supp/

Usage:
    cd /home/nidhal/PAD4_BENCH
    python paper_results_figures.py
"""

import json
import sys
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# -----------------------------------------------------------------------------
# Paths and config
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path("/home/nidhal/PAD4_BENCH")
MODELS_ROOT  = PROJECT_ROOT / "models_v1"
RESULTS_CSV  = MODELS_ROOT / "all_results.csv"
OUT_MAIN     = PROJECT_ROOT / "paper" / "figures" / "results" / "main"
OUT_SUPP     = PROJECT_ROOT / "paper" / "figures" / "results" / "supp"

STRATEGIES = ["random", "scaffold", "confirmed", "lead_opt", "similarity", "cliff_aware"]
VARIANTS   = ["full", "fingerprints", "physchem", "mordred", "fragments"]
STRAT_PRETTY = {
    "random": "Random", "scaffold": "Scaffold", "confirmed": "Confirmed",
    "lead_opt": "Lead-Opt", "similarity": "Similarity", "cliff_aware": "Cliff-Aware",
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

DPI_RASTER = 600

# Diverging colormap for heatmap: light at low values, dark at high
CMAP_R2 = LinearSegmentedColormap.from_list(
    "paper_blues", ["#f0f4f8", "#a6c0d8", "#5079a3", "#1f4e79", "#0a2540"])
CMAP_AUC = LinearSegmentedColormap.from_list(
    "paper_greens", ["#f0f6f1", "#aed1b4", "#5ba16f", "#2d7a4f", "#0f4824"])


# -----------------------------------------------------------------------------
# Style
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
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "lines.linewidth": 1.4,
        "patch.linewidth": 0.6,
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
    png = out_dir / f"{name}.png"
    pdf = out_dir / f"{name}.pdf"
    fig.savefig(png, dpi=DPI_RASTER)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"    wrote {png.relative_to(PROJECT_ROOT)} and .pdf", flush=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# -----------------------------------------------------------------------------
# Data loaders
# -----------------------------------------------------------------------------
def load_results() -> pd.DataFrame:
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(f"{RESULTS_CSV} missing — run paper1_reviewer_proof.py")
    df = pd.read_csv(RESULTS_CSV)
    # Drop any smoke-test rows
    df = df[~df["metrics_path"].str.contains("smoketest", na=False)]
    return df


def filter_main_sweep(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """Return one row per (strategy, variant, model) for the main sweep (no seeds)."""
    mask = ((df["task_dir"] == task) &
            (df["model"].isin(["xgboost", "elasticnet", "logreg_enet"])) &
            (df["seed_subdir"].isna() if "seed_subdir" in df.columns else True))
    return df[mask].copy()


# =============================================================================
# R1 - Headline heatmaps
# =============================================================================
def fig_R1_headline_heatmaps(df: pd.DataFrame) -> None:
    log("R1: headline heatmaps")

    reg = filter_main_sweep(df, "regression")
    reg = reg[reg["model"] == "xgboost"]
    cls = filter_main_sweep(df, "classification")
    cls = cls[cls["model"] == "xgboost"]

    # Pivot to matrix
    reg_mat = (reg.pivot_table(index="strategy", columns="variant",
                                values="test_r2", aggfunc="mean")
                  .reindex(index=STRATEGIES, columns=VARIANTS))
    cls_mat = (cls.pivot_table(index="strategy", columns="variant",
                                values="test_roc_auc", aggfunc="mean")
                  .reindex(index=STRATEGIES, columns=VARIANTS))

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))

    for ax, mat, title, cmap, vmin, vmax, fmt in [
        (axes[0], reg_mat, "Regression  (test R²)",   CMAP_R2,  0.55, 0.82, ".3f"),
        (axes[1], cls_mat, "Classification  (test ROC-AUC)", CMAP_AUC, 0.86, 0.98, ".3f"),
    ]:
        im = ax.imshow(mat.values, cmap=cmap, vmin=vmin, vmax=vmax,
                       aspect="auto")
        ax.set_xticks(np.arange(len(VARIANTS)))
        ax.set_xticklabels([VARIANT_PRETTY[v] for v in VARIANTS],
                           rotation=30, ha="right")
        ax.set_yticks(np.arange(len(STRATEGIES)))
        ax.set_yticklabels([STRAT_PRETTY[s] for s in STRATEGIES])
        ax.set_title(title)
        # Cell annotations with adaptive color
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat.values[i, j]
                if np.isnan(v):
                    continue
                # white text on dark cells, dark text on light cells
                norm_v = (v - vmin) / (vmax - vmin)
                color = "white" if norm_v > 0.55 else "#1a1a1a"
                ax.text(j, i, format(v, fmt), ha="center", va="center",
                        fontsize=8, weight="bold", color=color)
        ax.set_xticks(np.arange(-0.5, len(VARIANTS), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(STRATEGIES), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.0)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.grid(which="major", visible=False)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                            shrink=0.85)
        cbar.ax.tick_params(labelsize=7.5)
        cbar.outline.set_linewidth(0.6)

    fig.suptitle("Figure R1.  XGBoost test performance across split × feature variant",
                 fontsize=10.5, y=1.02)
    fig.tight_layout()
    save_fig(fig, OUT_MAIN, "R1_headline_heatmaps")


# =============================================================================
# R2 - Headline with CIs
# =============================================================================
def fig_R2_headline_with_ci(df: pd.DataFrame) -> None:
    log("R2: headline with 95% CIs")

    def collect(df_, task, metric_col, ci_lo_col, ci_hi_col):
        d = filter_main_sweep(df_, task)
        d = d[d["model"] == "xgboost"]
        d = d[["strategy", "variant", metric_col, ci_lo_col, ci_hi_col]].copy()
        d["strategy"] = pd.Categorical(d["strategy"], STRATEGIES, ordered=True)
        d["variant"]  = pd.Categorical(d["variant"], VARIANTS, ordered=True)
        d = d.sort_values(["strategy", "variant"])
        return d

    reg = collect(df, "regression", "test_r2", "test_ci_r2_lo", "test_ci_r2_hi")
    cls = collect(df, "classification", "test_roc_auc",
                   "test_ci_roc_auc_lo", "test_ci_roc_auc_hi")

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.6))

    for ax, d, ylabel, title, ylim in [
        (axes[0], reg, "test R²", "Regression  (XGBoost on tree-space)", (0.40, 0.90)),
        (axes[1], cls, "test ROC-AUC",
                       "Classification  (XGBoost on tree-space)", (0.83, 1.00)),
    ]:
        n_strat = len(STRATEGIES)
        n_var = len(VARIANTS)
        group_w = 0.85
        bar_w = group_w / n_var
        offsets = np.linspace(-group_w/2 + bar_w/2, group_w/2 - bar_w/2, n_var)
        x_base = np.arange(n_strat)

        for vi, variant in enumerate(VARIANTS):
            sub = d[d["variant"] == variant]
            # Reorder to match STRATEGIES
            sub = sub.set_index("strategy").reindex(STRATEGIES).reset_index()
            vals = sub.iloc[:, 2].values
            ci_lo = sub.iloc[:, 3].values
            ci_hi = sub.iloc[:, 4].values
            err_lo = vals - ci_lo
            err_hi = ci_hi - vals
            color = plt.cm.viridis(vi / max(1, n_var - 1)) if False else None
            colors = ["#1f4e79", "#2d7a4f", "#5a5a5a", "#e8b73a", "#c5504b"]
            ax.bar(x_base + offsets[vi], vals, bar_w,
                   yerr=[err_lo, err_hi],
                   color=colors[vi], edgecolor="white", linewidth=0.4,
                   error_kw=dict(elinewidth=0.7, ecolor="#222222", capsize=1.5),
                   label=VARIANT_PRETTY[variant])
        ax.set_xticks(x_base)
        ax.set_xticklabels([STRAT_PRETTY[s] for s in STRATEGIES])
        ax.set_ylabel(ylabel)
        ax.set_ylim(ylim)
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=7.5, ncol=5,
                  loc="upper center", bbox_to_anchor=(0.5, -0.10))
        ax.grid(True, axis="y", alpha=0.5)

    fig.suptitle("Figure R2.  Headline test metrics with bootstrap 95% CIs  "
                 "(1,000 bootstrap resamples per cell)",
                 fontsize=10.5, y=1.00)
    fig.tight_layout()
    save_fig(fig, OUT_MAIN, "R2_headline_with_ci")


# =============================================================================
# R3 - Linear vs tree
# =============================================================================
def fig_R3_linear_vs_tree(df: pd.DataFrame) -> None:
    log("R3: linear vs tree")

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))

    for ax, task, metric, model_pair, ylabel, ylim, title in [
        (axes[0], "regression", "test_r2",
         ("xgboost", "elasticnet"), "test R²", (0, 0.85),
         "Regression"),
        (axes[1], "classification", "test_roc_auc",
         ("xgboost", "logreg_enet"), "test ROC-AUC", (0.55, 1.0),
         "Classification"),
    ]:
        d = filter_main_sweep(df, task)
        # Average across variants per (strategy, model) to keep the figure
        # readable; full variant-level information is in R1/R2.
        agg = (d.groupby(["strategy", "model"])[metric].mean().reset_index())
        x = np.arange(len(STRATEGIES))
        width = 0.38
        tree_vals = []
        lin_vals = []
        for s in STRATEGIES:
            tv = agg[(agg["strategy"] == s) & (agg["model"] == model_pair[0])][metric]
            lv = agg[(agg["strategy"] == s) & (agg["model"] == model_pair[1])][metric]
            tree_vals.append(tv.iloc[0] if len(tv) else np.nan)
            lin_vals.append(lv.iloc[0] if len(lv) else np.nan)

        ax.bar(x - width/2, tree_vals, width,
               label=f"{model_pair[0]} (tree-space)",
               color=PALETTE["primary"], edgecolor="white", linewidth=0.4)
        ax.bar(x + width/2, lin_vals, width,
               label=f"{model_pair[1]} (linear-space)",
               color=PALETTE["accent"], edgecolor="white", linewidth=0.4)

        # Annotate gap
        for i, (t, l) in enumerate(zip(tree_vals, lin_vals)):
            if np.isfinite(t) and np.isfinite(l):
                gap = t - l
                ax.text(i, max(t, l) + (ylim[1] - ylim[0]) * 0.025,
                        f"+{gap:.2f}", ha="center", fontsize=7,
                        color=PALETTE["good"], weight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([STRAT_PRETTY[s] for s in STRATEGIES],
                           rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_ylim(ylim)
        ax.set_title(f"{title}  (mean across 5 variants)")
        ax.legend(frameon=False, fontsize=7.5, loc="lower left")

    fig.suptitle("Figure R3.  Tree-space (XGBoost) vs linear-space "
                 "regularized linear model — performance gap",
                 fontsize=10.5, y=1.02)
    fig.tight_layout()
    save_fig(fig, OUT_MAIN, "R3_linear_vs_tree")


# =============================================================================
# R4 - CV-vs-test gap
# =============================================================================
def fig_R4_cv_vs_test(df: pd.DataFrame) -> None:
    log("R4: CV-vs-test gap")

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4))

    for ax, task, cv_col, test_col, label, lim in [
        (axes[0], "regression",     "oof_r2",      "test_r2",      "R²",      (0.4, 0.9)),
        (axes[1], "classification", "oof_roc_auc", "test_roc_auc", "ROC-AUC", (0.85, 1.0)),
    ]:
        d = filter_main_sweep(df, task)
        d = d[d["model"] == "xgboost"]
        # OOF column naming — fall back to "oof_r2" if present, else try alternatives
        cv_candidates = [cv_col, cv_col.replace("oof_", "oof_mean_"),
                         f"val_{cv_col.replace('oof_', '')}"]
        cv_actual = None
        for c in cv_candidates:
            if c in d.columns:
                cv_actual = c
                break
        if cv_actual is None:
            ax.set_visible(False)
            continue
        for strategy in STRATEGIES:
            sub = d[d["strategy"] == strategy]
            ax.scatter(sub[cv_actual], sub[test_col],
                       s=55, alpha=0.85,
                       color=SPLIT_COLOR[strategy],
                       edgecolor="white", linewidth=0.7,
                       label=STRAT_PRETTY[strategy], zorder=3)
        ax.plot(lim, lim, color=PALETTE["neutral"], linestyle="--",
                linewidth=0.9, alpha=0.7, zorder=1, label="y = x")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel(f"5-fold CV {label}  (OOF on train)")
        ax.set_ylabel(f"test {label}  (test_locked)")
        ax.set_title(task.capitalize())
        ax.set_aspect("equal", adjustable="box")
        if task == "regression":
            ax.legend(frameon=False, fontsize=7, ncol=2,
                      loc="upper left", bbox_to_anchor=(0.02, 0.98))
        ax.grid(True, alpha=0.4)

    fig.suptitle("Figure R4.  Cross-validation vs test performance per cell  "
                 "(points below y=x indicate optimistic CV)",
                 fontsize=10.5, y=1.02)
    fig.tight_layout()
    save_fig(fig, OUT_MAIN, "R4_cv_vs_test")


# =============================================================================
# R5 - Calibration vs accuracy
# =============================================================================
def fig_R5_calibration_vs_auc(df: pd.DataFrame) -> None:
    log("R5: calibration vs accuracy")

    d = filter_main_sweep(df, "classification")
    d = d[d["model"] == "xgboost"]
    # Need test_calibration_ece column
    ece_col = "test_calibration_ece"
    if ece_col not in d.columns:
        log(f"  WARN: {ece_col} not in CSV — skipping R5")
        return

    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    for strategy in STRATEGIES:
        sub = d[d["strategy"] == strategy]
        ax.scatter(sub["test_roc_auc"], sub[ece_col],
                   s=110, alpha=0.85,
                   color=SPLIT_COLOR[strategy],
                   edgecolor="white", linewidth=0.7,
                   label=STRAT_PRETTY[strategy], zorder=3)
        # Annotate the variant for each point
        for _, row in sub.iterrows():
            ax.annotate(VARIANT_PRETTY[row["variant"]][:4],
                        xy=(row["test_roc_auc"], row[ece_col]),
                        xytext=(4, 3), textcoords="offset points",
                        fontsize=6.5, color=SPLIT_COLOR[strategy], alpha=0.85)
    ax.axhline(0.05, color=PALETTE["good"], linestyle="--", linewidth=0.7,
               alpha=0.6)
    ax.axhline(0.10, color=PALETTE["secondary"], linestyle="--", linewidth=0.7,
               alpha=0.6)
    ax.text(0.99, 0.052, "  ECE = 0.05 (well-calibrated)",
            ha="right", va="bottom", fontsize=7,
            color=PALETTE["good"], transform=ax.get_yaxis_transform())
    ax.text(0.99, 0.102, "  ECE = 0.10 (needs recalibration)",
            ha="right", va="bottom", fontsize=7,
            color=PALETTE["secondary"], transform=ax.get_yaxis_transform())
    ax.set_xlabel("test ROC-AUC")
    ax.set_ylabel("test ECE  (Expected Calibration Error)")
    ax.set_title("Higher AUC alone does not imply trustworthy probabilities", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, ncol=2,
              loc="upper left")
    ax.grid(True, alpha=0.4)

    fig.suptitle("Figure R5.  Ranking quality (AUC) vs probability quality (ECE) "
                 "per classification cell", fontsize=10.5, y=1.00)
    fig.tight_layout()
    save_fig(fig, OUT_MAIN, "R5_calibration_vs_auc")


# =============================================================================
# R6 - Stacking lift (supplementary)
# =============================================================================
def fig_R6_stacking_lift(df: pd.DataFrame) -> None:
    log("R6: stacking lift")

    # Stacking cells: task_dir == 'stacking'
    stk = df[df["task_dir"] == "stacking"].copy()
    if stk.empty:
        log("  WARN: no stacking cells found")
        return
    # Pull baseline classification AUC for the same (strategy, variant)
    base = df[(df["task_dir"] == "classification") & (df["model"] == "xgboost")].copy()
    base_map = base.set_index(["strategy", "variant"])["test_roc_auc"].to_dict()

    stk["baseline_auc"] = stk.apply(
        lambda r: base_map.get((r["strategy"], r["variant"]), np.nan), axis=1)
    stk["lift"] = stk["test_roc_auc"] - stk["baseline_auc"]
    stk["strategy"] = pd.Categorical(stk["strategy"], STRATEGIES, ordered=True)
    stk["variant"] = pd.Categorical(stk["variant"], VARIANTS, ordered=True)
    stk = stk.sort_values(["strategy", "variant"])

    fig, ax = plt.subplots(figsize=(10.0, 4.6))
    n_strat = len(STRATEGIES)
    n_var = len(VARIANTS)
    group_w = 0.85
    bar_w = group_w / n_var
    offsets = np.linspace(-group_w/2 + bar_w/2, group_w/2 - bar_w/2, n_var)
    x_base = np.arange(n_strat)
    colors = ["#1f4e79", "#2d7a4f", "#5a5a5a", "#e8b73a", "#c5504b"]

    for vi, variant in enumerate(VARIANTS):
        sub = stk[stk["variant"] == variant].set_index("strategy").reindex(STRATEGIES)
        vals = sub["lift"].values
        bar_colors = [colors[vi] if v >= 0 else "#aaaaaa" for v in vals]
        ax.bar(x_base + offsets[vi], vals, bar_w,
               color=bar_colors, edgecolor="white", linewidth=0.4,
               label=VARIANT_PRETTY[variant])
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(x_base)
    ax.set_xticklabels([STRAT_PRETTY[s] for s in STRATEGIES])
    ax.set_ylabel("Δ test ROC-AUC  (stacked − baseline classifier)")
    ax.legend(frameon=False, fontsize=7.5, ncol=5,
              loc="upper center", bbox_to_anchor=(0.5, -0.08))
    ax.grid(True, axis="y", alpha=0.5)
    mean_lift = stk["lift"].mean()
    n_pos = int((stk["lift"] > 0).sum())
    ax.text(0.02, 0.97,
            f"Mean Δ AUC: {mean_lift:+.4f}\n"
            f"Cells with positive lift: {n_pos}/{len(stk)}",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(facecolor="white", edgecolor=PALETTE["neutral"],
                      boxstyle="round,pad=0.4", linewidth=0.6))

    fig.suptitle("Figure R6.  Stacking (regression OOF + classification OOF) lift "
                 "over baseline classifier",
                 fontsize=10.5, y=1.00)
    fig.tight_layout()
    save_fig(fig, OUT_SUPP, "R6_stacking_lift")


# =============================================================================
# R7 - Seed robustness (supplementary)
# =============================================================================
def fig_R7_seed_robustness(df: pd.DataFrame) -> None:
    log("R7: seed robustness boxplots")

    # Pull from per-cell metrics for {seed_42 (main), seed_7, seed_1337}
    # Use random + scaffold only
    rows = []
    for task, metric_key in [("regression", "r2"),
                              ("classification", "roc_auc")]:
        for strategy in ["random", "scaffold"]:
            for variant in VARIANTS:
                base = MODELS_ROOT / task / strategy / variant / "xgboost"
                seed_paths = [(42, base / "metrics.json")]
                for seed in [7, 1337]:
                    seed_paths.append((seed, base / "seeds" / f"seed_{seed}" / "metrics.json"))
                for seed, p in seed_paths:
                    if p.exists():
                        m = json.loads(p.read_text())
                        val = m.get("test", {}).get(metric_key)
                        if val is not None:
                            rows.append({"task": task, "strategy": strategy,
                                          "variant": variant, "seed": seed,
                                          "value": float(val)})
    rd = pd.DataFrame(rows)
    if rd.empty:
        log("  WARN: no robustness data found")
        return

    fig, axes = plt.subplots(2, 1, figsize=(10.0, 6.4))
    for ax, task, ylabel in [
        (axes[0], "regression", "test R²"),
        (axes[1], "classification", "test ROC-AUC"),
    ]:
        sub = rd[rd["task"] == task]
        positions = []
        data_per_pos = []
        colors_per_pos = []
        labels_per_pos = []
        i = 0
        for strategy in ["random", "scaffold"]:
            for variant in VARIANTS:
                vals = sub[(sub["strategy"] == strategy) &
                            (sub["variant"] == variant)]["value"].values
                if len(vals) >= 2:
                    positions.append(i)
                    data_per_pos.append(vals)
                    colors_per_pos.append(SPLIT_COLOR[strategy])
                    labels_per_pos.append(f"{VARIANT_PRETTY[variant]}")
                i += 1
            i += 1  # gap between split groups
        bp = ax.boxplot(data_per_pos, positions=positions, widths=0.65,
                         patch_artist=True, showfliers=True)
        for patch, c in zip(bp["boxes"], colors_per_pos):
            patch.set_facecolor(c)
            patch.set_alpha(0.45)
            patch.set_edgecolor(c)
            patch.set_linewidth(0.8)
        for k in ("medians",):
            for line in bp[k]:
                line.set_color("#1a1a1a")
                line.set_linewidth(1.1)
        for k in ("whiskers", "caps"):
            for line in bp[k]:
                line.set_color("#444444")
                line.set_linewidth(0.7)
        # Overlay seed points
        for pos, vals, c in zip(positions, data_per_pos, colors_per_pos):
            jitter = np.random.RandomState(pos).uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(np.full(len(vals), pos) + jitter, vals,
                       s=20, color=c, alpha=0.9, edgecolor="white",
                       linewidth=0.4, zorder=3)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels_per_pos, rotation=35, ha="right", fontsize=7.5)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{task.capitalize()}  —  3-seed distribution per cell")
        # Split-group annotations
        ax.axvline(len(VARIANTS) + 0.5, color=PALETTE["neutral"],
                   linestyle=":", linewidth=0.7, alpha=0.5)
        ax.text(2, ax.get_ylim()[1] * 0.99 + ax.get_ylim()[0] * 0.01,
                "Random", ha="center", va="top", fontsize=8,
                color=SPLIT_COLOR["random"], weight="bold",
                transform=ax.get_xaxis_transform())
        ax.text(len(VARIANTS) + 3, ax.get_ylim()[1] * 0.99 + ax.get_ylim()[0] * 0.01,
                "Scaffold", ha="center", va="top", fontsize=8,
                color=SPLIT_COLOR["scaffold"], weight="bold",
                transform=ax.get_xaxis_transform())
        ax.grid(True, axis="y", alpha=0.4)

    fig.suptitle("Figure R7.  Seed robustness on random and scaffold splits  "
                 "(seeds 42, 7, 1337)", fontsize=10.5, y=1.00)
    fig.tight_layout()
    save_fig(fig, OUT_SUPP, "R7_seed_robustness")


# =============================================================================
# R8 - Threshold recalibration
# =============================================================================
def fig_R8_threshold_recal(df: pd.DataFrame) -> None:
    log("R8: threshold recalibration")

    d = filter_main_sweep(df, "classification")
    d = d[d["model"].isin(["xgboost", "logreg_enet"])]

    # Need default test_mcc and tuned test_tuned_mcc + tuned_threshold
    default_col = "test_mcc"
    tuned_col = "test_tuned_mcc"
    thr_col = "tuned_threshold"
    needed = [default_col, tuned_col, thr_col]
    if any(c not in d.columns for c in needed):
        log(f"  WARN: missing columns {needed} — skipping R8")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4))

    # Left: default vs tuned MCC per cell (scatter)
    ax = axes[0]
    for strategy in STRATEGIES:
        sub = d[d["strategy"] == strategy]
        ax.scatter(sub[default_col], sub[tuned_col],
                   s=55, alpha=0.85,
                   color=SPLIT_COLOR[strategy],
                   edgecolor="white", linewidth=0.6,
                   label=STRAT_PRETTY[strategy], zorder=3)
    lim = (-0.05, 1.0)
    ax.plot(lim, lim, color=PALETTE["neutral"], linestyle="--",
            linewidth=0.8, alpha=0.7, label="y = x")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("MCC at threshold = 0.5  (default)")
    ax.set_ylabel("MCC at Youden's J  (tuned)")
    ax.set_title("Recalibration impact on MCC")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper left")
    ax.grid(True, alpha=0.4)
    # Mark degenerate-at-default cells
    degen = d[d[default_col] == 0]
    for _, row in degen.iterrows():
        ax.annotate(f"{STRAT_PRETTY[row['strategy']][:3]}/{row['variant'][:4]}",
                    xy=(row[default_col], row[tuned_col]),
                    xytext=(7, -3), textcoords="offset points",
                    fontsize=7, color=PALETTE["secondary"])

    # Right: tuned threshold distribution per split
    ax = axes[1]
    positions = np.arange(len(STRATEGIES))
    box_data = [d[d["strategy"] == s][thr_col].dropna().values
                for s in STRATEGIES]
    bp = ax.boxplot(box_data, positions=positions, widths=0.55,
                     patch_artist=True, showfliers=True)
    for patch, s in zip(bp["boxes"], STRATEGIES):
        patch.set_facecolor(SPLIT_COLOR[s])
        patch.set_alpha(0.45)
        patch.set_edgecolor(SPLIT_COLOR[s])
        patch.set_linewidth(0.8)
    for line in bp["medians"]:
        line.set_color("#1a1a1a")
        line.set_linewidth(1.2)
    for k in ("whiskers", "caps"):
        for line in bp[k]:
            line.set_color("#444444")
            line.set_linewidth(0.7)
    # Overlay points
    for pos, vals, s in zip(positions, box_data, STRATEGIES):
        jitter = np.random.RandomState(pos).uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(np.full(len(vals), pos) + jitter, vals,
                   s=22, color=SPLIT_COLOR[s], alpha=0.85,
                   edgecolor="white", linewidth=0.4, zorder=3)
    ax.axhline(0.5, color=PALETTE["neutral"], linestyle="--", linewidth=0.7,
               alpha=0.6, label="default = 0.5")
    ax.set_xticks(positions)
    ax.set_xticklabels([STRAT_PRETTY[s] for s in STRATEGIES],
                       rotation=30, ha="right")
    ax.set_ylabel("Youden-tuned threshold")
    ax.set_title("Tuned threshold distribution per split")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    ax.grid(True, axis="y", alpha=0.4)

    fig.suptitle("Figure R8.  Decision-threshold policy: default 0.5 vs Youden's J",
                 fontsize=10.5, y=1.02)
    fig.tight_layout()
    save_fig(fig, OUT_SUPP, "R8_threshold_recal")


# =============================================================================
# R9 - ECE and AUC summary by split
# =============================================================================
def fig_R9_ece_auc_summary(df: pd.DataFrame) -> None:
    log("R9: ECE + AUC summary by split")

    d = filter_main_sweep(df, "classification")
    d = d[d["model"] == "xgboost"]
    ece_col = "test_calibration_ece"
    if ece_col not in d.columns:
        log(f"  WARN: {ece_col} missing — skipping R9")
        return

    # Mean and std across variants per split
    agg = d.groupby("strategy").agg(
        mean_auc=("test_roc_auc", "mean"),
        std_auc=("test_roc_auc", "std"),
        mean_ece=(ece_col, "mean"),
        std_ece=(ece_col, "std"),
    ).reindex(STRATEGIES).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))

    x = np.arange(len(STRATEGIES))
    colors = [SPLIT_COLOR[s] for s in STRATEGIES]

    ax = axes[0]
    bars = ax.bar(x, agg["mean_auc"], yerr=agg["std_auc"], color=colors,
                  edgecolor="white", linewidth=0.4, capsize=2,
                  error_kw=dict(elinewidth=0.7, ecolor="#333"))
    for b, v in zip(bars, agg["mean_auc"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.003,
                f"{v:.3f}", ha="center", fontsize=7.5, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([STRAT_PRETTY[s] for s in STRATEGIES],
                       rotation=25, ha="right")
    ax.set_ylim(0.85, 1.0)
    ax.set_ylabel("test ROC-AUC")
    ax.set_title("Ranking quality (mean ± std across variants)")
    ax.grid(True, axis="y", alpha=0.5)

    ax = axes[1]
    bars = ax.bar(x, agg["mean_ece"], yerr=agg["std_ece"], color=colors,
                  edgecolor="white", linewidth=0.4, capsize=2,
                  error_kw=dict(elinewidth=0.7, ecolor="#333"))
    for b, v in zip(bars, agg["mean_ece"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.004,
                f"{v:.3f}", ha="center", fontsize=7.5, weight="bold")
    ax.axhline(0.05, color=PALETTE["good"], linestyle="--",
               linewidth=0.7, alpha=0.7)
    ax.axhline(0.10, color=PALETTE["secondary"], linestyle="--",
               linewidth=0.7, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([STRAT_PRETTY[s] for s in STRATEGIES],
                       rotation=25, ha="right")
    ax.set_ylim(0, 0.20)
    ax.set_ylabel("test ECE")
    ax.set_title("Probability quality (mean ± std across variants)")
    ax.text(5.4, 0.052, "0.05", color=PALETTE["good"], fontsize=7,
            ha="right", va="bottom")
    ax.text(5.4, 0.102, "0.10", color=PALETTE["secondary"], fontsize=7,
            ha="right", va="bottom")
    ax.grid(True, axis="y", alpha=0.5)

    fig.suptitle("Figure R9.  Ranking and calibration follow the same split-difficulty "
                 "ordering, but calibration degrades faster",
                 fontsize=10.5, y=1.02)
    fig.tight_layout()
    save_fig(fig, OUT_SUPP, "R9_ece_auc_summary")


# =============================================================================
# Main
# =============================================================================
def main() -> int:
    set_paper_style()
    OUT_MAIN.mkdir(parents=True, exist_ok=True)
    OUT_SUPP.mkdir(parents=True, exist_ok=True)
    log(f"output (main): {OUT_MAIN}")
    log(f"output (supp): {OUT_SUPP}")

    df = load_results()
    log(f"loaded {len(df)} cell rows from {RESULTS_CSV.name}")

    fig_R1_headline_heatmaps(df)
    fig_R2_headline_with_ci(df)
    fig_R3_linear_vs_tree(df)
    fig_R4_cv_vs_test(df)
    fig_R5_calibration_vs_auc(df)

    fig_R6_stacking_lift(df)
    fig_R7_seed_robustness(df)
    fig_R8_threshold_recal(df)
    fig_R9_ece_auc_summary(df)

    log("DONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
