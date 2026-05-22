#!/usr/bin/env python3
"""
PAD4_BENCH Paper 1 - tables.

Produces every table a reviewer will expect. Three formats per table:
  - .md   for drafting (paste into paper markdown)
  - .tex  for submission (booktabs-style LaTeX)
  - .csv  for the data appendix

Main-text tables (paper/tables/main/):
  T1   dataset summary (per task × strategy)
  T2   regression headline R² + 95% CI
  T3   classification headline ROC-AUC + 95% CI
  T4   linear vs tree comparison
  T5   stacking experiment summary

Supplementary tables (paper/tables/supp/):
  T6   per-cell full results (data appendix)
  T7   leakage verification
  T8   seed robustness (mean ± std across 3 seeds)
  T9   threshold recalibration impact
  T10  calibration (ECE / MCE / Brier)
  T11  covalent accounting
  T12  best hyperparameters per cell

Usage:
    cd /home/nidhal/PAD4_BENCH
    python paper_tables.py
"""

import json
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_ROOT  = PROJECT_ROOT / "models_v1"
SPLITS_ROOT  = PROJECT_ROOT / "data" / "splits"
RESULTS_CSV  = MODELS_ROOT / "all_results.csv"
LEAKAGE_JSON = MODELS_ROOT / "leakage_verification.json"

OUT_MAIN = PROJECT_ROOT / "paper" / "tables" / "main"
OUT_SUPP = PROJECT_ROOT / "paper" / "tables" / "supp"

STRATEGIES = ["random", "scaffold", "confirmed", "lead_opt", "similarity", "cliff_aware"]
VARIANTS   = ["full", "fingerprints", "physchem", "mordred", "fragments"]
TASKS      = ["regression", "classification"]

STRAT_PRETTY = {
    "random": "Random", "scaffold": "Scaffold", "confirmed": "Confirmed",
    "lead_opt": "Lead-Opt", "similarity": "Similarity", "cliff_aware": "Cliff-Aware",
}
VARIANT_PRETTY = {
    "full": "Full", "fingerprints": "Fingerprints", "physchem": "PhysChem",
    "mordred": "Mordred", "fragments": "Fragments",
}


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------
def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save_table(df: pd.DataFrame, out_dir: Path, name: str,
                caption: str = "", label: str = "",
                index: bool = False) -> None:
    """Write a DataFrame as .md, .tex, and .csv into out_dir/<name>.{ext}."""
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"{name}.csv"
    df.to_csv(csv_path, index=index)

    md_path = out_dir / f"{name}.md"
    md = df.to_markdown(index=index, floatfmt=".4f")
    md_full = f"**{caption}**\n\n{md}\n" if caption else md
    md_path.write_text(md_full)

    tex_path = out_dir / f"{name}.tex"
    # Build LaTeX with booktabs
    cols = df.columns.tolist()
    col_spec = "l" + "r" * (len(cols) - 1) if not index else "l" + "r" * len(cols)
    tex_lines = [r"\begin{table}[ht]", r"\centering"]
    if caption:
        tex_lines.append(rf"\caption{{{caption}}}")
    if label:
        tex_lines.append(rf"\label{{{label}}}")
    tex_lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    tex_lines.append(r"\toprule")
    tex_lines.append(" & ".join(str(c) for c in cols) + r" \\")
    tex_lines.append(r"\midrule")
    for _, row in df.iterrows():
        cells = []
        for v in row.values:
            if isinstance(v, float):
                if np.isnan(v):
                    cells.append("--")
                else:
                    cells.append(f"{v:.4f}")
            else:
                # Escape underscores for LaTeX
                cells.append(str(v).replace("_", r"\_"))
        tex_lines.append(" & ".join(cells) + r" \\")
    tex_lines.append(r"\bottomrule")
    tex_lines.append(r"\end{tabular}")
    tex_lines.append(r"\end{table}")
    tex_path.write_text("\n".join(tex_lines))

    print(f"    wrote {csv_path.relative_to(PROJECT_ROOT)} (+ .md + .tex)",
          flush=True)


def fmt_with_ci(point: float, lo: float, hi: float, prec: int = 3) -> str:
    if any(pd.isna(x) for x in (point, lo, hi)):
        return "--"
    return f"{point:.{prec}f} [{lo:.{prec}f}, {hi:.{prec}f}]"


def fmt_meanstd(mean: float, std: float, prec: int = 3) -> str:
    if pd.isna(mean):
        return "--"
    if pd.isna(std):
        return f"{mean:.{prec}f}"
    return f"{mean:.{prec}f} ± {std:.{prec}f}"


def fmt_pct(v: float) -> str:
    return "--" if pd.isna(v) else f"{v:.1f}%"


# -----------------------------------------------------------------------------
# Loaders
# -----------------------------------------------------------------------------
def load_results() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_CSV)
    df = df[~df["metrics_path"].str.contains("smoketest", na=False)].copy()
    return df


# =============================================================================
# T1 - Dataset summary
# =============================================================================
def table_T1_dataset_summary() -> None:
    log("T1: dataset summary")
    ss = pd.read_csv(SPLITS_ROOT / "splits_summary.csv")
    # Regression sizes are already there. Get classification sizes by reading split CSVs.
    rows = []
    for strategy in STRATEGIES:
        r_train_sz = r_val_sz = r_test_sz = None
        c_train_sz = c_val_sz = c_test_sz = None
        for subset_name, attr in [("train", "n_tr"), ("val", "n_v"), ("test_locked", "n_te")]:
            rp = SPLITS_ROOT / "regression" / strategy / f"{subset_name}.csv"
            cp = SPLITS_ROOT / "classification" / strategy / f"{subset_name}.csv"
            if rp.exists():
                rn = len(pd.read_csv(rp, usecols=[0]))
                if subset_name == "train": r_train_sz = rn
                elif subset_name == "val": r_val_sz = rn
                else: r_test_sz = rn
            if cp.exists():
                cn = len(pd.read_csv(cp, usecols=[0]))
                if subset_name == "train": c_train_sz = cn
                elif subset_name == "val": c_val_sz = cn
                else: c_test_sz = cn

        ss_row = ss[ss["method"] == strategy]
        scaf_train = int(ss_row["n_scaffolds_train"].iloc[0]) if not ss_row.empty else None
        scaf_test  = int(ss_row["n_scaffolds_test"].iloc[0])  if not ss_row.empty else None
        pic_mean   = float(ss_row["test_pic50_mean"].iloc[0])  if not ss_row.empty else None
        pic_std    = float(ss_row["test_pic50_std"].iloc[0])   if not ss_row.empty else None
        pct_active = float(ss_row["test_pct_active"].iloc[0])  if not ss_row.empty else None
        cliff_cov  = ss_row["cliff_test_coverage_pct"].iloc[0] if not ss_row.empty else None

        rows.append({
            "Split":           STRAT_PRETTY[strategy],
            "Reg train":       r_train_sz,
            "Reg val":         r_val_sz,
            "Reg test":        r_test_sz,
            "Cls train":       c_train_sz,
            "Cls val":         c_val_sz,
            "Cls test":        c_test_sz,
            "Scaffolds train": scaf_train,
            "Scaffolds test":  scaf_test,
            "Test pIC50 (mean ± std)": f"{pic_mean:.3f} ± {pic_std:.3f}"
                                       if pic_mean is not None else "--",
            "Test % active":   fmt_pct(pct_active),
            "Cliff-derived test (%)": "--" if pd.isna(cliff_cov) else f"{cliff_cov:.1f}%",
        })
    df = pd.DataFrame(rows)
    save_table(df, OUT_MAIN, "T1_dataset_summary",
               caption="Dataset summary: per-split sizes, scaffold counts, "
                       "and test-set characteristics for the regression "
                       "(n=2,618) and classification (n=2,758) modeling sets.",
               label="tab:T1_dataset")


# =============================================================================
# T2 / T3 - Headline tables (regression and classification with CIs)
# =============================================================================
def _build_headline_table(df: pd.DataFrame, task: str, metric: str,
                           ci_lo: str, ci_hi: str, prec: int = 3) -> pd.DataFrame:
    sub = df[(df["task_dir"] == task) & (df["model"] == "xgboost")].copy()
    # Drop seed rows
    sub = sub[sub["seed_subdir"].isna()] if "seed_subdir" in sub.columns else sub
    rows = []
    for strategy in STRATEGIES:
        row = {"Split": STRAT_PRETTY[strategy]}
        for variant in VARIANTS:
            r = sub[(sub["strategy"] == strategy) & (sub["variant"] == variant)]
            if r.empty:
                row[VARIANT_PRETTY[variant]] = "--"
            else:
                pt = r[metric].iloc[0]
                lo = r[ci_lo].iloc[0]
                hi = r[ci_hi].iloc[0]
                row[VARIANT_PRETTY[variant]] = fmt_with_ci(pt, lo, hi, prec=prec)
        rows.append(row)
    return pd.DataFrame(rows)


def table_T2_regression_headline(df: pd.DataFrame) -> None:
    log("T2: regression headline (R² + 95% CI)")
    out = _build_headline_table(df, "regression", "test_r2",
                                  "test_ci_r2_lo", "test_ci_r2_hi", prec=3)
    save_table(out, OUT_MAIN, "T2_regression_headline_R2",
               caption="Regression test R² with 95\\% bootstrap CI (XGBoost on "
                       "tree-space, 1,000 resamples).",
               label="tab:T2_reg_R2")


def table_T3_classification_headline(df: pd.DataFrame) -> None:
    log("T3: classification headline (ROC-AUC + 95% CI)")
    out = _build_headline_table(df, "classification", "test_roc_auc",
                                  "test_ci_roc_auc_lo", "test_ci_roc_auc_hi",
                                  prec=3)
    save_table(out, OUT_MAIN, "T3_classification_headline_AUC",
               caption="Classification test ROC-AUC with 95\\% bootstrap CI "
                       "(XGBoost on tree-space, 1,000 resamples).",
               label="tab:T3_clf_AUC")


# =============================================================================
# T4 - Linear vs tree comparison
# =============================================================================
def table_T4_linear_vs_tree(df: pd.DataFrame) -> None:
    log("T4: linear vs tree comparison")
    rows = []
    sub = df.copy()
    sub = sub[sub["seed_subdir"].isna()] if "seed_subdir" in sub.columns else sub
    for strategy in STRATEGIES:
        for task, metric, tree_model, lin_model, label in [
            ("regression",     "test_r2",      "xgboost", "elasticnet",   "R²"),
            ("classification", "test_roc_auc", "xgboost", "logreg_enet",  "AUC"),
        ]:
            tree_vals, lin_vals = [], []
            for variant in VARIANTS:
                tr = sub[(sub["task_dir"] == task) & (sub["strategy"] == strategy) &
                          (sub["variant"] == variant) & (sub["model"] == tree_model)]
                ln = sub[(sub["task_dir"] == task) & (sub["strategy"] == strategy) &
                          (sub["variant"] == variant) & (sub["model"] == lin_model)]
                if not tr.empty:
                    tree_vals.append(tr[metric].iloc[0])
                if not ln.empty:
                    lin_vals.append(ln[metric].iloc[0])
            if tree_vals and lin_vals:
                rows.append({
                    "Split": STRAT_PRETTY[strategy],
                    "Task": task.capitalize(),
                    "Metric": label,
                    "XGBoost (tree)": fmt_meanstd(float(np.mean(tree_vals)),
                                                   float(np.std(tree_vals, ddof=1))),
                    "Linear-space": fmt_meanstd(float(np.mean(lin_vals)),
                                                  float(np.std(lin_vals, ddof=1))),
                    "Gap (tree − linear)": f"+{np.mean(tree_vals) - np.mean(lin_vals):.3f}",
                })
    out = pd.DataFrame(rows)
    save_table(out, OUT_MAIN, "T4_linear_vs_tree",
               caption="Tree-space (XGBoost) vs linear-space (ElasticNet / "
                       "LogisticRegression-ElasticNet) comparison, mean ± std "
                       "across 5 feature variants.",
               label="tab:T4_linear_tree")


# =============================================================================
# T5 - Stacking summary
# =============================================================================
def table_T5_stacking(df: pd.DataFrame) -> None:
    log("T5: stacking experiment summary")
    stk = df[df["task_dir"] == "stacking"].copy()
    if stk.empty:
        log("  WARN: no stacking cells found")
        return
    base = df[(df["task_dir"] == "classification") & (df["model"] == "xgboost")].copy()
    base_map = base.set_index(["strategy", "variant"])["test_roc_auc"].to_dict()

    rows = []
    for strategy in STRATEGIES:
        for variant in VARIANTS:
            stk_row = stk[(stk["strategy"] == strategy) &
                           (stk["variant"] == variant)]
            base_auc = base_map.get((strategy, variant))
            if stk_row.empty or base_auc is None:
                continue
            stk_auc = stk_row["test_roc_auc"].iloc[0]
            lift = stk_auc - base_auc
            rows.append({
                "Split": STRAT_PRETTY[strategy],
                "Variant": VARIANT_PRETTY[variant],
                "Baseline AUC": f"{base_auc:.4f}",
                "Stacked AUC":  f"{stk_auc:.4f}",
                "Δ AUC":         f"{lift:+.4f}",
            })
    out = pd.DataFrame(rows)
    mean_lift = float(out["Δ AUC"].apply(lambda s: float(s)).mean())
    n_pos = int(out["Δ AUC"].apply(lambda s: float(s) > 0).sum())
    save_table(out, OUT_MAIN, "T5_stacking",
               caption=f"Stacking experiment: per-cell stacked classifier "
                       f"(reg-OOF + cls-OOF → XGBoost) vs baseline classifier. "
                       f"Mean Δ AUC = {mean_lift:+.4f} across {len(out)} cells, "
                       f"{n_pos} cells with positive lift.",
               label="tab:T5_stacking")


# =============================================================================
# T6 - Per-cell full results
# =============================================================================
def table_T6_full_results(df: pd.DataFrame) -> None:
    log("T6: per-cell full results")
    keep_cols = [c for c in (
        "task_dir", "strategy", "variant", "model", "seed", "seed_subdir",
        "test_r2", "test_ci_r2_lo", "test_ci_r2_med", "test_ci_r2_hi",
        "test_rmse", "test_ci_rmse_lo", "test_ci_rmse_med", "test_ci_rmse_hi",
        "test_mae",
        "test_roc_auc", "test_ci_roc_auc_lo", "test_ci_roc_auc_med",
        "test_ci_roc_auc_hi",
        "test_balanced_accuracy", "test_mcc", "test_f1_macro", "test_brier",
        "test_tuned_balanced_accuracy", "test_tuned_mcc", "test_tuned_f1_macro",
        "tuned_threshold",
        "test_calibration_ece", "test_calibration_mce",
        "oof_r2", "oof_rmse", "oof_roc_auc",
        "val_r2", "val_rmse", "val_roc_auc",
        "metrics_path",
    ) if c in df.columns]
    out = df[keep_cols].copy()
    # Sort
    out["strategy"] = pd.Categorical(out["strategy"], STRATEGIES, ordered=True)
    out["variant"]  = pd.Categorical(out["variant"], VARIANTS, ordered=True)
    out = out.sort_values(["task_dir", "strategy", "variant", "model",
                            "seed_subdir"])
    save_table(out, OUT_SUPP, "T6_per_cell_full_results",
               caption="Per-cell full results: every modeling cell with its "
                       "headline metrics, bootstrap confidence intervals, "
                       "tuned-threshold metrics, calibration, and "
                       "cross-validation diagnostics.",
               label="tab:T6_full_results")


# =============================================================================
# T7 - Leakage verification
# =============================================================================
def table_T7_leakage() -> None:
    log("T7: leakage verification")
    if not LEAKAGE_JSON.exists():
        log(f"  WARN: {LEAKAGE_JSON} missing")
        return
    leak = json.loads(LEAKAGE_JSON.read_text())
    rows = []
    for cell in leak.get("cells", []):
        task = cell.get("task")
        strategy = cell.get("strategy")
        ik = cell.get("inchikey_disjointness", {})
        tan = cell.get("ecfp4_tanimoto_train_to_test", {})
        cliff = cell.get("cliff_check") or {}
        rows.append({
            "Task": task,
            "Split": STRAT_PRETTY.get(strategy, strategy),
            "n train": cell.get("n_train"),
            "n val":   cell.get("n_val"),
            "n test":  cell.get("n_test"),
            "InChIKey disjoint": "✓" if ik.get("all_disjoint") else "✗",
            "Scaffold overlap": cell.get("scaffold_overlap_train_test"),
            "Tanimoto max": f"{tan.get('max', np.nan):.3f}"
                            if tan.get("max") is not None else "--",
            "Tanimoto mean-of-max": f"{tan.get('mean_of_max_per_test', np.nan):.3f}"
                                     if tan.get("mean_of_max_per_test") is not None else "--",
            "Tanimoto p95": f"{tan.get('p95_of_max_per_test', np.nan):.3f}"
                            if tan.get("p95_of_max_per_test") is not None else "--",
            "Cliff-test fraction":
                "--" if not cliff or "fraction_cliff_derived_test" not in cliff
                else f"{cliff['fraction_cliff_derived_test']*100:.1f}%",
        })
    out = pd.DataFrame(rows)
    save_table(out, OUT_SUPP, "T7_leakage_verification",
               caption="Leakage verification: per-cell train/val/test "
                       "InChIKey disjointness, scaffold overlap, ECFP4 "
                       "Tanimoto distribution, and (for cliff\\_aware) "
                       "fraction of cliff-derived test compounds.",
               label="tab:T7_leakage")


# =============================================================================
# T8 - Seed robustness
# =============================================================================
def table_T8_seed_robustness() -> None:
    log("T8: seed robustness")
    rows = []
    for task, metric_key, label in [("regression", "r2", "R²"),
                                      ("classification", "roc_auc", "ROC-AUC")]:
        for strategy in ["random", "scaffold"]:
            for variant in VARIANTS:
                base = MODELS_ROOT / task / strategy / variant / "xgboost"
                seed_paths = [(42, base / "metrics.json")]
                for seed in [7, 1337]:
                    seed_paths.append((seed, base / "seeds" / f"seed_{seed}" / "metrics.json"))
                vals = []
                for seed, p in seed_paths:
                    if p.exists():
                        m = json.loads(p.read_text())
                        v = m.get("test", {}).get(metric_key)
                        if v is not None:
                            vals.append(float(v))
                if len(vals) >= 2:
                    rows.append({
                        "Task": task.capitalize(),
                        "Split": STRAT_PRETTY[strategy],
                        "Variant": VARIANT_PRETTY[variant],
                        "Metric": label,
                        "Mean": f"{np.mean(vals):.4f}",
                        "Std": f"{np.std(vals, ddof=1):.4f}",
                        "Min": f"{np.min(vals):.4f}",
                        "Max": f"{np.max(vals):.4f}",
                        "n seeds": len(vals),
                    })
    out = pd.DataFrame(rows)
    save_table(out, OUT_SUPP, "T8_seed_robustness",
               caption="Seed robustness across 3 seeds (42, 7, 1337) on "
                       "random and scaffold splits.",
               label="tab:T8_robustness")


# =============================================================================
# T9 - Threshold recalibration
# =============================================================================
def table_T9_threshold_recal(df: pd.DataFrame) -> None:
    log("T9: threshold recalibration impact")
    d = df[(df["task_dir"] == "classification") &
            (df["model"].isin(["xgboost", "logreg_enet"]))].copy()
    d = d[d["seed_subdir"].isna()] if "seed_subdir" in d.columns else d
    needed = ["test_mcc", "test_tuned_mcc", "tuned_threshold",
               "test_balanced_accuracy", "test_tuned_balanced_accuracy"]
    if any(c not in d.columns for c in needed):
        log(f"  WARN: missing one of {needed}")
        return
    d["strategy"] = pd.Categorical(d["strategy"], STRATEGIES, ordered=True)
    d["variant"]  = pd.Categorical(d["variant"], VARIANTS, ordered=True)
    d = d.sort_values(["strategy", "variant", "model"])

    rows = []
    for _, r in d.iterrows():
        rows.append({
            "Split":   STRAT_PRETTY[r["strategy"]],
            "Variant": VARIANT_PRETTY[r["variant"]],
            "Model":   r["model"],
            "Default MCC (thr=0.5)":  f"{r['test_mcc']:.3f}",
            "Tuned threshold":        f"{r['tuned_threshold']:.3f}",
            "Tuned MCC":              f"{r['test_tuned_mcc']:.3f}",
            "Δ MCC":                  f"{r['test_tuned_mcc'] - r['test_mcc']:+.3f}",
            "Tuned BalAcc":           f"{r['test_tuned_balanced_accuracy']:.3f}",
        })
    out = pd.DataFrame(rows)
    save_table(out, OUT_SUPP, "T9_threshold_recalibration",
               caption="Threshold recalibration impact: MCC and balanced "
                       "accuracy at default 0.5 vs Youden's J optimum threshold "
                       "(tuned on validation set).",
               label="tab:T9_threshold")


# =============================================================================
# T10 - Calibration
# =============================================================================
def table_T10_calibration(df: pd.DataFrame) -> None:
    log("T10: calibration per cell")
    d = df[(df["task_dir"] == "classification") &
            (df["model"].isin(["xgboost", "logreg_enet"]))].copy()
    d = d[d["seed_subdir"].isna()] if "seed_subdir" in d.columns else d
    if "test_calibration_ece" not in d.columns:
        log("  WARN: test_calibration_ece missing")
        return
    d["strategy"] = pd.Categorical(d["strategy"], STRATEGIES, ordered=True)
    d["variant"]  = pd.Categorical(d["variant"], VARIANTS, ordered=True)
    d = d.sort_values(["strategy", "variant", "model"])
    rows = []
    for _, r in d.iterrows():
        ece = r.get("test_calibration_ece", np.nan)
        mce = r.get("test_calibration_mce", np.nan)
        brier = r.get("test_brier", np.nan)
        verdict = ("well-calibrated" if (ece < 0.05) else
                   ("moderate"        if (ece < 0.10) else "poor"))
        rows.append({
            "Split":   STRAT_PRETTY[r["strategy"]],
            "Variant": VARIANT_PRETTY[r["variant"]],
            "Model":   r["model"],
            "ECE":     f"{ece:.4f}" if pd.notna(ece) else "--",
            "MCE":     f"{mce:.4f}" if pd.notna(mce) else "--",
            "Brier":   f"{brier:.4f}" if pd.notna(brier) else "--",
            "Verdict": verdict,
        })
    out = pd.DataFrame(rows)
    save_table(out, OUT_SUPP, "T10_calibration",
               caption="Per-cell calibration: Expected Calibration Error (ECE), "
                       "Maximum Calibration Error (MCE), and Brier score. "
                       "ECE \\textless 0.05 = well-calibrated, "
                       "0.05--0.10 = moderate, \\textgreater 0.10 = poor.",
               label="tab:T10_calibration")


# =============================================================================
# T11 - Covalent accounting
# =============================================================================
def table_T11_covalent() -> None:
    log("T11: covalent accounting")
    rows = [
        {"Set": "pad_t1_covalent (reference)",
         "n total": "36",
         "Irreversible covalent": "36",
         "Reversible covalent": "0",
         "Treatment": "EXCLUDED from all Paper 1 splits"},
        {"Set": "pad_t1_non_covalent (regression base)",
         "n total": "2,618",
         "Irreversible covalent": "0",
         "Reversible covalent": "9",
         "Treatment": "USED — base for random/scaffold/similarity/lead_opt/cliff_aware splits"},
        {"Set": "pad_classification_v17 (classification)",
         "n total": "2,758",
         "Irreversible covalent": "36",
         "Reversible covalent": "9",
         "Treatment": "USED — both classes retained for binary activity prediction"},
        {"Set": "pad_t1_confirmed (Confirmed split)",
         "n total": "1,623",
         "Irreversible covalent": "20",
         "Reversible covalent": "9",
         "Treatment": "USED — covalents retained as quality-controlled stratification subset"},
    ]
    out = pd.DataFrame(rows)
    save_table(out, OUT_SUPP, "T11_covalent_accounting",
               caption="Covalent-inhibitor accounting across Paper 1 data "
                       "tiers. Reversible covalent compounds (boronic acids / "
                       "activated nitriles) are retained because they bind "
                       "competitively in equilibrium; irreversible covalents "
                       "(acrylamides, vinyl sulfones) are excluded from the "
                       "regression base set.",
               label="tab:T11_covalent")


# =============================================================================
# T12 - Best hyperparameters per cell
# =============================================================================
def table_T12_hyperparameters() -> None:
    log("T12: best hyperparameters per cell")
    rows = []
    for task, models in [("regression", ["xgboost", "elasticnet"]),
                          ("classification", ["xgboost", "logreg_enet"])]:
        for strategy in STRATEGIES:
            for variant in VARIANTS:
                for model in models:
                    hp_path = MODELS_ROOT / task / strategy / variant / model / "hparams.json"
                    tn_path = MODELS_ROOT / task / strategy / variant / model / "tuning_results.json"
                    if not hp_path.exists():
                        continue
                    hp = json.loads(hp_path.read_text())
                    cv_score = None
                    if tn_path.exists():
                        tn = json.loads(tn_path.read_text())
                        cv_score = tn.get("best_cv_score")
                    rows.append({
                        "Task":     task,
                        "Split":    STRAT_PRETTY[strategy],
                        "Variant":  VARIANT_PRETTY[variant],
                        "Model":    model,
                        "CV score": f"{cv_score:.4f}" if cv_score is not None else "--",
                        "Hyperparameters": json.dumps(hp, separators=(",", ":")),
                    })
    out = pd.DataFrame(rows)
    save_table(out, OUT_SUPP, "T12_hyperparameters",
               caption="Best hyperparameters per cell from grid search "
                       "(18 configs for XGBoost, 15 configs for linear models, "
                       "5-fold CV on train, refit on full train).",
               label="tab:T12_hparams")


# =============================================================================
# Main
# =============================================================================
def main() -> int:
    OUT_MAIN.mkdir(parents=True, exist_ok=True)
    OUT_SUPP.mkdir(parents=True, exist_ok=True)
    log(f"output (main): {OUT_MAIN}")
    log(f"output (supp): {OUT_SUPP}")

    df = load_results()
    log(f"loaded {len(df)} cell rows from {RESULTS_CSV.name}")

    # Main
    table_T1_dataset_summary()
    table_T2_regression_headline(df)
    table_T3_classification_headline(df)
    table_T4_linear_vs_tree(df)
    table_T5_stacking(df)

    # Supplementary
    table_T6_full_results(df)
    table_T7_leakage()
    table_T8_seed_robustness()
    table_T9_threshold_recal(df)
    table_T10_calibration(df)
    table_T11_covalent()
    table_T12_hyperparameters()

    log("DONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
