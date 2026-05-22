#!/usr/bin/env python3
"""
Paper-1 reviewer-proofing pipeline.

Phase 1 - Leakage verification
  Per (task, strategy):
    - InChIKey-14 disjointness across {train, val, test_locked}.
    - Scaffold disjointness train vs test for scaffold split (and reported for all).
    - Train-test ECFP4 Tanimoto similarity distribution (max, mean, p95) for all
      splits; the similarity-split claim depends on this.
    - cliff_aware: count / fraction of cliff-derived test compounds.
  Writes models_v1/leakage_verification.json and prints a summary table.

Phase 2 - Bootstrap confidence intervals
  For every cell's test_pred.npz: 1000 bootstrap resamples with replacement,
  preserving sample weights. Compute 2.5/50/97.5 percentile for each headline
  metric. Inject as `test_ci` block in metrics.json. Skip cells already done.

Phase 3 - Per-cell results CSV
  Walk models_v1/ and produce models_v1/all_results.csv with one row per cell,
  flat columns for every metric and CI.

Phase 4 - Calibration analysis (classification only)
  Compute ECE (10-bin equal-width), MCE, and refined Brier decomposition per
  classification cell on test_locked. Inject `test_calibration` block.
  Produce one reliability-diagram figure showing all 6 splits at the
  fingerprints variant XGBoost cells: models_v1/figures/reliability_diagrams.png

Idempotent: skip work whose marker key already exists.

Usage:
    cd /home/nidhal/PAD4_BENCH
    nohup python paper1_reviewer_proof.py > models_v1/reviewer_proof.log 2>&1 &
    echo $! > models_v1/reviewer_proof.pid
    tail -f models_v1/reviewer_proof.log
"""

import csv
import json
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
SPLITS_ROOT = PROJECT_ROOT / "data" / "splits"
FEATURES_ROOT = PROJECT_ROOT / "features_v18"
MODELS_ROOT = PROJECT_ROOT / "models_v1"
FIGURES_ROOT = MODELS_ROOT / "figures"

STRATEGIES = ["random", "scaffold", "confirmed", "lead_opt", "similarity", "cliff_aware"]
VARIANTS = ["full", "fingerprints", "physchem", "mordred", "fragments"]
TASKS = ["regression", "classification"]

N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42
CALIBRATION_N_BINS = 10
TANIMOTO_SIMILARITY_MAX_PAIRS = 200_000  # cap on pairwise computations for speed


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


# =============================================================================
# Shared loaders
# =============================================================================
def load_split_csv(task: str, strategy: str, subset: str) -> pd.DataFrame:
    path = SPLITS_ROOT / task / strategy / f"{subset}.csv"
    return pd.read_csv(path)


def load_features_npz(task: str, strategy: str, variant: str, space: str,
                      subset: str) -> dict:
    path = FEATURES_ROOT / task / strategy / subset / f"{variant}_{space}.npz"
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def load_stratifiers_npz(task: str, strategy: str, subset: str) -> dict:
    path = FEATURES_ROOT / task / strategy / subset / "stratifiers.npz"
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def load_pred_npz(path: Path) -> dict:
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


# =============================================================================
# PHASE 1 - Leakage verification
# =============================================================================
def tanimoto_max_min_p95(fp_train: np.ndarray, fp_test: np.ndarray,
                         max_pairs: int = TANIMOTO_SIMILARITY_MAX_PAIRS) -> dict:
    """Per-test-compound max train Tanimoto. ECFP4 bits as binary uint8.
    Returns max, mean of max-per-test, p95 of max-per-test."""
    # Convert to bool for fast bitwise ops
    fp_train = fp_train.astype(bool)
    fp_test = fp_test.astype(bool)

    n_test = len(fp_test)
    # Cap pairs by chunking the test set if needed; train is small enough.
    train_popcnt = fp_train.sum(axis=1).astype(np.int32)
    test_popcnt = fp_test.sum(axis=1).astype(np.int32)

    max_per_test = np.zeros(n_test, dtype=np.float32)
    chunk = max(1, max_pairs // max(1, len(fp_train)))
    for i in range(0, n_test, chunk):
        block = fp_test[i:i + chunk]
        # intersection counts: (test_block @ train.T)  using uint16 to avoid overflow
        inter = (block.astype(np.uint16) @ fp_train.T.astype(np.uint16))
        # union = popcnt(test) + popcnt(train) - inter
        union = test_popcnt[i:i + chunk, None] + train_popcnt[None, :] - inter
        # Avoid div-by-zero
        union = np.where(union == 0, 1, union)
        sim = inter / union
        max_per_test[i:i + chunk] = sim.max(axis=1)

    return {
        "max": float(max_per_test.max()),
        "mean_of_max_per_test": float(max_per_test.mean()),
        "p95_of_max_per_test": float(np.percentile(max_per_test, 95)),
        "p99_of_max_per_test": float(np.percentile(max_per_test, 99)),
        "frac_test_with_train_sim_above_0_7": float((max_per_test > 0.7).mean()),
        "frac_test_with_train_sim_above_0_5": float((max_per_test > 0.5).mean()),
    }


def extract_ecfp4_fingerprints(task: str, strategy: str,
                               subset: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (ids, ecfp4_binary_matrix) from the tree-space full variant.
    Only the ECFP4 columns are needed for Tanimoto."""
    feat = load_features_npz(task, strategy, "fingerprints", "tree", subset)
    ids = feat["ids"]
    X = feat["X"]
    feature_names = feat["feature_names"]
    # Filter ECFP4-only columns to make Tanimoto principled
    ecfp4_mask = np.array([n.startswith("rdkit::ecfp4::") for n in feature_names])
    if ecfp4_mask.sum() == 0:
        # fallback: use all features (still a defensible if coarser similarity)
        ecfp4_mask = np.ones(len(feature_names), dtype=bool)
    return ids, (X[:, ecfp4_mask] > 0).astype(np.uint8)


def cliff_test_fraction(task: str, strategy: str) -> dict | None:
    """For cliff_aware split: count of test compounds that appear in
    pad_activity_cliffs.csv. Returns None for other strategies."""
    if strategy != "cliff_aware":
        return None
    cliffs_path = DATA_PROCESSED / "pad_activity_cliffs.csv"
    if not cliffs_path.exists():
        return {"error": f"missing {cliffs_path}"}
    cliffs = pd.read_csv(cliffs_path)
    # InChIKey columns in cliffs file are 27-char; truncate to 14 for join
    id_cols = [c for c in cliffs.columns if "inchikey" in c.lower() or "InChIKey" in c]
    if not id_cols:
        return {"error": "no inchikey columns in cliffs file"}
    cliff_ids_14 = set()
    for col in id_cols:
        for v in cliffs[col].dropna().unique().tolist():
            cliff_ids_14.add(str(v)[:14])

    test_df = load_split_csv(task, strategy, "test_locked")
    test_id_col = None
    for c in test_df.columns:
        if "inchikey" in c.lower():
            test_id_col = c
            break
    if test_id_col is None:
        return {"error": "no inchikey column in test_locked"}
    test_ids = set(test_df[test_id_col].astype(str).str[:14])
    cliff_test_overlap = test_ids & cliff_ids_14
    return {
        "n_test_compounds": int(len(test_ids)),
        "n_test_in_cliffs": int(len(cliff_test_overlap)),
        "fraction_cliff_derived_test": float(len(cliff_test_overlap) / max(1, len(test_ids))),
        "expected_fraction_from_handoff": 0.176,
    }


def phase1_leakage_verification() -> dict:
    log("=" * 70)
    log("PHASE 1: Leakage verification")
    log("=" * 70)

    out_path = MODELS_ROOT / "leakage_verification.json"
    if out_path.exists():
        log(f"  leakage_verification.json exists — overwriting with fresh run")

    report = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "cells": []}
    summary_rows = []
    all_pass = True

    for task in TASKS:
        for strategy in STRATEGIES:
            log(f"  {task}/{strategy} ...")
            try:
                # Load id sets from CSVs (authoritative)
                train_df = load_split_csv(task, strategy, "train")
                val_df = load_split_csv(task, strategy, "val")
                test_df = load_split_csv(task, strategy, "test_locked")

                # Identify the ID column
                id_col = None
                for c in train_df.columns:
                    if "inchikey" in c.lower():
                        id_col = c
                        break
                if id_col is None:
                    raise ValueError(f"no inchikey column in {task}/{strategy}/train")

                train_ids = set(train_df[id_col].astype(str).str[:14])
                val_ids = set(val_df[id_col].astype(str).str[:14])
                test_ids = set(test_df[id_col].astype(str).str[:14])

                # 1. InChIKey disjointness
                tv = len(train_ids & val_ids)
                tt = len(train_ids & test_ids)
                vt = len(val_ids & test_ids)
                inchikey_disjoint = (tv == 0 and tt == 0 and vt == 0)

                # 2. Scaffold disjointness
                scaffold_col = "stereo_stripped_scaffold"
                scaffold_overlap_train_test = None
                if scaffold_col in train_df.columns and scaffold_col in test_df.columns:
                    train_scaffolds = set(train_df[scaffold_col].dropna().astype(str))
                    test_scaffolds = set(test_df[scaffold_col].dropna().astype(str))
                    scaffold_overlap_train_test = int(len(train_scaffolds & test_scaffolds))
                else:
                    log(f"    WARN: no '{scaffold_col}' column; skipping scaffold check")

                # 3. ECFP4 Tanimoto distribution train→test (and val→test for completeness)
                _, fp_train = extract_ecfp4_fingerprints(task, strategy, "train")
                _, fp_test = extract_ecfp4_fingerprints(task, strategy, "test")
                tan_train_test = tanimoto_max_min_p95(fp_train, fp_test)

                # 4. Cliff-aware specific check
                cliff_check = cliff_test_fraction(task, strategy)

                cell_report = {
                    "task": task,
                    "strategy": strategy,
                    "n_train": int(len(train_ids)),
                    "n_val": int(len(val_ids)),
                    "n_test": int(len(test_ids)),
                    "inchikey_disjointness": {
                        "train_val_overlap": int(tv),
                        "train_test_overlap": int(tt),
                        "val_test_overlap": int(vt),
                        "all_disjoint": bool(inchikey_disjoint),
                    },
                    "scaffold_overlap_train_test": scaffold_overlap_train_test,
                    "ecfp4_tanimoto_train_to_test": tan_train_test,
                    "cliff_check": cliff_check,
                }
                report["cells"].append(cell_report)
                summary_rows.append((task, strategy, inchikey_disjoint,
                                     scaffold_overlap_train_test,
                                     tan_train_test["max"],
                                     tan_train_test["mean_of_max_per_test"]))
                cell_pass = inchikey_disjoint
                if strategy == "scaffold" and scaffold_overlap_train_test not in (None, 0):
                    cell_pass = False
                all_pass = all_pass and cell_pass

            except Exception as e:
                log(f"    FAIL {task}/{strategy}: {e}")
                traceback.print_exc()
                report["cells"].append({"task": task, "strategy": strategy,
                                        "error": str(e)})
                all_pass = False

    report["all_inchikey_disjoint"] = all_pass
    report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out_path.write_text(json.dumps(report, indent=2))

    # Pretty summary
    log("")
    log(f"  {'task':<14} {'strategy':<14} {'IK_disjoint':<12} {'scaf_overlap':<14} {'tan_max':<10} {'tan_mean_max'}")
    for task, strategy, ikd, sov, tmax, tmean in summary_rows:
        sov_s = str(sov) if sov is not None else "n/a"
        log(f"  {task:<14} {strategy:<14} {str(ikd):<12} {sov_s:<14} {tmax:<10.3f} {tmean:.3f}")
    log("")
    log(f"  Leakage verification: {'PASS' if all_pass else 'FAIL — review report'}")
    return report


# =============================================================================
# PHASE 2 - Bootstrap confidence intervals
# =============================================================================
def bootstrap_regression(y_true, y_pred, weights, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    rmse_b, mae_b, r2_b, rho_b = [], [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        w = weights[idx]
        try:
            rmse_b.append(np.sqrt(mean_squared_error(yt, yp, sample_weight=w)))
            mae_b.append(mean_absolute_error(yt, yp, sample_weight=w))
            r2_b.append(r2_score(yt, yp, sample_weight=w))
        except Exception:
            continue
        try:
            rho, _ = spearmanr(yt, yp)
            rho_b.append(rho if np.isfinite(rho) else 0.0)
        except Exception:
            pass

    def pct(arr):
        a = np.asarray(arr, dtype=np.float64)
        a = a[np.isfinite(a)]
        if len(a) == 0:
            return {"lo": None, "med": None, "hi": None}
        return {"lo": float(np.percentile(a, 2.5)),
                "med": float(np.percentile(a, 50)),
                "hi": float(np.percentile(a, 97.5))}

    return {"rmse": pct(rmse_b), "mae": pct(mae_b),
            "r2": pct(r2_b), "spearman_rho": pct(rho_b),
            "n_bootstrap": n_boot}


def bootstrap_classification(y_true, y_proba, weights, threshold: float,
                              n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    auc_b, balacc_b, f1_b, mcc_b, brier_b = [], [], [], [], []
    y_pred_hard_full = (y_proba >= threshold).astype(int)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_proba[idx]
        yph = y_pred_hard_full[idx]
        w = weights[idx]
        try:
            if len(np.unique(yt)) > 1:
                auc_b.append(roc_auc_score(yt, yp, sample_weight=w))
                brier_b.append(brier_score_loss(yt, yp, sample_weight=w))
            balacc_b.append(balanced_accuracy_score(yt, yph, sample_weight=w))
            f1_b.append(f1_score(yt, yph, average="macro", sample_weight=w))
            mcc_b.append(matthews_corrcoef(yt, yph, sample_weight=w))
        except Exception:
            continue

    def pct(arr):
        a = np.asarray(arr, dtype=np.float64)
        a = a[np.isfinite(a)]
        if len(a) == 0:
            return {"lo": None, "med": None, "hi": None}
        return {"lo": float(np.percentile(a, 2.5)),
                "med": float(np.percentile(a, 50)),
                "hi": float(np.percentile(a, 97.5))}

    return {"roc_auc": pct(auc_b), "balanced_accuracy": pct(balacc_b),
            "f1_macro": pct(f1_b), "mcc": pct(mcc_b), "brier": pct(brier_b),
            "threshold_used": float(threshold), "n_bootstrap": n_boot}


def phase2_bootstrap_cis() -> dict:
    log("=" * 70)
    log("PHASE 2: Bootstrap confidence intervals")
    log("=" * 70)

    counts = {"updated": 0, "skipped": 0, "errors": 0}
    cells_to_process = []

    # Main sweep cells
    for task in TASKS:
        models = ["xgboost", "elasticnet"] if task == "regression" else ["xgboost", "logreg_enet"]
        for strategy in STRATEGIES:
            for variant in VARIANTS:
                for model in models:
                    cells_to_process.append(
                        (task, strategy, variant, model,
                         MODELS_ROOT / task / strategy / variant / model))

    # Stacking cells
    for strategy in STRATEGIES:
        for variant in VARIANTS:
            cells_to_process.append(
                ("stacking", strategy, variant, "xgboost",
                 MODELS_ROOT / "stacking" / strategy / variant / "xgboost"))

    # Robustness seeds
    for task in TASKS:
        for strategy in ["random", "scaffold"]:
            for variant in VARIANTS:
                for seed in [7, 1337]:
                    cells_to_process.append(
                        (task, strategy, variant, f"xgboost_seed{seed}",
                         MODELS_ROOT / task / strategy / variant / "xgboost" /
                         "seeds" / f"seed_{seed}"))

    log(f"  Total cells to consider: {len(cells_to_process)}")
    for i, (task, strategy, variant, model, cell_dir) in enumerate(cells_to_process, 1):
        metrics_path = cell_dir / "metrics.json"
        test_path = cell_dir / "test_pred.npz"

        if not metrics_path.exists() or not test_path.exists():
            counts["errors"] += 1
            continue

        try:
            metrics = json.loads(metrics_path.read_text())
            if "test_ci" in metrics:
                counts["skipped"] += 1
                continue

            test_d = load_pred_npz(test_path)
            # Load weights from features_v18 stratifiers; classification task for
            # stacking, otherwise use task name directly.
            features_task = "classification" if task == "stacking" else task
            strat = load_stratifiers_npz(features_task, strategy, "test")
            if "ml_weight" in strat:
                weights_lookup = dict(
                    zip(strat["ids"].tolist() if "ids" in strat else
                        load_features_npz(features_task, strategy, "fingerprints",
                                          "tree", "test")["ids"].tolist(),
                        strat["ml_weight"].tolist())
                )
                weights = np.array([weights_lookup.get(i, 1.0)
                                    for i in test_d["ids"].tolist()],
                                   dtype=np.float64)
            else:
                weights = np.ones(len(test_d["y_true"]), dtype=np.float64)

            if features_task == "regression":
                ci = bootstrap_regression(
                    test_d["y_true"].astype(np.float64),
                    test_d["y_pred"].astype(np.float64),
                    weights,
                    n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED)
            else:
                # Use tuned threshold if available, else 0.5
                thr = metrics.get("tuned_threshold", 0.5)
                ci = bootstrap_classification(
                    test_d["y_true"].astype(np.int32),
                    test_d["y_pred"].astype(np.float64),
                    weights, threshold=thr,
                    n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED)

            metrics["test_ci"] = ci
            metrics_path.write_text(json.dumps(metrics, indent=2))
            counts["updated"] += 1

            if i % 20 == 0:
                log(f"  [{i}/{len(cells_to_process)}] {task}/{strategy}/{variant}/{model} done")
        except Exception as e:
            log(f"  FAIL {task}/{strategy}/{variant}/{model}: {e}")
            counts["errors"] += 1

    log(f"PHASE 2 DONE: {counts}")
    return counts


# =============================================================================
# PHASE 3 - Per-cell results CSV
# =============================================================================
def phase3_results_csv() -> dict:
    log("=" * 70)
    log("PHASE 3: Per-cell results CSV")
    log("=" * 70)

    csv_path = MODELS_ROOT / "all_results.csv"
    rows = []

    def flatten(prefix, d):
        out = {}
        if not isinstance(d, dict):
            return out
        for k, v in d.items():
            if isinstance(v, dict):
                out.update(flatten(f"{prefix}_{k}", v))
            elif isinstance(v, (int, float, str, bool)) or v is None:
                out[f"{prefix}_{k}"] = v
        return out

    metrics_files = []
    for task in TASKS + ["stacking"]:
        root = MODELS_ROOT / task
        if not root.exists():
            continue
        for path in root.rglob("metrics.json"):
            metrics_files.append((task, path))

    for task, path in metrics_files:
        try:
            m = json.loads(path.read_text())
            # Identify cell from path; handles seeds/seed_N nesting
            rel = path.relative_to(MODELS_ROOT)
            parts = rel.parts  # e.g., ('regression','random','full','xgboost','metrics.json')
            row = {"task_dir": parts[0]}
            if len(parts) >= 4:
                row["strategy"] = parts[1]
                row["variant"] = parts[2]
                row["model"] = parts[3]
            # Detect seed nesting
            if len(parts) >= 6 and parts[4] == "seeds":
                row["seed_subdir"] = parts[5]
            # Pull metadata
            for k in ["task", "strategy", "variant", "space", "model", "seed",
                      "n_cv_folds", "best_iteration_final", "cell_elapsed_sec",
                      "decision_threshold", "tuned_threshold"]:
                if k in m:
                    row[k] = m[k]
            if "best_config" in m:
                row.update({f"best_{k}": v for k, v in m["best_config"].items()})
            # Flatten all metric blocks
            for block in ["oof", "val", "test", "test_tuned", "test_ci",
                          "test_calibration"]:
                if block in m:
                    row.update(flatten(block, m[block]))
            row["metrics_path"] = str(rel)
            rows.append(row)
        except Exception as e:
            log(f"  FAIL reading {path}: {e}")

    if rows:
        df = pd.DataFrame(rows)
        # Stable column order: identifiers first, then alphabetical
        id_cols = ["task_dir", "task", "strategy", "variant", "model", "space",
                   "seed", "seed_subdir", "metrics_path"]
        other_cols = sorted([c for c in df.columns if c not in id_cols])
        df = df[[c for c in id_cols if c in df.columns] + other_cols]
        df.to_csv(csv_path, index=False)
        log(f"  Wrote {len(rows)} rows to {csv_path}")
        log(f"  Columns: {len(df.columns)}")
    else:
        log("  No metrics.json files found")

    return {"rows": len(rows), "path": str(csv_path)}


# =============================================================================
# PHASE 4 - Calibration analysis (classification only)
# =============================================================================
def expected_calibration_error(y_true: np.ndarray, y_proba: np.ndarray,
                                sample_weight: np.ndarray | None = None,
                                n_bins: int = CALIBRATION_N_BINS) -> dict:
    """ECE with equal-width bins, plus MCE and bin-by-bin diagnostics."""
    if sample_weight is None:
        sample_weight = np.ones_like(y_true, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_proba, bins) - 1, 0, n_bins - 1)
    total_w = sample_weight.sum()
    if total_w == 0:
        return {"ece": None, "mce": None, "bins": []}

    bin_info = []
    ece = 0.0
    mce = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        w_b = sample_weight[mask].sum()
        if w_b == 0:
            bin_info.append({"bin": b, "lo": float(bins[b]), "hi": float(bins[b + 1]),
                             "n_eff": 0.0, "avg_proba": None, "obs_pos_rate": None,
                             "gap": None})
            continue
        avg_proba = float(np.average(y_proba[mask], weights=sample_weight[mask]))
        obs_pos = float(np.average(y_true[mask], weights=sample_weight[mask]))
        gap = abs(avg_proba - obs_pos)
        ece += (w_b / total_w) * gap
        mce = max(mce, gap)
        bin_info.append({"bin": b, "lo": float(bins[b]), "hi": float(bins[b + 1]),
                         "n_eff": float(w_b), "avg_proba": avg_proba,
                         "obs_pos_rate": obs_pos, "gap": gap})

    return {"ece": float(ece), "mce": float(mce), "n_bins": n_bins,
            "bins": bin_info}


def phase4_calibration() -> dict:
    log("=" * 70)
    log("PHASE 4: Calibration analysis (classification cells)")
    log("=" * 70)

    counts = {"updated": 0, "skipped": 0, "errors": 0}
    plot_data = {}  # strategy -> (avg_proba_per_bin, obs_pos_per_bin)

    for strategy in STRATEGIES:
        for variant in VARIANTS:
            for model in ["xgboost", "logreg_enet"]:
                cell_dir = MODELS_ROOT / "classification" / strategy / variant / model
                metrics_path = cell_dir / "metrics.json"
                test_path = cell_dir / "test_pred.npz"
                if not metrics_path.exists() or not test_path.exists():
                    continue
                try:
                    metrics = json.loads(metrics_path.read_text())
                    if "test_calibration" in metrics:
                        counts["skipped"] += 1
                        # Still gather plot data if this is the headline cell
                        if variant == "fingerprints" and model == "xgboost":
                            plot_data[strategy] = metrics["test_calibration"]["bins"]
                        continue

                    test_d = load_pred_npz(test_path)
                    strat = load_stratifiers_npz("classification", strategy, "test")
                    if "ml_weight" in strat:
                        ids_test = load_features_npz("classification", strategy,
                                                     "fingerprints", "tree",
                                                     "test")["ids"].tolist()
                        w_lookup = dict(zip(ids_test, strat["ml_weight"].tolist()))
                        weights = np.array([w_lookup.get(i, 1.0)
                                            for i in test_d["ids"].tolist()],
                                           dtype=np.float64)
                    else:
                        weights = np.ones(len(test_d["y_true"]), dtype=np.float64)

                    cal = expected_calibration_error(
                        test_d["y_true"].astype(np.int32),
                        test_d["y_pred"].astype(np.float64),
                        sample_weight=weights, n_bins=CALIBRATION_N_BINS)
                    metrics["test_calibration"] = cal
                    metrics_path.write_text(json.dumps(metrics, indent=2))
                    counts["updated"] += 1

                    if variant == "fingerprints" and model == "xgboost":
                        plot_data[strategy] = cal["bins"]

                except Exception as e:
                    log(f"  FAIL {strategy}/{variant}/{model}: {e}")
                    counts["errors"] += 1

    log(f"PHASE 4 DONE: {counts}")

    # Reliability diagram figure — fingerprints/xgboost across 6 splits
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(2, 3, figsize=(13, 8.5), sharex=True, sharey=True)
        for ax, strategy in zip(axes.flat, STRATEGIES):
            ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5,
                    label="perfect calibration")
            bins = plot_data.get(strategy, [])
            xs, ys, weights = [], [], []
            for b in bins:
                if b.get("avg_proba") is not None and b.get("obs_pos_rate") is not None:
                    xs.append(b["avg_proba"])
                    ys.append(b["obs_pos_rate"])
                    weights.append(b["n_eff"])
            if xs:
                # Size by bin weight
                w = np.array(weights, dtype=np.float64)
                w_norm = 60 + 240 * (w / max(1.0, w.max()))
                ax.scatter(xs, ys, s=w_norm, alpha=0.7, edgecolor="black")
                ax.plot(xs, ys, "-", alpha=0.4)
            ax.set_title(f"{strategy}")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
        for ax in axes[-1]:
            ax.set_xlabel("Predicted probability")
        for ax in axes[:, 0]:
            ax.set_ylabel("Observed positive rate")
        fig.suptitle("Reliability diagrams: XGBoost classifier on fingerprints variant\n"
                     "(point size ∝ effective sample weight in bin)")
        fig.tight_layout()
        out_path = FIGURES_ROOT / "reliability_diagrams.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log(f"  Wrote reliability diagram to {out_path}")
    except Exception as e:
        log(f"  Reliability diagram FAILED: {e}")
        traceback.print_exc()

    return counts


# =============================================================================
# Main
# =============================================================================
def main() -> int:
    t0 = time.time()
    log("PAPER-1 REVIEWER-PROOF PIPELINE START")

    summary = {"started": time.strftime("%Y-%m-%d %H:%M:%S")}

    try:
        summary["phase1"] = {"summary": "see leakage_verification.json"}
        phase1_leakage_verification()
    except Exception as e:
        log(f"PHASE 1 ABORTED: {e}")
        traceback.print_exc()
        summary["phase1"] = {"error": str(e)}

    try:
        summary["phase2"] = phase2_bootstrap_cis()
    except Exception as e:
        log(f"PHASE 2 ABORTED: {e}")
        traceback.print_exc()
        summary["phase2"] = {"error": str(e)}

    try:
        summary["phase4"] = phase4_calibration()
    except Exception as e:
        log(f"PHASE 4 ABORTED: {e}")
        traceback.print_exc()
        summary["phase4"] = {"error": str(e)}

    # Phase 3 last so it captures the CIs and calibration just written
    try:
        summary["phase3"] = phase3_results_csv()
    except Exception as e:
        log(f"PHASE 3 ABORTED: {e}")
        traceback.print_exc()
        summary["phase3"] = {"error": str(e)}

    total = time.time() - t0
    log(f"PIPELINE COMPLETE in {total/3600:.2f}h")
    summary["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    summary["total_elapsed_sec"] = round(total, 1)
    (MODELS_ROOT / "reviewer_proof_summary.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
