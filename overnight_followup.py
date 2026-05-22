#!/usr/bin/env python3
"""
Overnight pipeline: 3 phases.

PHASE 1 - Threshold recalibration (~1 min)
  For each classification cell, pick threshold maximizing Youden's J on val
  probabilities. Recompute hard-label test metrics at the tuned threshold.
  Writes back to metrics.json under a new `test_tuned` block; preserves the
  original `test` block.

PHASE 2 - Stacking experiment (~1-2h)
  For each (strategy, variant), train an XGBClassifier on a 2-feature input:
  [regression OOF prediction, classification OOF prediction] from the
  already-trained tree-space base models. Tests stacking benefit.
  Output: models_v1/stacking/<strategy>/<variant>/xgboost/

PHASE 3 - Robustness seeds (~4-6h)
  For random and scaffold splits only, re-run XGBoost regression and
  classification with seeds {7, 1337} using each cell's best hparams from
  the main sweep (no grid re-tuning). 5-fold CV, refit, predict.
  Output: models_v1/<task>/<strategy>/<variant>/xgboost/seeds/seed_<N>/

Idempotent: skip work whose final metrics.json exists.

Usage:
    cd /home/nidhal/PAD4_BENCH
    nohup python overnight_followup.py > models_v1/overnight.log 2>&1 &
    echo $! > models_v1/overnight.pid
    tail -f models_v1/overnight.log
"""

import json
import pickle
import sys
import time
import traceback
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import KFold, StratifiedKFold

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path("/home/nidhal/PAD4_BENCH")
FEATURES_ROOT = PROJECT_ROOT / "features_v18"
MODELS_ROOT = PROJECT_ROOT / "models_v1"

STRATEGIES = ["random", "scaffold", "confirmed", "lead_opt", "similarity", "cliff_aware"]
VARIANTS = ["full", "fingerprints", "physchem", "mordred", "fragments"]
ROBUSTNESS_STRATEGIES = ["random", "scaffold"]
ROBUSTNESS_SEEDS = [7, 1337]

N_CV_FOLDS = 5
EARLY_STOPPING_ROUNDS = 50


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


# =============================================================================
# Shared loaders
# =============================================================================
def load_subset(task: str, strategy: str, variant: str, space: str, subset: str) -> dict:
    feat_path = FEATURES_ROOT / task / strategy / subset / f"{variant}_{space}.npz"
    strat_path = FEATURES_ROOT / task / strategy / subset / "stratifiers.npz"
    feat = np.load(feat_path, allow_pickle=True)
    strat = np.load(strat_path, allow_pickle=True)

    meta = json.loads(feat["meta"][0]) if "meta" in feat.files else {}
    if "ml_weight" in strat.files:
        weights = strat["ml_weight"].astype(np.float64)
    elif "weights" in feat.files:
        weights = feat["weights"].astype(np.float64)
    else:
        weights = np.ones(len(feat["y"]), dtype=np.float64)

    return {
        "X": feat["X"].astype(np.float32),
        "y": feat["y"],
        "ids": feat["ids"],
        "weights": weights,
        "meta": meta,
    }


def load_pred_npz(path: Path) -> dict:
    d = np.load(path, allow_pickle=True)
    out = {k: d[k] for k in d.files}
    return out


def save_predictions(path: Path, ids, y_true, y_pred, fold_idx=None,
                     y_dtype=np.float32) -> None:
    payload = {"ids": ids,
               "y_true": np.asarray(y_true, dtype=y_dtype if y_dtype else np.int8),
               "y_pred": np.asarray(y_pred, dtype=np.float32)}
    if fold_idx is not None:
        payload["fold_idx"] = np.asarray(fold_idx, dtype=np.int8)
    np.savez_compressed(path, **payload)


# =============================================================================
# Classification metrics (for phases 1 and 3-classification)
# =============================================================================
def clf_metrics(y_true, y_proba, sample_weight=None, threshold: float = 0.5) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    try:
        auc = float(roc_auc_score(y_true, y_proba, sample_weight=sample_weight))
    except ValueError:
        auc = float("nan")
    return {
        "roc_auc": auc,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred, sample_weight=sample_weight)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", sample_weight=sample_weight)),
        "mcc": float(matthews_corrcoef(y_true, y_pred, sample_weight=sample_weight)),
        "brier": float(brier_score_loss(y_true, y_proba, sample_weight=sample_weight)),
        "n": int(len(y_true)),
        "pos_rate": (float(np.average(y_true, weights=sample_weight))
                     if sample_weight is not None else float(np.mean(y_true))),
        "threshold": float(threshold),
    }


def reg_metrics(y_true, y_pred, sample_weight=None) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred, sample_weight=sample_weight)))
    mae = float(mean_absolute_error(y_true, y_pred, sample_weight=sample_weight))
    r2 = float(r2_score(y_true, y_pred, sample_weight=sample_weight))
    rho, _ = spearmanr(y_true, y_pred)
    return {"rmse": rmse, "mae": mae, "r2": r2,
            "spearman_rho": float(rho), "n": int(len(y_true))}


def youden_threshold(y_true: np.ndarray, y_proba: np.ndarray,
                     sample_weight: np.ndarray | None = None) -> float:
    """Threshold maximizing TPR - FPR on the given data."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, y_proba, sample_weight=sample_weight)
    j = tpr - fpr
    best_idx = int(np.argmax(j))
    thr = thresholds[best_idx]
    # roc_curve can return inf for the first threshold; clip to [0, 1].
    if not np.isfinite(thr):
        thr = 0.5
    return float(np.clip(thr, 0.0, 1.0))


# =============================================================================
# PHASE 1 - Threshold recalibration
# =============================================================================
def phase1_recalibrate_thresholds() -> dict:
    log("=" * 70)
    log("PHASE 1: Threshold recalibration on classification cells")
    log("=" * 70)

    clf_root = MODELS_ROOT / "classification"
    counts = {"updated": 0, "unchanged": 0, "missing": 0, "errors": 0}
    degenerate_cells_before = []
    degenerate_cells_after = []

    for strategy in STRATEGIES:
        for variant in VARIANTS:
            for model_name in ["xgboost", "logreg_enet"]:
                cell = clf_root / strategy / variant / model_name
                metrics_path = cell / "metrics.json"
                val_path = cell / "val_pred.npz"
                test_path = cell / "test_pred.npz"

                if not metrics_path.exists():
                    counts["missing"] += 1
                    continue
                if not val_path.exists() or not test_path.exists():
                    log(f"  WARN: {strategy}/{variant}/{model_name} has metrics but no preds — skipping")
                    counts["errors"] += 1
                    continue

                try:
                    metrics = json.loads(metrics_path.read_text())
                    # If already recalibrated, skip
                    if "test_tuned" in metrics:
                        counts["unchanged"] += 1
                        continue

                    val_d = load_pred_npz(val_path)
                    test_d = load_pred_npz(test_path)

                    # Load val weights for Youden's J
                    val_subset = load_subset("classification", strategy, variant, "tree", "val")
                    test_subset = load_subset("classification", strategy, variant, "tree", "test")

                    thr = youden_threshold(val_d["y_true"], val_d["y_pred"],
                                           sample_weight=val_subset["weights"])
                    test_tuned = clf_metrics(test_d["y_true"], test_d["y_pred"],
                                             sample_weight=test_subset["weights"],
                                             threshold=thr)

                    # Track degenerate cells
                    if metrics.get("test", {}).get("mcc", 0) == 0:
                        degenerate_cells_before.append(f"{strategy}/{variant}/{model_name}")
                    if test_tuned["mcc"] == 0:
                        degenerate_cells_after.append(f"{strategy}/{variant}/{model_name}")

                    metrics["tuned_threshold"] = thr
                    metrics["test_tuned"] = test_tuned
                    metrics_path.write_text(json.dumps(metrics, indent=2))
                    counts["updated"] += 1

                    log(f"  {strategy}/{variant}/{model_name}: thr={thr:.3f} | "
                        f"orig MCC={metrics['test']['mcc']:.3f} -> tuned MCC={test_tuned['mcc']:.3f} | "
                        f"BalAcc={metrics['test']['balanced_accuracy']:.3f} -> {test_tuned['balanced_accuracy']:.3f}")

                except Exception as e:
                    log(f"  FAIL {strategy}/{variant}/{model_name}: {e}")
                    counts["errors"] += 1

    log(f"PHASE 1 DONE: {counts}")
    log(f"  Degenerate cells before recalibration: {len(degenerate_cells_before)}")
    for c in degenerate_cells_before:
        log(f"    - {c}")
    log(f"  Still degenerate after recalibration: {len(degenerate_cells_after)}")
    for c in degenerate_cells_after:
        log(f"    - {c}")
    return counts


# =============================================================================
# PHASE 2 - Stacking experiment
# =============================================================================
def phase2_stacking() -> dict:
    log("=" * 70)
    log("PHASE 2: Stacking experiment (regression OOF + classification OOF -> classifier)")
    log("=" * 70)

    stacking_root = MODELS_ROOT / "stacking"
    stacking_root.mkdir(parents=True, exist_ok=True)
    counts = {"completed": 0, "skipped": 0, "failed": 0}

    summary = []

    # Stacking hparam grid: small since input is just 2 features
    grid = list(product(
        [3, 4, 5],            # max_depth
        [1, 5],               # min_child_weight
        [0.0, 0.1],           # reg_alpha
    ))
    base_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "n_estimators": 500,  # small input -> small tree count is fine
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 1.0,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
    }

    for strategy in STRATEGIES:
        for variant in VARIANTS:
            out_dir = stacking_root / strategy / variant / "xgboost"
            metrics_path = out_dir / "metrics.json"

            if metrics_path.exists():
                log(f"  SKIP {strategy}/{variant}: already done")
                counts["skipped"] += 1
                continue

            try:
                t0 = time.time()
                log(f"  START {strategy}/{variant}")

                # Load OOF + val + test predictions from base regression and classification XGB cells
                reg_cell = MODELS_ROOT / "regression" / strategy / variant / "xgboost"
                clf_cell = MODELS_ROOT / "classification" / strategy / variant / "xgboost"

                reg_oof = load_pred_npz(reg_cell / "oof_train.npz")
                reg_val = load_pred_npz(reg_cell / "val_pred.npz")
                reg_test = load_pred_npz(reg_cell / "test_pred.npz")
                clf_oof = load_pred_npz(clf_cell / "oof_train.npz")
                clf_val = load_pred_npz(clf_cell / "val_pred.npz")
                clf_test = load_pred_npz(clf_cell / "test_pred.npz")

                # Align by InChIKey-14. The regression and classification source data
                # are 78% overlapping, so we use the classification set's compounds as
                # the stacking universe (since we are predicting activity classes).
                # Build a regression-pred lookup, then for each classification compound
                # take its regression OOF/val/test prediction if available; otherwise
                # use the unconditional mean pIC50 as a fallback.

                def build_stacked_X(clf_ids, clf_proba, reg_ids, reg_pred,
                                    reg_train_mean: float | None = None):
                    """Returns (X stacked [n,2], indicator mask of regression coverage)."""
                    reg_map = dict(zip(reg_ids.tolist(), reg_pred.tolist()))
                    n = len(clf_ids)
                    X = np.zeros((n, 2), dtype=np.float32)
                    covered = np.zeros(n, dtype=bool)
                    for i, cid in enumerate(clf_ids.tolist()):
                        if cid in reg_map:
                            X[i, 0] = reg_map[cid]
                            covered[i] = True
                        else:
                            X[i, 0] = reg_train_mean if reg_train_mean is not None else 0.0
                        X[i, 1] = clf_proba[i]
                    return X, covered

                reg_train_mean = float(np.mean(reg_oof["y_true"]))

                X_train, cov_train = build_stacked_X(
                    clf_oof["ids"], clf_oof["y_pred"],
                    reg_oof["ids"], reg_oof["y_pred"], reg_train_mean)
                X_val, cov_val = build_stacked_X(
                    clf_val["ids"], clf_val["y_pred"],
                    reg_val["ids"], reg_val["y_pred"], reg_train_mean)
                X_test, cov_test = build_stacked_X(
                    clf_test["ids"], clf_test["y_pred"],
                    reg_test["ids"], reg_test["y_pred"], reg_train_mean)

                y_train = clf_oof["y_true"].astype(np.int32)
                y_val = clf_val["y_true"].astype(np.int32)
                y_test = clf_test["y_true"].astype(np.int32)

                # Load classification weights for honest weighted CV
                clf_train_subset = load_subset("classification", strategy, variant, "tree", "train")
                clf_val_subset = load_subset("classification", strategy, variant, "tree", "val")
                clf_test_subset = load_subset("classification", strategy, variant, "tree", "test")

                # Align weights to the OOF id order (classification IDs ARE the train order)
                # Build a dict for safety
                w_train_map = dict(zip(clf_train_subset["ids"].tolist(),
                                       clf_train_subset["weights"].tolist()))
                w_val_map = dict(zip(clf_val_subset["ids"].tolist(),
                                     clf_val_subset["weights"].tolist()))
                w_test_map = dict(zip(clf_test_subset["ids"].tolist(),
                                      clf_test_subset["weights"].tolist()))
                w_train = np.array([w_train_map[i] for i in clf_oof["ids"].tolist()],
                                   dtype=np.float64)
                w_val = np.array([w_val_map[i] for i in clf_val["ids"].tolist()],
                                 dtype=np.float64)
                w_test = np.array([w_test_map[i] for i in clf_test["ids"].tolist()],
                                  dtype=np.float64)

                log(f"    train n={len(y_train)} reg_coverage={cov_train.mean():.3f} "
                    f"val n={len(y_val)} cov={cov_val.mean():.3f} "
                    f"test n={len(y_test)} cov={cov_test.mean():.3f}")

                # CV grid search
                grid_results = []
                for max_d, min_cw, reg_a in grid:
                    params = {**base_params, "max_depth": max_d,
                              "min_child_weight": min_cw, "reg_alpha": reg_a}
                    skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=42)
                    aucs = []
                    for tr_idx, va_idx in skf.split(X_train, y_train):
                        m = xgb.XGBClassifier(**params,
                                              early_stopping_rounds=EARLY_STOPPING_ROUNDS)
                        m.fit(X_train[tr_idx], y_train[tr_idx], sample_weight=w_train[tr_idx],
                              eval_set=[(X_train[va_idx], y_train[va_idx])],
                              sample_weight_eval_set=[w_train[va_idx]], verbose=False)
                        proba = m.predict_proba(X_train[va_idx])[:, 1]
                        aucs.append(roc_auc_score(y_train[va_idx], proba,
                                                  sample_weight=w_train[va_idx]))
                    grid_results.append({
                        "config": {"max_depth": max_d, "min_child_weight": min_cw,
                                   "reg_alpha": reg_a},
                        "cv_mean_auc": float(np.mean(aucs)),
                    })

                best = max(grid_results, key=lambda r: r["cv_mean_auc"])
                best_params = {**base_params, **best["config"]}

                final = xgb.XGBClassifier(**best_params,
                                          early_stopping_rounds=EARLY_STOPPING_ROUNDS)
                final.fit(X_train, y_train, sample_weight=w_train,
                          eval_set=[(X_val, y_val)],
                          sample_weight_eval_set=[w_val], verbose=False)

                val_proba = final.predict_proba(X_val)[:, 1]
                test_proba = final.predict_proba(X_test)[:, 1]
                val_m = clf_metrics(y_val, val_proba, sample_weight=w_val)
                test_m_default = clf_metrics(y_test, test_proba, sample_weight=w_test)
                thr = youden_threshold(y_val, val_proba, sample_weight=w_val)
                test_m_tuned = clf_metrics(y_test, test_proba, sample_weight=w_test, threshold=thr)

                # Baseline comparison: just the classification model alone
                base_clf_metrics = json.loads((clf_cell / "metrics.json").read_text())
                base_test_auc = base_clf_metrics["test"]["roc_auc"]
                base_test_mcc = base_clf_metrics["test"]["mcc"]

                lift_auc = test_m_default["roc_auc"] - base_test_auc

                out_dir.mkdir(parents=True, exist_ok=True)
                save_predictions(out_dir / "val_pred.npz",
                                 clf_val["ids"], y_val, val_proba)
                save_predictions(out_dir / "test_pred.npz",
                                 clf_test["ids"], y_test, test_proba)
                with open(out_dir / "model.pkl", "wb") as f:
                    pickle.dump(final, f)
                (out_dir / "tuning_results.json").write_text(json.dumps({
                    "grid_results": grid_results,
                    "best_config": best["config"],
                }, indent=2))
                (out_dir / "metrics.json").write_text(json.dumps({
                    "task": "stacking",
                    "strategy": strategy,
                    "variant": variant,
                    "n_train": int(len(y_train)),
                    "n_val": int(len(y_val)),
                    "n_test": int(len(y_test)),
                    "regression_coverage": {
                        "train": float(cov_train.mean()),
                        "val": float(cov_val.mean()),
                        "test": float(cov_test.mean()),
                    },
                    "best_config": best["config"],
                    "cell_elapsed_sec": round(time.time() - t0, 1),
                    "val": val_m,
                    "test": test_m_default,
                    "test_tuned": test_m_tuned,
                    "tuned_threshold": thr,
                    "baseline_xgb_clf_test_auc": base_test_auc,
                    "baseline_xgb_clf_test_mcc": base_test_mcc,
                    "auc_lift_over_baseline": lift_auc,
                }, indent=2))

                summary.append({
                    "strategy": strategy, "variant": variant,
                    "stacked_test_auc": test_m_default["roc_auc"],
                    "baseline_test_auc": base_test_auc,
                    "auc_lift": lift_auc,
                })

                log(f"    DONE in {time.time() - t0:.1f}s | "
                    f"stacked AUC={test_m_default['roc_auc']:.4f} "
                    f"baseline={base_test_auc:.4f} lift={lift_auc:+.4f}")
                counts["completed"] += 1

            except Exception as e:
                log(f"  FAIL {strategy}/{variant}: {e}")
                traceback.print_exc()
                counts["failed"] += 1

    (stacking_root / "summary.json").write_text(json.dumps({
        "completed": counts["completed"],
        "skipped": counts["skipped"],
        "failed": counts["failed"],
        "cells": summary,
    }, indent=2))

    log(f"PHASE 2 DONE: {counts}")
    if summary:
        lifts = [s["auc_lift"] for s in summary]
        log(f"  Mean stacking AUC lift: {np.mean(lifts):+.4f}")
        log(f"  Cells with positive lift: {sum(1 for l in lifts if l > 0)}/{len(lifts)}")
    return counts


# =============================================================================
# PHASE 3 - Robustness seeds
# =============================================================================
def _xgb_robustness_regression(strategy: str, variant: str, seed: int,
                                best_config: dict, out_dir: Path) -> None:
    train = load_subset("regression", strategy, variant, "tree", "train")
    val = load_subset("regression", strategy, variant, "tree", "val")
    test = load_subset("regression", strategy, variant, "tree", "test")

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "n_estimators": 2000,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_jobs": -1,
        "verbosity": 0,
        **best_config,
    }

    # 5-fold CV for OOF (same logic as main sweep)
    kf = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=seed)
    y_tr_f = train["y"].astype(np.float32)
    oof = np.full(len(y_tr_f), np.nan, dtype=np.float32)
    fold_idx_arr = np.full(len(y_tr_f), -1, dtype=np.int8)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(train["X"])):
        m = xgb.XGBRegressor(**params, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
        m.fit(train["X"][tr_idx], y_tr_f[tr_idx],
              sample_weight=train["weights"][tr_idx],
              eval_set=[(train["X"][va_idx], y_tr_f[va_idx])],
              sample_weight_eval_set=[train["weights"][va_idx]], verbose=False)
        oof[va_idx] = m.predict(train["X"][va_idx])
        fold_idx_arr[va_idx] = fold

    oof_m = reg_metrics(y_tr_f, oof, sample_weight=train["weights"])

    final = xgb.XGBRegressor(**params, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    final.fit(train["X"], y_tr_f, sample_weight=train["weights"],
              eval_set=[(val["X"], val["y"].astype(np.float32))],
              sample_weight_eval_set=[val["weights"]], verbose=False)

    val_pred = final.predict(val["X"])
    test_pred = final.predict(test["X"])
    val_m = reg_metrics(val["y"].astype(np.float32), val_pred, sample_weight=val["weights"])
    test_m = reg_metrics(test["y"].astype(np.float32), test_pred, sample_weight=test["weights"])

    out_dir.mkdir(parents=True, exist_ok=True)
    save_predictions(out_dir / "oof_train.npz",
                     train["ids"], y_tr_f, oof, fold_idx_arr)
    save_predictions(out_dir / "val_pred.npz",
                     val["ids"], val["y"].astype(np.float32), val_pred)
    save_predictions(out_dir / "test_pred.npz",
                     test["ids"], test["y"].astype(np.float32), test_pred)
    with open(out_dir / "model.pkl", "wb") as f:
        pickle.dump(final, f)
    (out_dir / "metrics.json").write_text(json.dumps({
        "task": "regression",
        "strategy": strategy,
        "variant": variant,
        "model": "xgboost",
        "seed": seed,
        "best_config_from_seed42_sweep": best_config,
        "oof": oof_m,
        "val": val_m,
        "test": test_m,
    }, indent=2))


def _xgb_robustness_classification(strategy: str, variant: str, seed: int,
                                    best_config: dict, out_dir: Path) -> None:
    train = load_subset("classification", strategy, variant, "tree", "train")
    val = load_subset("classification", strategy, variant, "tree", "val")
    test = load_subset("classification", strategy, variant, "tree", "test")

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "n_estimators": 2000,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_jobs": -1,
        "verbosity": 0,
        **best_config,
    }

    y_tr = train["y"].astype(np.int32)
    skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(len(y_tr), np.nan, dtype=np.float32)
    fold_idx_arr = np.full(len(y_tr), -1, dtype=np.int8)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train["X"], y_tr)):
        m = xgb.XGBClassifier(**params, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
        m.fit(train["X"][tr_idx], y_tr[tr_idx],
              sample_weight=train["weights"][tr_idx],
              eval_set=[(train["X"][va_idx], y_tr[va_idx])],
              sample_weight_eval_set=[train["weights"][va_idx]], verbose=False)
        oof[va_idx] = m.predict_proba(train["X"][va_idx])[:, 1]
        fold_idx_arr[va_idx] = fold

    oof_m = clf_metrics(y_tr, oof, sample_weight=train["weights"])

    final = xgb.XGBClassifier(**params, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    final.fit(train["X"], y_tr, sample_weight=train["weights"],
              eval_set=[(val["X"], val["y"].astype(np.int32))],
              sample_weight_eval_set=[val["weights"]], verbose=False)

    val_proba = final.predict_proba(val["X"])[:, 1]
    test_proba = final.predict_proba(test["X"])[:, 1]
    val_m = clf_metrics(val["y"].astype(np.int32), val_proba, sample_weight=val["weights"])
    test_m = clf_metrics(test["y"].astype(np.int32), test_proba, sample_weight=test["weights"])

    # Tuned-threshold metrics for parity with the main sweep's test_tuned
    thr = youden_threshold(val["y"].astype(np.int32), val_proba,
                           sample_weight=val["weights"])
    test_m_tuned = clf_metrics(test["y"].astype(np.int32), test_proba,
                               sample_weight=test["weights"], threshold=thr)

    out_dir.mkdir(parents=True, exist_ok=True)
    save_predictions(out_dir / "oof_train.npz",
                     train["ids"], y_tr, oof, fold_idx_arr, y_dtype=np.int8)
    save_predictions(out_dir / "val_pred.npz",
                     val["ids"], val["y"].astype(np.int32), val_proba, y_dtype=np.int8)
    save_predictions(out_dir / "test_pred.npz",
                     test["ids"], test["y"].astype(np.int32), test_proba, y_dtype=np.int8)
    with open(out_dir / "model.pkl", "wb") as f:
        pickle.dump(final, f)
    (out_dir / "metrics.json").write_text(json.dumps({
        "task": "classification",
        "strategy": strategy,
        "variant": variant,
        "model": "xgboost",
        "seed": seed,
        "best_config_from_seed42_sweep": best_config,
        "oof": oof_m,
        "val": val_m,
        "test": test_m,
        "test_tuned": test_m_tuned,
        "tuned_threshold": thr,
    }, indent=2))


def phase3_robustness() -> dict:
    log("=" * 70)
    log("PHASE 3: Robustness seeds (random + scaffold, seeds 7 and 1337)")
    log("=" * 70)

    counts = {"completed": 0, "skipped": 0, "failed": 0}

    for task, runner in [("regression", _xgb_robustness_regression),
                          ("classification", _xgb_robustness_classification)]:
        for strategy in ROBUSTNESS_STRATEGIES:
            for variant in VARIANTS:
                # Load best config from the seed-42 main sweep
                main_metrics_path = (MODELS_ROOT / task / strategy / variant /
                                     "xgboost" / "metrics.json")
                if not main_metrics_path.exists():
                    log(f"  WARN: missing seed-42 sweep for {task}/{strategy}/{variant} — skipping")
                    counts["failed"] += 1
                    continue
                # Best config is nested under tuning_results
                tuning_path = (MODELS_ROOT / task / strategy / variant /
                               "xgboost" / "tuning_results.json")
                tuning = json.loads(tuning_path.read_text())
                best_config = tuning["best_config"]

                for seed in ROBUSTNESS_SEEDS:
                    out_dir = (MODELS_ROOT / task / strategy / variant /
                               "xgboost" / "seeds" / f"seed_{seed}")
                    metrics_path = out_dir / "metrics.json"

                    if metrics_path.exists():
                        log(f"  SKIP {task}/{strategy}/{variant}/seed_{seed}")
                        counts["skipped"] += 1
                        continue

                    try:
                        t0 = time.time()
                        log(f"  START {task}/{strategy}/{variant}/seed_{seed} "
                            f"config={best_config}")
                        runner(strategy, variant, seed, best_config, out_dir)
                        m = json.loads(metrics_path.read_text())
                        if task == "regression":
                            log(f"    DONE in {time.time() - t0:.1f}s | "
                                f"test RMSE={m['test']['rmse']:.4f} "
                                f"R2={m['test']['r2']:.4f}")
                        else:
                            log(f"    DONE in {time.time() - t0:.1f}s | "
                                f"test AUC={m['test']['roc_auc']:.4f} "
                                f"MCC={m['test']['mcc']:.4f} "
                                f"tuned_MCC={m['test_tuned']['mcc']:.4f}")
                        counts["completed"] += 1
                    except Exception as e:
                        log(f"  FAIL {task}/{strategy}/{variant}/seed_{seed}: {e}")
                        traceback.print_exc()
                        counts["failed"] += 1

    log(f"PHASE 3 DONE: {counts}")
    return counts


# =============================================================================
# Main
# =============================================================================
def main() -> int:
    t0 = time.time()
    log("OVERNIGHT FOLLOWUP START")
    log(f"  Strategies: {STRATEGIES}")
    log(f"  Variants: {VARIANTS}")
    log(f"  Robustness: {ROBUSTNESS_STRATEGIES} x seeds {ROBUSTNESS_SEEDS}")

    summary = {"started": time.strftime("%Y-%m-%d %H:%M:%S")}

    try:
        summary["phase1"] = phase1_recalibrate_thresholds()
    except Exception as e:
        log(f"PHASE 1 ABORTED: {e}")
        traceback.print_exc()
        summary["phase1"] = {"error": str(e)}

    try:
        summary["phase2"] = phase2_stacking()
    except Exception as e:
        log(f"PHASE 2 ABORTED: {e}")
        traceback.print_exc()
        summary["phase2"] = {"error": str(e)}

    try:
        summary["phase3"] = phase3_robustness()
    except Exception as e:
        log(f"PHASE 3 ABORTED: {e}")
        traceback.print_exc()
        summary["phase3"] = {"error": str(e)}

    total = time.time() - t0
    log(f"OVERNIGHT FOLLOWUP COMPLETE in {total/3600:.2f}h")
    summary["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    summary["total_elapsed_sec"] = round(total, 1)
    (MODELS_ROOT / "overnight_summary.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
