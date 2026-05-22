#!/usr/bin/env python3
"""
Classification sweep: XGBoost (tree) + LogisticRegression-ElasticNet (linear)
across all (strategy, variant).

Mirrors sweep_regression.py:
  - Small hparam grid -> 5-fold CV on train, pick best by mean ROC-AUC (maximize).
  - Refit best config on full train (val for early stopping where applicable).
  - 5-fold OOF predictions (probabilities) with best config for downstream stacking.
  - Predict val and test_locked probabilities once each.
  - Persist 7 artifacts to models_v1/classification/<strategy>/<variant>/<model>/.

Predictions are stored as P(class=1) probabilities, not hard labels.

Idempotent: a cell with a final metrics.json is skipped on rerun. Delete that
file to force re-execution of a single cell.

Usage:
    cd /home/nidhal/PAD4_BENCH
    nohup python sweep_classification.py > models_v1/classification/sweep.log 2>&1 &
    echo $! > models_v1/classification/sweep.pid
    tail -f models_v1/classification/sweep.log
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

# SAGA can emit ConvergenceWarning at extreme regularization; suppress for clean log.
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TASK = "classification"
FEATURES_ROOT = PROJECT_ROOT / "features_v18" / TASK
MODELS_ROOT = PROJECT_ROOT / "models_v1" / TASK
SUMMARY_PATH = MODELS_ROOT / "sweep_summary.json"

STRATEGIES = ["random", "scaffold", "confirmed", "lead_opt", "similarity", "cliff_aware"]
VARIANTS = ["full", "fingerprints", "physchem", "mordred", "fragments"]

SEED = 42
N_CV_FOLDS = 5
EARLY_STOPPING_ROUNDS = 50
DECISION_THRESHOLD = 0.5  # for hard-label metrics (BalAcc, F1, MCC); probabilities are saved separately

# XGBoost grid: 18 configs.
XGB_BASE = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "random_state": SEED,
    "n_jobs": -1,
    "verbosity": 0,
}
XGB_GRID = list(product(
    [4, 6, 8],          # max_depth
    [1, 5, 10],         # min_child_weight
    [0.0, 0.1],         # reg_alpha
))

# LogisticRegression-ElasticNet grid: 15 configs.
# Note: sklearn uses C = 1/alpha, so we sweep C directly. l1_ratio in [0, 1].
LOGREG_BASE = {
    "penalty": "elasticnet",
    "solver": "saga",
    "max_iter": 20000,
    "tol": 1e-4,
    "random_state": SEED,
    "n_jobs": -1,
}
LOGREG_GRID = list(product(
    [0.01, 0.1, 1.0, 10.0, 100.0],   # C  (inverse of regularization strength)
    [0.1, 0.5, 0.9],                  # l1_ratio
))


# -----------------------------------------------------------------------------
# IO
# -----------------------------------------------------------------------------
def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def load_subset(strategy: str, variant: str, space: str, subset: str) -> dict:
    feat_path = FEATURES_ROOT / strategy / subset / f"{variant}_{space}.npz"
    strat_path = FEATURES_ROOT / strategy / subset / "stratifiers.npz"
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
        "y": feat["y"].astype(np.int32),
        "ids": feat["ids"],
        "weights": weights,
        "meta": meta,
    }


def compute_metrics(y_true, y_proba, sample_weight=None,
                    threshold: float = DECISION_THRESHOLD) -> dict:
    """Classification metrics. ROC-AUC and Brier use probabilities;
    BalAcc / F1-macro / MCC use thresholded labels."""
    y_pred = (y_proba >= threshold).astype(int)
    try:
        auc = float(roc_auc_score(y_true, y_proba, sample_weight=sample_weight))
    except ValueError:
        # only one class present in y_true
        auc = float("nan")
    return {
        "roc_auc": auc,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred, sample_weight=sample_weight)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", sample_weight=sample_weight)),
        "mcc": float(matthews_corrcoef(y_true, y_pred, sample_weight=sample_weight)),
        "brier": float(brier_score_loss(y_true, y_proba, sample_weight=sample_weight)),
        "n": int(len(y_true)),
        "pos_rate": float(np.average(y_true, weights=sample_weight)) if sample_weight is not None else float(np.mean(y_true)),
        "threshold": float(threshold),
    }


def save_predictions(path: Path, ids, y_true, y_proba, fold_idx=None) -> None:
    payload = {"ids": ids,
               "y_true": np.asarray(y_true, dtype=np.int8),
               "y_pred": np.asarray(y_proba, dtype=np.float32)}  # probabilities
    if fold_idx is not None:
        payload["fold_idx"] = np.asarray(fold_idx, dtype=np.int8)
    np.savez_compressed(path, **payload)


# -----------------------------------------------------------------------------
# Per-cell training
# -----------------------------------------------------------------------------
def cv_score_xgb(X, y, w, params) -> tuple[float, list[int]]:
    """5-fold stratified CV mean weighted ROC-AUC for an XGB config."""
    skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=SEED)
    aucs, best_iters = [], []
    for tr_idx, va_idx in skf.split(X, y):
        m = xgb.XGBClassifier(**params, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
        m.fit(X[tr_idx], y[tr_idx], sample_weight=w[tr_idx],
              eval_set=[(X[va_idx], y[va_idx])],
              sample_weight_eval_set=[w[va_idx]], verbose=False)
        proba = m.predict_proba(X[va_idx])[:, 1]
        aucs.append(roc_auc_score(y[va_idx], proba, sample_weight=w[va_idx]))
        best_iters.append(int(m.best_iteration) if hasattr(m, "best_iteration") else params["n_estimators"])
    return float(np.mean(aucs)), best_iters


def cv_score_logreg(X, y, w, params) -> float:
    """5-fold stratified CV mean weighted ROC-AUC for a LogReg config."""
    skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=SEED)
    aucs = []
    for tr_idx, va_idx in skf.split(X, y):
        m = LogisticRegression(**params)
        m.fit(X[tr_idx], y[tr_idx], sample_weight=w[tr_idx])
        proba = m.predict_proba(X[va_idx])[:, 1]
        aucs.append(roc_auc_score(y[va_idx], proba, sample_weight=w[va_idx]))
    return float(np.mean(aucs))


def generate_oof_xgb(X, y, w, params) -> tuple[np.ndarray, np.ndarray]:
    """5-fold stratified OOF probabilities for the chosen XGB config."""
    skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=SEED)
    oof = np.full(len(y), np.nan, dtype=np.float32)
    fold_idx = np.full(len(y), -1, dtype=np.int8)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        m = xgb.XGBClassifier(**params, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
        m.fit(X[tr_idx], y[tr_idx], sample_weight=w[tr_idx],
              eval_set=[(X[va_idx], y[va_idx])],
              sample_weight_eval_set=[w[va_idx]], verbose=False)
        oof[va_idx] = m.predict_proba(X[va_idx])[:, 1]
        fold_idx[va_idx] = fold
    return oof, fold_idx


def generate_oof_logreg(X, y, w, params) -> tuple[np.ndarray, np.ndarray]:
    """5-fold stratified OOF probabilities for the chosen LogReg config."""
    skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=SEED)
    oof = np.full(len(y), np.nan, dtype=np.float32)
    fold_idx = np.full(len(y), -1, dtype=np.int8)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        m = LogisticRegression(**params)
        m.fit(X[tr_idx], y[tr_idx], sample_weight=w[tr_idx])
        oof[va_idx] = m.predict_proba(X[va_idx])[:, 1]
        fold_idx[va_idx] = fold
    return oof, fold_idx


def run_xgb_cell(strategy: str, variant: str) -> dict:
    train = load_subset(strategy, variant, "tree", "train")
    val = load_subset(strategy, variant, "tree", "val")
    test = load_subset(strategy, variant, "tree", "test")

    log(f"  XGB train shape: {train['X'].shape}  pos_rate={np.mean(train['y']):.3f}")

    log(f"  XGB grid search: {len(XGB_GRID)} configs")
    grid_results = []
    for i, (max_d, min_cw, reg_a) in enumerate(XGB_GRID):
        params = {**XGB_BASE, "max_depth": max_d, "min_child_weight": min_cw,
                  "reg_alpha": reg_a}
        t0 = time.time()
        mean_auc, best_iters = cv_score_xgb(train["X"], train["y"], train["weights"], params)
        elapsed = time.time() - t0
        grid_results.append({
            "config": {"max_depth": max_d, "min_child_weight": min_cw, "reg_alpha": reg_a},
            "cv_mean_auc": mean_auc,
            "median_best_iter": int(np.median(best_iters)),
            "elapsed_sec": round(elapsed, 1),
        })
        log(f"    [{i+1}/{len(XGB_GRID)}] depth={max_d} mcw={min_cw} alpha={reg_a} "
            f"-> AUC={mean_auc:.4f} ({elapsed:.1f}s)")

    # Maximize AUC.
    best = max(grid_results, key=lambda r: r["cv_mean_auc"])
    log(f"  XGB best: {best['config']} AUC={best['cv_mean_auc']:.4f}")
    best_params = {**XGB_BASE, **best["config"]}

    log(f"  XGB generating OOF probabilities with best config")
    oof_pred, fold_idx = generate_oof_xgb(train["X"], train["y"], train["weights"], best_params)
    oof_metrics = compute_metrics(train["y"], oof_pred, sample_weight=train["weights"])

    log(f"  XGB refitting on full train")
    final = xgb.XGBClassifier(**best_params, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    final.fit(train["X"], train["y"], sample_weight=train["weights"],
              eval_set=[(val["X"], val["y"])],
              sample_weight_eval_set=[val["weights"]], verbose=False)
    best_iter_final = int(final.best_iteration) if hasattr(final, "best_iteration") else None

    val_proba = final.predict_proba(val["X"])[:, 1]
    test_proba = final.predict_proba(test["X"])[:, 1]
    val_metrics = compute_metrics(val["y"], val_proba, sample_weight=val["weights"])
    test_metrics = compute_metrics(test["y"], test_proba, sample_weight=test["weights"])

    return {
        "best_config": best["config"],
        "best_params": best_params,
        "best_iteration_final": best_iter_final,
        "grid_results": grid_results,
        "oof_metrics": oof_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "model": final,
        "oof_pred": oof_pred,
        "fold_idx": fold_idx,
        "val_pred": val_proba,
        "test_pred": test_proba,
        "train_ids": train["ids"], "train_y": train["y"],
        "val_ids": val["ids"], "val_y": val["y"],
        "test_ids": test["ids"], "test_y": test["y"],
        "meta": train["meta"],
    }


def run_logreg_cell(strategy: str, variant: str) -> dict:
    train = load_subset(strategy, variant, "linear", "train")
    val = load_subset(strategy, variant, "linear", "val")
    test = load_subset(strategy, variant, "linear", "test")

    log(f"  LOGREG train shape: {train['X'].shape}  pos_rate={np.mean(train['y']):.3f}")

    log(f"  LOGREG grid search: {len(LOGREG_GRID)} configs")
    grid_results = []
    for i, (C, l1r) in enumerate(LOGREG_GRID):
        params = {**LOGREG_BASE, "C": C, "l1_ratio": l1r}
        t0 = time.time()
        mean_auc = cv_score_logreg(train["X"], train["y"], train["weights"], params)
        elapsed = time.time() - t0
        grid_results.append({
            "config": {"C": C, "l1_ratio": l1r},
            "cv_mean_auc": mean_auc,
            "elapsed_sec": round(elapsed, 1),
        })
        log(f"    [{i+1}/{len(LOGREG_GRID)}] C={C} l1={l1r} "
            f"-> AUC={mean_auc:.4f} ({elapsed:.1f}s)")

    best = max(grid_results, key=lambda r: r["cv_mean_auc"])
    log(f"  LOGREG best: {best['config']} AUC={best['cv_mean_auc']:.4f}")
    best_params = {**LOGREG_BASE, **best["config"]}

    log(f"  LOGREG generating OOF probabilities")
    oof_pred, fold_idx = generate_oof_logreg(train["X"], train["y"], train["weights"], best_params)
    oof_metrics = compute_metrics(train["y"], oof_pred, sample_weight=train["weights"])

    log(f"  LOGREG refitting on full train")
    final = LogisticRegression(**best_params)
    final.fit(train["X"], train["y"], sample_weight=train["weights"])

    val_proba = final.predict_proba(val["X"])[:, 1]
    test_proba = final.predict_proba(test["X"])[:, 1]
    val_metrics = compute_metrics(val["y"], val_proba, sample_weight=val["weights"])
    test_metrics = compute_metrics(test["y"], test_proba, sample_weight=test["weights"])

    return {
        "best_config": best["config"],
        "best_params": best_params,
        "best_iteration_final": None,
        "grid_results": grid_results,
        "oof_metrics": oof_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "model": final,
        "oof_pred": oof_pred,
        "fold_idx": fold_idx,
        "val_pred": val_proba,
        "test_pred": test_proba,
        "train_ids": train["ids"], "train_y": train["y"],
        "val_ids": val["ids"], "val_y": val["y"],
        "test_ids": test["ids"], "test_y": test["y"],
        "meta": train["meta"],
    }


def persist_cell(out_dir: Path, model_name: str, strategy: str, variant: str,
                 space: str, result: dict, cell_elapsed: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    save_predictions(out_dir / "oof_train.npz",
                     result["train_ids"], result["train_y"],
                     result["oof_pred"], result["fold_idx"])
    save_predictions(out_dir / "val_pred.npz",
                     result["val_ids"], result["val_y"], result["val_pred"])
    save_predictions(out_dir / "test_pred.npz",
                     result["test_ids"], result["test_y"], result["test_pred"])

    with open(out_dir / "model.pkl", "wb") as f:
        pickle.dump(result["model"], f)

    (out_dir / "tuning_results.json").write_text(json.dumps({
        "grid_results": result["grid_results"],
        "best_config": result["best_config"],
    }, indent=2))

    (out_dir / "hparams.json").write_text(json.dumps({
        "model": model_name,
        "best_params": {k: (v if not isinstance(v, np.generic) else v.item())
                        for k, v in result["best_params"].items()},
        "seed": SEED,
        "n_cv_folds": N_CV_FOLDS,
        "decision_threshold": DECISION_THRESHOLD,
        "feature_pipeline_version": result["meta"].get("pipeline_version"),
    }, indent=2))

    # metrics.json LAST -> presence = "done" marker for idempotent resume.
    (out_dir / "metrics.json").write_text(json.dumps({
        "task": TASK,
        "strategy": strategy,
        "variant": variant,
        "space": space,
        "model": model_name,
        "seed": SEED,
        "n_cv_folds": N_CV_FOLDS,
        "decision_threshold": DECISION_THRESHOLD,
        "best_iteration_final": result["best_iteration_final"],
        "cell_elapsed_sec": round(cell_elapsed, 1),
        "oof": result["oof_metrics"],
        "val": result["val_metrics"],
        "test": result["test_metrics"],
    }, indent=2))


def update_summary(summary: dict) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))


# -----------------------------------------------------------------------------
# Sweep driver
# -----------------------------------------------------------------------------
def main() -> int:
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    log(f"Classification sweep start. Strategies={STRATEGIES} Variants={VARIANTS}")
    log(f"Total cells: {len(STRATEGIES) * len(VARIANTS) * 2}")

    summary = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "cells": []}
    if SUMMARY_PATH.exists():
        try:
            summary = json.loads(SUMMARY_PATH.read_text())
        except Exception:
            pass

    sweep_t0 = time.time()
    cell_count = 0
    completed = 0
    skipped = 0
    failed = 0

    for strategy in STRATEGIES:
        for variant in VARIANTS:
            for model_name, space, runner in [
                ("xgboost", "tree", run_xgb_cell),
                ("logreg_enet", "linear", run_logreg_cell),
            ]:
                cell_count += 1
                out_dir = MODELS_ROOT / strategy / variant / model_name
                metrics_path = out_dir / "metrics.json"
                tag = f"[{cell_count}] {strategy}/{variant}/{model_name}"

                if metrics_path.exists():
                    log(f"{tag} SKIP (already complete)")
                    skipped += 1
                    continue

                log(f"{tag} START")
                t0 = time.time()
                try:
                    result = runner(strategy, variant)
                    cell_elapsed = time.time() - t0
                    persist_cell(out_dir, model_name, strategy, variant, space,
                                 result, cell_elapsed)
                    completed += 1
                    log(f"{tag} DONE in {cell_elapsed:.1f}s | "
                        f"test AUC={result['test_metrics']['roc_auc']:.4f} "
                        f"BalAcc={result['test_metrics']['balanced_accuracy']:.4f} "
                        f"MCC={result['test_metrics']['mcc']:.4f}")

                    summary["cells"].append({
                        "strategy": strategy,
                        "variant": variant,
                        "model": model_name,
                        "elapsed_sec": round(cell_elapsed, 1),
                        "test_roc_auc": result["test_metrics"]["roc_auc"],
                        "test_balanced_accuracy": result["test_metrics"]["balanced_accuracy"],
                        "test_f1_macro": result["test_metrics"]["f1_macro"],
                        "test_mcc": result["test_metrics"]["mcc"],
                        "test_brier": result["test_metrics"]["brier"],
                        "val_roc_auc": result["val_metrics"]["roc_auc"],
                        "oof_roc_auc": result["oof_metrics"]["roc_auc"],
                        "best_config": result["best_config"],
                    })
                    update_summary(summary)

                except Exception as e:
                    failed += 1
                    log(f"{tag} FAILED after {time.time() - t0:.1f}s: {e}")
                    traceback.print_exc()

    total = time.time() - sweep_t0
    log(f"SWEEP COMPLETE in {total/3600:.2f}h | "
        f"completed={completed} skipped={skipped} failed={failed}")
    summary["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    summary["total_elapsed_sec"] = round(total, 1)
    summary["completed"] = completed
    summary["skipped"] = skipped
    summary["failed"] = failed
    update_summary(summary)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
