#!/usr/bin/env python3
"""
diagnose_all_splits_v6.py – Batch cliff diagnosis over all (task, strategy,
variant, space) combinations using the statistically‑calibrated v6 engine.
"""

from __future__ import annotations
import argparse, logging, sys
from pathlib import Path
import pandas as pd
import numpy as np

# v6 engine functions
from diagnose_cliffs_v6 import (
    load_split,
    load_cliffs,
    train_models,
    predict_all,
    local_neighbour_distribution,
    density_percentile,
    diagnose_pair_v4_precomputed,
    summarise_v6,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("batch_v6")

TASKS = ["regression", "classification"]
STRATEGIES = ["scaffold", "random", "similarity", "lead_opt", "confirmed"]
VARIANTS = ["full", "fingerprints", "physchem", "mordred", "fragments"]
SPACES = ["tree", "linear"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cliff_file", required=True,
                    help="pad_activity_cliffs.csv")
    ap.add_argument("--features_root", default="features_v18")
    ap.add_argument("--smiles_map", default="",
                    help="CSV with inchikey_14,smiles columns (optional)")
    ap.add_argument("--output_prefix", default="batch_v6")
    ap.add_argument("--percentile", type=float, default=95,
                    help="Percentile for density sparsity threshold")
    args = ap.parse_args()

    # Load SMILES map if provided
    smiles_map = {}
    if args.smiles_map:
        smi_df = pd.read_csv(args.smiles_map)
        if "inchikey_14" in smi_df.columns and "smiles" in smi_df.columns:
            smiles_map = dict(zip(smi_df["inchikey_14"], smi_df["smiles"]))
            log.info(f"Loaded {len(smiles_map)} SMILES for structural Tanimoto.")
        else:
            log.warning("SMILES map must have inchikey_14 and smiles columns; ignoring.")

    cliffs = load_cliffs(Path(args.cliff_file))

    all_rows = []
    summary_rows = []

    for task in TASKS:
        for strategy in STRATEGIES:
            feat_dir = Path(args.features_root) / task / strategy
            if not feat_dir.exists():
                log.warning(f"Directory missing, skipping: {feat_dir}")
                continue

            for variant in VARIANTS:
                for space in SPACES:
                    try:
                        subsets = load_split(feat_dir, variant, space)
                    except SystemExit:
                        log.info(f"  No data for {feat_dir}/{variant}_{space}")
                        continue

                    # Train models and precompute predictions (once per combo)
                    X_train = subsets["train"]["X"]
                    y_train = subsets["train"]["y"]
                    models = train_models(X_train, y_train)
                    train_preds = predict_all(models, X_train)
                    test_preds  = predict_all(models, subsets["test"]["X"])

                    # Statistical calibrations
                    train_null_dists = local_neighbour_distribution(X_train)
                    density_threshold = density_percentile(X_train, args.percentile)

                    log.info(f"Processing {task}/{strategy}/{variant}/{space} "
                             f"density_thresh={density_threshold:.3f}")

                    combo_rows = []
                    for _, r in cliffs.iterrows():
                        res = diagnose_pair_v4_precomputed(
                            r["ik14_a"], r["ik14_b"], r["true_delta"],
                            subsets, train_preds, test_preds,
                            train_null_dists, density_threshold,
                            smiles_map, cliffs
                        )
                        if res:
                            res["task"] = task
                            res["strategy"] = strategy
                            res["variant"] = variant
                            res["space"] = space
                            combo_rows.append(res)

                    combo_df = pd.DataFrame(combo_rows)
                    summ = summarise_v6(combo_df)
                    summ.update({"task": task, "strategy": strategy,
                                 "variant": variant, "space": space})
                    summary_rows.append(summ)
                    all_rows.extend(combo_rows)

                    # Save per‑combination CSV
                    out_combo = f"{args.output_prefix}_{task}_{strategy}_{variant}_{space}.csv"
                    combo_df.to_csv(out_combo, index=False)

    # Global outputs
    summary_df = pd.DataFrame(summary_rows)
    detail_df = pd.DataFrame(all_rows)

    summary_csv = args.output_prefix + "_summary.csv"
    detail_csv = args.output_prefix + "_all_pairs.csv"
    summary_df.to_csv(summary_csv, index=False)
    detail_df.to_csv(detail_csv, index=False)

    log.info(f"Wrote summary: {summary_csv}")
    log.info(f"Wrote all details: {detail_csv}")

    # Quick overview
    print("\n=== Mean failure counts across all splits ===")
    grp = summary_df.groupby(["task", "strategy", "variant", "space"])[
        ["resolved", "feature_collision", "model_averaging",
         "model_instability", "sparse_region", "epistemic_uncertainty",
         "overfit_memorized"]
    ].mean().round(1)
    print(grp.to_string())


if __name__ == "__main__":
    main()