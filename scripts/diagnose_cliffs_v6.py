#!/usr/bin/env python3
"""
diagnose_cliffs_v6.py – Statistically‑calibrated cliff failure diagnostics
with NaN‑filtered loading, multi‑model predictions, local‑neighbour null
distributions, normalised instability, and percentile‑based sparsity.
"""

from __future__ import annotations
import argparse, json, logging, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("v6")
warnings.filterwarnings("ignore")


# -------------------------------------------------------------------------
# Loading helpers (with NaN filtering)
# -------------------------------------------------------------------------
def load_npz(path):
    """Load X, y, ids from npz; remove rows with NaN/inf targets."""
    d = np.load(path, allow_pickle=True)
    X = d["X"].astype(np.float32)
    y = d["y"].astype(float)
    ids = np.array(d["ids"]).astype(str)

    valid = np.isfinite(y)
    if not valid.all():
        n_bad = (~valid).sum()
        log.warning(f"{path}: removing {n_bad} rows with NaN/inf targets")
        X = X[valid]
        y = y[valid]
        ids = ids[valid]

    return X, y, ids


def load_split(base, variant, space):
    """Load train/val/test feature blocks for a variant."""
    subsets = {}
    for split in ("train", "val", "test"):
        p = Path(base) / split / f"{variant}_{space}.npz"
        if p.exists():
            X, y, ids = load_npz(p)
            subsets[split] = {
                "X": X, "y": y, "ids": ids,
                "map": {i: idx for idx, i in enumerate(ids)}
            }
            log.info(f"  {split}: {X.shape}  ids={len(ids)}")
        else:
            log.warning(f"  Missing {p}")
    if not subsets:
        sys.exit("No feature files found.")
    return subsets


def load_cliffs(path):
    """Normalise cliff CSV to ik14_a, ik14_b, true_delta, optional tanimoto."""
    df = pd.read_csv(path)
    log.info(f"Cliff file: {len(df)} pairs, cols={list(df.columns)}")
    col_a = next((c for c in df.columns if "inchikey" in c.lower() and "1" in c), None)
    col_b = next((c for c in df.columns if "inchikey" in c.lower() and "2" in c), None)
    col_d = next((c for c in df.columns if "delta" in c.lower()), None)
    col_t = next((c for c in df.columns if "tanimoto" in c.lower()), None)
    if col_a is None or col_b is None:
        sys.exit("Cliff file missing InChIKey columns.")
    out = pd.DataFrame({
        "ik14_a": df[col_a].astype(str).str[:14],
        "ik14_b": df[col_b].astype(str).str[:14],
    })
    if col_d:
        out["true_delta"] = df[col_d].astype(float).abs()
    else:
        out["true_delta"] = np.nan
    if col_t:
        out["tanimoto"] = df[col_t].astype(float)
    return out


# -------------------------------------------------------------------------
# Optional structural Tanimoto (RDKit)
# -------------------------------------------------------------------------
def compute_morgan_tanimoto(smiles_a, smiles_b, radius=2, nbits=2048):
    if not HAS_RDKIT:
        return None
    try:
        mol_a = Chem.MolFromSmiles(smiles_a)
        mol_b = Chem.MolFromSmiles(smiles_b)
        if mol_a is None or mol_b is None:
            return None
        fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, radius, nBits=nbits)
        fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, radius, nBits=nbits)
        return DataStructs.TanimotoSimilarity(fp_a, fp_b)
    except:
        return None


# -------------------------------------------------------------------------
# Multi‑model training & prediction (precomputed once)
# -------------------------------------------------------------------------
def train_models(X_train, y_train):
    """Train RF, XGB (if available), and ElasticNet on training set."""
    models = {}
    # Random Forest
    rf = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    models["RF"] = rf

    # XGBoost (optional)
    if HAS_XGB:
        xgbm = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                                subsample=0.8, colsample_bytree=0.8,
                                random_state=42, n_jobs=-1, verbosity=0)
        xgbm.fit(X_train, y_train)
        models["XGB"] = xgbm
    else:
        models["XGB"] = None

    # ElasticNet (linear baseline with scaling)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    en = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)
    en.fit(X_scaled, y_train)
    models["EN"] = (scaler, en)

    return models


def predict_all(models, X):
    """
    Return dict with:
        "RF": point predictions
        "RF_std": per‑sample tree standard deviation
        "XGB": XGB predictions (or NaN if unavailable)
        "EN": ElasticNet predictions
    """
    preds = {}
    # RF + tree variance
    rf_preds = models["RF"].predict(X)
    tree_preds = np.array([t.predict(X) for t in models["RF"].estimators_])
    rf_std = np.std(tree_preds, axis=0)
    preds["RF"] = rf_preds
    preds["RF_std"] = rf_std

    # XGBoost
    if models["XGB"] is not None:
        preds["XGB"] = models["XGB"].predict(X)
    else:
        preds["XGB"] = np.full_like(rf_preds, np.nan)

    # ElasticNet
    scaler, en = models["EN"]
    X_scaled = scaler.transform(X)
    preds["EN"] = en.predict(X_scaled)

    return preds


# -------------------------------------------------------------------------
# Statistical calibrations (train‑based)
# -------------------------------------------------------------------------
def local_neighbour_distribution(X_train, k=5):
    """
    Distribution of distances to the nearest neighbour (within train)
    for a random subset – used as the null for collision Z‑score.
    """
    n_sample = min(1000, len(X_train))
    idx = np.random.choice(len(X_train), n_sample, replace=False)
    nn = NearestNeighbors(n_neighbors=min(k+1, len(X_train)), metric="euclidean").fit(X_train)
    # Index 0 is self; index 1 is nearest *other* neighbour
    dists, _ = nn.kneighbors(X_train[idx])
    local_dists = dists[:, 1]   # distance to closest other train point
    return local_dists


def density_percentile(X_train, percentile=95):
    """Return the density threshold at the given percentile of train‑train mean NN distances."""
    nn = NearestNeighbors(n_neighbors=min(6, len(X_train)), metric="euclidean").fit(X_train)
    distances, _ = nn.kneighbors(X_train)
    # Exclude self (distance 0)
    mean_dists = distances[:, 1:].mean(axis=1)
    return float(np.percentile(mean_dists, percentile))


def density(X_train, vec, k=5):
    """Mean Euclidean distance to k nearest neighbours in training set."""
    if len(X_train) == 0:
        return np.nan
    nn = NearestNeighbors(n_neighbors=min(k, len(X_train)), metric="euclidean").fit(X_train)
    d, _ = nn.kneighbors(vec.reshape(1, -1))
    return float(d.mean())


def compute_pair_collision_zscore(train_null_dists, v_a, v_b):
    """
    Compare cliff‑pair distance to the distribution of local neighbour distances
    in the training set. Returns Z‑score and collision flag (Z < –1).
    """
    pair_dist = float(np.linalg.norm(v_a - v_b))
    mu = float(np.mean(train_null_dists))
    sigma = float(np.std(train_null_dists))
    z = (pair_dist - mu) / sigma if sigma > 0 else 0.0
    is_collision = z < -1.0   # unusually close
    return {
        "pair_distance": pair_dist,
        "null_mean": mu,
        "null_std": sigma,
        "z_score": round(z, 3),
        "is_feature_collision": is_collision,
    }


# -------------------------------------------------------------------------
# Root‑cause classifier (normalised thresholds)
# -------------------------------------------------------------------------
def classify_root_cause(attenuation, is_collision, density, density_threshold,
                        rf_uncertainty, model_disagreement_norm, memorized, struct_tani):
    """
    model_disagreement_norm = std(model Δs) / true_delta
    Returns (verdict, explanation, recommendation).
    """
    # Resolved
    if attenuation is not None and not np.isnan(attenuation) and attenuation >= 0.7:
        return ("resolved",
                "Model captures the activity difference well.",
                "No action needed.")

    # Overfitting / memorization
    if memorized and attenuation is not None and attenuation < 0.5:
        return ("overfit_memorized",
                "Cliff resolved in training set but not on test – overfitting or domain shift.",
                "Increase regularization, check for data leakage, or collect more diverse training data.")

    # Feature collision
    if is_collision:
        return ("feature_collision",
                "Compounds are indistinguishable in the current feature space (local Z‑score collision).",
                "Featurization bottleneck: add 3D descriptors, electrostatic features, or use pretrained representations.")

    # Model instability (normalised)
    if model_disagreement_norm > 0.5:   # Δ std > 50% of true delta
        return ("model_instability",
                "Different model types give very different Δ predictions; cliff is sensitive to inductive bias.",
                "Ensemble models and calibrate uncertainty; consider Bayesian methods.")

    # Epistemic uncertainty (high RF variance)
    if rf_uncertainty > 1.0 and density < density_threshold:
        return ("epistemic_uncertainty",
                "High ensemble variance suggests insufficient support; model is uncertain.",
                "Add more training data in this region or use a more expressive architecture.")

    # Sparse region (density above percentile threshold)
    if density > density_threshold:
        return ("sparse_region",
                "Compounds lie far from training data; model extrapolating.",
                "Collect more training compounds in this chemical region.")

    # Default: model averaging
    return ("model_averaging",
            "Features differ but model still predicts similar activities.",
            "Try a more expressive model (e.g., deeper GNN) or add more training data.")


# -------------------------------------------------------------------------
# Main pair diagnosis (uses precomputed predictions)
# -------------------------------------------------------------------------
def diagnose_pair_v4_precomputed(a, b, true_delta, subsets,
                                 train_preds, test_preds,
                                 train_null_dists, density_threshold,
                                 smiles_map=None, cliffs_df=None):
    """
    Diagnose one cliff pair using precomputed predictions and calibrations.
    """
    test = subsets["test"]
    train = subsets["train"]
    if a not in test["map"] or b not in test["map"]:
        return None

    ia, ib = test["map"][a], test["map"][b]
    va, vb = test["X"][ia], test["X"][ib]

    # Test predictions
    pa_rf = test_preds["RF"][ia]
    pb_rf = test_preds["RF"][ib]
    pred_delta_rf = abs(pa_rf - pb_rf)
    att_rf = pred_delta_rf / true_delta if true_delta > 0 else np.nan

    rf_uncertainty = np.sqrt(test_preds["RF_std"][ia]**2 + test_preds["RF_std"][ib]**2)

    pa_xgb = test_preds["XGB"][ia]
    pb_xgb = test_preds["XGB"][ib]
    pred_delta_xgb = abs(pa_xgb - pb_xgb) if not np.isnan(pa_xgb) else np.nan
    att_xgb = pred_delta_xgb / true_delta if true_delta > 0 and not np.isnan(pred_delta_xgb) else np.nan

    pa_en = test_preds["EN"][ia]
    pb_en = test_preds["EN"][ib]
    pred_delta_en = abs(pa_en - pb_en)
    att_en = pred_delta_en / true_delta if true_delta > 0 else np.nan

    # Model disagreement normalised by true delta
    deltas = [pred_delta_rf]
    if not np.isnan(pred_delta_xgb):
        deltas.append(pred_delta_xgb)
    deltas.append(pred_delta_en)
    model_disagreement_norm = float(np.std(deltas) / true_delta) if true_delta > 0 else 0.0

    # Feature collision Z‑score
    coll = compute_pair_collision_zscore(train_null_dists, va, vb)

    # Density
    d_a = density(train["X"], va)
    d_b = density(train["X"], vb)
    mean_density = (d_a + d_b) / 2

    # Memorization
    train_att = None
    memorized = False
    if a in train["map"] and b in train["map"]:
        idx_a_tr = train["map"][a]
        idx_b_tr = train["map"][b]
        p_tr_a = train_preds["RF"][idx_a_tr]
        p_tr_b = train_preds["RF"][idx_b_tr]
        train_pred_delta = abs(p_tr_a - p_tr_b)
        train_att = train_pred_delta / true_delta if true_delta > 0 else np.nan
        memorized = train_att is not None and not np.isnan(train_att) and train_att >= 0.7

    # Structural Tanimoto
    struct_tani = None
    if cliffs_df is not None and "tanimoto" in cliffs_df.columns:
        row = cliffs_df[(cliffs_df["ik14_a"] == a) & (cliffs_df["ik14_b"] == b)]
        if not row.empty:
            struct_tani = row.iloc[0]["tanimoto"]
    if smiles_map and a in smiles_map and b in smiles_map:
        t = compute_morgan_tanimoto(smiles_map[a], smiles_map[b])
        if t is not None:
            struct_tani = t

    # Root cause
    verdict, explanation, recommendation = classify_root_cause(
        att_rf, coll["is_feature_collision"], mean_density, density_threshold,
        rf_uncertainty, model_disagreement_norm, memorized, struct_tani,
    )

    return {
        "a": a, "b": b,
        "true_delta": true_delta,
        "pred_delta_rf": round(pred_delta_rf, 4),
        "attenuation_rf": round(att_rf, 4) if not np.isnan(att_rf) else None,
        "pred_delta_xgb": round(pred_delta_xgb, 4) if not np.isnan(pred_delta_xgb) else None,
        "attenuation_xgb": round(att_xgb, 4) if not np.isnan(att_xgb) else None,
        "pred_delta_en": round(pred_delta_en, 4),
        "attenuation_en": round(att_en, 4) if not np.isnan(att_en) else None,
        "rf_uncertainty": round(rf_uncertainty, 4),
        "model_disagreement_norm": round(model_disagreement_norm, 4),
        "collision_z_score": coll["z_score"],
        "is_feature_collision": coll["is_feature_collision"],
        "density_a": round(d_a, 4), "density_b": round(d_b, 4),
        "mean_density": round(mean_density, 4),
        "density_percentile_threshold": round(density_threshold, 4),
        "train_attenuation_rf": round(train_att, 4) if train_att is not None else None,
        "memorized_in_train": memorized,
        "structural_tanimoto": round(struct_tani, 4) if struct_tani is not None else None,
        "verdict": verdict,
        "explanation": explanation,
        "recommendation": recommendation,
    }


# -------------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------------
def summarise_v6(df):
    """Aggregate counts for each verdict category."""
    n = len(df)
    verdict_counts = df["verdict"].value_counts().to_dict()
    out = {"n_pairs": n}
    for v in ["resolved", "feature_collision", "model_averaging",
              "model_instability", "sparse_region", "epistemic_uncertainty",
              "overfit_memorized"]:
        out[v] = verdict_counts.get(v, 0)
    out["median_attenuation_rf"] = round(df["attenuation_rf"].median(), 3)
    return out


# -------------------------------------------------------------------------
# Main (single split execution)
# -------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cliff_file", required=True)
    ap.add_argument("--features_dir", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--space", default="tree")
    ap.add_argument("--smiles_map", default="",
                    help="CSV with inchikey_14,smiles columns (optional)")
    ap.add_argument("--output", default="diagnosis_v6")
    args = ap.parse_args()

    subsets = load_split(Path(args.features_dir), args.variant, args.space)
    cliffs = load_cliffs(Path(args.cliff_file))

    # Train models and precompute predictions once
    X_train = subsets["train"]["X"]
    y_train = subsets["train"]["y"]
    models = train_models(X_train, y_train)
    log.info("Trained RF, XGB (if available), and ElasticNet.")

    train_preds = predict_all(models, X_train)
    test_preds  = predict_all(models, subsets["test"]["X"])
    log.info("Precomputed all predictions.")

    # Statistical calibrations
    train_null_dists = local_neighbour_distribution(X_train)
    density_threshold = density_percentile(X_train, percentile=95)
    log.info(f"Collision null: mean={train_null_dists.mean():.3f}, std={train_null_dists.std():.3f}")
    log.info(f"Density sparsity threshold (95th pctile): {density_threshold:.3f}")

    # Load SMILES map if provided
    smiles_map = {}
    if args.smiles_map:
        smi_df = pd.read_csv(args.smiles_map)
        if "inchikey_14" in smi_df.columns and "smiles" in smi_df.columns:
            smiles_map = dict(zip(smi_df["inchikey_14"], smi_df["smiles"]))
            log.info(f"Loaded {len(smiles_map)} SMILES.")

    rows = []
    for _, r in cliffs.iterrows():
        res = diagnose_pair_v4_precomputed(
            r["ik14_a"], r["ik14_b"], r["true_delta"],
            subsets, train_preds, test_preds,
            train_null_dists, density_threshold,
            smiles_map, cliffs
        )
        if res:
            rows.append(res)

    df = pd.DataFrame(rows)
    summary = summarise_v6(df)

    df.to_csv(args.output + ".csv", index=False)
    with open(args.output + ".json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Cliff Diagnostic Summary (v6) ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    unresolved = df[df["verdict"] != "resolved"]
    if len(unresolved) > 0:
        worst = unresolved.nlargest(5, "true_delta")
        print("\nTop 5 unresolved cliffs:")
        for _, row in worst.iterrows():
            print(f"  {row['a'][:10]}/{row['b'][:10]}  "
                  f"Δtrue={row['true_delta']:.2f}  RF att={row['attenuation_rf']:.3f}  "
                  f"[{row['verdict']}] → {row['recommendation'][:80]}")


if __name__ == "__main__":
    main()