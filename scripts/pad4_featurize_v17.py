#!/usr/bin/env python3
"""
PAD4 Featurization Pipeline v17.0 — Patched Frozen Benchmark Release
====================================================================

Patches over v16.0. Three correctness fixes that affect benchmark
integrity. No score/feature-pipeline changes; outputs differ from v16
only where v16 was wrong.

Changes from v16 → v17
-----------------------
  [FIX-A] Subset re-slicing bug REMOVED. v16 concatenated train+val+test
          into a single dataframe, ran precompute_* on the concatenation,
          then re-split via index arithmetic on dict insertion order.
          This was fragile (depended on Python dict ordering AND on
          concat producing exactly the original row spans) and had an
          asymmetric "train is first" special case. v17 computes global
          aggregates ONCE and joins them per-subset by inchikey_14. No
          concat, no re-slice, no index arithmetic.

  [FIX-B] Assay-consistency stats now fit on TRAIN ONLY. v16 grouped
          across the full concatenated dataframe, so per-compound stats
          like `assay_disagreement_score` could in principle be informed
          by test-set measurements (in practice limited by inchikey-14
          dedup, but the principle was wrong). v17 fits aggregates on
          train data only and looks them up for val/test; compounds
          unseen in train get zero-valued defaults (signal "no train
          evidence", not "perfect consistency").

  [FIX-C] VIF candidate selection uses a per-fold RNG. v16 hardcoded
          `random_state=42` for `mutual_info_regression` inside
          ScopedVIFFilter, so every outer fold's MI ranking was computed
          on different row subsets but with the same seed, producing
          near-identical 400-feature candidate pools and a stable
          ~298 features removed per fold. v17 threads the master RNG
          into ScopedVIFFilter so per-fold variation is real, not
          cosmetic.

  [DIAG-A] Manifest records per-stage dropped feature counts per
          variant (zero_variance / corr / fragment_min_support / mi /
          vif), so the pipeline state is auditable without re-running.

Usage
-----
  python pad4_featurize_v17.py \
      --split_dir ./data/splits/regression/scaffold \
      --output_dir ./features_v17/scaffold \
      --variants full fingerprints physchem mordred fragments \
      --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import subprocess
import sys
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# S0  Logging, reproducibility, exceptions
# ==============================================================================
DEFAULT_SEED = 42
PIPELINE_VERSION = "17.0"

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, *a, **kw):
        return it


class PipelineError(Exception):
    pass


class FeatureSpaceMismatch(PipelineError):
    pass


class RowCountMismatch(PipelineError):
    pass


class NoValidMolecules(PipelineError):
    pass


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    import random as _r
    _r.seed(seed)


def get_git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


# ==============================================================================
# S1  RDKit + optional dependencies
# ==============================================================================
try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, Descriptors, MACCSkeys, rdMolDescriptors
    from rdkit.Chem.rdFingerprintGenerator import (
        GetAtomPairGenerator,
        GetTopologicalTorsionGenerator,
    )
    from rdkit.DataStructs import ConvertToNumpyArray
    RDLogger.DisableLog("rdApp.*")
    HAS_RDKIT = True
except ImportError:
    log.error("RDKit is required")
    sys.exit(1)

try:
    from mordred import Calculator as MordredCalculator
    from mordred import descriptors as mordred_desc
    HAS_MORDRED = True
except ImportError:
    HAS_MORDRED = False
    log.warning("Mordred not installed; will be skipped")

HAS_DOPTOOLS = False
HAS_CHYTHON = False
try:
    from doptools import ChythonCircus, ChythonLinear
    CircuS, ChyLine = ChythonCircus, ChythonLinear
    HAS_DOPTOOLS = True
except ImportError:
    pass
try:
    from chython import smiles as chython_smiles
    HAS_CHYTHON = True
except ImportError:
    pass

if HAS_DOPTOOLS and HAS_CHYTHON:
    log.info("DOPtools + Chython available")
else:
    log.warning("DOPtools/Chython unavailable; fragment features will be skipped")


# ==============================================================================
# S2  Pipeline configuration
# ==============================================================================
@dataclass
class PipelineConfig:
    split_dir: str = ""
    output_dir: str = ""
    cache_dir: str = ""
    cliff_file: str = ""
    smiles_col: str = "canonical_smiles"
    id_col: str = "inchikey_14"
    activity_col: str = "pIC50"
    activity_col_fallbacks: List[str] = field(default_factory=lambda: [
        "pKi", "Ki_nM", "IC50_nM", "ki", "ic50"
    ])
    assay_type_col: str = "assay_type"
    weight_col: str = "ml_weight"
    test_filename: str = "test_locked.csv"

    ad_score_col: str = "t1_self_tanimoto"
    ad_in_domain_col: str = ""
    ad_score_col_fallback: str = "t1_novelty_score"
    ad_threshold: float = 0.35

    variants: List[str] = field(default_factory=lambda: [
        "full", "fingerprints", "physchem", "mordred", "fragments"
    ])

    mordred_nan_pct: float = 0.05
    doptools_fit_samples: int = 5000
    use_mordred: bool = True
    use_doptools: bool = True

    variance_threshold: float = 1e-6
    corr_threshold: float = 0.95
    fp_mi_target_k: int = 2048
    fragment_min_support: float = 0.01

    vif_threshold: float = 10.0
    vif_max_candidates: int = 400
    apply_vif_to_fingerprints: bool = False

    n_features_linear: int = 120
    n_outer_cv: int = 5
    n_inner_cv: int = 3
    stability_n_bootstrap: int = 50
    global_stability_min_folds: int = 3

    uncertainty_k: int = 5
    compute_uncertainty: bool = True
    compute_assay_consistency: bool = True

    seed: int = DEFAULT_SEED
    stratifiers_only: bool = False

    def validate(self) -> None:
        if not self.split_dir:
            raise PipelineError("--split_dir is required")
        if not self.output_dir:
            raise PipelineError("--output_dir is required")
        if not (1 <= self.global_stability_min_folds <= self.n_outer_cv):
            raise PipelineError("global_stability_min_folds must be in [1, n_outer_cv]")
        valid_variants = {"full", "fingerprints", "physchem", "mordred", "fragments"}
        for v in self.variants:
            if v not in valid_variants:
                raise PipelineError(f"Unknown variant: {v}")

    @property
    def config_hash(self) -> str:
        return hashlib.md5(
            json.dumps(asdict(self), sort_keys=True, default=str).encode()
        ).hexdigest()[:12]


FAMILY_BY_NAMESPACE = {
    "ecfp4": "fingerprint",
    "ecfp6": "fingerprint",
    "maccs": "fingerprint",
    "ap": "fingerprint",
    "tt": "fingerprint",
    "physchem": "physchem",
    "vsa": "physchem",
    "smarts": "physchem",
    "meta": "physchem",
    "mordred": "mordred",
    "frag": "fragment",
}

VARIANT_FAMILIES = {
    "full":         {"fingerprint", "physchem", "mordred", "fragment"},
    "fingerprints": {"fingerprint"},
    "physchem":     {"physchem"},
    "mordred":      {"mordred"},
    "fragments":    {"fragment"},
}


# ==============================================================================
# S3  Descriptor constants
# ==============================================================================
ECFP4_BITS = 2048
ECFP6_BITS = 2048
MACCS_BITS = 167
AP_BITS = 2048
TT_BITS = 2048

PHYSCHEM_LIST = [
    ("MolWt", Descriptors.MolWt),
    ("MolLogP", Descriptors.MolLogP),
    ("TPSA", Descriptors.TPSA),
    ("NumHAcceptors", Descriptors.NumHAcceptors),
    ("NumHDonors", Descriptors.NumHDonors),
    ("NumRotatableBonds", Descriptors.NumRotatableBonds),
    ("NumAromaticRings", Descriptors.NumAromaticRings),
    ("NumAliphaticRings", Descriptors.NumAliphaticRings),
    ("NumSaturatedRings", Descriptors.NumSaturatedRings),
    ("RingCount", Descriptors.RingCount),
    ("FractionCSP3", Descriptors.FractionCSP3),
    ("NumHeteroatoms", Descriptors.NumHeteroatoms),
    ("HeavyAtomCount", Descriptors.HeavyAtomCount),
    ("NHOHCount", Descriptors.NHOHCount),
    ("NOCount", Descriptors.NOCount),
    ("NumAliphaticCarbocycles", Descriptors.NumAliphaticCarbocycles),
    ("NumAromaticCarbocycles", Descriptors.NumAromaticCarbocycles),
    ("NumAromaticHeterocycles", Descriptors.NumAromaticHeterocycles),
    ("BertzCT", Descriptors.BertzCT),
    ("LabuteASA", Descriptors.LabuteASA),
    ("BalabanJ", Descriptors.BalabanJ),
    ("Kappa1", Descriptors.Kappa1),
    ("Kappa2", Descriptors.Kappa2),
    ("Kappa3", Descriptors.Kappa3),
    ("HallKierAlpha", Descriptors.HallKierAlpha),
    ("Chi0n", Descriptors.Chi0n),
    ("Chi1n", Descriptors.Chi1n),
    ("Chi2n", Descriptors.Chi2n),
    ("Chi3n", Descriptors.Chi3n),
    ("Chi4n", Descriptors.Chi4n),
]

VSA_SPECS = [
    ("SlogP_VSA", 12, rdMolDescriptors.SlogP_VSA_),
    ("SMR_VSA", 10, rdMolDescriptors.SMR_VSA_),
    ("PEOE_VSA", 14, rdMolDescriptors.PEOE_VSA_),
]
VSA_NAMES = [f"{p}_{i}" for p, n, _ in VSA_SPECS for i in range(n)]
VSA_TOTAL = sum(n for _, n, _ in VSA_SPECS)

PAD4_SMARTS = {
    "guanidine": "[NX3][CX3](=[NX3])[NX3]",
    "amidine": "[NX3][CX3](=[NX2])[NX2]",
    "primary_amine": "[NX3H2][CX4]",
    "secondary_amine": "[NX3H1]([CX4])[CX4]",
    "amide": "[NX3][CX3](=[OX1])[CX4,CX3]",
    "ester": "[OX2][CX3](=[OX1])[CX4,CX3]",
    "ketone": "[CX4,CX3][CX3](=[OX1])[CX4,CX3]",
    "aldehyde": "[CX3H1](=[OX1])[CX4,CX3]",
    "phenyl": "c1ccccc1",
    "pyridine": "c1ccncc1",
    "pyrimidine": "c1ccncn1",
    "indole": "c1ccc2[nH]ccc2c1",
    "quinoline": "c1ccc2ncccc2c1",
    "piperazine": "N1CCNCC1",
    "piperidine": "N1CCCCC1",
    "morpholine": "O1CCNCC1",
    "thiazole": "c1cncs1",
    "oxazole": "c1cnco1",
    "aryl_chloride": "c[Cl]",
    "aryl_fluoride": "c[F]",
    "alkyl_chloride": "[CX4][Cl]",
    "ether": "[CX4][OX2][CX4]",
    "sulfonamide": "[SX4](=[OX1])(=[OX1])[NX3]",
    "urea": "[NX3][CX3](=[OX1])[NX3]",
    "acrylamide": "[NX3][CX3](=[OX1])[CX3]=[CX3]",
    "nitrile": "[CX2]#[NX1]",
    "hydroxamate": "[NX3][CX3](=[OX1])[OX2H]",
    "carboxylate": "[CX3](=[OX1])[OH]",
}

METADATA_SPEC = {
    "is_covalent":           ("is_covalent", 0.0),
    "covalent_confidence":   ("covalent_confidence", 0.0),
    "stereo_defined":        ("stereo_defined_flag", 0.0),
    "complexity_score":      ("complexity_score", 0.3),
    "is_frequent_scaffold":  ("is_frequent_scaffold", 0.0),
}


# ==============================================================================
# S4  Molecule parsing + stratifier precomputation
# ==============================================================================
def parse_mol(smi: str):
    if not isinstance(smi, str) or not smi.strip():
        return None
    return Chem.MolFromSmiles(smi.strip())


def compile_smarts(smarts_dict: dict) -> dict:
    out = {}
    for name, pat in smarts_dict.items():
        m = Chem.MolFromSmarts(pat)
        if m is not None:
            out[name] = m
        else:
            log.warning(f"Invalid SMARTS: {name}")
    return out


def _safe_float(v, default: float = 0.0) -> float:
    try:
        if pd.isna(v):
            return float(default)
        f = float(v)
        return f if np.isfinite(f) else float(default)
    except Exception:
        return float(default)


def _first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c and c in df.columns:
            return c
    return None


def _resolve_activity(row: pd.Series, cfg: PipelineConfig) -> float:
    """Resolve activity value with fallback chain."""
    for col in [cfg.activity_col] + cfg.activity_col_fallbacks:
        if col in row.index and pd.notna(row[col]):
            v = float(row[col])
            if np.isfinite(v):
                # Normalize Ki_nM / IC50_nM to pKi / pIC50 if needed
                if col in ("Ki_nM", "IC50_nM") and v > 0:
                    v = 9.0 - math.log10(v)
                return v
    return np.nan


def _resolve_assay_type(row: pd.Series, cfg: PipelineConfig) -> str:
    if cfg.assay_type_col in row.index and pd.notna(row[cfg.assay_type_col]):
        return str(row[cfg.assay_type_col]).strip().lower()
    if "pKi" in row.index and pd.notna(row["pKi"]):
        return "ki"
    if "Ki_nM" in row.index and pd.notna(row["Ki_nM"]):
        return "ki"
    return "ic50"


@dataclass
class ParsedSubset:
    mols: List[Any]
    smiles: List[str]
    ids: np.ndarray
    y: np.ndarray
    weights: np.ndarray
    rows: List[dict]
    stratifiers: Dict[str, np.ndarray]
    n_valid: int
    n_total: int


# ── FIX-B: assay-consistency aggregates fit on TRAIN ONLY ─────────────
def fit_assay_consistency(train_df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """
    Compute per-inchikey-14 assay aggregates from the TRAINING set only.
    Returns a small dataframe keyed on `cfg.id_col` with columns:
      ki_ic50_delta, assay_disagreement_score, multi_assay_variance,
      has_ki, has_ic50

    These are properties of each compound across its train-side
    measurements. For val/test compounds the aggregates are looked up
    (apply_assay_consistency); compounds unseen in train get zeros,
    which correctly signals "no train evidence".
    """
    if not cfg.compute_assay_consistency or cfg.id_col not in train_df.columns:
        return pd.DataFrame(columns=[
            cfg.id_col, "ki_ic50_delta", "assay_disagreement_score",
            "multi_assay_variance", "has_ki", "has_ic50",
        ])

    df = train_df.copy()
    df["_act"] = df.apply(lambda r: _resolve_activity(r, cfg), axis=1)
    df["_assay"] = df.apply(lambda r: _resolve_assay_type(r, cfg), axis=1)

    grouped = df.groupby(cfg.id_col).agg(
        n_assays=("_assay", "nunique"),
        act_std=("_act", lambda s: s.std() if len(s.dropna()) > 1 else 0.0),
        act_max=("_act", "max"),
        act_min=("_act", "min"),
        has_ki=("_assay", lambda s: int("ki" in set(s))),
        has_ic50=("_assay", lambda s: int("ic50" in set(s))),
    ).reset_index()
    grouped["ki_ic50_delta"] = (grouped["act_max"] - grouped["act_min"]).fillna(0.0)
    grouped["assay_disagreement_score"] = grouped["act_std"].fillna(0.0)
    grouped["multi_assay_variance"] = grouped["act_std"].fillna(0.0) ** 2

    return grouped[[
        cfg.id_col, "ki_ic50_delta", "assay_disagreement_score",
        "multi_assay_variance", "has_ki", "has_ic50",
    ]]


def apply_assay_consistency(
    df: pd.DataFrame,
    aggregates: pd.DataFrame,
    cfg: PipelineConfig,
) -> pd.DataFrame:
    """
    Merge train-fitted assay aggregates onto a subset.
    Unseen compounds get zero defaults (signal: no train evidence).
    """
    if not cfg.compute_assay_consistency or cfg.id_col not in df.columns:
        df = df.copy()
        df["assay_disagreement_score"] = 0.0
        df["ki_ic50_delta"] = 0.0
        df["multi_assay_variance"] = 0.0
        df["is_ki"] = 0.0
        df["is_ic50"] = 0.0
        return df

    df = df.merge(aggregates, on=cfg.id_col, how="left")
    df["ki_ic50_delta"] = df["ki_ic50_delta"].fillna(0.0)
    df["assay_disagreement_score"] = df["assay_disagreement_score"].fillna(0.0)
    df["multi_assay_variance"] = df["multi_assay_variance"].fillna(0.0)
    df["is_ki"] = df["has_ki"].fillna(0.0).astype(float)
    df["is_ic50"] = df["has_ic50"].fillna(0.0).astype(float)
    df = df.drop(columns=[c for c in ("has_ki", "has_ic50") if c in df.columns])
    return df


# ── FIX-A + cleanup: uncertainty features fit on TRAIN, applied per-subset ──
@dataclass
class UncertaintyState:
    """Frozen state from training: train fingerprints, train labels,
    per-scaffold label variance. Applied to each subset independently."""
    train_fps: list  # RDKit fingerprint objects (preserved for BulkTanimoto)
    y_train: np.ndarray
    scaffold_var: Dict[str, float]
    k: int


def fit_uncertainty(train_df: pd.DataFrame, cfg: PipelineConfig) -> Optional[UncertaintyState]:
    """Fit uncertainty features (kNN std, scaffold variance, density)
    on the TRAIN subset. Returned state is then applied to val/test
    separately (no concat-and-reslice)."""
    if not cfg.compute_uncertainty or cfg.id_col not in train_df.columns:
        return None

    # Parse once, store one fingerprint per valid molecule along with its y.
    train_fps = []
    y_vals = []
    scaf_vals = []
    scaf_col = "stereo_stripped_scaffold"

    for _, row in train_df.iterrows():
        smi = row.get(cfg.smiles_col)
        mol = parse_mol(str(smi) if pd.notna(smi) else "")
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, 2, nBits=ECFP4_BITS, useChirality=True
        )
        y = _resolve_activity(row, cfg)
        if not np.isfinite(y):
            continue
        train_fps.append(fp)
        y_vals.append(y)
        scaf_vals.append(str(row.get(scaf_col, "")))

    if len(train_fps) < cfg.uncertainty_k + 1:
        return None

    y_train = np.array(y_vals, dtype=float)

    # Per-scaffold variance from train labels
    scaffold_var: Dict[str, float] = {}
    if any(s for s in scaf_vals):
        tmp = pd.DataFrame({"_scaf": scaf_vals, "_y": y_train})
        scaffold_var = tmp.groupby("_scaf")["_y"].var().fillna(0.0).to_dict()

    return UncertaintyState(
        train_fps=train_fps,
        y_train=y_train,
        scaffold_var=scaffold_var,
        k=cfg.uncertainty_k,
    )


def apply_uncertainty(
    df: pd.DataFrame,
    state: Optional[UncertaintyState],
    cfg: PipelineConfig,
) -> pd.DataFrame:
    """Apply train-fitted uncertainty state to a subset, in-place
    semantics returning a new dataframe. Compounds with unparseable
    SMILES get zero defaults."""
    df = df.copy()
    if state is None:
        df["uncertainty_knn_std"] = 0.0
        df["uncertainty_scaffold_variance"] = 0.0
        df["feature_space_density"] = 1.0
        return df

    from rdkit import DataStructs
    scaf_col = "stereo_stripped_scaffold"
    k = state.k
    y_train = state.y_train

    knn_stds, densities, sc_vars = [], [], []
    for _, row in df.iterrows():
        smi = row.get(cfg.smiles_col)
        mol = parse_mol(str(smi)) if pd.notna(smi) else None
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, 2, nBits=ECFP4_BITS, useChirality=True
            )
            sims = np.array(DataStructs.BulkTanimotoSimilarity(fp, state.train_fps))
            topk = np.argsort(sims)[-k:]
            vals = y_train[topk]
            vals = vals[np.isfinite(vals)]
            knn_stds.append(float(np.std(vals)) if len(vals) > 1 else 0.0)
            densities.append(float(np.mean(sims[topk])))
        else:
            knn_stds.append(0.0)
            densities.append(0.0)
        sc = str(row.get(scaf_col, ""))
        sc_vars.append(_safe_float(state.scaffold_var.get(sc), 0.0))

    df["uncertainty_knn_std"] = knn_stds
    df["uncertainty_scaffold_variance"] = sc_vars
    df["feature_space_density"] = densities
    return df


def parse_subset(
    df: pd.DataFrame,
    cfg: PipelineConfig,
    cliff_ids: Set[str],
    label: str,
) -> ParsedSubset:
    mols, smiles_list, row_dicts = [], [], []
    ys, ws, ids = [], [], []

    col_ad_score = _first_present(df, [cfg.ad_score_col, cfg.ad_score_col_fallback,
                                       "ad_score", "t1_self_tanimoto"])
    col_ad_indom = _first_present(df, [cfg.ad_in_domain_col, "ad_in_domain"])
    col_label_unc = _first_present(df, ["label_uncertainty_score_v2", "label_uncertainty", "label_noise"])
    col_cliff = _first_present(df, ["is_activity_cliff", "is_cliff"])
    col_cliff_sev = _first_present(df, ["cliff_severity"])
    col_cliff_partners = _first_present(df, ["n_cliff_partners"])
    col_novelty = _first_present(df, ["t1_novelty_score"])
    col_fidelity = _first_present(df, ["fidelity_level"])
    col_confidence = _first_present(df, ["confidence_weight", "source_reliability_weight"])
    col_pic50_std = _first_present(df, ["pIC50_std"])

    col_is_ki = "is_ki" if "is_ki" in df.columns else None
    col_is_ic50 = "is_ic50" if "is_ic50" in df.columns else None
    col_assay_conf = "assay_confidence_score" if "assay_confidence_score" in df.columns else None
    col_ki_ic50_delta = "ki_ic50_delta" if "ki_ic50_delta" in df.columns else None
    col_assay_disagree = "assay_disagreement_score" if "assay_disagreement_score" in df.columns else None
    col_multi_var = "multi_assay_variance" if "multi_assay_variance" in df.columns else None
    col_unc_knn = "uncertainty_knn_std" if "uncertainty_knn_std" in df.columns else None
    col_unc_scaf = "uncertainty_scaffold_variance" if "uncertainty_scaffold_variance" in df.columns else None
    col_fsd = "feature_space_density" if "feature_space_density" in df.columns else None

    strat_keys = [
        "ad_score", "ad_in_domain",
        "ml_weight", "label_uncertainty",
        "is_cliff", "cliff_severity", "n_cliff_partners",
        "is_covalent", "is_frequent_scaffold",
        "complexity_score", "t1_novelty_score",
        "fidelity_level", "confidence_weight", "pIC50_std",
        "is_ki", "is_ic50", "assay_confidence_score",
        "ki_ic50_delta", "assay_disagreement_score", "multi_assay_variance",
        "uncertainty_knn_std", "uncertainty_scaffold_variance", "feature_space_density",
    ]
    stratifiers: Dict[str, List[float]] = {k: [] for k in strat_keys}
    scaffold_ids: List[str] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  {label}", leave=False):
        smi = row.get(cfg.smiles_col)
        mol = parse_mol(str(smi) if pd.notna(smi) else "")
        if mol is None:
            continue
        mols.append(mol)
        smiles_list.append(str(smi).strip())
        row_dicts.append(row.to_dict())

        ys.append(_resolve_activity(row, cfg))
        w_val = row.get(cfg.weight_col)
        ws.append(float(w_val) if pd.notna(w_val) else 1.0)
        rid = row.get(cfg.id_col) or f"idx_{len(ids)}"
        ids.append(str(rid))

        ad_s = _safe_float(row.get(col_ad_score)) if col_ad_score else 0.0
        if col_ad_indom:
            ad_d = _safe_float(row.get(col_ad_indom))
        else:
            ad_d = 1.0 if (col_ad_score and ad_s >= cfg.ad_threshold) else 0.0
        stratifiers["ad_score"].append(ad_s)
        stratifiers["ad_in_domain"].append(ad_d)

        stratifiers["ml_weight"].append(_safe_float(row.get(cfg.weight_col), 1.0))
        stratifiers["label_uncertainty"].append(
            _safe_float(row.get(col_label_unc)) if col_label_unc else 0.0
        )

        if col_cliff:
            stratifiers["is_cliff"].append(_safe_float(row.get(col_cliff)))
        else:
            stratifiers["is_cliff"].append(1.0 if str(rid) in cliff_ids else 0.0)
        stratifiers["cliff_severity"].append(
            _safe_float(row.get(col_cliff_sev)) if col_cliff_sev else 0.0
        )
        stratifiers["n_cliff_partners"].append(
            _safe_float(row.get(col_cliff_partners)) if col_cliff_partners else 0.0
        )

        stratifiers["is_covalent"].append(_safe_float(row.get("is_covalent")))
        stratifiers["is_frequent_scaffold"].append(
            _safe_float(row.get("is_frequent_scaffold"))
        )
        stratifiers["complexity_score"].append(
            _safe_float(row.get("complexity_score"), 0.3)
        )
        stratifiers["t1_novelty_score"].append(
            _safe_float(row.get(col_novelty)) if col_novelty else 0.0
        )
        stratifiers["fidelity_level"].append(
            _safe_float(row.get(col_fidelity)) if col_fidelity else 0.0
        )
        stratifiers["confidence_weight"].append(
            _safe_float(row.get(col_confidence), 1.0) if col_confidence else 1.0
        )
        stratifiers["pIC50_std"].append(
            _safe_float(row.get(col_pic50_std)) if col_pic50_std else 0.0
        )

        stratifiers["is_ki"].append(_safe_float(row.get(col_is_ki)) if col_is_ki else 0.0)
        stratifiers["is_ic50"].append(_safe_float(row.get(col_is_ic50)) if col_is_ic50 else 0.0)
        stratifiers["assay_confidence_score"].append(
            _safe_float(row.get(col_assay_conf), 1.0) if col_assay_conf else 1.0
        )
        stratifiers["ki_ic50_delta"].append(
            _safe_float(row.get(col_ki_ic50_delta)) if col_ki_ic50_delta else 0.0
        )
        stratifiers["assay_disagreement_score"].append(
            _safe_float(row.get(col_assay_disagree)) if col_assay_disagree else 0.0
        )
        stratifiers["multi_assay_variance"].append(
            _safe_float(row.get(col_multi_var)) if col_multi_var else 0.0
        )
        stratifiers["uncertainty_knn_std"].append(
            _safe_float(row.get(col_unc_knn)) if col_unc_knn else 0.0
        )
        stratifiers["uncertainty_scaffold_variance"].append(
            _safe_float(row.get(col_unc_scaf)) if col_unc_scaf else 0.0
        )
        stratifiers["feature_space_density"].append(
            _safe_float(row.get(col_fsd), 1.0) if col_fsd else 1.0
        )

        scaffold_ids.append(str(row.get("stereo_stripped_scaffold", "")))

    if not mols:
        raise NoValidMolecules(f"No valid molecules in {label}")

    stratifiers_arr = {k: np.array(v, dtype=np.float32) for k, v in stratifiers.items()}
    stratifiers_arr["scaffold_id"] = np.array(scaffold_ids, dtype=object)

    return ParsedSubset(
        mols=mols, smiles=smiles_list,
        ids=np.array(ids, dtype=object),
        y=np.array(ys, dtype=np.float32),
        weights=np.array(ws, dtype=np.float32),
        rows=row_dicts,
        stratifiers=stratifiers_arr,
        n_valid=len(mols),
        n_total=len(df),
    )


# ==============================================================================
# S5  RDKit descriptor block
# ==============================================================================
def compute_vsa(mol) -> np.ndarray:
    arr = np.zeros(VSA_TOTAL, np.float32)
    pos = 0
    for _, n_bins, fn in VSA_SPECS:
        try:
            vals = list(fn(mol))[:n_bins]
            arr[pos:pos + len(vals)] = [v if np.isfinite(v) else 0.0 for v in vals]
        except Exception:
            pass
        pos += n_bins
    return arr


def _covalent_confidence_num(val) -> float:
    cov_map = {"high": 1.0, "medium": 0.5, "low": 0.25, "none": 0.0, "": 0.0}
    if isinstance(val, str):
        return cov_map.get(val.lower(), 0.0)
    return _safe_float(val, 0.0)


def compute_rdkit_matrix(mols: List[Any], rows: List[dict]) -> Tuple[np.ndarray, List[str], List[str], List[str]]:
    smarts_pats = compile_smarts(PAD4_SMARTS)
    ap_gen = GetAtomPairGenerator(fpSize=AP_BITS)
    tt_gen = GetTopologicalTorsionGenerator(fpSize=TT_BITS)

    names: List[str] = []
    block_types: List[str] = []
    namespaces: List[str] = []

    for i in range(ECFP4_BITS):
        names.append(f"rdkit::ecfp4::{i}"); block_types.append("fingerprint"); namespaces.append("ecfp4")
    for i in range(ECFP6_BITS):
        names.append(f"rdkit::ecfp6::{i}"); block_types.append("fingerprint"); namespaces.append("ecfp6")
    for i in range(MACCS_BITS):
        names.append(f"rdkit::maccs::{i}"); block_types.append("fingerprint"); namespaces.append("maccs")
    for i in range(AP_BITS):
        names.append(f"rdkit::ap::{i}"); block_types.append("fingerprint"); namespaces.append("ap")
    for i in range(TT_BITS):
        names.append(f"rdkit::tt::{i}"); block_types.append("fingerprint"); namespaces.append("tt")

    for nm, _ in PHYSCHEM_LIST:
        names.append(f"rdkit::physchem::{nm}"); block_types.append("continuous"); namespaces.append("physchem")
    for nm in VSA_NAMES:
        names.append(f"rdkit::vsa::{nm}"); block_types.append("continuous"); namespaces.append("vsa")
    for nm in PAD4_SMARTS.keys():
        names.append(f"rdkit::smarts::{nm}"); block_types.append("continuous"); namespaces.append("smarts")
    for nm in METADATA_SPEC.keys():
        names.append(f"meta::{nm}"); block_types.append("continuous"); namespaces.append("meta")

    vectors: List[np.ndarray] = []
    for mol, row in zip(mols, rows):
        parts: List[np.ndarray] = []

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=ECFP4_BITS, useChirality=True)
        arr = np.zeros(ECFP4_BITS, np.float32); ConvertToNumpyArray(fp, arr); parts.append(arr)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=ECFP6_BITS, useChirality=True)
        arr = np.zeros(ECFP6_BITS, np.float32); ConvertToNumpyArray(fp, arr); parts.append(arr)
        fp = MACCSkeys.GenMACCSKeys(mol)
        arr = np.zeros(MACCS_BITS, np.float32); ConvertToNumpyArray(fp, arr); parts.append(arr)
        try:
            arr = np.zeros(AP_BITS, np.float32)
            ConvertToNumpyArray(ap_gen.GetFingerprint(mol), arr)
        except Exception:
            arr = np.zeros(AP_BITS, np.float32)
        parts.append(arr)
        try:
            arr = np.zeros(TT_BITS, np.float32)
            ConvertToNumpyArray(tt_gen.GetFingerprint(mol), arr)
        except Exception:
            arr = np.zeros(TT_BITS, np.float32)
        parts.append(arr)

        arr = np.zeros(len(PHYSCHEM_LIST), np.float32)
        for j, (_, fn) in enumerate(PHYSCHEM_LIST):
            try:
                v = float(fn(mol)); arr[j] = v if np.isfinite(v) else 0.0
            except Exception:
                pass
        parts.append(arr)

        parts.append(compute_vsa(mol))

        arr = np.zeros(len(smarts_pats), np.float32)
        for j, (_, pat) in enumerate(smarts_pats.items()):
            try:
                arr[j] = float(len(mol.GetSubstructMatches(pat)))
            except Exception:
                pass
        parts.append(arr)

        arr = np.zeros(len(METADATA_SPEC), np.float32)
        for idx, (key, (col, default)) in enumerate(METADATA_SPEC.items()):
            if key == "covalent_confidence":
                arr[idx] = _covalent_confidence_num(row.get(col, default))
            else:
                arr[idx] = _safe_float(row.get(col, default))
        parts.append(arr)

        vectors.append(np.concatenate(parts).astype(np.float32))

    X = np.vstack(vectors).astype(np.float32)
    assert X.shape[1] == len(names) == len(block_types) == len(namespaces)
    return X, names, block_types, namespaces


# ==============================================================================
# S6  Mordred with versioned per-molecule cache
# ==============================================================================
class MordredBlock:
    def __init__(self, nan_threshold: float, cache_dir: Optional[Path]):
        self.nan_threshold = nan_threshold
        self.cache_dir = cache_dir
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)

        self._calc = None
        self._keep_mask: Optional[np.ndarray] = None
        self._col_names: List[str] = []
        self._raw_names: List[str] = []
        self._cache_suffix = ""
        if HAS_MORDRED:
            try:
                self._calc = MordredCalculator(mordred_desc, ignore_3D=True)
                self._raw_names = [str(d) for d in self._calc.descriptors]
                self._n_raw = len(self._raw_names)
                desc_hash = hashlib.md5(
                    json.dumps(self._raw_names, sort_keys=True).encode()
                ).hexdigest()[:6]
                self._cache_suffix = f"_{desc_hash}"
            except Exception as e:
                log.warning(f"Mordred init failed: {e}")

    @property
    def available(self) -> bool:
        return self._calc is not None

    def _mol_to_vec(self, mol) -> np.ndarray:
        if self._calc is None:
            return np.array([], dtype=np.float32)
        try:
            result = self._calc(mol)
            vals = []
            for v in result:
                try:
                    f = float(v)
                    vals.append(f if np.isfinite(f) else np.nan)
                except Exception:
                    vals.append(np.nan)
            return np.array(vals, dtype=np.float32)
        except Exception:
            return np.full(self._n_raw, np.nan, dtype=np.float32)

    def compute(self, mols: List[Any], ids: np.ndarray, label: str) -> np.ndarray:
        if not self.available or not mols:
            n_cols = self._n_raw if self.available else 0
            return np.zeros((len(mols), n_cols), dtype=np.float32)

        out = np.full((len(mols), self._n_raw), np.nan, dtype=np.float32)
        missing: List[Tuple[int, Any]] = []
        cache_hits = 0

        for i, (mol, mol_id) in enumerate(zip(mols, ids)):
            cache_path = self._cache_path(mol_id)
            if cache_path is not None and cache_path.exists():
                try:
                    out[i] = np.load(cache_path)
                    cache_hits += 1
                    continue
                except Exception:
                    pass
            missing.append((i, mol, mol_id))

        if missing:
            log.info(f"Mordred {label}: cache hits={cache_hits}/{len(mols)}, computing {len(missing)} missing")
            for i, mol, mol_id in tqdm(missing, desc=f"Mordred {label}", leave=False):
                vec = self._mol_to_vec(mol)
                out[i] = vec
                cache_path = self._cache_path(mol_id)
                if cache_path is not None:
                    try:
                        np.save(cache_path, vec)
                    except Exception:
                        pass
        else:
            log.info(f"Mordred {label}: all {len(mols)} from cache")

        log.info(f"Mordred {label}: {out.shape}  NaN rate={np.isnan(out).mean():.3f}")
        return out

    def _cache_path(self, mol_id: str) -> Optional[Path]:
        if self.cache_dir is None or not mol_id or mol_id.startswith("idx_"):
            return None
        safe_id = "".join(c for c in str(mol_id) if c.isalnum() or c in "-_")
        return self.cache_dir / f"{safe_id}{self._cache_suffix}.npy"

    def fit_filter(self, M_train: np.ndarray) -> None:
        if M_train.shape[1] == 0:
            self._keep_mask = np.array([], dtype=bool)
            return
        nan_rate = np.isnan(M_train).mean(axis=0)
        self._keep_mask = nan_rate <= self.nan_threshold
        self._col_names = [
            f"mordred::{n}" for n, keep in zip(self._raw_names, self._keep_mask) if keep
        ]
        log.info(f"Mordred NaN filter: dropped {(~self._keep_mask).sum()}, kept {len(self._col_names)}")

    def apply_filter(self, M: np.ndarray) -> np.ndarray:
        if self._keep_mask is None or M.shape[1] == 0:
            return np.zeros((M.shape[0], 0), dtype=np.float32)
        filtered = M[:, self._keep_mask]
        np.nan_to_num(filtered, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return filtered.astype(np.float32)

    @property
    def feature_names(self) -> List[str]:
        return self._col_names


# ==============================================================================
# S7  DOPtools fragments
# ==============================================================================
class DOPtoolsBlock:
    CONFIGS = {"circus": [(0, 2), (0, 3)], "chyline": [(2, 4), (2, 6)]}

    def __init__(self, fit_samples: int = 5000, seed: int = 42):
        self.fit_samples = fit_samples
        self.seed = seed
        self._fitters: List[Tuple[str, Any]] = []
        self._col_sets: Dict[str, List[str]] = {}
        self._all_cols: List[str] = []
        self._fitted = False

    @property
    def available(self) -> bool:
        return HAS_DOPTOOLS and HAS_CHYTHON

    def _to_chython(self, smiles_list: List[str]):
        mols, idx = [], []
        for i, smi in enumerate(smiles_list):
            try:
                mol = chython_smiles(smi.strip())
                if mol is not None:
                    mols.append(mol); idx.append(i)
            except Exception:
                pass
        return mols, idx

    def fit(self, train_smiles: List[str]) -> None:
        if not self.available:
            return
        rng = np.random.default_rng(self.seed)
        if len(train_smiles) > self.fit_samples:
            sample = list(rng.choice(train_smiles, size=self.fit_samples, replace=False))
        else:
            sample = train_smiles
        mols, _ = self._to_chython(sample)
        if not mols:
            log.warning("DOPtools: no valid chython molecules")
            return

        for kind, configs in self.CONFIGS.items():
            Cls = CircuS if kind == "circus" else ChyLine
            prefix = "circus_r" if kind == "circus" else "chyline_l"
            for lo, hi in configs:
                label = f"{prefix}{lo}_{hi}"
                try:
                    fitter = Cls(lower=lo, upper=hi)
                    fitter.fit(mols)
                    probe = fitter.transform(mols[: min(5, len(mols))])
                    self._fitters.append((label, fitter))
                    self._col_sets[label] = list(probe.columns)
                    log.info(f"DOPtools {label}: {len(probe.columns)} fragments")
                except Exception as e:
                    log.warning(f"DOPtools {label} fit failed: {e}")

        self._all_cols = []
        for lbl, _ in self._fitters:
            for c in self._col_sets.get(lbl, []):
                self._all_cols.append(f"{lbl}::{c}")
        self._fitted = bool(self._fitters)

    def transform(self, smiles_list: List[str], label: str) -> np.ndarray:
        n = len(smiles_list)
        if not self._fitted:
            return np.zeros((n, 0), dtype=np.float32)
        X_out = np.zeros((n, len(self._all_cols)), dtype=np.float32)
        mols, valid_idx = self._to_chython(smiles_list)
        if not mols:
            return X_out
        col_cursor = 0
        for lbl, fitter in self._fitters:
            cols_lbl = self._col_sets.get(lbl, [])
            n_cols = len(cols_lbl)
            try:
                df = fitter.transform(mols).reindex(columns=cols_lbl, fill_value=0.0)
                arr = np.nan_to_num(df.values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                for row_i, global_i in enumerate(valid_idx):
                    X_out[global_i, col_cursor: col_cursor + n_cols] = arr[row_i]
            except Exception as e:
                log.warning(f"DOPtools {lbl} transform failed: {e}")
            col_cursor += n_cols
        density = (X_out != 0).mean()
        log.info(f"DOPtools {label}: {n}x{len(self._all_cols)} density={density:.4f}")
        return X_out

    @property
    def feature_names(self) -> List[str]:
        return [f"frag::{c}" for c in self._all_cols]


# ==============================================================================
# S8  Filters
# ==============================================================================
def remove_zero_variance(X: np.ndarray, threshold: float = 1e-6) -> np.ndarray:
    variances = np.var(X, axis=0)
    keep = variances > threshold
    log.info(f"  zero-variance: removed {(~keep).sum()}, kept {keep.sum()}")
    return keep


def blockwise_corr_filter(X: np.ndarray, block_types: List[str], threshold: float = 0.95) -> np.ndarray:
    keep_mask = np.ones(X.shape[1], dtype=bool)
    for block in ("fingerprint", "continuous", "fragment"):
        idx = np.array([i for i, t in enumerate(block_types) if t == block])
        if len(idx) < 2:
            continue
        X_b = X[:, idx]
        C = np.abs(np.corrcoef(X_b, rowvar=False))
        C = np.nan_to_num(C, nan=0.0)
        np.fill_diagonal(C, 0.0)
        remove: Set[int] = set()
        for i in range(len(idx)):
            if i in remove:
                continue
            for j in range(i + 1, len(idx)):
                if j in remove:
                    continue
                if C[i, j] >= threshold:
                    remove.add(j)
        removed_global = [int(idx[i]) for i in remove]
        for gi in removed_global:
            keep_mask[gi] = False
        log.info(f"  corr[{block}]: removed {len(removed_global)}")
    return keep_mask


def fragment_min_support_filter(X: np.ndarray, block_types: List[str], min_support: float = 0.01) -> np.ndarray:
    keep = np.ones(X.shape[1], dtype=bool)
    frag_idx = np.array([i for i, t in enumerate(block_types) if t == "fragment"])
    if len(frag_idx) == 0 or min_support <= 0.0:
        return keep
    n_rows = X.shape[0]
    min_count = max(1, int(np.ceil(min_support * n_rows)))
    support = (X[:, frag_idx] != 0).sum(axis=0)
    drop_local = support < min_count
    for local_i, drop in enumerate(drop_local):
        if drop:
            keep[frag_idx[local_i]] = False
    log.info(f"  fragment min-support ({min_support:.1%}, min_count={min_count}): removed {int(drop_local.sum())}/{len(frag_idx)}")
    return keep


def reduce_fingerprint_mi(X: np.ndarray, y: np.ndarray, block_types: List[str], k: int = 2048,
                           random_state: int = 42) -> np.ndarray:
    """MI-based fingerprint reduction. Note: this runs ONCE per variant
    (tree-space), on the full training set. random_state is the global
    pipeline seed because we want the tree-space matrix to be a
    deterministic function of (data, config). Per-fold MI inside VIF
    uses a per-fold rng instead — that's where Bug 5 was."""
    fp_idx = np.array([i for i, t in enumerate(block_types) if t == "fingerprint"])
    keep = np.ones(X.shape[1], dtype=bool)
    if len(fp_idx) <= k:
        return keep
    y_arr = np.asarray(y, dtype=float)
    valid = np.isfinite(y_arr)
    if not valid.all():
        n_bad = int((~valid).sum())
        log.info(f"  FP MI: dropping {n_bad} rows with NaN/Inf target for MI computation")
    X_valid = X[valid][:, fp_idx]
    y_valid = y_arr[valid]
    if len(y_valid) < 10:
        log.warning(f"  FP MI: only {len(y_valid)} valid-target rows; skipping reduction")
        return keep
    mi = mutual_info_regression(X_valid, y_valid, random_state=random_state)
    mi = np.nan_to_num(mi)
    top_local = np.argsort(mi)[-k:]
    top_global = set(int(fp_idx[i]) for i in top_local)
    for i, gi in enumerate(fp_idx):
        if int(gi) not in top_global:
            keep[gi] = False
    log.info(f"  FP MI-reduction: {len(fp_idx)} -> {k}")
    return keep


class ScopedVIFFilter(BaseEstimator, TransformerMixin):
    def __init__(self, block_types: List[str], y: np.ndarray,
                 vif_threshold: float = 10.0, max_candidates: int = 400,
                 apply_to_fingerprints: bool = False,
                 mi_random_state: int = 42):
        # FIX-C: mi_random_state threaded in from caller, varies per fold.
        self.block_types = block_types
        self.y = y
        self.vif_threshold = vif_threshold
        self.max_candidates = max_candidates
        self.apply_to_fingerprints = apply_to_fingerprints
        self.mi_random_state = mi_random_state
        self.support_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y=None):
        n = X.shape[1]
        self.support_ = np.ones(n, dtype=bool)
        if self.apply_to_fingerprints:
            cands = np.arange(n)
        else:
            cands = np.array([i for i, t in enumerate(self.block_types) if t == "continuous"])
        if len(cands) == 0:
            return self

        if len(cands) > self.max_candidates:
            # FIX-C: per-fold random state, not hardcoded 42
            mi = mutual_info_regression(X[:, cands], self.y, random_state=self.mi_random_state)
            mi = np.nan_to_num(mi)
            top = np.argsort(mi)[-self.max_candidates:]
            cands = cands[top]
            log.info(f"  VIF candidates pre-reduced by MI: -> {len(cands)}")

        remaining = list(map(int, cands))
        removed: Set[int] = set()
        while len(remaining) >= 2:
            X_sub = X[:, remaining]
            X_std = StandardScaler().fit_transform(X_sub)
            R = np.corrcoef(X_std.T)
            R = np.nan_to_num(R)
            np.fill_diagonal(R, 1.0)
            try:
                R_ridge = R + np.eye(R.shape[0]) * 1e-6
                vif = np.diag(np.linalg.inv(R_ridge))
            except np.linalg.LinAlgError:
                vif = np.diag(np.linalg.pinv(R))
            if vif.max() <= self.vif_threshold:
                break
            worst = int(np.argmax(vif))
            removed.add(remaining.pop(worst))
        for gi in removed:
            self.support_[gi] = False
        log.info(f"  ScopedVIF: removed {len(removed)}")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.support_]


# ==============================================================================
# S9  Stability Ridge selector and nested CV
# ==============================================================================
class StabilityRidgeSelector(BaseEstimator, TransformerMixin):
    def __init__(self, n_bootstrap: int = 50, subsample_frac: float = 0.8, random_state: int = 42):
        self.n_bootstrap = n_bootstrap
        self.subsample_frac = subsample_frac
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray):
        rng = np.random.RandomState(self.random_state)
        n, p = X.shape
        n_sub = int(n * self.subsample_frac)
        coefs = np.zeros((self.n_bootstrap, p))
        for i in range(self.n_bootstrap):
            idx = rng.choice(n, n_sub, replace=False)
            m = RidgeCV(alphas=np.logspace(-2, 2, 10), cv=3)
            m.fit(X[idx], y[idx])
            coefs[i] = m.coef_
        mean_abs = np.abs(coefs).mean(axis=0)
        std = coefs.std(axis=0)
        self.stability_ = mean_abs / (std + 1e-8)
        self.ranking_ = np.argsort(self.stability_)[::-1]
        return self

    def get_support(self, n_features: int) -> np.ndarray:
        sup = np.zeros(len(self.ranking_), dtype=bool)
        sup[self.ranking_[:n_features]] = True
        return sup


def nested_cv_select(X: np.ndarray, y: np.ndarray, block_types: List[str], cfg: PipelineConfig) -> Dict[str, Any]:
    y = np.asarray(y, dtype=float)
    finite_mask = np.isfinite(y)
    n_bad = int((~finite_mask).sum())
    if n_bad > 0:
        log.info(f"  nested_cv: dropping {n_bad} rows with NaN/Inf target")
        X = X[finite_mask]
        y = y[finite_mask]

    rng = np.random.RandomState(cfg.seed)
    outer = KFold(cfg.n_outer_cv, shuffle=True, random_state=rng.randint(2**31))
    all_selected: List[Set[int]] = []
    outer_scores: List[float] = []

    for fold_i, (tr_idx, te_idx) in enumerate(outer.split(X, y)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # FIX-C: per-fold MI seed for VIF candidate selection
        vif = ScopedVIFFilter(
            block_types=block_types, y=y_tr,
            vif_threshold=cfg.vif_threshold,
            max_candidates=cfg.vif_max_candidates,
            apply_to_fingerprints=cfg.apply_vif_to_fingerprints,
            mi_random_state=rng.randint(2**31),
        ).fit(X_tr)
        X_tr_v = vif.transform(X_tr)
        X_te_v = vif.transform(X_te)
        vif_kept_idx = np.where(vif.support_)[0]

        inner = KFold(cfg.n_inner_cv, shuffle=True, random_state=rng.randint(2**31))
        inner_selections: List[Set[int]] = []
        for in_tr_idx, in_val_idx in inner.split(X_tr_v, y_tr):
            X_in, X_val = X_tr_v[in_tr_idx], X_tr_v[in_val_idx]
            y_in, y_val = y_tr[in_tr_idx], y_tr[in_val_idx]
            scaler = StandardScaler().fit(X_in)
            X_in_s = scaler.transform(X_in)
            sel = StabilityRidgeSelector(
                n_bootstrap=cfg.stability_n_bootstrap,
                random_state=rng.randint(2**31)
            ).fit(X_in_s, y_in)
            sup = sel.get_support(cfg.n_features_linear)
            sel_local = np.where(sup)[0]
            sel_global = set(int(vif_kept_idx[i]) for i in sel_local)
            inner_selections.append(sel_global)

        counts = Counter(f for s in inner_selections for f in s)
        consensus = [f for f, c in counts.items() if c >= cfg.global_stability_min_folds]
        consensus = sorted(consensus, key=lambda f: -counts[f])[:cfg.n_features_linear]
        if not consensus:
            consensus = list(inner_selections[0])[:cfg.n_features_linear]

        vif_idx_map = {int(g): i for i, g in enumerate(vif_kept_idx)}
        cons_local = [vif_idx_map[f] for f in consensus if f in vif_idx_map]
        if cons_local:
            scaler_out = StandardScaler().fit(X_tr_v)
            X_tr_s = scaler_out.transform(X_tr_v)
            X_te_s = scaler_out.transform(X_te_v)
            m = RidgeCV().fit(X_tr_s[:, cons_local], y_tr)
            score = m.score(X_te_s[:, cons_local], y_te)
        else:
            score = np.nan
        outer_scores.append(score)
        all_selected.append(set(consensus))
        log.info(f"    fold {fold_i+1}: {len(consensus)} features, R2={score:.3f}")

    freq = Counter(f for s in all_selected for f in s)
    stable = [f for f, c in freq.items() if c >= cfg.global_stability_min_folds]
    stable = sorted(stable, key=lambda f: -freq[f])[:cfg.n_features_linear]

    return {
        "stable_features": stable,
        "outer_scores": outer_scores,
        "mean_r2": float(np.nanmean(outer_scores)),
        "std_r2": float(np.nanstd(outer_scores)),
        "feature_frequency": dict(freq),
    }


# ==============================================================================
# S10  Variant processing
# ==============================================================================
def process_variant(
    variant: str,
    X_by_subset: Dict[str, np.ndarray],
    y_train: np.ndarray,
    feature_names: List[str],
    block_types: List[str],
    namespaces: List[str],
    cfg: PipelineConfig,
    out_root: Path,
    subset_data: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> Dict[str, Any]:
    t0 = time.time()
    log.info(f"\n========== Variant: {variant} ==========")

    families = VARIANT_FAMILIES[variant]
    col_mask = np.array([FAMILY_BY_NAMESPACE.get(ns, "unknown") in families for ns in namespaces])
    if col_mask.sum() == 0:
        log.warning(f"Variant '{variant}' has no columns -- skipping")
        return {"variant": variant, "skipped": True}

    X_tr_raw = X_by_subset["train"][:, col_mask]
    names = [n for n, k in zip(feature_names, col_mask) if k]
    btypes = [t for t, k in zip(block_types, col_mask) if k]
    n_start = X_tr_raw.shape[1]
    log.info(f"  starting features: {n_start} (families={sorted(families)})")

    # DIAG-A: track per-stage dropped counts
    stage_drops: Dict[str, int] = {}

    # ---- Step 1: zero-variance ----
    zv_keep = remove_zero_variance(X_tr_raw, cfg.variance_threshold)
    stage_drops["zero_variance"] = int((~zv_keep).sum())
    X_tr = X_tr_raw[:, zv_keep]
    names = [n for n, k in zip(names, zv_keep) if k]
    btypes = [t for t, k in zip(btypes, zv_keep) if k]

    # ---- Step 2: block-wise correlation ----
    cc_keep = blockwise_corr_filter(X_tr, btypes, cfg.corr_threshold)
    stage_drops["correlation"] = int((~cc_keep).sum())
    X_tr = X_tr[:, cc_keep]
    names = [n for n, k in zip(names, cc_keep) if k]
    btypes = [t for t, k in zip(btypes, cc_keep) if k]

    # ---- Step 3: fragment min-support ----
    if any(t == "fragment" for t in btypes) and cfg.fragment_min_support > 0.0:
        fs_keep = fragment_min_support_filter(X_tr, btypes, min_support=cfg.fragment_min_support)
        stage_drops["fragment_min_support"] = int((~fs_keep).sum())
        X_tr = X_tr[:, fs_keep]
        names = [n for n, k in zip(names, fs_keep) if k]
        btypes = [t for t, k in zip(btypes, fs_keep) if k]
    else:
        fs_keep = np.ones(X_tr.shape[1], dtype=bool)
        stage_drops["fragment_min_support"] = 0

    # ---- Step 4: FP MI-reduction ----
    if any(t == "fingerprint" for t in btypes):
        mi_keep = reduce_fingerprint_mi(X_tr, y_train, btypes, k=cfg.fp_mi_target_k,
                                         random_state=cfg.seed)
        stage_drops["fp_mi_reduction"] = int((~mi_keep).sum())
        X_tr = X_tr[:, mi_keep]
        names = [n for n, k in zip(names, mi_keep) if k]
        btypes = [t for t, k in zip(btypes, mi_keep) if k]
    else:
        mi_keep = np.ones(X_tr.shape[1], dtype=bool)
        stage_drops["fp_mi_reduction"] = 0

    log.info(f"  tree-space features: {X_tr.shape[1]}")

    X_tree: Dict[str, np.ndarray] = {}
    for sub in X_by_subset:
        X_sub = X_by_subset[sub][:, col_mask]
        X_sub = X_sub[:, zv_keep]
        X_sub = X_sub[:, cc_keep]
        X_sub = X_sub[:, fs_keep]
        X_sub = X_sub[:, mi_keep]
        X_tree[sub] = X_sub.astype(np.float32)

    variant_meta: Dict[str, Any] = {"variant": variant, "skipped": False}
    variant_meta["n_starting_features"] = n_start
    variant_meta["n_tree_features"] = X_tree["train"].shape[1]
    variant_meta["stage_drops"] = stage_drops
    variant_meta["tree_feature_names"] = list(names)

    for sub, X_sub in X_tree.items():
        y_s, w_s, ids_s = subset_data[sub]
        path = out_root / sub / f"{variant}_tree.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        save_npz(
            path, X_sub, y_s, w_s, ids_s, names,
            variant=variant, space="tree", subset=sub,
            pipeline_version=PIPELINE_VERSION,
        )
        log.info(f"  saved {sub}/{variant}_tree.npz  shape={X_sub.shape}")

    log.info(f"  linear-space nested CV ({cfg.n_outer_cv}x{cfg.n_inner_cv} folds)")
    nested = nested_cv_select(X_tree["train"], y_train, btypes, cfg)
    stable = nested["stable_features"]

    if not stable:
        log.warning(f"  variant {variant}: no stable features; skipping linear-space")
        variant_meta["linear_skipped"] = True
        variant_meta["nested_cv"] = {k: v for k, v in nested.items() if k != "feature_frequency"}
        variant_meta["elapsed_sec"] = round(time.time() - t0, 1)
        return variant_meta

    linear_mask = np.zeros(X_tree["train"].shape[1], dtype=bool)
    linear_mask[stable] = True
    linear_names = [n for n, k in zip(names, linear_mask) if k]

    for sub, X_sub in X_tree.items():
        X_lin = X_sub[:, linear_mask]
        y_s, w_s, ids_s = subset_data[sub]
        path = out_root / sub / f"{variant}_linear.npz"
        save_npz(
            path, X_lin, y_s, w_s, ids_s, linear_names,
            variant=variant, space="linear", subset=sub,
            pipeline_version=PIPELINE_VERSION,
        )
        log.info(f"  saved {sub}/{variant}_linear.npz  shape={X_lin.shape}")

    variant_meta["n_linear_features"] = int(linear_mask.sum())
    variant_meta["linear_feature_names"] = linear_names
    variant_meta["nested_cv"] = {
        "mean_r2": nested["mean_r2"],
        "std_r2": nested["std_r2"],
        "outer_scores": nested["outer_scores"],
    }
    variant_meta["elapsed_sec"] = round(time.time() - t0, 1)
    log.info(f"  variant {variant} done in {variant_meta['elapsed_sec']}s")
    return variant_meta


# ==============================================================================
# S11  Save utilities
# ==============================================================================
def save_npz(path: Path, X, y, weights, ids, feature_names, **meta):
    np.savez_compressed(
        path,
        X=X.astype(np.float32),
        y=y.astype(np.float32),
        weights=weights.astype(np.float32),
        ids=np.array(ids, dtype=object),
        feature_names=np.array(feature_names, dtype=object),
        meta=np.array([json.dumps(meta)], dtype=object),
    )


def save_stratifiers(path: Path, stratifiers: Dict[str, np.ndarray], ids: np.ndarray):
    arrs = {"ids": ids}
    arrs.update(stratifiers)
    np.savez_compressed(path, **arrs)


# ==============================================================================
# S12  Leakage audit + output artifacts
# ==============================================================================
def run_leakage_audit(
    parsed: Dict[str, ParsedSubset],
    out_root: Path,
    cfg: PipelineConfig,
) -> dict:
    report = {
        "pipeline_version": PIPELINE_VERSION,
        "leakage": "PASS",
        "checks": {},
    }

    if "train" in parsed and "test" in parsed:
        tr_ids = set(parsed["train"].ids)
        te_ids = set(parsed["test"].ids)
        va_ids = set(parsed["val"].ids) if "val" in parsed else set()
        overlap_tt = len(tr_ids & te_ids)
        overlap_tv = len(tr_ids & va_ids)
        report["checks"]["inchikey14_train_test_overlap"] = overlap_tt
        report["checks"]["inchikey14_train_val_overlap"] = overlap_tv
        if overlap_tt > 0:
            report["leakage"] = "FAIL"
            log.error(f"LEAKAGE: {overlap_tt} InChIKey-14 overlap train/test")
        else:
            log.info("Leakage audit: InChIKey-14 train/test OK")

    scaf_col = "scaffold_id"
    if scaf_col in parsed["train"].stratifiers and scaf_col in parsed["test"].stratifiers:
        tr_sc = set(parsed["train"].stratifiers[scaf_col])
        te_sc = set(parsed["test"].stratifiers[scaf_col])
        overlap_sc = len(tr_sc & te_sc)
        report["checks"]["scaffold_train_test_overlap"] = overlap_sc
        if overlap_sc > 0:
            log.warning(f"Leakage audit: {overlap_sc} scaffolds shared train/test (expected for random/sim/lead_opt)")
        else:
            log.info("Leakage audit: scaffold train/test disjoint")

    for sub, p in parsed.items():
        if "is_ki" in p.stratifiers:
            ki_pct = float(p.stratifiers["is_ki"].mean() * 100)
            report["checks"][f"{sub}_ki_pct"] = round(ki_pct, 2)

    ad_report = {}
    for sub, p in parsed.items():
        if "ad_score" in p.stratifiers:
            ad_report[sub] = {
                "mean_ad_score": float(p.stratifiers["ad_score"].mean()),
                "in_domain_pct": float(p.stratifiers["ad_in_domain"].mean() * 100),
            }
    with open(out_root / "ad_report.json", "w") as f:
        json.dump(ad_report, f, indent=2)

    ac_report = {}
    for sub, p in parsed.items():
        if "assay_disagreement_score" in p.stratifiers:
            ac_report[sub] = {
                "mean_disagreement": float(p.stratifiers["assay_disagreement_score"].mean()),
                "mean_ki_ic50_delta": float(p.stratifiers["ki_ic50_delta"].mean()),
            }
    with open(out_root / "assay_consistency_report.json", "w") as f:
        json.dump(ac_report, f, indent=2)

    with open(out_root / "leakage_report.json", "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Leakage report: {out_root / 'leakage_report.json'}")
    return report


def write_feature_schema(
    out_root: Path,
    feature_names: List[str],
    block_types: List[str],
    namespaces: List[str],
):
    schema = {
        "pipeline_version": PIPELINE_VERSION,
        "n_features": len(feature_names),
        "features": [
            {"index": i, "name": n, "block_type": b, "namespace": ns}
            for i, (n, b, ns) in enumerate(zip(feature_names, block_types, namespaces))
        ],
    }
    with open(out_root / "feature_schema.json", "w") as f:
        json.dump(schema, f, indent=2)


def write_feature_stats(
    out_root: Path,
    X_train: np.ndarray,
    feature_names: List[str],
):
    stats = []
    for i, name in enumerate(feature_names):
        col = X_train[:, i]
        finite = col[np.isfinite(col)]
        stats.append({
            "index": i,
            "name": name,
            "mean": float(finite.mean()) if len(finite) else None,
            "std": float(finite.std()) if len(finite) else None,
            "missing_rate": float(np.isnan(col).mean()),
            "min": float(finite.min()) if len(finite) else None,
            "max": float(finite.max()) if len(finite) else None,
        })
    with open(out_root / "feature_stats.json", "w") as f:
        json.dump(stats, f, indent=2)


# ==============================================================================
# S13  Main
# ==============================================================================
def main():
    p = argparse.ArgumentParser(
        description=f"PAD4 Featurization v{PIPELINE_VERSION}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--split_dir", default="")
    p.add_argument("--output_dir", default="")
    p.add_argument("--cache_dir", default="")
    p.add_argument("--cliff_file", default="")
    p.add_argument("--smiles_col", default="canonical_smiles")
    p.add_argument("--id_col", default="inchikey_14")
    p.add_argument("--activity_col", default="pIC50")
    p.add_argument("--assay_type_col", default="assay_type")
    p.add_argument("--weight_col", default="ml_weight")
    p.add_argument("--test_filename", default="test_locked.csv")
    p.add_argument("--ad_score_col", default="t1_self_tanimoto")
    p.add_argument("--ad_in_domain_col", default="")
    p.add_argument("--ad_threshold", type=float, default=0.35)
    p.add_argument("--variants", nargs="+",
                   default=["full", "fingerprints", "physchem", "mordred", "fragments"],
                   choices=["full", "fingerprints", "physchem", "mordred", "fragments"])
    p.add_argument("--no_mordred", action="store_true")
    p.add_argument("--no_doptools", action="store_true")
    p.add_argument("--mordred_nan_pct", type=float, default=0.05)
    p.add_argument("--doptools_fit_samples", type=int, default=5000)
    p.add_argument("--corr_threshold", type=float, default=0.95)
    p.add_argument("--fp_mi_target_k", type=int, default=2048)
    p.add_argument("--fragment_min_support", type=float, default=0.01)
    p.add_argument("--vif_threshold", type=float, default=10.0)
    p.add_argument("--vif_max_candidates", type=int, default=400)
    p.add_argument("--vif_on_fingerprints", action="store_true")
    p.add_argument("--n_features_linear", type=int, default=120)
    p.add_argument("--n_outer_cv", type=int, default=5)
    p.add_argument("--n_inner_cv", type=int, default=3)
    p.add_argument("--stability_n_bootstrap", type=int, default=50)
    p.add_argument("--global_stability_min_folds", type=int, default=3)
    p.add_argument("--uncertainty_k", type=int, default=5)
    p.add_argument("--no_uncertainty", action="store_true")
    p.add_argument("--no_assay_consistency", action="store_true")
    p.add_argument("--stratifiers_only", action="store_true")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)

    args = p.parse_args()

    cfg = PipelineConfig(
        split_dir=args.split_dir,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        cliff_file=args.cliff_file,
        smiles_col=args.smiles_col,
        id_col=args.id_col,
        activity_col=args.activity_col,
        assay_type_col=args.assay_type_col,
        weight_col=args.weight_col,
        test_filename=args.test_filename,
        ad_score_col=args.ad_score_col,
        ad_in_domain_col=args.ad_in_domain_col,
        ad_threshold=args.ad_threshold,
        variants=args.variants,
        mordred_nan_pct=args.mordred_nan_pct,
        doptools_fit_samples=args.doptools_fit_samples,
        use_mordred=not args.no_mordred,
        use_doptools=not args.no_doptools,
        corr_threshold=args.corr_threshold,
        fp_mi_target_k=args.fp_mi_target_k,
        fragment_min_support=args.fragment_min_support,
        vif_threshold=args.vif_threshold,
        vif_max_candidates=args.vif_max_candidates,
        apply_vif_to_fingerprints=args.vif_on_fingerprints,
        n_features_linear=args.n_features_linear,
        n_outer_cv=args.n_outer_cv,
        n_inner_cv=args.n_inner_cv,
        stability_n_bootstrap=args.stability_n_bootstrap,
        global_stability_min_folds=args.global_stability_min_folds,
        uncertainty_k=args.uncertainty_k,
        compute_uncertainty=not args.no_uncertainty,
        compute_assay_consistency=not args.no_assay_consistency,
        stratifiers_only=args.stratifiers_only,
        seed=args.seed,
    )
    cfg.validate()
    set_global_seed(cfg.seed)

    split_dir = Path(cfg.split_dir)
    out_root = Path(cfg.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cfg.cache_dir) if cfg.cache_dir else (out_root / ".cache")

    git_hash = get_git_hash()
    log.info("=" * 68)
    log.info(f"PAD4 FEATURIZATION v{PIPELINE_VERSION}")
    log.info(f"Config hash: {cfg.config_hash}  Seed: {cfg.seed}  Git: {git_hash}")
    log.info(f"Variants: {cfg.variants}")
    log.info(f"Cache:    {cache_dir}")

    cliff_ids: Set[str] = set()
    if cfg.cliff_file:
        try:
            cliff_df = pd.read_csv(cfg.cliff_file)
            if cfg.id_col in cliff_df.columns:
                cliff_ids = set(cliff_df[cfg.id_col].astype(str))
            elif "inchikey_14" in cliff_df.columns:
                cliff_ids = set(cliff_df["inchikey_14"].astype(str))
            log.info(f"Loaded cliff IDs: {len(cliff_ids)}")
        except Exception as e:
            log.warning(f"Could not read cliff file: {e}")

    subsets: Dict[str, pd.DataFrame] = {}
    for sub in ("train", "val", "test"):
        path = split_dir / (cfg.test_filename if sub == "test" else f"{sub}.csv")
        if sub == "test" and not path.exists():
            path = split_dir / "test.csv"
        if path.exists():
            df_sub = pd.read_csv(path)
            if sub == "val" and len(df_sub) == 0:
                log.warning(f"  val.csv is empty; skipping validation subset")
                continue
            subsets[sub] = df_sub
            log.info(f"Loaded {sub}: {len(subsets[sub])} rows from {path.name}")

    if "train" not in subsets:
        raise PipelineError(f"train.csv not found in {split_dir}")

    # ── FIX-A + FIX-B: per-subset application of train-fitted state ───
    # No concat. No reslice. Each subset is processed independently
    # using state fit exclusively on training data.

    if cfg.compute_assay_consistency:
        log.info("Fitting assay-consistency aggregates on TRAIN only...")
        assay_aggs = fit_assay_consistency(subsets["train"], cfg)
        for sub in subsets:
            subsets[sub] = apply_assay_consistency(subsets[sub], assay_aggs, cfg)

    if cfg.compute_uncertainty:
        log.info("Fitting uncertainty state on TRAIN only...")
        unc_state = fit_uncertainty(subsets["train"], cfg)
        for sub in subsets:
            subsets[sub] = apply_uncertainty(subsets[sub], unc_state, cfg)

    # Sanity: train/test ID overlap (should already be guaranteed by splits)
    if "test" in subsets:
        train_ids = set(subsets["train"][cfg.id_col])
        test_ids = set(subsets["test"][cfg.id_col])
        overlap = train_ids & test_ids
        if overlap:
            raise PipelineError(f"Data leakage: {len(overlap)} IDs overlap between train and test")
        log.info("Train/test ID overlap check: PASSED")

    parsed: Dict[str, ParsedSubset] = {}
    for sub, df in subsets.items():
        parsed[sub] = parse_subset(df, cfg, cliff_ids, sub)
        log.info(f"  {sub}: {parsed[sub].n_valid}/{parsed[sub].n_total} valid molecules")

    if cfg.stratifiers_only:
        log.info("\n--stratifiers_only: rewriting stratifiers.npz only")
        for sub in subsets:
            sub_dir = out_root / sub
            sub_dir.mkdir(parents=True, exist_ok=True)
            save_stratifiers(
                sub_dir / "stratifiers.npz",
                parsed[sub].stratifiers,
                parsed[sub].ids,
            )
            log.info(f"  wrote {sub}/stratifiers.npz")
        run_leakage_audit(parsed, out_root, cfg)
        log.info("=" * 68)
        log.info(f"PIPELINE v{PIPELINE_VERSION} COMPLETE (stratifiers only)")
        log.info(f"Output: {out_root}")
        log.info("=" * 68)
        return

    log.info("\nComputing descriptor blocks (one-time per split)")
    t_desc = time.time()

    rdkit_X: Dict[str, np.ndarray] = {}
    rdkit_names: List[str] = []
    rdkit_btypes: List[str] = []
    rdkit_ns: List[str] = []
    for sub in subsets:
        t0 = time.time()
        X_r, names_r, btypes_r, ns_r = compute_rdkit_matrix(parsed[sub].mols, parsed[sub].rows)
        rdkit_X[sub] = X_r
        if not rdkit_names:
            rdkit_names, rdkit_btypes, rdkit_ns = names_r, btypes_r, ns_r
        log.info(f"  RDKit {sub}: {X_r.shape}  ({time.time()-t0:.1f}s)")

    mordred_X: Dict[str, np.ndarray] = {}
    mordred_names: List[str] = []
    if cfg.use_mordred and HAS_MORDRED:
        mrd = MordredBlock(cfg.mordred_nan_pct, cache_dir / "mordred")
        if mrd.available:
            raw: Dict[str, np.ndarray] = {}
            for sub in subsets:
                raw[sub] = mrd.compute(parsed[sub].mols, parsed[sub].ids, sub)
            mrd.fit_filter(raw["train"])
            for sub in subsets:
                mordred_X[sub] = mrd.apply_filter(raw[sub])
            mordred_names = mrd.feature_names

    dop_X: Dict[str, np.ndarray] = {}
    dop_names: List[str] = []
    if cfg.use_doptools and HAS_DOPTOOLS and HAS_CHYTHON:
        dop = DOPtoolsBlock(cfg.doptools_fit_samples, seed=cfg.seed)
        dop.fit(parsed["train"].smiles)
        if dop._fitted:
            for sub in subsets:
                dop_X[sub] = dop.transform(parsed[sub].smiles, sub)
            dop_names = dop.feature_names

    log.info(f"Descriptor computation total: {time.time()-t_desc:.1f}s")

    X_by_subset: Dict[str, np.ndarray] = {}
    feature_names = list(rdkit_names)
    block_types = list(rdkit_btypes)
    namespaces = list(rdkit_ns)

    if mordred_names:
        feature_names += mordred_names
        block_types += ["continuous"] * len(mordred_names)
        namespaces += ["mordred"] * len(mordred_names)
    if dop_names:
        feature_names += dop_names
        block_types += ["fragment"] * len(dop_names)
        namespaces += ["frag"] * len(dop_names)

    for sub in subsets:
        parts = [rdkit_X[sub]]
        if sub in mordred_X and mordred_X[sub].shape[1] > 0:
            parts.append(mordred_X[sub])
        if sub in dop_X and dop_X[sub].shape[1] > 0:
            parts.append(dop_X[sub])
        X_by_subset[sub] = np.concatenate(parts, axis=1).astype(np.float32)
        log.info(f"  concatenated {sub}: {X_by_subset[sub].shape}")

    for sub in X_by_subset:
        if X_by_subset[sub].shape[1] != len(feature_names):
            raise FeatureSpaceMismatch(
                f"{sub} has {X_by_subset[sub].shape[1]} cols but {len(feature_names)} names"
            )

    for sub in subsets:
        strat_path = out_root / sub
        strat_path.mkdir(parents=True, exist_ok=True)
        save_stratifiers(
            strat_path / "stratifiers.npz",
            parsed[sub].stratifiers,
            parsed[sub].ids,
        )
        log.info(f"  saved {sub}/stratifiers.npz")

    write_feature_schema(out_root, feature_names, block_types, namespaces)
    write_feature_stats(out_root, X_by_subset["train"], feature_names)

    subset_data = {
        sub: (parsed[sub].y, parsed[sub].weights, parsed[sub].ids)
        for sub in subsets
    }
    y_train = parsed["train"].y

    manifest: Dict[str, Any] = {
        "pipeline_version": PIPELINE_VERSION,
        "generated": pd.Timestamp.now().isoformat(),
        "git_hash": git_hash,
        "config_hash": cfg.config_hash,
        "config": asdict(cfg),
        "n_train": parsed["train"].n_valid,
        "n_val": parsed["val"].n_valid if "val" in parsed else 0,
        "n_test": parsed["test"].n_valid if "test" in parsed else 0,
        "variants": {},
    }

    for variant in cfg.variants:
        vmeta = process_variant(
            variant=variant,
            X_by_subset=X_by_subset,
            y_train=y_train,
            feature_names=feature_names,
            block_types=block_types,
            namespaces=namespaces,
            cfg=cfg,
            out_root=out_root,
            subset_data=subset_data,
        )
        manifest["variants"][variant] = vmeta

    run_leakage_audit(parsed, out_root, cfg)

    manifest_path = out_root / "feature_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    log.info(f"\nManifest: {manifest_path}")
    log.info("=" * 68)
    log.info(f"PIPELINE v{PIPELINE_VERSION} COMPLETE")
    log.info(f"Output: {out_root}")
    log.info("=" * 68)


if __name__ == "__main__":
    main()