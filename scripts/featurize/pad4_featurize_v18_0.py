#!/usr/bin/env python3
"""
PAD4 Featurization Pipeline v18.0 - Task-Aware Regression + Classification
===========================================================================

Major changes from v17.0 (=v13.0 internally):

  [NEW-1] AUTO-DETECTED TASK. The script now resolves whether the data
          is a regression or classification benchmark from the split
          directory path. `data/splits/regression/<strategy>/` → regression;
          `data/splits/classification/<strategy>/` → classification. An
          explicit `--task` flag overrides the auto-detection. This avoids
          the previous failure mode where classification splits produced
          NaN targets because the script hard-coded `--activity_col pIC50`.

  [NEW-2] TASK-SPECIFIC TARGET COLUMN. Regression targets `pIC50` by
          default; classification targets `activity_class` (binary 0/1).
          `--activity_col` still overrides both.

  [NEW-3] HARD ERROR ON NaN TARGETS. After parsing, the script aborts if
          any target value is NaN, listing the first offending IDs. No
          silent drop. Forces upstream data hygiene.

  [NEW-4] CLASSIFICATION-CORRECT FEATURE SELECTION. For classification
          tasks, the following estimators are swapped automatically:
            - mutual_info_regression -> mutual_info_classif
            - RidgeCV                -> LogisticRegressionCV
            - StabilityRidgeSelector -> StabilityLogisticSelector
            - r2_score               -> roc_auc_score
          The stability metric (mean_abs_coef / std_coef) is preserved;
          only the underlying estimator changes.

  [NEW-5] activity_class SAFETY CHECK. On classification tasks the parser
          verifies activity_class is binary (values ⊆ {0, 1}), warns if
          the minority class is below 5%, and aborts if there is only
          one class.

  [NEW-6] TASK IN CONFIG HASH. PipelineConfig.task is part of config_hash
          so classification and regression runs do not collide in caches
          or manifests.

  [NEW-7] METADATA AWARENESS. Tree/linear NPZ metadata records "task"
          and "target_column". The manifest records the resolved task.

  [NEW-8] STRATIFIER COMPATIBILITY. The pIC50_active derived stratifier
          is skipped for classification (target already encodes activity).

Preserved from v17 (unchanged unless noted)
-------------------------------------------
  - All descriptor blocks (RDKit, Mordred with per-molecule cache, DOPtools)
  - All filters (zero-variance, MI FP reduction, fragment min-support,
    blockwise correlation, scoped VIF)
  - Variant system (full / fingerprints / physchem / mordred / fragments)
  - Tree-space and linear-space NPZ outputs
  - Stratifiers.npz (per subset, once per split)
  - Manifest, runner generator, cache, all CLI flags

Usage
-----
  # Regression (task inferred from /regression/ in path)
  python pad4_featurize_v18.py \\
      --split_dir data/splits/regression/scaffold \\
      --output_dir features_v18/regression/scaffold \\
      --variants full fingerprints physchem mordred fragments \\
      --seed 42

  # Classification (task inferred from /classification/ in path)
  python pad4_featurize_v18.py \\
      --split_dir data/splits/classification/scaffold \\
      --output_dir features_v18/classification/scaffold \\
      --variants full fingerprints physchem mordred fragments \\
      --seed 42

  # Explicit override
  python pad4_featurize_v18.py --task classification ...

  # Generate runner script for all (split-base x task) combinations
  python pad4_featurize_v18.py --generate_runner \\
      --split_base data/splits --output_base features_v18
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import pickle
import re
import subprocess
import sys
import time
import warnings
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# S0  Logging, reproducibility, exceptions
# ==============================================================================
DEFAULT_SEED = 42
PIPELINE_VERSION = "18.0"

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
    """Base for all pipeline-specific errors."""


class FeatureSpaceMismatch(PipelineError):
    pass


class RowCountMismatch(PipelineError):
    pass


class NoValidMolecules(PipelineError):
    pass


class TaskResolutionError(PipelineError):
    """Raised when the task (regression/classification) cannot be determined."""


class TargetColumnError(PipelineError):
    """Raised when the target column is missing or contains NaN."""


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
    log.error("RDKit is required; install with: conda install -c conda-forge rdkit")
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
# S1.5  [NEW-1, NEW-2] Task resolution
# ==============================================================================
# Defaults that depend on the resolved task.
TASK_DEFAULT_TARGET = {
    "regression":     "pIC50",
    "classification": "activity_class",
}

VALID_TASKS = ("regression", "classification")


def resolve_task(
    split_dir: str,
    task_flag: str,
) -> str:
    """
    Resolve task from --task flag or by parsing --split_dir.

    Priority:
      1. Explicit --task value (regression|classification) wins.
      2. --task auto inspects split_dir for /regression/ or /classification/
         path segments.
      3. Failure raises TaskResolutionError with actionable guidance.
    """
    if task_flag in VALID_TASKS:
        log.info(f"  task resolution: explicit --task {task_flag}")
        return task_flag

    if task_flag != "auto":
        raise TaskResolutionError(
            f"Invalid --task value '{task_flag}'. "
            f"Use 'auto', 'regression', or 'classification'."
        )

    # Normalize separators and split into segments
    norm = str(Path(split_dir).resolve()).replace("\\", "/")
    parts = norm.split("/")
    has_reg = "regression" in parts
    has_cls = "classification" in parts

    if has_reg and has_cls:
        raise TaskResolutionError(
            f"Path '{split_dir}' contains BOTH 'regression' and 'classification' "
            f"segments. Cannot auto-detect task. Pass --task explicitly."
        )
    if has_reg:
        log.info(f"  task resolution: auto-detected 'regression' from path")
        return "regression"
    if has_cls:
        log.info(f"  task resolution: auto-detected 'classification' from path")
        return "classification"

    raise TaskResolutionError(
        f"Cannot infer task from path '{split_dir}'. "
        f"Path must contain a '/regression/' or '/classification/' segment, "
        f"or pass --task explicitly."
    )


# ==============================================================================
# S2  Pipeline configuration
# ==============================================================================
@dataclass
class PipelineConfig:
    """All tunable pipeline parameters."""
    # I/O
    split_dir: str = ""
    output_dir: str = ""
    cache_dir: str = ""
    cliff_file: str = ""
    smiles_col: str = "canonical_smiles"
    id_col: str = "inchikey_14"

    # [NEW-2] Target column is resolved per task. activity_col=="" means
    # "use the task default" (pIC50 for regression, activity_class for
    # classification). A non-empty value overrides this.
    activity_col: str = ""
    weight_col: str = "ml_weight"
    test_filename: str = "test_locked.csv"

    # [NEW-1] Task: "auto", "regression", or "classification"
    task: str = "auto"

    # AD columns
    ad_score_col: str = "t1_self_tanimoto"
    ad_in_domain_col: str = ""
    ad_score_col_fallback: str = "t1_novelty_score"
    ad_in_domain_col_fallback: str = ""
    ad_threshold: float = 0.35

    # Variants
    variants: List[str] = field(default_factory=lambda: [
        "full", "fingerprints", "physchem", "mordred", "fragments"
    ])

    # Descriptor controls
    mordred_nan_pct: float = 0.05
    doptools_fit_samples: int = 5000
    use_mordred: bool = True
    use_doptools: bool = True

    # Filters
    variance_threshold: float = 1e-6
    corr_threshold: float = 0.95
    fp_mi_target_k: int = 2048
    fragment_min_support: float = 0.01

    # VIF
    vif_threshold: float = 10.0
    vif_max_candidates: int = 400
    apply_vif_to_fingerprints: bool = False

    # Nested CV
    n_features_linear: int = 120
    n_outer_cv: int = 5
    n_inner_cv: int = 3
    stability_n_bootstrap: int = 50
    global_stability_min_folds: int = 3

    # Reproducibility
    seed: int = DEFAULT_SEED

    # Operating mode
    stratifiers_only: bool = False

    def validate(self) -> None:
        if not self.split_dir:
            raise PipelineError("--split_dir is required")
        if not self.output_dir:
            raise PipelineError("--output_dir is required")
        if self.task not in ("auto",) + VALID_TASKS:
            raise PipelineError(
                f"--task must be 'auto', 'regression', or 'classification'; got '{self.task}'"
            )
        if not (1 <= self.global_stability_min_folds <= self.n_outer_cv):
            raise PipelineError(
                "global_stability_min_folds must be in [1, n_outer_cv]"
            )
        valid_variants = {"full", "fingerprints", "physchem", "mordred", "fragments"}
        for v in self.variants:
            if v not in valid_variants:
                raise PipelineError(f"Unknown variant: {v}")

    @property
    def config_hash(self) -> str:
        """[NEW-6] task is included so cls/reg runs hash distinctly."""
        return hashlib.md5(
            json.dumps(asdict(self), sort_keys=True, default=str).encode()
        ).hexdigest()[:12]


# Feature family map. Unchanged from v17.
FAMILY_BY_NAMESPACE = {
    "ecfp4": "fingerprint", "ecfp6": "fingerprint",
    "maccs": "fingerprint", "ap": "fingerprint", "tt": "fingerprint",
    "physchem": "physchem", "vsa": "physchem", "smarts": "physchem",
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
# S3  Descriptor constants  (unchanged from v17)
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
# S4  Molecule parsing
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


@dataclass
class ParsedSubset:
    """One parsed subset (train/val/test) with carried metadata."""
    mols: List[Any]
    smiles: List[str]
    ids: np.ndarray
    y: np.ndarray
    weights: np.ndarray
    rows: List[dict]
    stratifiers: Dict[str, np.ndarray]
    n_valid: int
    n_total: int


def _first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c and c in df.columns:
            return c
    return None


def _safe_float(v, default: float = 0.0) -> float:
    try:
        if pd.isna(v):
            return float(default)
        f = float(v)
        return f if np.isfinite(f) else float(default)
    except Exception:
        return float(default)


# [NEW-5] activity_class safety check
def _validate_classification_target(y: np.ndarray, ids: np.ndarray, label: str) -> None:
    """
    Assert that a classification target is well-formed:
      - Values must be binary (subset of {0, 1})
      - At least two classes must be present
      - Warn if minority class < 5%
    """
    if y.size == 0:
        raise TargetColumnError(f"{label}: classification target is empty")

    # Check binary-ness (allow ints stored as floats, e.g., 0.0/1.0)
    unique_vals = np.unique(y[~np.isnan(y)])
    bad = [v for v in unique_vals if v not in (0.0, 1.0)]
    if bad:
        raise TargetColumnError(
            f"{label}: activity_class must be binary {{0, 1}}; "
            f"found non-binary values: {bad[:10]} ... "
            f"(unique value count = {len(unique_vals)})"
        )

    # Class balance
    pos = int((y == 1.0).sum())
    neg = int((y == 0.0).sum())
    n = pos + neg
    if pos == 0 or neg == 0:
        raise TargetColumnError(
            f"{label}: only one class present (pos={pos}, neg={neg}). "
            f"Cannot fit a classifier."
        )
    minority_frac = min(pos, neg) / max(1, n)
    log.info(
        f"  {label}: classification target balance -> "
        f"pos={pos} ({100*pos/n:.1f}%), neg={neg} ({100*neg/n:.1f}%)"
    )
    if minority_frac < 0.05:
        log.warning(
            f"  {label}: minority class is {100*minority_frac:.1f}% (<5%); "
            "consider class-weighted training or resampling downstream."
        )


def parse_subset(
    df: pd.DataFrame,
    cfg: PipelineConfig,
    target_col: str,
    cliff_ids: Set[str],
    label: str,
) -> ParsedSubset:
    """Parse SMILES + extract target/weights/ids and stratifier columns."""
    mols, smiles_list, row_dicts = [], [], []
    ys, ws, ids = [], [], []

    # Column resolution (unchanged from v17)
    col_ad_score = _first_present(df, [cfg.ad_score_col, cfg.ad_score_col_fallback,
                                       "ad_score", "t1_self_tanimoto"])
    col_ad_indom = _first_present(df, [cfg.ad_in_domain_col,
                                       cfg.ad_in_domain_col_fallback,
                                       "ad_in_domain"])
    col_label_unc = _first_present(df, ["label_uncertainty_score_v2",
                                        "label_uncertainty", "label_noise"])
    col_cliff = _first_present(df, ["is_activity_cliff", "is_cliff"])
    col_cliff_sev = _first_present(df, ["cliff_severity"])
    col_cliff_partners = _first_present(df, ["n_cliff_partners"])
    col_novelty = _first_present(df, ["t1_novelty_score"])
    col_fidelity = _first_present(df, ["fidelity_level"])
    col_confidence = _first_present(df, ["confidence_weight",
                                         "source_reliability_weight"])
    col_pic50_std = _first_present(df, ["pIC50_std"])

    found = {
        "target_col": target_col,
        "ad_score": col_ad_score, "ad_in_domain": col_ad_indom,
        "label_uncertainty": col_label_unc, "is_cliff": col_cliff,
        "cliff_severity": col_cliff_sev, "n_cliff_partners": col_cliff_partners,
        "t1_novelty": col_novelty, "fidelity": col_fidelity,
        "confidence_weight": col_confidence, "pIC50_std": col_pic50_std,
    }
    log.info(f"  {label}: resolved cols -> "
             + ", ".join(f"{k}={v}" for k, v in found.items() if v))
    missing = [k for k, v in found.items() if not v]
    if missing:
        log.warning(f"  {label}: not found in CSV: {missing} (will be zero)")

    # Verify the target column exists in the CSV
    if target_col not in df.columns:
        raise TargetColumnError(
            f"{label}: target column '{target_col}' is missing from CSV. "
            f"Available columns: {list(df.columns)[:30]}..."
        )

    strat_keys = [
        "ad_score", "ad_in_domain",
        "ml_weight", "label_uncertainty",
        "is_cliff", "cliff_severity", "n_cliff_partners",
        "is_covalent", "is_frequent_scaffold",
        "complexity_score", "t1_novelty_score",
        "fidelity_level", "confidence_weight", "pIC50_std",
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

        # [NEW-2/3] Use the resolved target column; KEEP NaN here — we
        # validate after parsing all rows so the error message can list
        # offending IDs.
        y_val = row.get(target_col)
        ys.append(float(y_val) if pd.notna(y_val) else np.nan)

        w_val = row.get(cfg.weight_col)
        ws.append(float(w_val) if pd.notna(w_val) else 1.0)
        rid = row.get(cfg.id_col) or f"idx_{len(ids)}"
        ids.append(str(rid))

        # AD
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

        scaffold_ids.append(str(row.get("scaffold_id", "")))

    if not mols:
        raise NoValidMolecules(f"No valid molecules in {label}")

    y_arr = np.array(ys, dtype=np.float32)
    ids_arr = np.array(ids, dtype=object)

    # [NEW-3] HARD ERROR on NaN targets. Report offending IDs.
    nan_mask = np.isnan(y_arr)
    if nan_mask.any():
        n_nan = int(nan_mask.sum())
        offenders = ids_arr[nan_mask][:5].tolist()
        raise TargetColumnError(
            f"{label}: target column '{target_col}' has {n_nan} NaN values "
            f"(out of {len(y_arr)}). First offending IDs: {offenders}. "
            f"This usually means the wrong target column was selected for "
            f"this task, or the upstream CSV has unlabeled rows. "
            f"Clean the data upstream or pick the right --activity_col."
        )

    stratifiers_arr = {k: np.array(v, dtype=np.float32) for k, v in stratifiers.items()}
    stratifiers_arr["scaffold_id"] = np.array(scaffold_ids, dtype=object)

    return ParsedSubset(
        mols=mols, smiles=smiles_list,
        ids=ids_arr,
        y=y_arr,
        weights=np.array(ws, dtype=np.float32),
        rows=row_dicts,
        stratifiers=stratifiers_arr,
        n_valid=len(mols),
        n_total=len(df),
    )


# ==============================================================================
# S5  RDKit descriptor block  (unchanged from v17)
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


def compute_rdkit_matrix(
    mols: List[Any],
    rows: List[dict],
) -> Tuple[np.ndarray, List[str], List[str], List[str]]:
    """Compute RDKit-based descriptors. Returns (X, names, block_types, namespaces)."""
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
        arr[0] = _safe_float(row.get("is_covalent"))
        arr[1] = _covalent_confidence_num(row.get("covalent_confidence"))
        arr[2] = _safe_float(row.get("stereo_defined_flag"))
        arr[3] = _safe_float(row.get("complexity_score"), 0.3)
        arr[4] = _safe_float(row.get("is_frequent_scaffold"))
        parts.append(arr)

        vectors.append(np.concatenate(parts).astype(np.float32))

    X = np.vstack(vectors).astype(np.float32)
    assert X.shape[1] == len(names) == len(block_types) == len(namespaces)
    return X, names, block_types, namespaces


# ==============================================================================
# S6  Mordred  (unchanged from v17)
# ==============================================================================
class MordredBlock:
    """Mordred descriptors with NaN filter fitted on train, cached per-InChIKey."""

    def __init__(self, nan_threshold: float, cache_dir: Optional[Path]):
        self.nan_threshold = nan_threshold
        self.cache_dir = cache_dir
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)

        self._calc = None
        self._keep_mask: Optional[np.ndarray] = None
        self._col_names: List[str] = []
        self._raw_names: List[str] = []
        if HAS_MORDRED:
            try:
                self._calc = MordredCalculator(mordred_desc, ignore_3D=True)
                self._raw_names = [str(d) for d in self._calc.descriptors]
                self._n_raw = len(self._raw_names)
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
            return np.zeros((len(mols), self._n_raw if self.available else 0),
                            dtype=np.float32)

        out = np.full((len(mols), self._n_raw), np.nan, dtype=np.float32)
        missing: List[Tuple[int, Any, Any]] = []
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
            log.info(f"Mordred {label}: cache hits={cache_hits}/{len(mols)}, "
                     f"computing {len(missing)} missing")
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
        return self.cache_dir / f"{safe_id}.npy"

    def fit_filter(self, M_train: np.ndarray) -> None:
        if M_train.shape[1] == 0:
            self._keep_mask = np.array([], dtype=bool)
            return
        nan_rate = np.isnan(M_train).mean(axis=0)
        self._keep_mask = nan_rate <= self.nan_threshold
        self._col_names = [
            f"mordred::{n}" for n, keep in zip(self._raw_names, self._keep_mask) if keep
        ]
        log.info(f"Mordred NaN filter: dropped {(~self._keep_mask).sum()}, "
                 f"kept {len(self._col_names)}")

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
# S7  DOPtools fragments  (unchanged from v17)
# ==============================================================================
class DOPtoolsBlock:
    CONFIGS = {"circus": [(0, 2), (0, 3)], "chyline": [(2, 4), (2, 6)]}

    def __init__(self, fit_samples: int = 5000):
        self.fit_samples = fit_samples
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
        sample = train_smiles[: self.fit_samples]
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
                arr = np.nan_to_num(df.values.astype(np.float32),
                                    nan=0.0, posinf=0.0, neginf=0.0)
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
# S8  Filters  ([NEW-4] task-aware MI for FP reduction and VIF)
# ==============================================================================
def remove_zero_variance(X: np.ndarray, threshold: float = 1e-6) -> np.ndarray:
    variances = np.var(X, axis=0)
    keep = variances > threshold
    log.info(f"  zero-variance: removed {(~keep).sum()}, kept {keep.sum()}")
    return keep


def _mi_for_task(X: np.ndarray, y: np.ndarray, task: str,
                 random_state: int = 42) -> np.ndarray:
    """[NEW-4] Pick the right MI estimator for the task."""
    if task == "classification":
        return mutual_info_classif(X, y.astype(int), random_state=random_state)
    return mutual_info_regression(X, y, random_state=random_state)


def reduce_fingerprint_mi(
    X: np.ndarray,
    y: np.ndarray,
    block_types: List[str],
    task: str,
    k: int = 2048,
) -> np.ndarray:
    """Keep top-k fingerprint features by MI with y; keep all non-FP features."""
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
        log.warning(f"  FP MI: only {len(y_valid)} valid-target rows; "
                    f"skipping MI reduction, keeping all {len(fp_idx)} fingerprint features")
        return keep
    mi = _mi_for_task(X_valid, y_valid, task)
    mi = np.nan_to_num(mi)
    top_local = np.argsort(mi)[-k:]
    top_global = set(int(fp_idx[i]) for i in top_local)
    for i, gi in enumerate(fp_idx):
        if int(gi) not in top_global:
            keep[gi] = False
    log.info(f"  FP MI-reduction ({task}): {len(fp_idx)} -> {k}")
    return keep


def fragment_min_support_filter(
    X: np.ndarray,
    block_types: List[str],
    min_support: float = 0.01,
) -> np.ndarray:
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
    log.info(f"  fragment min-support ({min_support:.1%}, min_count={min_count}): "
             f"removed {int(drop_local.sum())}/{len(frag_idx)}")
    return keep


def blockwise_corr_filter(
    X: np.ndarray,
    block_types: List[str],
    threshold: float = 0.95,
) -> np.ndarray:
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


class ScopedVIFFilter(BaseEstimator, TransformerMixin):
    """Iterative VIF removal on continuous features. Pre-reduced by task-aware MI."""
    def __init__(self, block_types: List[str], y: np.ndarray,
                 task: str = "regression",
                 vif_threshold: float = 10.0, max_candidates: int = 400,
                 apply_to_fingerprints: bool = False):
        self.block_types = block_types
        self.y = y
        self.task = task
        self.vif_threshold = vif_threshold
        self.max_candidates = max_candidates
        self.apply_to_fingerprints = apply_to_fingerprints
        self.support_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y=None):
        n = X.shape[1]
        self.support_ = np.ones(n, dtype=bool)
        if self.apply_to_fingerprints:
            cands = np.arange(n)
        else:
            cands = np.array([i for i, t in enumerate(self.block_types)
                              if t == "continuous"])
        if len(cands) == 0:
            return self

        if len(cands) > self.max_candidates:
            mi = _mi_for_task(X[:, cands], self.y, self.task)
            mi = np.nan_to_num(mi)
            top = np.argsort(mi)[-self.max_candidates:]
            cands = cands[top]
            log.info(f"  VIF candidates pre-reduced by MI ({self.task}): -> {len(cands)}")

        remaining = list(map(int, cands))
        removed: Set[int] = set()
        while len(remaining) >= 2:
            X_sub = X[:, remaining]
            X_std = StandardScaler().fit_transform(X_sub)
            R = np.corrcoef(X_std.T); R = np.nan_to_num(R); np.fill_diagonal(R, 1.0)
            try:
                vif = np.diag(np.linalg.inv(R))
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
# S9  Stability selectors and nested CV ([NEW-4] task-aware)
# ==============================================================================
class StabilityRidgeSelector(BaseEstimator, TransformerMixin):
    """Stability selection via bootstrapped RidgeCV (regression)."""
    def __init__(self, n_bootstrap: int = 50, subsample_frac: float = 0.8,
                 random_state: int = 42):
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


class StabilityLogisticSelector(BaseEstimator, TransformerMixin):
    """
    [NEW-4] Stability selection via bootstrapped LogisticRegressionCV.

    Mirrors StabilityRidgeSelector exactly:
      - same bootstrap loop
      - same stability metric (mean |coef| / std coef)
      - same ranking and get_support signature
    Differences:
      - LogisticRegressionCV with L2 penalty (Cs swept on log scale)
      - Coefs come from model.coef_[0] (binary classification, single row)
      - Inputs are scaled the same way upstream as the Ridge path
    """
    def __init__(self, n_bootstrap: int = 50, subsample_frac: float = 0.8,
                 random_state: int = 42):
        self.n_bootstrap = n_bootstrap
        self.subsample_frac = subsample_frac
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray):
        rng = np.random.RandomState(self.random_state)
        n, p = X.shape
        n_sub = int(n * self.subsample_frac)
        coefs = np.zeros((self.n_bootstrap, p))
        y_int = y.astype(int)
        # Guard: each bootstrap sample must contain both classes.
        # If a draw misses a class, redraw (up to 10 attempts per iteration).
        for i in range(self.n_bootstrap):
            for _attempt in range(10):
                idx = rng.choice(n, n_sub, replace=False)
                if len(np.unique(y_int[idx])) >= 2:
                    break
            else:
                # Degenerate fold: keep zero coefs, contributing nothing.
                continue
            m = LogisticRegressionCV(
                Cs=np.logspace(-2, 2, 10),
                cv=3,
                penalty="l2",
                solver="lbfgs",
                max_iter=2000,
                scoring="roc_auc",
            )
            m.fit(X[idx], y_int[idx])
            coefs[i] = m.coef_[0]
        mean_abs = np.abs(coefs).mean(axis=0)
        std = coefs.std(axis=0)
        self.stability_ = mean_abs / (std + 1e-8)
        self.ranking_ = np.argsort(self.stability_)[::-1]
        return self

    def get_support(self, n_features: int) -> np.ndarray:
        sup = np.zeros(len(self.ranking_), dtype=bool)
        sup[self.ranking_[:n_features]] = True
        return sup


def _make_outer_cv(task: str, n_splits: int, random_state: int):
    """[NEW-4] StratifiedKFold for classification, KFold for regression."""
    if task == "classification":
        return StratifiedKFold(n_splits, shuffle=True, random_state=random_state)
    return KFold(n_splits, shuffle=True, random_state=random_state)


def _make_inner_cv(task: str, n_splits: int, random_state: int):
    if task == "classification":
        return StratifiedKFold(n_splits, shuffle=True, random_state=random_state)
    return KFold(n_splits, shuffle=True, random_state=random_state)


def _fit_eval_outer(X_tr: np.ndarray, y_tr: np.ndarray,
                    X_te: np.ndarray, y_te: np.ndarray,
                    task: str) -> float:
    """[NEW-4] Evaluate the consensus feature set on the outer test fold."""
    if task == "classification":
        m = LogisticRegressionCV(
            Cs=np.logspace(-2, 2, 10),
            cv=3, penalty="l2", solver="lbfgs",
            max_iter=2000, scoring="roc_auc",
        ).fit(X_tr, y_tr.astype(int))
        # ROC-AUC needs both classes in y_te
        if len(np.unique(y_te.astype(int))) < 2:
            return float("nan")
        proba = m.predict_proba(X_te)[:, 1]
        return float(roc_auc_score(y_te.astype(int), proba))
    else:
        m = RidgeCV().fit(X_tr, y_tr)
        return float(m.score(X_te, y_te))


def nested_cv_select(
    X: np.ndarray,
    y: np.ndarray,
    block_types: List[str],
    cfg: PipelineConfig,
    task: str,
) -> Dict[str, Any]:
    """
    [NEW-4] Task-aware nested CV.
      - Regression:     KFold / RidgeCV / mutual_info_regression / R^2
      - Classification: StratifiedKFold / LogisticRegressionCV / mutual_info_classif / ROC-AUC
    """
    y = np.asarray(y, dtype=float)
    finite_mask = np.isfinite(y)
    n_bad = int((~finite_mask).sum())
    if n_bad > 0:
        log.info(f"  nested_cv: dropping {n_bad} rows with NaN/Inf target "
                 f"(feature selection only)")
        X = X[finite_mask]
        y = y[finite_mask]

    rng = np.random.RandomState(cfg.seed)
    outer = _make_outer_cv(task, cfg.n_outer_cv, rng.randint(2**31))
    all_selected: List[Set[int]] = []
    outer_scores: List[float] = []
    score_name = "roc_auc" if task == "classification" else "r2"

    # StratifiedKFold needs y; KFold also accepts y. Pass y everywhere.
    for fold_i, (tr_idx, te_idx) in enumerate(outer.split(X, y.astype(int) if task == "classification" else y)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        vif = ScopedVIFFilter(
            block_types=block_types, y=y_tr, task=task,
            vif_threshold=cfg.vif_threshold,
            max_candidates=cfg.vif_max_candidates,
            apply_to_fingerprints=cfg.apply_vif_to_fingerprints,
        ).fit(X_tr)
        X_tr_v = vif.transform(X_tr)
        X_te_v = vif.transform(X_te)
        vif_kept_idx = np.where(vif.support_)[0]

        inner = _make_inner_cv(task, cfg.n_inner_cv, rng.randint(2**31))
        inner_selections: List[Set[int]] = []

        inner_y_for_split = y_tr.astype(int) if task == "classification" else y_tr
        for in_tr_idx, in_val_idx in inner.split(X_tr_v, inner_y_for_split):
            X_in, X_val = X_tr_v[in_tr_idx], X_tr_v[in_val_idx]
            y_in, y_val = y_tr[in_tr_idx], y_tr[in_val_idx]
            scaler = StandardScaler().fit(X_in)
            X_in_s = scaler.transform(X_in)
            if task == "classification":
                sel = StabilityLogisticSelector(
                    n_bootstrap=cfg.stability_n_bootstrap,
                    random_state=rng.randint(2**31)
                ).fit(X_in_s, y_in)
            else:
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
            score = _fit_eval_outer(
                X_tr_s[:, cons_local], y_tr,
                X_te_s[:, cons_local], y_te,
                task,
            )
        else:
            score = float("nan")
        outer_scores.append(score)
        all_selected.append(set(consensus))
        log.info(f"    fold {fold_i+1}: {len(consensus)} features, {score_name}={score:.3f}")

    freq = Counter(f for s in all_selected for f in s)
    stable = [f for f, c in freq.items() if c >= cfg.global_stability_min_folds]
    stable = sorted(stable, key=lambda f: -freq[f])[:cfg.n_features_linear]

    return {
        "stable_features": stable,
        "outer_scores": outer_scores,
        f"mean_{score_name}": float(np.nanmean(outer_scores)),
        f"std_{score_name}": float(np.nanstd(outer_scores)),
        "score_name": score_name,
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
    task: str,
    target_col: str,
    out_root: Path,
    subset_data: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> Dict[str, Any]:
    t0 = time.time()
    log.info(f"\n========== Variant: {variant} ==========")

    families = VARIANT_FAMILIES[variant]
    col_mask = np.array([
        FAMILY_BY_NAMESPACE.get(ns, "unknown") in families
        for ns in namespaces
    ])
    if col_mask.sum() == 0:
        log.warning(f"Variant '{variant}' has no columns -- skipping")
        return {"variant": variant, "skipped": True}

    X_tr_raw = X_by_subset["train"][:, col_mask]
    names = [n for n, k in zip(feature_names, col_mask) if k]
    btypes = [t for t, k in zip(block_types, col_mask) if k]
    log.info(f"  starting features: {X_tr_raw.shape[1]} "
             f"(families={sorted(families)})")

    zv_keep = remove_zero_variance(X_tr_raw, cfg.variance_threshold)
    X_tr = X_tr_raw[:, zv_keep]
    names = [n for n, k in zip(names, zv_keep) if k]
    btypes = [t for t, k in zip(btypes, zv_keep) if k]

    if any(t == "fingerprint" for t in btypes):
        mi_keep = reduce_fingerprint_mi(X_tr, y_train, btypes, task=task,
                                         k=cfg.fp_mi_target_k)
        X_tr = X_tr[:, mi_keep]
        names = [n for n, k in zip(names, mi_keep) if k]
        btypes = [t for t, k in zip(btypes, mi_keep) if k]
    else:
        mi_keep = np.ones(X_tr.shape[1], dtype=bool)

    if any(t == "fragment" for t in btypes) and cfg.fragment_min_support > 0.0:
        fs_keep = fragment_min_support_filter(
            X_tr, btypes, min_support=cfg.fragment_min_support
        )
        X_tr = X_tr[:, fs_keep]
        names = [n for n, k in zip(names, fs_keep) if k]
        btypes = [t for t, k in zip(btypes, fs_keep) if k]
    else:
        fs_keep = np.ones(X_tr.shape[1], dtype=bool)

    cc_keep = blockwise_corr_filter(X_tr, btypes, cfg.corr_threshold)
    X_tr = X_tr[:, cc_keep]
    names = [n for n, k in zip(names, cc_keep) if k]
    btypes = [t for t, k in zip(btypes, cc_keep) if k]

    log.info(f"  tree-space features: {X_tr.shape[1]}")

    X_tree: Dict[str, np.ndarray] = {}
    for sub in X_by_subset:
        X_sub = X_by_subset[sub][:, col_mask]
        X_sub = X_sub[:, zv_keep]
        X_sub = X_sub[:, mi_keep]
        X_sub = X_sub[:, fs_keep]
        X_sub = X_sub[:, cc_keep]
        X_tree[sub] = X_sub.astype(np.float32)

    variant_meta: Dict[str, Any] = {"variant": variant, "skipped": False}
    variant_meta["n_tree_features"] = X_tree["train"].shape[1]
    variant_meta["tree_feature_names"] = list(names)

    for sub, X_sub in X_tree.items():
        y_s, w_s, ids_s = subset_data[sub]
        path = out_root / sub / f"{variant}_tree.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        save_npz(
            path, X_sub, y_s, w_s, ids_s, names,
            variant=variant, space="tree", subset=sub,
            pipeline_version=PIPELINE_VERSION,
            task=task, target_column=target_col,                  # [NEW-7]
        )
        log.info(f"  saved {sub}/{variant}_tree.npz  shape={X_sub.shape}")

    log.info(f"  linear-space nested CV ({cfg.n_outer_cv}x{cfg.n_inner_cv} folds, task={task})")
    nested = nested_cv_select(X_tree["train"], y_train, btypes, cfg, task)
    stable = nested["stable_features"]

    if not stable:
        log.warning(f"  variant {variant}: no stable features; skipping linear-space")
        variant_meta["linear_skipped"] = True
        variant_meta["nested_cv"] = {
            k: v for k, v in nested.items() if k != "feature_frequency"
        }
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
            task=task, target_column=target_col,                  # [NEW-7]
        )
        log.info(f"  saved {sub}/{variant}_linear.npz  shape={X_lin.shape}")

    variant_meta["n_linear_features"] = int(linear_mask.sum())
    variant_meta["linear_feature_names"] = linear_names
    # Preserve task-aware metric keys
    variant_meta["nested_cv"] = {
        k: v for k, v in nested.items()
        if k in ("score_name", "outer_scores")
        or k.startswith("mean_") or k.startswith("std_")
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
# S12  Runner script generator  ([NEW-1] task-aware runner)
# ==============================================================================
def generate_runner(split_base: str, output_base: str,
                    script_path: str = "run_featurize_v18.sh") -> None:
    """
    Walks <split_base>/{regression,classification}/<strategy>/ and emits
    one invocation per (task, strategy) pair. The script writes outputs
    to <output_base>/<task>/<strategy>/.
    """
    split_base_p = Path(split_base)
    if not split_base_p.exists():
        log.error(f"split_base does not exist: {split_base_p}")
        sys.exit(1)

    invocations: List[Tuple[str, str]] = []  # (task, strategy)
    for task in VALID_TASKS:
        task_dir = split_base_p / task
        if not task_dir.exists():
            log.warning(f"  no {task} splits under {task_dir}; skipping")
            continue
        for strat_dir in sorted(task_dir.iterdir()):
            if strat_dir.is_dir() and (strat_dir / "train.csv").exists():
                invocations.append((task, strat_dir.name))

    if not invocations:
        log.error(f"No valid split dirs under {split_base_p}/{{regression,classification}}/")
        sys.exit(1)
    log.info(f"Found {len(invocations)} (task, strategy) pairs:")
    for t, s in invocations:
        log.info(f"  {t}/{s}")

    lines = [
        "#!/usr/bin/env bash",
        f"# Auto-generated by pad4_featurize_v{PIPELINE_VERSION}",
        "set -euo pipefail",
        "",
        f'SPLIT_BASE="{split_base_p.resolve()}"',
        f'OUTPUT_BASE="{Path(output_base).resolve()}"',
        f'CACHE_DIR="{Path(output_base).resolve()}/.cache_mordred"',
        f'SCRIPT="$(cd "$(dirname "$0")" && pwd)/pad4_featurize_v{PIPELINE_VERSION.replace(".", "_")}.py"',
        "",
        '# Each entry: "task strategy"',
        "INVOCATIONS=(",
    ]
    for task, strat in invocations:
        lines.append(f'    "{task} {strat}"')
    lines += [
        ")", "",
        'for entry in "${INVOCATIONS[@]}"; do',
        '    read -r task strategy <<< "$entry"',
        '    split_dir="${SPLIT_BASE}/${task}/${strategy}"',
        '    out_dir="${OUTPUT_BASE}/${task}/${strategy}"',
        '    if [ ! -d "$split_dir" ]; then',
        '        echo "ERROR: $split_dir does not exist" >&2',
        '        exit 1',
        '    fi',
        '    echo "=== Processing: ${task}/${strategy} ==="',
        '    python "$SCRIPT" \\',
        '        --split_dir "$split_dir" \\',
        '        --output_dir "$out_dir" \\',
        '        --cache_dir "${CACHE_DIR}" \\',
        '        --task auto \\',
        '        --test_filename test_locked.csv \\',
        '        --variants full fingerprints physchem mordred fragments \\',
        '        --ad_score_col t1_self_tanimoto \\',
        '        --ad_threshold 0.35 \\',
        '        --fragment_min_support 0.01 \\',
        '        --n_features_linear 120 \\',
        '        --n_outer_cv 5 \\',
        '        --n_inner_cv 3 \\',
        '        --global_stability_min_folds 3 \\',
        '        --seed 42',
        'done',
        '',
        'echo "All (task, strategy) pairs complete."',
        '',
    ]
    with open(script_path, "w") as f:
        f.write("\n".join(lines))
    Path(script_path).chmod(0o755)
    log.info(f"Runner script written to: {script_path}")


# ==============================================================================
# S13  Main
# ==============================================================================
def main():
    p = argparse.ArgumentParser(
        description=f"PAD4 Featurization v{PIPELINE_VERSION} (task-aware)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # I/O
    p.add_argument("--split_dir", default="")
    p.add_argument("--output_dir", default="")
    p.add_argument("--cache_dir", default="",
                   help="Mordred per-molecule cache (default: <output_dir>/.cache)")
    p.add_argument("--cliff_file", default="",
                   help="Optional pad_activity_cliffs.csv for is_cliff stratifier")
    p.add_argument("--smiles_col", default="canonical_smiles")
    p.add_argument("--id_col", default="inchikey_14")

    # [NEW-1] Task
    p.add_argument("--task", default="auto",
                   choices=["auto", "regression", "classification"],
                   help="Task type. 'auto' infers from split_dir path "
                        "(/regression/ or /classification/ segment).")

    # [NEW-2] Target column. Empty means use task default.
    p.add_argument("--activity_col", default="",
                   help="Target column override. Empty = task default "
                        "(pIC50 for regression, activity_class for classification).")
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
    p.add_argument("--stratifiers_only", action="store_true")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--generate_runner", action="store_true")
    p.add_argument("--split_base", default="")
    p.add_argument("--output_base", default="")

    args = p.parse_args()

    if args.generate_runner:
        if not args.split_base or not args.output_base:
            p.error("--generate_runner requires --split_base and --output_base")
        generate_runner(args.split_base, args.output_base)
        return

    # [NEW-1] Resolve task
    try:
        resolved_task = resolve_task(args.split_dir, args.task)
    except TaskResolutionError as e:
        log.error(str(e))
        sys.exit(2)

    # [NEW-2] Resolve target column from task default unless overridden
    if args.activity_col:
        resolved_target = args.activity_col
        log.info(f"  target column: explicit --activity_col {resolved_target}")
    else:
        resolved_target = TASK_DEFAULT_TARGET[resolved_task]
        log.info(f"  target column: task default '{resolved_target}' "
                 f"(task={resolved_task})")

    cfg = PipelineConfig(
        split_dir=args.split_dir,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        cliff_file=args.cliff_file,
        smiles_col=args.smiles_col,
        id_col=args.id_col,
        activity_col=resolved_target,   # stored resolved value in config
        task=resolved_task,             # [NEW-1, NEW-6]
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
    log.info(f"PAD4 FEATURIZATION v{PIPELINE_VERSION}  (task-aware)")
    log.info(f"Task:        {cfg.task}")
    log.info(f"Target col:  {cfg.activity_col}")
    log.info(f"Config hash: {cfg.config_hash}  Seed: {cfg.seed}  Git: {git_hash}")
    log.info(f"Variants:    {cfg.variants}")
    log.info(f"Cache:       {cache_dir}")

    # Cliffs (optional)
    cliff_ids: Set[str] = set()
    if cfg.cliff_file:
        try:
            cliff_df = pd.read_csv(cfg.cliff_file)
            for cand in (cfg.id_col, "inchikey_14", "inchikey_1"):
                if cand in cliff_df.columns:
                    cliff_ids = set(cliff_df[cand].astype(str).str[:14])
                    break
            log.info(f"Loaded cliff IDs: {len(cliff_ids)}")
        except Exception as e:
            log.warning(f"Could not read cliff file: {e}")

    # Load splits
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

    # Sanity: train/test ID overlap
    if "test" in subsets:
        train_ids = set(subsets["train"][cfg.id_col])
        test_ids = set(subsets["test"][cfg.id_col])
        overlap = train_ids & test_ids
        if overlap:
            raise PipelineError(
                f"Data leakage: {len(overlap)} IDs overlap between train and test"
            )
        log.info("Train/test ID overlap check: PASSED")

    # Parse all subsets. [NEW-3] NaN targets raise here.
    parsed: Dict[str, ParsedSubset] = {}
    for sub, df in subsets.items():
        parsed[sub] = parse_subset(df, cfg, resolved_target, cliff_ids, sub)
        log.info(f"  {sub}: {parsed[sub].n_valid}/{parsed[sub].n_total} valid molecules")

        # [NEW-5] activity_class safety check
        if cfg.task == "classification":
            _validate_classification_target(parsed[sub].y, parsed[sub].ids, sub)

    # Fast path: stratifiers only
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
            log.info(f"  wrote {sub}/stratifiers.npz  "
                     f"keys={list(parsed[sub].stratifiers.keys())}  "
                     f"n={len(parsed[sub].ids)}")
        log.info("=" * 68)
        log.info(f"PIPELINE v{PIPELINE_VERSION} COMPLETE (stratifiers only)")
        log.info(f"Output: {out_root}")
        log.info("=" * 68)
        return

    # Compute descriptor blocks
    log.info("\nComputing descriptor blocks (one-time per split)")
    t_desc = time.time()

    rdkit_X: Dict[str, np.ndarray] = {}
    rdkit_names: List[str] = []
    rdkit_btypes: List[str] = []
    rdkit_ns: List[str] = []
    for sub in subsets:
        t0 = time.time()
        X_r, names_r, btypes_r, ns_r = compute_rdkit_matrix(parsed[sub].mols,
                                                            parsed[sub].rows)
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
        dop = DOPtoolsBlock(cfg.doptools_fit_samples)
        dop.fit(parsed["train"].smiles)
        if dop._fitted:
            for sub in subsets:
                dop_X[sub] = dop.transform(parsed[sub].smiles, sub)
            dop_names = dop.feature_names

    log.info(f"Descriptor computation total: {time.time()-t_desc:.1f}s")

    # Concatenate
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
                f"{sub} has {X_by_subset[sub].shape[1]} cols but "
                f"{len(feature_names)} names"
            )

    # Stratifiers
    for sub in subsets:
        strat_path = out_root / sub
        strat_path.mkdir(parents=True, exist_ok=True)
        save_stratifiers(
            strat_path / "stratifiers.npz",
            parsed[sub].stratifiers,
            parsed[sub].ids,
        )
        log.info(f"  saved {sub}/stratifiers.npz  keys={list(parsed[sub].stratifiers.keys())}")

    # Per-variant processing
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
        "task": cfg.task,                         # [NEW-7]
        "target_column": resolved_target,         # [NEW-7]
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
            task=cfg.task,
            target_col=resolved_target,
            out_root=out_root,
            subset_data=subset_data,
        )
        manifest["variants"][variant] = vmeta

    manifest_path = out_root / "feature_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    log.info(f"\nManifest: {manifest_path}")
    log.info("=" * 68)
    log.info(f"PIPELINE v{PIPELINE_VERSION} COMPLETE  (task={cfg.task})")
    log.info(f"Output: {out_root}")
    log.info("=" * 68)


if __name__ == "__main__":
    main()