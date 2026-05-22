#!/usr/bin/env python3
"""
PAD4 Inhibitor Data Curation Pipeline — v17.2 (Repo-clean + E/Z fix)
=====================================================================
Changes from v17.1 (2026-05):
• [BUG FIX] _get_stereo_info_v17 now compatible with RDKit ≥ 2025.x.
  v17.0/v17.1 used the old StereoInfo.Type.Bond / .Specified.Yes API,
  which raises AttributeError on modern RDKit. The except block silently
  swallowed the error and fell through to the legacy fallback, which
  has no E/Z detection. Net result on v17.1: zero E/Z compounds reported
  dataset-wide, even though the data contains hundreds.
  v17.2 detects whichever API is available and binds enum constants once
  at import time. Behavior on RDKit ≤ 2024 is unchanged.

Changes from v17.0 (cleanup pass):
• Validation-set scaffolding removed (AID 1805620 now treated as a normal
  Format-B PubChem assay; see PAD4-Bench paper §6.3 for v1 status).
• Duplicate `_enumerate_stereo_canonical` and `_enumerate_stereo_worker`
  definitions removed.
• Two competing Stage 2.6 implementations consolidated.
• Stage 8 `agg_dict` duplicate-key collisions cleaned up.
• Repo-aware path defaults (anchored to script location, not CWD).
• Stage 1 globs recurse into pubchem/, bindingdb/, chembl/ subdirs.
• Single RANDOM_SEED constant, seeded explicitly in main().

All numerical pipeline behavior on T1/T2/T3/Ki tier ASSIGNMENT is preserved.
The v17.2 fix will populate `stereo_has_ez=True` for compounds with E/Z
stereo (it was always False in v17.1 outputs), which propagates through:
  - stereo_priority_score (E/Z bonus +0.10 per compound)
  - ml_weight_raw (small downstream shift)
  - dataset_summary.json T1_stereo.ez_present count
  - Stage 13 QC log line
Tier counts (T1/T2/T3/Ki) are NOT affected by the E/Z fix.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import logging
import os
import random
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Optional imports ──────────────────────────────────────────────────
try:
    from sklearn.covariance import EmpiricalCovariance
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False

# ─────────────────────────────────────────────────────────────────────
# Dimorphite-DL integration (robust, version-agnostic, production-safe)
# ─────────────────────────────────────────────────────────────────────
_DIMORPHITE_OK = False
_DIMORPHITE_VERSION = "unknown"
_dimorphite_fn = None  # callable: (smiles: str, ph: float) -> list[str]

def _init_dimorphite() -> None:
    """Detect Dimorphite-DL installed version and bind the correct API."""
    global _DIMORPHITE_OK, _DIMORPHITE_VERSION, _dimorphite_fn
    import importlib.metadata
    try:
        _DIMORPHITE_VERSION = importlib.metadata.version("dimorphite-dl")
    except Exception:
        _DIMORPHITE_VERSION = "unknown"

    # 1️⃣ Try top-level API
    try:
        from dimorphite_dl import protonate_smiles as fn
        def _call(smi: str, ph: float) -> list:
            return fn(smi, ph_min=ph, ph_max=ph, precision=1.0, max_variants=1)
        _dimorphite_fn = _call
        _DIMORPHITE_OK = True
        logging.info("  Dimorphite-DL: using top-level protonate_smiles API")
        return
    except Exception:
        pass
    # 2️⃣ Try alternative v2 module layout
    try:
        from dimorphite_dl.protonate import protonate_smiles as fn
        def _call(smi: str, ph: float) -> list:
            return fn(smi, ph_min=ph, ph_max=ph, precision=1.0, max_variants=1)
        _dimorphite_fn = _call
        _DIMORPHITE_OK = True
        logging.info("  Dimorphite-DL: using protonate module API")
        return
    except Exception:
        pass
    # 3️⃣ Try legacy v1 class API
    try:
        from dimorphite_dl import DimorphiteDL
        def _call(smi: str, ph: float) -> list:
            dl = DimorphiteDL(min_ph=ph, max_ph=ph, pka_precision=1.0,
                              max_variants=1, silent=True)
            return list(dl.protonate(smi))
        _dimorphite_fn = _call
        _DIMORPHITE_OK = True
        logging.info("  Dimorphite-DL: using v1 class API")
        return
    except Exception:
        pass
    logging.warning("  Dimorphite-DL not available — using fallback (no protonation)")

_init_dimorphite()

def protonate_with_dimorphite(smiles: str, ph: float = 7.4) -> str:
    """Protonate SMILES at target pH using detected Dimorphite-DL API."""
    if not _DIMORPHITE_OK or _dimorphite_fn is None:
        return smiles
    try:
        results = _dimorphite_fn(smiles, ph)
        return results[0] if results else smiles
    except Exception as e:
        logging.debug(f"dimorphite failed for '{smiles[:40]}': {e}")
        return smiles

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, inchi as InchiMod
from rdkit.Chem import SanitizeFlags
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

# v17 / v17.2: Import FindPotentialStereo + the StereoType/StereoSpecified
# enums needed to interpret its output.
#
# RDKit changed this API around release 2025.x. Two compatibility shims:
#   • "new" API (RDKit ≥ ~2025.03): StereoType / StereoSpecified are
#     module-level enums in rdkit.Chem.rdchem, with values like
#     StereoType.Bond_Double and StereoSpecified.Specified.
#   • "old" API (RDKit ≤ ~2024):    nested as StereoInfo.Type.Bond and
#     StereoInfo.Specified.Yes.
#
# v17.1 of this script only handled the OLD API, so on modern RDKit every
# call to FindPotentialStereo fell through to the legacy fallback path
# (which doesn't detect E/Z), producing 0 E/Z compounds dataset-wide.
# v17.2 (this version) probes both and binds the right one.
_FIND_POTENTIAL_STEREO_OK = False
_STEREO_BOND_TYPES: tuple = ()       # set of enum values that mean "bond stereo"
_STEREO_SPECIFIED_VALUE = None       # enum value that means "specified"
_STEREO_UNSPECIFIED_VALUE = None     # enum value that means "unspecified"
try:
    from rdkit.Chem import FindPotentialStereo
    # Try new API first
    try:
        from rdkit.Chem.rdchem import StereoType, StereoSpecified
        # Bond-type stereo descriptors: double-bond E/Z and atropisomerism.
        # Use getattr to be tolerant of versions that don't expose all members.
        _STEREO_BOND_TYPES = tuple(
            t for t in (
                getattr(StereoType, "Bond_Double",       None),
                getattr(StereoType, "Bond_Cis_Trans",    None),  # alt naming
                getattr(StereoType, "Bond_Atropisomer",  None),
            ) if t is not None
        )
        _STEREO_SPECIFIED_VALUE   = StereoSpecified.Specified
        _STEREO_UNSPECIFIED_VALUE = StereoSpecified.Unspecified
        _FIND_POTENTIAL_STEREO_OK = True
        _STEREO_API = "new"
    except ImportError:
        # Fall back to old API
        from rdkit.Chem.rdchem import StereoInfo as _StereoInfo
        _STEREO_BOND_TYPES        = (_StereoInfo.Type.Bond,)
        _STEREO_SPECIFIED_VALUE   = _StereoInfo.Specified.Yes
        _STEREO_UNSPECIFIED_VALUE = _StereoInfo.Specified.No
        _FIND_POTENTIAL_STEREO_OK = True
        _STEREO_API = "legacy"
except ImportError:
    _STEREO_API = "unavailable"

try:
    from rdkit.Chem.FilterCatalogs import FilterCatalog, FilterCatalogParams
except ImportError:
    from rdkit.Chem.rdfiltercatalog import FilterCatalog, FilterCatalogParams

RDLogger.logger().setLevel(RDLogger.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────────────────────────────────────────────────
# rdMolStandardize singletons
# ─────────────────────────────────────────────────────────────────────
_LARGEST_FRAG  = rdMolStandardize.LargestFragmentChooser()
_UNCHARGER     = rdMolStandardize.Uncharger()
_NORMALIZER    = rdMolStandardize.Normalizer()
_REIONIZER     = rdMolStandardize.Reionizer()
_TAUTOMER_ENUM = rdMolStandardize.TautomerEnumerator()
_PAINS_PARAMS = FilterCatalogParams()
_PAINS_PARAMS.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
_PAINS_CATALOG = FilterCatalog(_PAINS_PARAMS)
try:
    _BRENK_PARAMS = FilterCatalogParams()
    _BRENK_PARAMS.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
    _BRENK_CATALOG = FilterCatalog(_BRENK_PARAMS)
    _BRENK_OK = True
except Exception:
    _BRENK_CATALOG = None
    _BRENK_OK = False

# ══════════════════════════════════════════════════════════════════════
# TOOLING STATUS LOG
# ══════════════════════════════════════════════════════════════════════
def _log_tooling_status() -> None:
    """Emit structured tooling availability block at pipeline startup."""
    import rdkit
    lines = [
        "",
        "┌─ TOOLING STATUS ──────────────────────────────────────────────────┐",
        f"│  RDKit              : {rdkit.__version__:<46} │",
        f"│  pandas             : {pd.__version__:<46} │",
        f"│  numpy              : {np.__version__:<46} │",
        f"│  Dimorphite-DL      : {'✓ v=' + _DIMORPHITE_VERSION if _DIMORPHITE_OK else '✗ NOT FOUND  →  RDKit Uncharger fallback':<46} │",
        f"│  scikit-learn       : {'✓ Mahalanobis AD enabled' if _SKLEARN_OK  else '✗ NOT FOUND  →  Tanimoto-only AD (install scikit-learn)':<46} │",
        f"│  Brenk catalog      : {'✓' if _BRENK_OK else '✗ NOT FOUND  →  is_brenk = False for all compounds':<46} │",
        f"│  FindPotentialStereo: {('✓ E/Z + tetrahedral [' + _STEREO_API + ' API]') if _FIND_POTENTIAL_STEREO_OK else '✗ NOT FOUND  →  legacy stereo census':<46} │",
        "└───────────────────────────────────────────────────────────────────┘",
    ]
    for line in lines:
        logging.info(line)
    if not _DIMORPHITE_OK:
        logging.warning("  Dimorphite-DL unavailable — protonation uses RDKit Uncharger (no pH 7.4 correction). Install: pip install dimorphite-dl")
    else:
        logging.info(f"  Dimorphite-DL version : {_DIMORPHITE_VERSION}")
        logging.info("  Dimorphite-DL: ready")
    if not _SKLEARN_OK:
        logging.warning("  scikit-learn unavailable — AD uses Tanimoto kNN only (no Mahalanobis). Install: pip install scikit-learn")
    if not _FIND_POTENTIAL_STEREO_OK:
        logging.warning("  FindPotentialStereo unavailable — using legacy stereo census. Upgrade RDKit ≥ 2020 for E/Z stereo detection.")

# ══════════════════════════════════════════════════════════════════════
# SECTION 0 — SMARTS SAFETY LAYER
# ══════════════════════════════════════════════════════════════════════
def safe_smarts_compile(name: str, smarts: str) -> Optional[Chem.Mol]:
    if not smarts or not smarts.strip():
        logging.error(f"  SMARTS [{name}]: empty — skipped"); return None
    try:
        pat = Chem.MolFromSmarts(smarts)
        if pat is None:
            logging.error(f"  SMARTS [{name}]: RDKit returned None — skipped"); return None
        _ = Chem.MolFromSmiles("CC").HasSubstructMatch(pat)
        return pat
    except Exception as e:
        logging.error(f"  SMARTS [{name}]: {e} — skipped"); return None

def validate_smarts_library(library: Dict[str, str]) -> Dict[str, Chem.Mol]:
    compiled: Dict[str, Chem.Mol] = {}
    for name, smarts in library.items():
        pat = safe_smarts_compile(name, smarts)
        if pat is not None:
            compiled[name] = pat
    n_fail = len(library) - len(compiled)
    if n_fail:
        logging.warning(f"  SMARTS: {len(compiled)}/{len(library)} compiled ({n_fail} FAILED)")
    else:
        logging.info(f"  SMARTS validation: all {len(compiled)} patterns compiled OK")
    return compiled

# ══════════════════════════════════════════════════════════════════════
# SECTION 1 — CONSTANTS (v17.1)
# ══════════════════════════════════════════════════════════════════════
PIPELINE_VERSION = "17.2"   # v17.1 + E/Z stereo detection bug fix
RANDOM_SEED      = 42       # seeded in main()

# Activity
PACTIVITY_THRESHOLD  = 6.0
PACTIVITY_VALID_MIN  = 2.0
PACTIVITY_VALID_MAX  = 12.0

# Uncertainty
HIGH_VARIANCE_STD = 0.50
LABEL_NOISE_STD   = 0.70
STD_EPSILON       = 0.10
MAD_FACTOR        = 1.4826

# HTS
HTS_NOISE_FLOOR      = 1.0
HTS_MIN_INFORMATIVE  = 5.0
HTS_ACTIVE_THRESHOLD = 30.0
FREQUENT_HITTER_FRAC = 0.40
T3_INACTIVE_RATIO    = 10

# Outlier
OUTLIER_IQR_FACTOR = 1.5
OUTLIER_ZSCORE_CUT = 2.5

# Druglikeness
DRUGLIKE_MW_MAX  = 900.0
DRUGLIKE_LOGP_MAX = 8.0
DRUGLIKE_LOGP_MIN = -4.0
DRUGLIKE_HBD_MAX = 10
DRUGLIKE_HBA_MAX = 15
MW_FLAG_HEAVY_CUT = 700.0

# Scaffold
FREQUENT_SCAFFOLD_MIN = 10
MAX_SCAFFOLD_CAP      = 15
SCAFFOLD_WEIGHT_DECAY = 0.5

# Fidelity thresholds
T1_HIGH_N_MEAS      = 3
T1_CONFIRMED_N_MEAS = 2
T1_HIGH_LU_MAX      = 0.15
T1_HIGH_STD_MAX     = 0.50
T1_CONFIRMED_STD_MAX = 0.50

# Label uncertainty v2 weights
LU_W_STD     = 0.35
LU_W_ASSAY   = 0.25
LU_W_SRC     = 0.15
LU_W_UNIT    = 0.15
LU_W_CONTEXT = 0.10
LU_STD_REF   = 1.0

# Unit normalization
UNIT_CONVERSION: Dict[str, float] = {
    "nm": 1.0, "nanomolar": 1.0, "nano molar": 1.0,
    "um": 1e3, "µm": 1e3, "micromolar": 1e3, "micro molar": 1e3, "μm": 1e3,
    "mm": 1e6, "millimolar": 1e6,
    "m":  1e9, "molar": 1e9,
    "pm": 1e-3, "picomolar": 1e-3, "pico molar": 1e-3,
}
UNIT_CONFIDENCE: Dict[str, float] = {
    "nm": 1.00, "nanomolar": 1.00,
    "um": 0.95, "µm": 0.95, "micromolar": 0.95, "μm": 0.95,
    "mm": 0.90, "millimolar": 0.90,
    "m":  0.85, "molar": 0.85,
    "pm": 0.90, "picomolar": 0.90,
    "inferred_nm": 0.60,
    "unknown": 0.30,
}

# AD v2
AD_KNN_K        = 5
AD_TANIMOTO_CUT = 0.35
AD_FP_RADIUS    = 2
AD_FP_BITS      = 1024
AD_DENSITY_K    = 10

# T3 self-AD thresholds
T3_SELF_AD_KNN_K   = 10
T3_SELF_AD_TAN_CUT = 0.25
MAX_T3_SELF_AD_TRAIN = 5000

# Activity cliff detection
ACTIVITY_CLIFF_TANIMOTO_MIN = 0.80
ACTIVITY_CLIFF_DPIC50_MIN   = 1.5
ACTIVITY_CLIFF_SEVERITY_BINS = [1.5, 2.0, 3.0]

# Cross-source consensus
CROSS_SOURCE_AGREEMENT_BONUS = 0.15
CROSS_SOURCE_MIN_SOURCES     = 2

# Source reliability
SOURCE_RELIABILITY: Dict[str, float] = {
    "BindingDB": 1.00, "ChEMBL": 0.90, "PubChem": 0.60,
}

# Assay confidence base
ASSAY_CONFIDENCE: Dict[str, float] = {
    "biochemical_confirmatory": 1.00, "biochemical_single_point": 0.80,
    "binding": 0.75, "confirmatory": 1.00, "panel": 0.60,
    "HTS": 0.30, "hts_percent_inhibition": 0.30, "unknown": 0.50,
}

# Assay context scoring rules
ASSAY_CONTEXT_RULES: Dict[str, float] = {
    "has_dose_response": +0.30, "has_doi": +0.20, "has_pmid": +0.10,
    "is_biochemical": +0.15, "is_cell_based": +0.05,
    "is_hts_penalty": -0.30, "is_rnai": -0.40, "is_non_human": -0.05,
}

COVALENT_CONFIDENCE_FOR_SPLIT = {"high", "medium"}

# Fidelity levels
FIDELITY_LEVELS = {
    "T1_high":      1.0,
    "T1_confirmed": 0.9,
    "T1_standard":  0.8,
    "T2_censored":  0.5,
    "T3_weak":      0.3,
}

# Target maps
UNIPROT_TO_ISOFORM: Dict[str, str] = {
    "Q9ULC6": "PAD1", "Q9Y2J8": "PAD2", "Q9ULW8": "PAD3",
    "Q9UM07": "PAD4", "Q6TGC4": "PAD6", "Q9Z183": "PAD4",
    "NP_037490.2": "PAD1", "NP_031391.2": "PAD2",
    "NP_057317.2": "PAD3", "NP_036519.2": "PAD4",
}
UNIPROT_TO_ORGANISM: Dict[str, str] = {
    "Q9ULC6": "human", "Q9Y2J8": "human", "Q9ULW8": "human",
    "Q9UM07": "human", "Q6TGC4": "human", "Q9Z183": "mouse",
}
CHEMBL_TO_ISOFORM: Dict[str, str] = {
    "CHEMBL1909486": "PAD1", "CHEMBL1909487": "PAD2",
    "CHEMBL1909488": "PAD3", "CHEMBL6111": "PAD4", "CHEMBL3638347": "PAD6",
}

# ── PubChem AID format groups ──
# Note: AID 1805620 was previously tagged as a held-out validation assay,
# but no end-to-end validation pipeline exists in v1. It is now treated as
# a normal Format-B PubChem assay. See PAD4-Bench paper §6.3.
CHEMBL_MIRROR_SOURCE_ID = "37"
FORMAT_A_AIDS = {"AID_1330527","AID_1813806","AID_1875531","AID_2071731","AID_2134413"}
FORMAT_B_AIDS = {"AID_1804546","AID_1805620","AID_1806182","AID_1806183",
                 "AID_1806764","AID_1806765","AID_1920046","AID_2202442"}
FORMAT_C_AIDS = {"AID_1919095","AID_1920200","AID_1963715","AID_1804627"}
FORMAT_D_AIDS = {"AID_463073","AID_485272","AID_488796"}
FORMAT_E_AIDS = {"AID_588487","AID_588560"}
FORMAT_F_AIDS = {"AID_492970"}
ASSAY_TYPE_MAP: Dict[str, str] = {
    "AID_492970": "biochemical_confirmatory",
    "AID_463073": "hts_percent_inhibition",
    "AID_485272": "hts_percent_inhibition",
    "AID_488796": "hts_percent_inhibition",
    "AID_588487": "panel",
    "AID_588560": "panel",
}

# Warhead SMARTS
WARHEAD_SMARTS_HIGH: Dict[str, str] = {
    "chloroacetamide":   "[NX3H1,NX3H2][CX3](=[OX1])[CH2][Cl,Br]",
    "vinyl_sulfone":     "[SX4](=[OX1])(=[OX1])[CX3]=[CX3]",
    "sulfonyl_fluoride": "[SX4](=[OX1])(=[OX1])[FX1]",
    "epoxide":           "[OX2r3]1[CX4r3][CX4r3]1",
    "alpha_halo_ketone": "[CX3](=[OX1])[CX4;!$(CC=O)][Cl,Br,I]",
    "isocyanate":        "[NX2]=[CX2]=[OX1]",
    "isothiocyanate":    "[NX2]=[CX2]=[SX1]",
}
WARHEAD_SMARTS_MEDIUM: Dict[str, str] = {
    "acrylamide":    "[NX3][CX3](=[OX1])[CX3;!$(C~a)]=[CX3;!$(C~a)]",
    "enone_refined": (
        "[CX3;!$(C~a);!$(C([NX3])=O);!$(C([OX2])=O)](=[OX1])"
        "[CX3;!$(C~a);!$(C~[NX3]);!$(C~[OX2])]=[CX3;!$(C~a)]"
    ),
    "aldehyde":      "[CX3H1](=[OX1])[CX4,CX3;!$(C~a)]",
    "beta_lactone":  "[OX2r4]1[CX3r4](=[OX1])[CX4r4][CX4r4]1",
    "disulfide":     "[SX2H0][SX2H0]",
}
WARHEAD_SMARTS_LOW: Dict[str, str] = {
    "boronic_acid":      "[BX3]([OX2H])[OX2H]",
    "boronate_ester":    "[BX3]([OX2])[OX2]",
    "activated_nitrile": "[NX3;H1,H2][CX4][CX2]#[NX1]",
    "cyanamide":         "[NX3H1][CX2]#[NX1]",
    "alpha_ketoamide":   "[CX3](=[OX1])[CX3](=[OX1])[NX3]",
}
REVERSIBLE_COVALENT_WARHEADS = {
    "boronic_acid", "boronate_ester", "activated_nitrile",
    "cyanamide", "alpha_ketoamide",
}
CONTEXT_EXCLUSION_SMARTS: Dict[str, str] = {
    "aromatic_conjugated_enone": "a[CX3]=[CX3][CX3](=[OX1])",
    "aryl_nitrile":              "a[CX2]#[NX1]",
    "alkyl_nitrile_plain":       "[CX4;!$(C[NX3])][CX2]#[NX1]",
    "hetaryl_nitrile":           "[n,nH][c][CX2]#[NX1]",
    "aryl_vinyl":                "a[CX3]=[CX3]",
    "full_amide_resonance":      "[NX3][CX3](=[OX1])[NX3]",
}
AGGREGATOR_SMARTS: List[str] = [
    "[CX4,CX3]~[CX4,CX3]~[CX4,CX3]~[CX4,CX3]~[CX4,CX3]~[CX4,CX3]~[CX4,CX3]~[CX4,CX3]",
    "[$(c1ccc2cc3ccccc3cc2c1)]",
    "[$(c1cc(-c2cc(=O)c3c(o2)cccc3)ccc1O)]",
    "[SX2r5]1[CX4r5][CX3r5](=[OX1])[NX3r5][CX3r5]1=[SX1]",
    "[NX3][CX3](=[SX1])[NX3]",
]
_TD_PATTERN = re.compile(
    "|".join(f"(?:{p})" for p in [
        r"preincubat", r"time[\s\-]?depend", r"irreversib",
        r"covalent", r"jump[\s\-]?dilut", r"washout", r"kinact",
    ]), flags=re.IGNORECASE,
)

# ── Compile SMARTS at startup ─────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S")
_all_smarts = {
    **{f"HIGH_{k}": v for k, v in WARHEAD_SMARTS_HIGH.items()},
    **{f"MEDIUM_{k}": v for k, v in WARHEAD_SMARTS_MEDIUM.items()},
    **{f"LOW_{k}": v for k, v in WARHEAD_SMARTS_LOW.items()},
    **{f"EXCL_{k}": v for k, v in CONTEXT_EXCLUSION_SMARTS.items()},
    **{f"AGG_{i}": v for i, v in enumerate(AGGREGATOR_SMARTS)},
}
_VALIDATED = validate_smarts_library(_all_smarts)
_WARHEAD_HIGH:    Dict[str, Chem.Mol] = {k[5:]:  _VALIDATED[k] for k in _VALIDATED if k.startswith("HIGH_")}
_WARHEAD_MEDIUM:  Dict[str, Chem.Mol] = {k[7:]:  _VALIDATED[k] for k in _VALIDATED if k.startswith("MEDIUM_")}
_WARHEAD_LOW:     Dict[str, Chem.Mol] = {k[4:]:  _VALIDATED[k] for k in _VALIDATED if k.startswith("LOW_")}
_CONTEXT_EXCL:    List[Chem.Mol]      = [_VALIDATED[k] for k in _VALIDATED if k.startswith("EXCL_")]
_AGGREGATOR_PATS: List[Chem.Mol]      = [_VALIDATED[k] for k in _VALIDATED if k.startswith("AGG_")]
_STAGE_LOG: List[dict] = []

# ══════════════════════════════════════════════════════════════════════
# v17 STEREO CONSTANTS
# ══════════════════════════════════════════════════════════════════════
STEREO_CROSS_RESOLVE      = True   # Enable cross-source stereo propagation
STEREO_ENUM_MAX_ISOMERS   = 4      # Max stereoisomers to enumerate per compound
STEREO_ENUM_WORKERS       = 4      # Parallel workers for enumeration step
STEREO_PENALTY_WEIGHT_V17 = 0.60   # Up from 0.40 — stereo is the biggest risk

# ══════════════════════════════════════════════════════════════════════
# SECTION 2 — LOGGING HELPERS
# ══════════════════════════════════════════════════════════════════════
def setup_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    for h in root.handlers[:]: root.removeHandler(h)
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

def log_stage(stage: str, rows_in: int, rows_out: int, notes: str = "") -> None:
    global _STAGE_LOG
    drop = rows_in - rows_out
    entry = {"ts": datetime.now().isoformat(), "stage": stage,
             "in": rows_in, "out": rows_out, "drop": drop, "notes": notes}
    _STAGE_LOG.append(entry)
    if rows_in == 0:
        logging.info(f"  [{stage}]  (no row-level operation)  {notes}")
    else:
        drop_pct = drop / rows_in * 100
        logging.info(
            f"  [{stage}] {rows_in:>8,} → {rows_out:>8,}  "
            f"(−{drop:,}, {drop_pct:.1f}%)  {notes}"
        )

def _table(rows: List[Tuple], headers: List[str], indent: int = 4) -> str:
    pad = " " * indent
    widths = [max(len(str(r[i])) for r in [headers] + list(rows))
              for i in range(len(headers))]
    fmt = pad + "  ".join(f"{{:<{w}}}" for w in widths)
    sep = pad + "  ".join("-" * w for w in widths)
    lines = [fmt.format(*headers), sep]
    for row in rows:
        lines.append(fmt.format(*[str(v) for v in row]))
    return "\n".join(lines)

# ══════════════════════════════════════════════════════════════════════
# v17 STEREO HELPER
# ══════════════════════════════════════════════════════════════════════
def _get_stereo_info_v17(mol):
    """
    v17.2: Accurate stereo census using FindPotentialStereo, with
    RDKit-version-agnostic enum handling.

    Returns: (n_specified, n_unspecified, has_ez, n_total, ez_specified)
    """
    if mol is None:
        return 0, 0, False, 0, 0
    if _FIND_POTENTIAL_STEREO_OK:
        try:
            si = FindPotentialStereo(mol)
            n_specified   = sum(1 for s in si if s.specified == _STEREO_SPECIFIED_VALUE)
            n_unspecified = sum(1 for s in si if s.specified == _STEREO_UNSPECIFIED_VALUE)
            ez_info       = [s for s in si if s.type in _STEREO_BOND_TYPES]
            has_ez        = len(ez_info) > 0
            ez_specified  = sum(1 for s in ez_info if s.specified == _STEREO_SPECIFIED_VALUE)
            return n_specified, n_unspecified, has_ez, len(si), ez_specified
        except Exception:
            pass
    # Legacy fallback (no FindPotentialStereo at all)
    try:
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        centres   = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        n_undef   = sum(1 for _, s in centres if s == "?")
        n_def     = len(centres) - n_undef
        return n_def, n_undef, False, len(centres), 0
    except Exception:
        return 0, 0, False, 0, 0

_STEREO_TIER = {
    "defined":           4,
    "achiral":           4,
    "partial_undefined": 2,
    "fully_undefined":   1,
    "unknown":           0,
}
_SRC_SCORE = {
    "BindingDB": 1.00,
    "ChEMBL":    0.90,
    "PubChem":   0.60,
}

def _pick_stereo_representative(grp: pd.DataFrame) -> str:
    """
    v17: Select the canonical SMILES that best represents a replicate group,
    giving priority to defined-stereo records over undefined ones.
    """
    if len(grp) == 0:
        return ""
    grp = grp.copy()
    grp["_stier"] = grp.get(
        "stereo_flag", pd.Series("unknown", index=grp.index)
    ).map(_STEREO_TIER).fillna(0)
    if "stereo_cross_resolved" in grp.columns:
        cross = grp["stereo_cross_resolved"].fillna(False)
        grp.loc[cross & (grp["_stier"] < 3), "_stier"] = 3
    grp["_sscore"] = grp.get(
        "source_db", pd.Series("PubChem", index=grp.index)
    ).apply(lambda s: _SRC_SCORE.get(str(s), 0.60) if isinstance(s, str) else 0.60)
    grp["_total"] = grp["_stier"] * 10 + grp["_sscore"]
    best_idx      = grp["_total"].idxmax()
    if (
        "stereo_cross_resolved" in grp.columns
        and grp.at[best_idx, "stereo_cross_resolved"]
        and "stereo_cross_smiles" in grp.columns
        and isinstance(grp.at[best_idx, "stereo_cross_smiles"], str)
        and grp.at[best_idx, "stereo_cross_smiles"].strip()
    ):
        return grp.at[best_idx, "stereo_cross_smiles"]
    smi = grp.at[best_idx, "canonical_smiles"]
    return smi if isinstance(smi, str) else ""

# ══════════════════════════════════════════════════════════════════════
# STAGE 1 — DATA LOADING
# ══════════════════════════════════════════════════════════════════════
def _skiprows(aid: str) -> List[int]:
    skip_5 = FORMAT_A_AIDS | FORMAT_B_AIDS | FORMAT_F_AIDS
    return [1, 2, 3, 4, 5] if aid in skip_5 else [1, 2, 3, 4]

def _assay_meta(aid: str, source_db: str) -> dict:
    return {"assay_id": aid,
            "assay_type": ASSAY_TYPE_MAP.get(aid, "biochemical_single_point"),
            "source_db": source_db}

def _row(smi, nM, orig_val, orig_unit, conv, qual, mtype,
         isoform, uniprot, source_db, aid, fname,
         pct, sd, meta,
         doi="", pmid="", has_dr=False, bioassay_name="",
         screening_conc=np.nan, raw_unit="") -> dict:
    """Build a record dict. v17.1: is_validation parameter removed."""
    return {
        "raw_smiles": smi, "activity_value_nM": nM,
        "original_activity_value": orig_val, "original_unit": orig_unit,
        "unit_conversion_factor": conv, "qualifier": qual,
        "measurement_type": mtype, "target_isoform": isoform,
        "target_uniprot": uniprot, "source_db": source_db,
        "source_id": aid, "source_file": fname,
        "pct_inhibition": pct, "sd_reported": sd,
        "is_artifact_high": False, "is_artifact_low": False,
        "doi": doi, "pmid": pmid, "has_dose_response": has_dr,
        "bioassay_name": bioassay_name,
        "screening_concentration_uM": screening_conc,
        "raw_unit_string": raw_unit,
        **meta,
    }

def _parse_a(fp: str, aid: str) -> pd.DataFrame:
    df = pd.read_csv(fp, skiprows=_skiprows(aid), low_memory=False)
    df.columns = df.columns.str.strip()
    meta = _assay_meta(aid, "PubChem"); fname = Path(fp).name; rows = []
    for _, r in df.iterrows():
        smi = r.get("PUBCHEM_EXT_DATASOURCE_SMILES", "")
        if not isinstance(smi, str) or not smi.strip(): continue
        std_type  = str(r.get("Standard Type", "")).lower()
        qual      = str(r.get("Standard Relation", "=")).strip() or "="
        std_units = str(r.get("Standard Units", "")).strip().lower()
        std_val   = r.get("Standard Value")
        pub_val   = r.get("PubChem Standard Value")
        nM, orig_val, orig_unit, conv = np.nan, np.nan, "", 1
        raw_unit = std_units or "unknown"
        if std_units == "nm" and pd.notna(std_val):
            nM = float(std_val); orig_val = float(std_val); orig_unit = "nM"
        elif pd.notna(pub_val):
            nM = float(pub_val) * 1000; orig_val = float(pub_val)
            orig_unit = "µM"; conv = 1000; raw_unit = "µM"
        if np.isnan(nM) or nM <= 0: continue
        mtype = ("IC50" if "ic50" in std_type else
                 "Ki"   if "ki"   in std_type else std_type.upper())
        doi  = str(r.get("DOI", r.get("doi", ""))).strip()
        pmid = str(r.get("PMID", r.get("pmid", ""))).strip()
        bname = str(r.get("BioAssay Name", r.get("Assay Name", ""))).strip()
        rows.append(_row(smi, nM, orig_val, orig_unit, conv, qual, mtype,
                         "PAD4", "Q9UM07", "PubChem", aid, fname,
                         np.nan, np.nan, meta,
                         doi=doi, pmid=pmid, bioassay_name=bname, raw_unit=raw_unit))
    return pd.DataFrame(rows)

def _parse_b(fp: str, aid: str) -> pd.DataFrame:
    df = pd.read_csv(fp, skiprows=_skiprows(aid), low_memory=False)
    df.columns = df.columns.str.strip()
    meta = _assay_meta(aid, "PubChem"); fname = Path(fp).name
    if "Target Accession(s)" in df.columns:
        df = df[df["Target Accession(s)"].astype(str).str.contains("Q9UM07", na=False)].copy()
    rows = []
    for _, r in df.iterrows():
        smi = r.get("PUBCHEM_EXT_DATASOURCE_SMILES", "")
        if not isinstance(smi, str) or not smi.strip(): continue
        ic50 = r.get("IC50")
        if pd.isna(ic50): continue
        try:    nM = float(ic50)
        except: continue
        if nM <= 0: continue
        qual = str(r.get("IC50 Qualifier", r.get("Standard Relation", "="))).strip()
        qual = qual.replace(".0", "").strip()
        if qual not in ("=", ">", "<", ">=", "<="): qual = "="
        acc     = str(r.get("Target Accession(s)", "Q9UM07")).strip()
        isoform = UNIPROT_TO_ISOFORM.get(acc, "PAD4")
        has_dr  = bool(r.get("Has Dose Response", False) or
                       str(r.get("Curve Class", "")).strip() not in ("", "nan"))
        sc = np.nan
        conc_col = next((c for c in df.columns if "conc" in c.lower()), None)
        if conc_col:
            try: sc = float(r.get(conc_col, np.nan))
            except: pass
        rows.append(_row(smi, nM, nM, "nM", 1, qual, "IC50", isoform, acc,
                         "PubChem", aid, fname,
                         np.nan, np.nan, meta,
                         doi=str(r.get("DOI", "")).strip(),
                         pmid=str(r.get("PMID", "")).strip(),
                         has_dr=has_dr,
                         bioassay_name=str(r.get("BioAssay Name", "")).strip(),
                         screening_conc=sc, raw_unit="nM"))
    return pd.DataFrame(rows)

def _parse_c(fp: str, aid: str) -> pd.DataFrame:
    df = pd.read_csv(fp, skiprows=_skiprows(aid), low_memory=False)
    df.columns = df.columns.str.strip()
    meta = _assay_meta(aid, "PubChem"); fname = Path(fp).name; rows = []
    for _, r in df.iterrows():
        smi = r.get("PUBCHEM_EXT_DATASOURCE_SMILES", "")
        if not isinstance(smi, str) or not smi.strip(): continue
        acc     = str(r.get("Target Accession(s)", "")).strip()
        isoform = UNIPROT_TO_ISOFORM.get(acc, None)
        if isoform is None:
            for k in UNIPROT_TO_ISOFORM:
                if k in acc: isoform = UNIPROT_TO_ISOFORM[k]; break
        isoform = isoform or "PAD4"
        for col, mtype in [("Ki", "Ki"), ("IC50", "IC50")]:
            if col in df.columns and pd.notna(r.get(col)):
                try:    nM = float(r[col])
                except: continue
                if nM > 0:
                    rows.append(_row(smi, nM, nM, "nM", 1, "=", mtype, isoform, acc,
                                     "PubChem", aid, fname,
                                     np.nan, np.nan, meta, raw_unit="nM"))
                break
    return pd.DataFrame(rows)

def _parse_d(fp: str, aid: str) -> pd.DataFrame:
    df = pd.read_csv(fp, skiprows=_skiprows(aid), low_memory=False)
    df.columns = df.columns.str.strip()
    meta = _assay_meta(aid, "PubChem"); fname = Path(fp).name
    inh_col = next((c for c in df.columns if "inhibition" in c.lower()), None)
    if inh_col is None: return pd.DataFrame()
    conc_col = next((c for c in df.columns if "conc" in c.lower()), None)
    rows = []
    for _, r in df.iterrows():
        smi = r.get("PUBCHEM_EXT_DATASOURCE_SMILES", "")
        if not isinstance(smi, str) or not smi.strip(): continue
        try:    raw = float(r[inh_col])
        except: continue
        if np.isnan(raw) or abs(raw) < HTS_MIN_INFORMATIVE: continue
        sc = np.nan
        if conc_col:
            try: sc = float(r.get(conc_col, np.nan))
            except: pass
        raw_clipped = float(np.clip(raw, 0, 100))
        row = _row(smi, np.nan, raw, "%", 1, "=", "pct_inhibition",
                   "PAD4", "Q9UM07", "PubChem", aid, fname,
                   raw_clipped, np.nan, meta, screening_conc=sc)
        row["is_artifact_high"] = bool(raw > 100)
        row["is_artifact_low"]  = bool(raw < 0)
        rows.append(row)
    return pd.DataFrame(rows)

def _parse_e(fp: str, aid: str) -> pd.DataFrame:
    df = pd.read_csv(fp, skiprows=_skiprows(aid), low_memory=False)
    df.columns = df.columns.str.strip()
    meta = _assay_meta(aid, "PubChem"); fname = Path(fp).name
    inh_col = next((c for c in df.columns if "inhibition" in c.lower()), None)
    if inh_col is None: return pd.DataFrame()
    rows = []
    for _, r in df.iterrows():
        smi = r.get("PUBCHEM_EXT_DATASOURCE_SMILES", "")
        if not isinstance(smi, str) or not smi.strip(): continue
        try:    raw = float(r[inh_col])
        except: continue
        if np.isnan(raw) or abs(raw) < HTS_MIN_INFORMATIVE: continue
        panel   = str(r.get("Panel Name", "")).strip()
        target  = str(r.get("Panel Target", "")).strip()
        isoform = None
        for k, iso in UNIPROT_TO_ISOFORM.items():
            if k in target: isoform = iso; break
        if isoform is None:
            for tag in ["PAD1","PAD2","PAD3","PAD4"]:
                if tag in panel: isoform = tag; break
        if isoform is None: continue
        row = _row(smi, np.nan, raw, "%", 1, "=", "pct_inhibition",
                   isoform, target, "PubChem", aid, fname,
                   float(np.clip(raw, 0, 100)), np.nan, meta)
        row["is_artifact_high"] = bool(raw > 100)
        row["is_artifact_low"]  = bool(raw < 0)
        rows.append(row)
    return pd.DataFrame(rows)

def _parse_f(fp: str, aid: str) -> pd.DataFrame:
    df = pd.read_csv(fp, skiprows=_skiprows(aid), low_memory=False)
    df.columns = df.columns.str.strip()
    meta = _assay_meta(aid, "PubChem"); fname = Path(fp).name; rows = []
    for _, r in df.iterrows():
        smi = r.get("PUBCHEM_EXT_DATASOURCE_SMILES", "")
        if not isinstance(smi, str) or not smi.strip(): continue
        try:    uM = float(r.get("Average IC50", np.nan))
        except: continue
        if np.isnan(uM) or uM <= 0: continue
        nM = uM * 1000
        qual = str(r.get("Qualifier", "=")).strip()
        qual = qual if qual in ("=", ">", "<", ">=", "<=") else "="
        sd = np.nan
        try:
            sd_uM = float(r.get("Standard Deviation", np.nan))
            if not np.isnan(sd_uM): sd = sd_uM * 1000
        except Exception: pass
        rows.append(_row(smi, nM, uM, "µM", 1000, qual, "IC50",
                         "PAD4", "Q9UM07", "PubChem", aid, fname,
                         np.nan, sd, meta,
                         doi=str(r.get("DOI", "")).strip(),
                         pmid=str(r.get("PMID", "")).strip(),
                         bioassay_name=str(r.get("BioAssay Name", "")).strip(),
                         raw_unit="µM"))
    return pd.DataFrame(rows)

def _parse_bindingdb(fp: str) -> pd.DataFrame:
    fname    = Path(fp).name
    iso_hint = next((i for i in ["PAD1","PAD2","PAD3","PAD4","PAD6"] if i in fname), None)
    aid      = fname.replace(".tsv", "")
    meta     = _assay_meta(aid, "BindingDB")
    try:
        header = pd.read_csv(fp, sep="\t", nrows=0)
    except Exception as e:
        logging.error(f"  BindingDB header failed {fname}: {e}"); return pd.DataFrame()
    needed   = ["Ligand SMILES", "Ki (nM)", "IC50 (nM)",
                "UniProt (SwissProt) Primary ID of Target Chain 1"]
    use_cols = [h for c in needed
                for h in [next((x for x in header.columns if c in x), None)] if h]
    try:
        df = pd.read_csv(fp, sep="\t", usecols=use_cols, low_memory=False, on_bad_lines="skip")
    except Exception as e:
        logging.error(f"  BindingDB parse failed {fname}: {e}"); return pd.DataFrame()
    rows = []
    for _, r in df.iterrows():
        smi     = r.get("Ligand SMILES", "")
        if not isinstance(smi, str) or not smi.strip(): continue
        uniprot = str(r.get("UniProt (SwissProt) Primary ID of Target Chain 1","")).strip()
        isoform = UNIPROT_TO_ISOFORM.get(uniprot, iso_hint)
        doi     = str(r.get("DOI", r.get("Article DOI", ""))).strip()
        pmid    = str(r.get("PMID", r.get("PubMed ID", ""))).strip()
        for col, mtype in [("IC50 (nM)", "IC50"), ("Ki (nM)", "Ki")]:
            raw = r.get(col)
            if pd.isna(raw): continue
            s = str(raw).strip(); qual = "="; num = s
            for p in [">=", "<=", ">", "<"]:
                if s.startswith(p): qual = p; num = s[len(p):].strip(); break
            try:    nM = float(num)
            except: continue
            if nM <= 0: continue
            rows.append(_row(smi, nM, nM, "nM", 1, qual, mtype,
                             isoform or "PAD4", uniprot, "BindingDB", aid, fname,
                             np.nan, np.nan, meta,
                             doi=doi, pmid=pmid, raw_unit="nM"))
    return pd.DataFrame(rows)

def _parse_chembl(fp: str) -> Tuple[pd.DataFrame, int]:
    fname    = Path(fp).name
    iso_hint = next((i for i in ["PAD1","PAD2","PAD3","PAD4","PAD6"] if i in fname), None)
    aid      = fname.replace(".tsv", "")
    meta     = _assay_meta(aid, "ChEMBL")
    n_mirror = 0
    try:
        with open(fp, encoding="utf-8") as f: hdr = f.readline()
        hdr_fixed = hdr.replace('"Target ChEMBL ID""Target Name"',
                                '"Target ChEMBL ID"\t"Target Name"')
        hdr_fixed = re.sub(r'""([A-Z])', '"\t"\\1', hdr_fixed)
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False,
                                         encoding="utf-8") as tmp:
            tmp.write(hdr_fixed)
            with open(fp, encoding="utf-8") as f:
                f.readline()
                for line in f: tmp.write(line)
            tmp_path = tmp.name
        df = pd.read_csv(tmp_path, sep="\t", quotechar='"', low_memory=False)
        os.unlink(tmp_path)
    except Exception as e:
        logging.error(f"  ChEMBL parse failed {fname}: {e}"); return pd.DataFrame(), 0
    df.columns = df.columns.str.strip().str.strip('"')
    if "Standard Type" in df.columns:
        df["Standard Type"] = df["Standard Type"].astype(str).str.strip().str.strip("'")
    df = df[df["Standard Type"].isin(["IC50", "Ki"])].copy()
    if "Standard Units" in df.columns:
        df["Standard Units"] = df["Standard Units"].astype(str).str.strip().str.strip("'")
    df = df[df["Standard Units"].str.lower() == "nm"].copy()
    if "Source ID" in df.columns:
        df["Source ID"] = df["Source ID"].astype(str).str.strip().str.strip("'")
    n_before = len(df)
    df       = df[df["Source ID"] != CHEMBL_MIRROR_SOURCE_ID].copy()
    n_mirror = n_before - len(df)
    rows = []
    for _, r in df.iterrows():
        smi = r.get("Smiles", "")
        if not isinstance(smi, str) or not smi.strip(): continue
        try:    nM = float(r.get("Standard Value", np.nan))
        except: continue
        if np.isnan(nM) or nM <= 0: continue
        qual  = str(r.get("Standard Relation", "=")).strip().strip("'")
        qual  = qual if qual in ("=", ">", "<", ">=", "<=") else "="
        mtype = str(r.get("Standard Type", "")).strip().strip("'")
        if mtype not in ("IC50", "Ki"): continue
        tid     = str(r.get("Target ChEMBL ID", "")).strip().strip("'")
        isoform = CHEMBL_TO_ISOFORM.get(tid, iso_hint)
        rows.append(_row(smi, nM, nM, "nM", 1, qual, mtype,
                         isoform or "PAD4", tid, "ChEMBL", aid, fname,
                         np.nan, np.nan, meta,
                         doi=str(r.get("DOI", r.get("Document DOI", ""))).strip(),
                         pmid=str(r.get("PMID", r.get("Pubmed ID", ""))).strip(),
                         bioassay_name=str(r.get("Assay Description", "")).strip(),
                         raw_unit="nM"))
    return pd.DataFrame(rows), n_mirror

def _classify_format(aid: str) -> str:
    if aid in FORMAT_A_AIDS: return "A"
    if aid in FORMAT_B_AIDS: return "B"
    if aid in FORMAT_C_AIDS: return "C"
    if aid in FORMAT_D_AIDS: return "D"
    if aid in FORMAT_E_AIDS: return "E"
    if aid in FORMAT_F_AIDS: return "F"
    return "?"

def _find_input_files(input_dir: str) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Find PubChem CSVs, BindingDB TSVs, and ChEMBL TSVs in input_dir.

    v17.1: searches both flat and nested layouts to support both the legacy
    single-directory layout and the PAD4_BENCH repo layout
    (data/raw/{pubchem,bindingdb,chembl}/).
    """
    p = Path(input_dir)
    pubchem  = sorted(set(list(p.glob("AID_*_datatable_all.csv")) +
                          list(p.glob("**/AID_*_datatable_all.csv"))))
    bindingdb = sorted(set(list(p.glob("BindingDB_PAD*.tsv")) +
                           list(p.glob("**/BindingDB_PAD*.tsv"))))
    chembl    = sorted(set(list(p.glob("CHEMBL_PAD*.tsv")) +
                           list(p.glob("**/CHEMBL_PAD*.tsv"))))
    return pubchem, bindingdb, chembl

def stage1_load(input_dir: str) -> Tuple[pd.DataFrame, List[dict], int]:
    logging.info("\n" + "═" * 60)
    logging.info("STAGE 1 — Data Loading & Source Harmonization")
    logging.info("═" * 60)
    all_dfs: List[pd.DataFrame] = []
    manifest: List[dict]        = []
    n_mirror = 0

    pubchem_files, bindingdb_files, chembl_files = _find_input_files(input_dir)
    logging.info(f"  Discovered: {len(pubchem_files)} PubChem CSVs, "
                 f"{len(bindingdb_files)} BindingDB TSVs, "
                 f"{len(chembl_files)} ChEMBL TSVs")

    for fp in tqdm(pubchem_files, desc="  PubChem CSVs", leave=False):
        aid = fp.stem.replace("_datatable_all", "")
        fmt = _classify_format(aid)
        try:
            parsers = {"A":_parse_a,"B":_parse_b,"C":_parse_c,
                       "D":_parse_d,"E":_parse_e,"F":_parse_f}
            if fmt not in parsers:
                logging.warning(f"  Unknown AID {aid} — skipped"); continue
            df = parsers[fmt](str(fp), aid)
        except Exception as e:
            logging.error(f"  FAILED {fp.name}: {e}"); df = pd.DataFrame()
        manifest.append({"filename": fp.name, "path": str(fp.resolve()),
                         "rows": len(df), "format": fmt, "source": "PubChem"})
        if len(df): all_dfs.append(df)

    for fp in tqdm(bindingdb_files, desc="  BindingDB", leave=False):
        try:    df = _parse_bindingdb(str(fp))
        except Exception as e:
            logging.error(f"  FAILED {fp.name}: {e}"); df = pd.DataFrame()
        manifest.append({"filename": fp.name, "path": str(fp.resolve()),
                         "rows": len(df), "format": "BindingDB", "source": "BindingDB"})
        if len(df): all_dfs.append(df)

    for fp in tqdm(chembl_files, desc="  ChEMBL", leave=False):
        try:    df, n_m = _parse_chembl(str(fp))
        except Exception as e:
            logging.error(f"  FAILED {fp.name}: {e}"); df, n_m = pd.DataFrame(), 0
        n_mirror += n_m
        manifest.append({"filename": fp.name, "path": str(fp.resolve()),
                         "rows": len(df), "format": "ChEMBL", "source": "ChEMBL"})
        if len(df): all_dfs.append(df)

    if not all_dfs:
        raise RuntimeError(
            f"No data loaded from {input_dir}. Expected layout: "
            "<dir>/{AID_*_datatable_all.csv,BindingDB_PAD*.tsv,CHEMBL_PAD*.tsv} "
            "either flat or in pubchem/, bindingdb/, chembl/ subdirectories."
        )
    combined = pd.concat(all_dfs, ignore_index=True)
    for col in ("is_artifact_high","is_artifact_low"):
        combined[col] = combined.get(col, False).fillna(False)
    for col in ("doi","pmid","bioassay_name","raw_unit_string"):
        if col not in combined.columns: combined[col] = ""
    for col in ("has_dose_response","screening_concentration_uM"):
        if col not in combined.columns:
            combined[col] = False if col == "has_dose_response" else np.nan
    logging.info(_table([(k,f"{v:,}") for k,v in combined["source_db"].value_counts().items()], ["Source","Rows"]))
    log_stage("1_load", len(combined), len(combined),
              f"loaded {len(manifest)} files, {n_mirror:,} ChEMBL mirrors removed")
    return combined, manifest, n_mirror

# ══════════════════════════════════════════════════════════════════════
# STAGE 2 — CHEMICAL STRUCTURE STANDARDIZATION
# ══════════════════════════════════════════════════════════════════════
def _stereo_stripped_scaffold(mol: Chem.Mol) -> str:
    if mol is None: return ""
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        if scaf is None: return ""
        rw = Chem.RWMol(scaf)
        Chem.RemoveStereochemistry(rw)
        Chem.SanitizeMol(rw)
        return Chem.MolToSmiles(rw, isomericSmiles=False)
    except Exception: return ""

def _normalize_protonation(mol: Chem.Mol, smi: str) -> Tuple[Chem.Mol, str, bool]:
    """Apply pH 7.4 protonation. Uses dimorphite helper."""
    if mol is None: return mol, smi, False
    if _DIMORPHITE_OK:
        try:
            smi_ph = protonate_with_dimorphite(smi, ph=7.4)
            if smi_ph and smi_ph != smi:
                mol_ph = Chem.MolFromSmiles(smi_ph)
                if mol_ph is not None:
                    Chem.SanitizeMol(mol_ph)
                    return mol_ph, Chem.MolToSmiles(mol_ph, isomericSmiles=True), True
        except Exception as e:
            logging.debug(f"  Dimorphite fallback: {e}")
    try:
        mol_uc = _UNCHARGER.uncharge(mol)
        if mol_uc is None: return mol, smi, False
        Chem.SanitizeMol(mol_uc)
        smi_uc = Chem.MolToSmiles(mol_uc, isomericSmiles=True)
        return mol_uc, smi_uc, smi_uc != smi
    except Exception:
        return mol, smi, False

def _try_sanitize(mol: Chem.Mol) -> Tuple[Optional[Chem.Mol], str]:
    try:
        Chem.SanitizeMol(mol); return mol, "clean"
    except Exception: pass
    try:
        mol2 = Chem.RWMol(mol)
        Chem.SanitizeMol(mol2, SanitizeFlags.SANITIZE_PROPERTIES |
                         SanitizeFlags.SANITIZE_SYMMRINGS)
        Chem.SetAromaticity(mol2)
        return mol2, "partial_sanitization"
    except Exception: pass
    return None, "failed_kekulization"

def _standardize_one(raw_smi: str) -> dict:
    """Full rdMolStandardize pipeline."""
    out = dict(
        canonical_smiles=None, inchikey=None, inchikey_14=None, _mol=None,
        error_reason=None, structure_ok=False,
        salt_removed=False, num_fragments=1,
        n_chiral_centres=0, n_undefined_centres=0,
        stereo_flag="achiral", tautomer_hash="", steps_applied="",
        smiles_pre_ph=None, protonation_modified=False,
        structure_quality_flag="clean", structure_quality_score=1.0,
        stereo_has_ez=False, n_ez_specified=0,
    )
    if not isinstance(raw_smi, str) or not raw_smi.strip():
        out["error_reason"] = "empty_smiles"; return out
    smi = raw_smi.split("|")[0].strip()
    mol = Chem.MolFromSmiles(smi, sanitize=False)
    if mol is None:
        out["error_reason"] = "parse_failed"; return out
    steps: List[str] = []
    mol, quality_flag = _try_sanitize(mol)
    if quality_flag == "failed_kekulization":
        out.update(structure_quality_flag="failed_kekulization",
                   structure_quality_score=0.0, error_reason="failed_kekulization")
        return out
    if quality_flag == "partial_sanitization":
        out.update(structure_quality_flag="partial_sanitization",
                   structure_quality_score=0.7)
        steps.append(f"sanitize:{quality_flag}")
    out["num_fragments"] = len(Chem.GetMolFrags(mol))
    try:
        mol = _NORMALIZER.normalize(mol); steps.append("normalize")
    except Exception as e:
        logging.debug(f"  Normalize failed: {e}")
    try:
        mol = _REIONIZER.reionize(mol); steps.append("reionize")
    except Exception as e:
        logging.debug(f"  Reionize failed: {e}")
    try:
        mol_s = _LARGEST_FRAG.choose(mol)
        if out["num_fragments"] > 1:
            out["salt_removed"] = True; steps.append("salt_strip")
        mol = mol_s
    except Exception as e:
        out["error_reason"] = f"salt_strip:{str(e)[:60]}"; return out
    try:
        mol_n = _UNCHARGER.uncharge(mol)
        if Chem.GetFormalCharge(mol) != Chem.GetFormalCharge(mol_n): steps.append("neutralize")
        mol = mol_n
    except Exception as e:
        out["error_reason"] = f"neutralize:{str(e)[:60]}"; return out
    try:
        mol = rdMolStandardize.Cleanup(mol); steps.append("cleanup")
    except Exception as e:
        out["error_reason"] = f"cleanup:{str(e)[:60]}"; return out
    if quality_flag == "clean":
        try:
            mol_tau = _TAUTOMER_ENUM.Canonicalize(mol)
            test = Chem.RWMol(mol_tau)
            Chem.Kekulize(test, clearAromaticFlags=False)
            mol = mol_tau; steps.append("tautomer_canon")
        except Exception:
            out["structure_quality_flag"]  = "tautomer_kekulize_failed"
            out["structure_quality_score"] = min(out["structure_quality_score"], 0.85)
            steps.append("tautomer_skipped")
    for atom in mol.GetAtoms(): atom.SetIsotope(0)
    try:
        pre_ph_smi = Chem.MolToSmiles(mol, isomericSmiles=True)
        mol_ph, smi_ph, ph_mod = _normalize_protonation(mol, pre_ph_smi)
        if ph_mod: mol = mol_ph; steps.append("protonation_ph74")
        out["smiles_pre_ph"] = pre_ph_smi
        out["protonation_modified"] = ph_mod
    except Exception as e:
        out["error_reason"] = f"protonation:{str(e)[:60]}"; return out
    try:
        n_spec, n_unspec, has_ez, n_total, ez_spec = _get_stereo_info_v17(mol)
        out["n_chiral_centres"] = n_total
        out["n_undefined_centres"] = n_unspec
        if   n_total == 0:   out["stereo_flag"] = "achiral"
        elif n_unspec == 0:  out["stereo_flag"] = "defined"
        elif n_spec   == 0:  out["stereo_flag"] = "fully_undefined"
        else:                out["stereo_flag"] = "partial_undefined"
        out["stereo_has_ez"] = has_ez
        out["n_ez_specified"] = ez_spec
    except Exception:
        out["stereo_flag"] = "unknown"
    try:
        can_smi  = Chem.MolToSmiles(mol, isomericSmiles=True)
        inchi    = InchiMod.MolToInchi(mol)
        if inchi is None: out["error_reason"] = "inchi_failed"; return out
        inchikey = InchiMod.InchiToInchiKey(inchi)
        if inchikey is None: out["error_reason"] = "inchikey_failed"; return out
        out.update(
            canonical_smiles=can_smi, inchikey=inchikey,
            inchikey_14=inchikey[:14], _mol=mol, structure_ok=True,
            tautomer_hash=hashlib.md5(can_smi.encode()).hexdigest()[:12],
            steps_applied=";".join(steps) or "none",
        )
    except Exception as e:
        out["error_reason"] = f"identifier:{str(e)[:60]}"
    return out

def _mol_props(mol: Chem.Mol) -> dict:
    p: dict = {}
    for k, fn in [
        ("mw",          Descriptors.ExactMolWt),
        ("heavy_atoms", lambda m: m.GetNumHeavyAtoms()),
        ("logP",        Descriptors.MolLogP),
        ("tpsa",        Descriptors.TPSA),
        ("hbd",         Descriptors.NumHDonors),
        ("hba",         Descriptors.NumHAcceptors),
        ("rot_bonds",   Descriptors.NumRotatableBonds),
        ("arom_rings",  Descriptors.NumAromaticRings),
        ("ring_count",  Descriptors.RingCount),
        ("frac_sp3",    Descriptors.FractionCSP3),
        ("formal_charge", Chem.GetFormalCharge),
    ]:
        try:    p[k] = fn(mol)
        except: p[k] = np.nan
    try:    p["scaffold"] = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except: p["scaffold"] = ""
    p["stereo_stripped_scaffold"] = _stereo_stripped_scaffold(mol)
    try:    p["is_pains"] = _PAINS_CATALOG.HasMatch(mol)
    except: p["is_pains"] = False
    p["is_brenk"] = (_BRENK_CATALOG.HasMatch(mol) if _BRENK_OK and _BRENK_CATALOG else False)
    try:
        p["is_aggregator"] = any(mol.HasSubstructMatch(pat) for pat in _AGGREGATOR_PATS)
    except Exception: p["is_aggregator"] = False
    p["substructure_alert_count"] = int(p["is_pains"]) + int(p["is_brenk"]) + int(p["is_aggregator"])
    mw, lp, hbd, hba = (p.get("mw",np.nan), p.get("logP",np.nan),
                        p.get("hbd",np.nan), p.get("hba",np.nan))
    p["flag_mw_extreme"]   = bool(pd.notna(mw)  and mw   > DRUGLIKE_MW_MAX)
    p["flag_logp_extreme"] = bool(pd.notna(lp)  and not (DRUGLIKE_LOGP_MIN <= lp <= DRUGLIKE_LOGP_MAX))
    p["flag_hbd_extreme"]  = bool(pd.notna(hbd) and hbd  > DRUGLIKE_HBD_MAX)
    p["flag_hba_extreme"]  = bool(pd.notna(hba) and hba  > DRUGLIKE_HBA_MAX)
    p["n_druglike_flags"]  = sum([p["flag_mw_extreme"], p["flag_logp_extreme"],
                                  p["flag_hbd_extreme"], p["flag_hba_extreme"]])
    p["ro5_violations"]    = (int(pd.notna(mw) and mw>500) + int(pd.notna(lp) and lp>5) +
                              int(pd.notna(hbd) and hbd>5) + int(pd.notna(hba) and hba>10))
    p["lipinski_violations"] = p["ro5_violations"]
    p["mw_flag_heavy"]       = bool(pd.notna(mw) and mw > MW_FLAG_HEAVY_CUT)
    p["ro3_violations"] = (
        int(pd.notna(mw)  and mw  > 300) +
        int(pd.notna(lp)  and lp  > 3)   +
        int(pd.notna(hbd) and hbd > 3)   +
        int(pd.notna(hba) and hba > 3)
    )
    p["fsp3_flag"] = bool(pd.notna(p.get("frac_sp3")) and p["frac_sp3"] < 0.25)
    p["stereo_defined_flag"]       = None
    p["stereo_completeness_score"] = None
    return p

def _resolve_inchikey_conflicts(df: pd.DataFrame) -> pd.DataFrame:
    conflicts = df.groupby("inchikey")["canonical_smiles"].nunique()
    bad = conflicts[conflicts > 1]
    if bad.empty: logging.info("  No InChIKey→SMILES conflicts"); return df
    logging.warning(f"  {len(bad)} InChIKey conflicts — keeping lowest-charge SMILES")
    for ik in bad.index:
        mask = df["inchikey"] == ik
        smis = df.loc[mask, "canonical_smiles"].unique().tolist()
        def _score(s):
            m = Chem.MolFromSmiles(s)
            return (sum(abs(a.GetFormalCharge()) for a in m.GetAtoms()) if m else 999, len(s))
        df.loc[mask, "canonical_smiles"] = min(smis, key=_score)
    return df

def stage2_standardize(df: pd.DataFrame, n_workers: int = 1) -> pd.DataFrame:
    logging.info("\n" + "═" * 60)
    logging.info("STAGE 2 — Chemical Structure Standardization")
    logging.info("═" * 60)
    n_in = len(df); df = df.copy()
    df["original_smiles"] = df["raw_smiles"].astype(str)
    ph_method = "Dimorphite-DL" if _DIMORPHITE_OK else "RDKit-Uncharger-fallback"
    logging.info(f"  Standardizing {n_in:,} | pH method: {ph_method} | workers: {n_workers}")
    logging.info("  Pipeline: sanitize → normalize → reionize → salt_strip → uncharge → cleanup → tautomer → protonation")
    if n_workers > 1:
        from multiprocessing import Pool
        with Pool(processes=n_workers) as pool:
            result_list = pool.map(_standardize_one, df["raw_smiles"].tolist())
            results = pd.Series(result_list, index=df.index)
    else:
        results = df["raw_smiles"].apply(_standardize_one)
    for key in ["canonical_smiles","inchikey","inchikey_14","_mol","error_reason",
                "structure_ok","salt_removed","num_fragments","n_chiral_centres",
                "n_undefined_centres","stereo_flag","tautomer_hash","steps_applied",
                "smiles_pre_ph","protonation_modified",
                "structure_quality_flag","structure_quality_score",
                "stereo_has_ez","n_ez_specified"]:
        df[key] = results.apply(lambda r, k=key: r[k])
    df["standardized_smiles"] = df["canonical_smiles"]
    mask_keku = df["structure_quality_flag"] == "failed_kekulization"
    n_keku    = int(mask_keku.sum())
    n_other   = int((~df["structure_ok"] & ~mask_keku).sum())
    df        = df[df["structure_ok"]].copy()
    n_ph_mod  = int(df["protonation_modified"].sum())
    props_df = pd.DataFrame(df["_mol"].apply(_mol_props).tolist(), index=df.index)
    for col in props_df.columns: df[col] = props_df[col]
    df["stereo_defined_flag"] = df["stereo_flag"].isin(["achiral","defined"])
    df["stereo_completeness_score"] = df.apply(
        lambda r: 1.0 if r["stereo_flag"] in ("achiral","defined")
        else 0.0 if r["stereo_flag"] == "fully_undefined"
        else (1.0 - r["n_undefined_centres"]/max(r["n_chiral_centres"],1))
        if r["stereo_flag"] == "partial_undefined" else 0.5,
        axis=1
    ).round(3)
    df["filter_heavy"]  = df["heavy_atoms"] < 5
    df["filter_mw"]     = df["mw"] > 1000
    df["filter_passed"] = ~(df["filter_heavy"] | df["filter_mw"])
    df = _resolve_inchikey_conflicts(df)
    df.drop(columns=["_mol","filter_heavy","filter_mw"], inplace=True)
    stereo_dist = df["stereo_flag"].value_counts()
    logging.info(_table([(k,f"{v:,}",f"{v/len(df)*100:.1f}%") for k,v in stereo_dist.items()],
                        ["Stereo flag","Count","Pct"]))
    logging.info(_table([
        ("Clean structures",       f"{(df['structure_quality_flag']=='clean').sum():,}"),
        ("Partial sanitization",   f"{(df['structure_quality_flag']=='partial_sanitization').sum():,}"),
        ("Tautomer kekulize skip", f"{(df['structure_quality_flag']=='tautomer_kekulize_failed').sum():,}"),
        ("Kekulization hard fail", f"{n_keku:,}  (retained as flagged)"),
        ("Hard failures (dropped)",f"{n_other:,}"),
        ("Stereo fully defined",   f"{df['stereo_defined_flag'].sum():,} ({df['stereo_defined_flag'].mean()*100:.1f}%)"),
        ("E/Z stereo present",     f"{int(df.get('stereo_has_ez',pd.Series(False)).sum()):,}"),
        ("pH modified",            f"{n_ph_mod:,}"),
        ("Normalize applied",      f"{df['steps_applied'].str.contains('normalize').sum():,}"),
        ("Reionize applied",       f"{df['steps_applied'].str.contains('reionize').sum():,}"),
    ], ["Category","Count"]))
    log_stage("2_standardize", n_in, len(df),
              f"failed={n_in-len(df):,}, ph_mod={n_ph_mod:,}")
    return df

# ══════════════════════════════════════════════════════════════════════
# STAGE 2.5 — COVALENT WARHEAD DETECTION
# ══════════════════════════════════════════════════════════════════════
def _detect_warheads_one(smi: str) -> dict:
    out = {"is_covalent": False, "covalent_confidence": "none",
           "covalent_type": "", "warhead_count": 0,
           "is_reversible_covalent": False, "covalent_context_excluded": False}
    if not isinstance(smi, str) or not smi.strip(): return out
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return out
    context_excluded = any(mol.HasSubstructMatch(pat) for pat in _CONTEXT_EXCL if pat is not None)
    matched_high: List[str] = []; matched_medium: List[str] = []; matched_low: List[str] = []
    for wname, pat in _WARHEAD_HIGH.items():
        try:
            if mol.HasSubstructMatch(pat): matched_high.append(wname)
        except Exception: continue
    for wname, pat in _WARHEAD_MEDIUM.items():
        try:
            if mol.HasSubstructMatch(pat): matched_medium.append(wname)
        except Exception: continue
    for wname, pat in _WARHEAD_LOW.items():
        try:
            if mol.HasSubstructMatch(pat): matched_low.append(wname)
        except Exception: continue
    if context_excluded and matched_high:
        matched_medium = matched_high + matched_medium; matched_high = []
        out["covalent_context_excluded"] = True
    NITRILE_WARHEADS = {"activated_nitrile","cyanamide"}
    all_matched = matched_high + matched_medium + matched_low
    if all_matched and all(w in NITRILE_WARHEADS for w in all_matched) and context_excluded:
        return out
    if not all_matched: return out
    confidence = "high" if matched_high else "medium" if matched_medium else "low"
    out.update(is_covalent=True, covalent_confidence=confidence,
               covalent_type=",".join(all_matched), warhead_count=len(all_matched),
               is_reversible_covalent=all(w in REVERSIBLE_COVALENT_WARHEADS for w in all_matched))
    return out

def stage2_5_covalent(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("\n" + "═" * 60)
    logging.info("STAGE 2.5 — Covalent Warhead Detection")
    logging.info("═" * 60)
    n_in = len(df); df = df.copy()
    results = df["canonical_smiles"].apply(_detect_warheads_one)
    for col in ["is_covalent","covalent_confidence","covalent_type",
                "warhead_count","is_reversible_covalent","covalent_context_excluded"]:
        df[col] = results.apply(lambda r, c=col: r[c])
    n_high = int((df["covalent_confidence"]=="high").sum())
    n_med  = int((df["covalent_confidence"]=="medium").sum())
    n_low  = int((df["covalent_confidence"]=="low").sum())
    n_cov  = int(df["is_covalent"].sum())
    n_excl = int(df["covalent_context_excluded"].sum())
    logging.info(_table([
        ("HIGH covalent",   f"{n_high:,}  ({n_high/max(n_in,1)*100:.1f}%)"),
        ("MEDIUM covalent", f"{n_med:,}  ({n_med/max(n_in,1)*100:.1f}%)"),
        ("LOW covalent",    f"{n_low:,}  ({n_low/max(n_in,1)*100:.1f}%)"),
        ("Context-excluded",f"{n_excl:,}"),
        ("Non-covalent",    f"{n_in-n_cov:,}  ({(n_in-n_cov)/max(n_in,1)*100:.1f}%)"),
    ], ["Category","Count"]))
    log_stage("2.5_covalent", n_in, n_in,
              f"cov={n_cov:,} (H={n_high},M={n_med},L={n_low}), excl={n_excl}")
    return df

# ══════════════════════════════════════════════════════════════════════
# STAGE 2.6 — STEREO CROSS-SOURCE RESOLUTION
# ══════════════════════════════════════════════════════════════════════
# v17.1: Consolidated from two separate implementations in v17.0.
# Combines:
#   • False-undefined cleanup (from v17.1 enhanced version)
#   • Cross-source resolution via InChIKey_14 with source-weighted voting
#     (from v17 base version; sets stereo_cross_inchikey + source_db)
#   • E/Z stereo census (from both)
#   • Confidence scoring + resolution-method tracking
# ══════════════════════════════════════════════════════════════════════

def _is_non_stereogenic_center(mol: Chem.Mol, atom_idx: int) -> bool:
    """Check if a chiral center is actually non-stereogenic due to symmetry."""
    try:
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED:
            neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
            if len(neighbors) < 4: return True
            orig_smi = Chem.MolToSmiles(mol, isomericSmiles=True)
            for i in range(len(neighbors)):
                for j in range(i+1, len(neighbors)):
                    tmp = Chem.RWMol(mol)
                    tmp.GetAtomWithIdx(neighbors[i]).SetIsotope(99)
                    tmp.GetAtomWithIdx(neighbors[j]).SetIsotope(100)
                    tmp_smi = Chem.MolToSmiles(tmp, isomericSmiles=True)
                    tmp.GetAtomWithIdx(neighbors[i]).SetIsotope(0)
                    tmp.GetAtomWithIdx(neighbors[j]).SetIsotope(0)
                    if orig_smi == tmp_smi:
                        return True
        return False
    except Exception:
        return False

def stage2_6_stereo_resolution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stage 2.6 — Stereo Cross-Source Resolution (consolidated v17.1).

    Three-pass resolution:
      1. False-undefined cleanup (symmetry-equivalent centers reclassified achiral)
      2. Exact cross-source via InChIKey_14 (high confidence, 0.95)
      3. Scaffold-match fallback (lower confidence, 0.65)

    Always populates:
      stereo_cross_resolved, stereo_cross_inchikey, stereo_cross_smiles,
      stereo_cross_source_db, stereo_resolution_confidence, stereo_resolved_by,
      stereo_has_ez, n_ez_specified
    """
    logging.info("\n" + "═" * 60)
    logging.info("STAGE 2.6 — Stereo Cross-Source Resolution [v17.1 consolidated]")
    logging.info("═" * 60)
    n_in = len(df)
    df = df.copy()

    # Initialise columns with deterministic defaults
    df["stereo_cross_resolved"]         = False
    df["stereo_cross_inchikey"]         = ""
    df["stereo_cross_smiles"]           = ""
    df["stereo_cross_source_db"]        = ""
    df["stereo_resolution_confidence"]  = 0.0
    df["stereo_resolved_by"]            = "none"

    # ── E/Z stereo census ────────────────────────────────────────────
    # If Stage 2 already populated these, keep them; otherwise compute.
    if "stereo_has_ez" not in df.columns or "n_ez_specified" not in df.columns:
        logging.info("  Computing E/Z stereo census…")
        has_ez_list, n_ez_spec_list = [], []
        for smi in df["canonical_smiles"]:
            mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
            _, _, has_ez, _, ez_spec = _get_stereo_info_v17(mol)
            has_ez_list.append(has_ez)
            n_ez_spec_list.append(ez_spec)
        df["stereo_has_ez"]  = has_ez_list
        df["n_ez_specified"] = n_ez_spec_list
    n_ez_total = int(df["stereo_has_ez"].fillna(False).sum())
    logging.info(f"  E/Z stereo present: {n_ez_total:,} compounds "
                 f"({n_ez_total/max(n_in,1)*100:.1f}%)")

    # ── Pass 1: False-undefined cleanup ──────────────────────────────
    logging.info("  Pass 1: filtering false undefined stereocenters…")
    n_cleaned = 0
    for idx in df.index:
        flag = df.at[idx, "stereo_flag"]
        if flag not in ("fully_undefined", "partial_undefined"):
            continue
        smi = df.at[idx, "canonical_smiles"]
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            continue
        try:
            centres = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        except Exception:
            continue
        false_undef = [aidx for aidx, _ in centres
                       if _is_non_stereogenic_center(mol, aidx)]
        if not false_undef:
            continue
        new_undef = max(0, int(df.at[idx, "n_undefined_centres"]) - len(false_undef))
        df.at[idx, "n_undefined_centres"] = new_undef
        if new_undef == 0:
            df.at[idx, "stereo_flag"]          = "achiral"
            df.at[idx, "stereo_defined_flag"]  = True
            n_cleaned += 1
    logging.info(f"  Reclassified {n_cleaned:,} false-undefined → achiral")

    if not STEREO_CROSS_RESOLVE:
        logging.info("  Cross-resolution disabled (STEREO_CROSS_RESOLVE=False)")
        log_stage("2.6_stereo_resolution", n_in, n_in,
                  f"cleanup={n_cleaned:,}, cross-resolve skipped")
        return df

    if "inchikey_14" not in df.columns:
        logging.warning("  inchikey_14 missing — cross-resolution skipped")
        log_stage("2.6_stereo_resolution", n_in, n_in,
                  f"cleanup={n_cleaned:,}, no inchikey_14")
        return df

    # ── Pass 2: Exact InChIKey_14 cross-source resolution ────────────
    logging.info("  Pass 2: exact InChIKey_14 cross-source matching…")
    defined_mask = df["stereo_flag"] == "defined"
    undef_mask   = df["stereo_flag"].isin(["fully_undefined", "partial_undefined"])
    defined_df = df.loc[defined_mask,
                        ["inchikey_14", "inchikey", "canonical_smiles", "source_db"]].copy()
    undef_idx  = df.index[undef_mask]

    n_exact = 0
    n_conflicting = 0
    if len(defined_df) > 0 and len(undef_idx) > 0:
        # Source-weighted majority vote per inchikey_14
        defined_df["_src_w"] = defined_df["source_db"].apply(
            lambda s: _SRC_SCORE.get(str(s), 0.60) if isinstance(s, str) else 0.60
        )
        # Build undefined → indices map
        undef_ik14 = (
            df.loc[undef_idx, ["inchikey_14"]]
              .reset_index()
              .groupby("inchikey_14")["index"]
              .apply(list)
              .to_dict()
        )
        for ik14, grp_def in defined_df.groupby("inchikey_14"):
            if ik14 not in undef_ik14:
                continue
            target_idx = undef_ik14[ik14]
            ik_votes = (
                grp_def
                .groupby("inchikey")
                .agg(count     =("inchikey",      "count"),
                     src_w_sum =("_src_w",         "sum"),
                     smiles    =("canonical_smiles","first"),
                     src_db    =("source_db",       "first"))
                .reset_index()
            )
            ik_votes["score"] = ik_votes["count"] + ik_votes["src_w_sum"]
            ik_votes = ik_votes.sort_values("score", ascending=False)
            conflicting = len(ik_votes) > 1
            best = ik_votes.iloc[0]
            src_label = f"conflicting({best['src_db']})" if conflicting else best["src_db"]
            for ti in target_idx:
                df.at[ti, "stereo_cross_resolved"]        = True
                df.at[ti, "stereo_cross_inchikey"]        = best["inchikey"]
                df.at[ti, "stereo_cross_smiles"]          = best["smiles"]
                df.at[ti, "stereo_cross_source_db"]       = src_label
                df.at[ti, "stereo_resolution_confidence"] = 0.95
                df.at[ti, "stereo_resolved_by"]           = "exact_ik14"
                n_exact += 1
            if conflicting:
                n_conflicting += 1

    # ── Pass 3: Scaffold-match fallback ──────────────────────────────
    logging.info("  Pass 3: scaffold-match fallback…")
    n_scaf = 0
    if len(defined_df) > 0 and len(undef_idx) > 0 and "scaffold" in df.columns:
        scaf_uniq = (
            df.loc[defined_mask]
              .groupby("scaffold")
              .agg(smiles=("canonical_smiles", "first"),
                   src_db=("source_db", "first"),
                   ik    =("inchikey", "first"),
                   count =("inchikey", "count"))
              .reset_index()
        )
        # Only use scaffolds with a single defined exemplar (low ambiguity)
        scaf_unique = scaf_uniq[scaf_uniq["count"] == 1].set_index("scaffold")
        for ti in undef_idx:
            if df.at[ti, "stereo_cross_resolved"]:
                continue
            scaf_u = df.at[ti, "scaffold"]
            if not isinstance(scaf_u, str) or scaf_u not in scaf_unique.index:
                continue
            rec = scaf_unique.loc[scaf_u]
            df.at[ti, "stereo_cross_resolved"]        = True
            df.at[ti, "stereo_cross_inchikey"]        = rec["ik"]
            df.at[ti, "stereo_cross_smiles"]          = rec["smiles"]
            df.at[ti, "stereo_cross_source_db"]       = str(rec["src_db"])
            df.at[ti, "stereo_resolution_confidence"] = 0.65
            df.at[ti, "stereo_resolved_by"]           = "scaffold_match"
            n_scaf += 1

    n_total_resolved = n_exact + n_scaf
    pct_resolved = n_total_resolved / max(len(undef_idx), 1) * 100
    logging.info(_table([
        ("Undefined records eligible",  f"{len(undef_idx):,}"),
        ("Resolved by exact ik14",      f"{n_exact:,}"),
        ("Resolved by scaffold match",  f"{n_scaf:,}"),
        ("Total resolved",              f"{n_total_resolved:,}  ({pct_resolved:.1f}%)"),
        ("Conflicting ik14 groups",     f"{n_conflicting:,}"),
        ("Remaining unresolvable",      f"{len(undef_idx) - n_total_resolved:,}"),
    ], ["Metric", "Value"]))

    log_stage("2.6_stereo_resolution", n_in, n_in,
              f"cleanup={n_cleaned:,}, exact={n_exact:,}, "
              f"scaffold={n_scaf:,}, conflicting={n_conflicting:,}, "
              f"ez_detected={n_ez_total:,}")
    return df

# ══════════════════════════════════════════════════════════════════════
# STAGE 2.7 — UNIT NORMALIZATION & VALIDATION
# ══════════════════════════════════════════════════════════════════════
def _parse_unit(raw_unit: str, value_nM: float) -> Tuple[str, float, float]:
    unit_clean = str(raw_unit).strip().lower().replace(" ", "")
    if not unit_clean or unit_clean in ("nan","none","","unknown"):
        if pd.notna(value_nM) and 0.001 <= value_nM <= 1e6:
            return "inferred_nm", value_nM, UNIT_CONFIDENCE["inferred_nm"]
        return "unknown", value_nM, UNIT_CONFIDENCE["unknown"]
    for key in UNIT_CONVERSION:
        if unit_clean == key or unit_clean.replace("molar","") == key.replace("molar",""):
            conf = UNIT_CONFIDENCE.get(key, 0.80)
            if key == "nm" and pd.notna(value_nM) and not (0.001 <= value_nM <= 1e7):
                conf = min(conf, 0.50)
            return key, value_nM, conf
    return "unknown", value_nM, UNIT_CONFIDENCE["unknown"]

def stage2_7_unit_normalization(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("\n" + "═" * 60)
    logging.info("STAGE 2.7 — Unit Normalization & Validation")
    logging.info("═" * 60)
    n_in = len(df); df = df.copy()
    unit_orig = []; unit_std = []; unit_conf = []; unit_flag_list = []
    for _, row in df.iterrows():
        raw_unit = str(row.get("raw_unit_string","")).strip()
        val_nM   = row.get("activity_value_nM", np.nan)
        mtype    = row.get("measurement_type","")
        if mtype == "pct_inhibition":
            unit_orig.append("%"); unit_std.append("%"); unit_conf.append(1.0); unit_flag_list.append("ok")
            continue
        u_std, _, u_conf = _parse_unit(raw_unit, val_nM)
        unit_orig.append(raw_unit or "unknown"); unit_std.append(u_std); unit_conf.append(u_conf)
        if u_std == "unknown":
            flag = "unknown"
        elif u_std == "inferred_nm":
            flag = "inferred"
        elif pd.notna(val_nM) and not (0.001 <= val_nM <= 1e7):
            flag = "suspicious"
        else:
            flag = "ok"
        unit_flag_list.append(flag)
    df["activity_unit_original"]     = unit_orig
    df["activity_unit_standardized"] = unit_std
    df["unit_confidence_score"]      = unit_conf
    df["unit_flag"]                  = unit_flag_list
    mask_exact = (df["measurement_type"].isin(["IC50","Ki"]) &
                  (df["qualifier"] == "=") & (df["activity_value_nM"] > 0) &
                  df["unit_flag"].isin(["ok","inferred"]))
    df["pIC50_unit_validated"] = np.nan
    df.loc[mask_exact, "pIC50_unit_validated"] = 9.0 - np.log10(df.loc[mask_exact, "activity_value_nM"])
    flag_counts = pd.Series(unit_flag_list).value_counts()
    logging.info(_table([(k,f"{v:,}",f"{v/n_in*100:.1f}%") for k,v in flag_counts.items()],
                        ["Unit flag","Count","Pct"]))
    log_stage("2.7_units", n_in, n_in,
              f"ok={flag_counts.get('ok',0):,}, suspicious={flag_counts.get('suspicious',0):,}")
    return df

# ══════════════════════════════════════════════════════════════════════
# STAGE 2.8 — MOLECULAR COMPLEXITY & DRUGLIKENESS SCORING
# ══════════════════════════════════════════════════════════════════════
def _complexity_score(mol: Chem.Mol) -> float:
    if mol is None: return 0.0
    try:
        n_rings    = Descriptors.RingCount(mol)
        n_arom     = Descriptors.NumAromaticRings(mol)
        n_rot      = Descriptors.NumRotatableBonds(mol)
        fsp3       = Descriptors.FractionCSP3(mol)
        n_chiral   = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        ring_comp  = min(n_rings / 6.0, 1.0) * 0.30
        arom_comp  = min(n_arom / 4.0, 1.0) * 0.15
        rot_pen    = min(n_rot  / 12.0, 1.0) * 0.10
        sp3_comp   = float(fsp3) * 0.25
        chiral_comp= min(n_chiral / 4.0, 1.0) * 0.20
        return round(float(ring_comp + arom_comp - rot_pen + sp3_comp + chiral_comp), 4)
    except Exception:
        return 0.0

def stage2_8_complexity(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("\n" + "═" * 60)
    logging.info("STAGE 2.8 — Molecular Complexity & Druglikeness Scoring")
    logging.info("═" * 60)
    n_in = len(df); df = df.copy()
    complexity = []
    np_like    = []
    for smi in df["canonical_smiles"]:
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        cs = _complexity_score(mol)
        complexity.append(cs)
        npl = False
        if mol is not None:
            try:
                fsp3     = Descriptors.FractionCSP3(mol)
                n_rings  = Descriptors.RingCount(mol)
                n_chiral = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
                npl      = bool(fsp3 > 0.5 and n_rings >= 3 and n_chiral >= 2)
            except Exception:
                pass
        np_like.append(npl)
    df["complexity_score"]              = complexity
    df["natural_product_likeness_flag"] = np_like
    if "fsp3_flag" not in df.columns:
        df["fsp3_flag"] = df.get("frac_sp3", pd.Series(0.5, index=df.index)).fillna(0.5) < 0.25
    if "ro3_violations" not in df.columns:
        df["ro3_violations"] = 0
    logging.info(_table([
        ("Complexity mean",                f"{df['complexity_score'].mean():.3f}"),
        ("Complexity > 0.6",               f"{(df['complexity_score']>0.6).sum():,}"),
        ("NP-like (fsp3>0.5, rings≥3, chiral≥2)", f"{df['natural_product_likeness_flag'].sum():,}"),
        ("Low sp3 (fsp3<0.25)",            f"{df.get('fsp3_flag',pd.Series(False)).sum():,}"),
    ], ["Metric","Value"]))
    log_stage("2.8_complexity", n_in, n_in)
    return df

# ══════════════════════════════════════════════════════════════════════
# STAGE 2.9 — STEREO QUALITY REMEDIATION
# v17.1: deduplicated worker functions; behavior unchanged.
# ══════════════════════════════════════════════════════════════════════
def _enumerate_stereo_canonical(smi: str, max_isomers: int = 4) -> List[str]:
    """Enumerate defined stereoisomers for a compound with undefined centres."""
    mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    if mol is None:
        return []
    try:
        from rdkit.Chem import EnumerateStereoisomers
        opts = EnumerateStereoisomers.StereoEnumerationOptions()
        opts.maxIsomers = max_isomers
        opts.tryEmbedding = False
        opts.onlyUnassigned = True
        isomers = list(EnumerateStereoisomers.EnumerateStereoisomers(mol, options=opts))
        return [Chem.MolToSmiles(m, isomericSmiles=True) for m in isomers if m is not None]
    except Exception:
        return []

def _enumerate_stereo_worker(args):
    """Top-level picklable worker for parallel stereo enumeration."""
    idx, smi = args
    return idx, _enumerate_stereo_canonical(smi, max_isomers=STEREO_ENUM_MAX_ISOMERS)

def stage2_9_stereo_remediation(df: pd.DataFrame, n_workers: int = 1) -> pd.DataFrame:
    """Stage 2.9 — Stereo Quality Remediation (parallel + FindPotentialStereo)."""
    logging.info("\n" + "═" * 60)
    logging.info("STAGE 2.9 — Stereo Quality Remediation")
    logging.info("═" * 60)
    n_in = len(df)
    df   = df.copy()

    df["stereo_remediation_status"] = "resolved"
    df["stereo_isomer_count"]       = 1
    df["stereo_isomer_smiles"]      = ""
    df["stereo_modeling_penalty"]   = 0.0

    if "stereo_has_ez" not in df.columns:
        has_ez_list = []
        for smi in df["canonical_smiles"]:
            mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
            _, _, has_ez, _, _ = _get_stereo_info_v17(mol)
            has_ez_list.append(has_ez)
        df["stereo_has_ez"] = has_ez_list

    stereo_flag = df.get("stereo_flag",
                          pd.Series("achiral", index=df.index))
    n_undef     = df.get("n_undefined_centres",
                          pd.Series(0, index=df.index)).fillna(0).astype(int)
    n_chiral    = df.get("n_chiral_centres",
                          pd.Series(0, index=df.index)).fillna(0).astype(int)
    cross_res   = df.get("stereo_cross_resolved",
                          pd.Series(False, index=df.index)).fillna(False)

    is_resolved    = stereo_flag.isin(["achiral", "defined"])
    is_fully_undef = stereo_flag == "fully_undefined"
    is_partial     = stereo_flag == "partial_undefined"
    is_enumerable  = (is_partial | is_fully_undef) & (n_undef <= 3) & (n_undef > 0)
    is_complex     = (is_partial | is_fully_undef) & (n_undef > 3)

    df.loc[is_enumerable, "stereo_remediation_status"] = "enumerable"
    df.loc[is_complex,    "stereo_remediation_status"] = "complex"
    df.loc[cross_res & ~is_resolved, "stereo_remediation_status"] = "cross_resolved"

    df["stereo_isomer_count"] = np.where(
        n_undef > 0,
        np.minimum(2 ** n_undef.astype(float), 64).astype(int),
        1,
    )

    penalty = pd.Series(0.0, index=df.index)
    partial_frac = np.minimum(
        n_undef[is_partial].astype(float)
        / np.maximum(n_chiral[is_partial].astype(float), 1.0),
        1.0,
    )
    penalty[is_partial]     = (partial_frac * 0.30).clip(0.0, 0.30)
    penalty[is_fully_undef] = 0.50
    penalty[is_complex]     = 0.55
    penalty[cross_res & ~is_resolved] = 0.10
    df["stereo_modeling_penalty"] = penalty.round(3)

    pic50_col = next(
        (c for c in ("pIC50", "pIC50_unit_validated") if c in df.columns), None
    )
    if pic50_col:
        high_value = is_enumerable & df[pic50_col].fillna(0.0).gt(6.0)
    else:
        high_value = is_enumerable

    enum_targets = [(idx, df.at[idx, "canonical_smiles"])
                    for idx in df.loc[high_value].index]

    n_enumerated = 0
    if enum_targets:
        workers = min(n_workers, STEREO_ENUM_WORKERS, len(enum_targets))
        if workers > 1:
            from multiprocessing import Pool
            with Pool(processes=workers) as pool:
                results = pool.map(_enumerate_stereo_worker, enum_targets)
        else:
            results = [_enumerate_stereo_worker(a) for a in enum_targets]
        for idx, isomers in results:
            if isomers:
                df.at[idx, "stereo_isomer_smiles"] = ";".join(isomers[:4])
                n_enumerated += 1

    if pic50_col:
        low_act = df[pic50_col].fillna(0.0).lt(5.0)
        is_unresolvable = is_fully_undef & low_act & ~cross_res
        df.loc[is_unresolvable, "stereo_remediation_status"] = "unresolvable"

    status_counts = df["stereo_remediation_status"].value_counts()
    n_cross_res   = int(cross_res.sum())
    logging.info(_table([
        (k, f"{v:,}", f"{v/n_in*100:.1f}%")
        for k, v in status_counts.items()
    ], ["Stereo Status", "Count", "Pct"]))
    logging.info(_table([
        ("Cross-resolved (from Stage 2.6)",  f"{n_cross_res:,}"),
        ("Isomers enumerated (high-value)",  f"{n_enumerated:,}"),
        ("E/Z stereo present",
         f"{int(df.get('stereo_has_ez', pd.Series(False)).sum()):,}"),
        ("Mean modeling penalty",            f"{df['stereo_modeling_penalty'].mean():.3f}"),
        ("Compounds with penalty > 0",
         f"{int((df['stereo_modeling_penalty']>0).sum()):,} "
         f"({(df['stereo_modeling_penalty']>0).mean()*100:.1f}%)"),
    ], ["Metric", "Value"]))

    log_stage("2.9_stereo_remediation", n_in, n_in,
              f"cross_resolved={n_cross_res:,}, "
              f"enumerable={status_counts.get('enumerable',0):,}, "
              f"complex={status_counts.get('complex',0):,}, "
              f"unresolvable={status_counts.get('unresolvable',0):,}, "
              f"isomers_enumerated={n_enumerated:,}")
    return df

# ══════════════════════════════════════════════════════════════════════
# STAGE 3 — ACTIVITY STANDARDIZATION
# ══════════════════════════════════════════════════════════════════════
def stage3_activity(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("\n" + "═" * 60)
    logging.info("STAGE 3 — Activity Standardization")
    logging.info("═" * 60)
    n_in = len(df); df = df.copy()
    df["qualifier"] = df["qualifier"].fillna("=").astype(str).str.strip()
    df.loc[~df["qualifier"].isin(["=",">","<",">=","<="]), "qualifier"] = "="
    df["is_censored"] = df["qualifier"].isin([">","<",">=","<="])
    mask_exact = (~df["is_censored"] & (df["activity_value_nM"]>0) &
                  df["measurement_type"].isin(["IC50","Ki"]))
    df["pIC50"] = np.nan
    df.loc[mask_exact,"pIC50"] = 9.0 - np.log10(df.loc[mask_exact,"activity_value_nM"])
    in_range = df["pIC50"].between(PACTIVITY_VALID_MIN, PACTIVITY_VALID_MAX)
    df["activity_outlier"] = mask_exact & ~in_range
    n_out = int(df["activity_outlier"].sum())
    if n_out: logging.warning(f"  {n_out:,} pIC50 outliers flagged")
    df["pIC50_lower"] = np.nan; df["pIC50_upper"] = np.nan
    df["activity_lower_bound_nM"] = np.nan; df["activity_upper_bound_nM"] = np.nan
    mask_gt = df["qualifier"].isin([">",">="]) & (df["activity_value_nM"]>0)
    mask_lt = df["qualifier"].isin(["<","<="]) & (df["activity_value_nM"]>0)
    df.loc[mask_gt,"pIC50_upper"]            = 9.0 - np.log10(df.loc[mask_gt,"activity_value_nM"])
    df.loc[mask_gt,"activity_lower_bound_nM"] = df.loc[mask_gt,"activity_value_nM"]
    df.loc[mask_gt,"activity_upper_bound_nM"] = np.inf
    df.loc[mask_lt,"pIC50_lower"]            = 9.0 - np.log10(df.loc[mask_lt,"activity_value_nM"])
    df.loc[mask_lt,"activity_upper_bound_nM"] = df.loc[mask_lt,"activity_value_nM"]
    df.loc[mask_lt,"activity_lower_bound_nM"] = 0.0
    df["activity_task"] = df["measurement_type"].map({
        "IC50": "ic50_regression", "Ki": "ki_regression",
        "pct_inhibition": "classification",
    }).fillna("ic50_regression")
    log_stage("3_activity", n_in, n_in,
              f"outlier_flags={n_out:,}, censored={int(df['is_censored'].sum()):,}")
    return df

# ══════════════════════════════════════════════════════════════════════
# STAGE 3.5 — % INHIBITION STRICT PROCESSING
# ══════════════════════════════════════════════════════════════════════
def stage3_5_pct_inhibition(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("\n" + "═" * 60)
    logging.info("STAGE 3.5 — % Inhibition Strict Processing")
    logging.info("═" * 60)
    n_in = len(df); df = df.copy()
    mask_pct = df["measurement_type"] == "pct_inhibition"
    n_pct    = int(mask_pct.sum())
    df["is_weak_label"]                  = False
    df["pct_inhibition_normalized_flag"] = False
    df["fidelity_level"]                 = "T1_standard"
    if n_pct > 0:
        df.loc[mask_pct, "is_weak_label"]    = True
        df.loc[mask_pct, "fidelity_level"]   = "T3_weak"
        sc      = df.loc[mask_pct, "screening_concentration_uM"]
        has_conc = sc.notna() & (sc > 0)
        n_conc   = int(has_conc.sum())
        if n_conc > 0:
            df.loc[has_conc[has_conc].index, "pct_inhibition_normalized_flag"] = True
        logging.info(f"  {n_conc:,} pct_inhibition records with concentration data")
        df.loc[mask_pct, "binary_label_pct"] = np.where(
            df.loc[mask_pct, "pct_inhibition"] >= HTS_ACTIVE_THRESHOLD, 1, 0
        )
    log_stage("3.5_pct_inhibition", n_in, n_in, f"pct_rows={n_pct:,}")
    return df

# ══════════════════════════════════════════════════════════════════════
# STAGE 4 — ASSAY ANNOTATION
# ══════════════════════════════════════════════════════════════════════
def _detect_time_dependence(text: str) -> Tuple[bool, bool]:
    if not isinstance(text, str) or not text.strip(): return False, False
    has_td     = bool(_TD_PATTERN.search(text))
    has_preinc = bool(re.search(r"preincubat", text, re.IGNORECASE))
    return has_td, has_preinc

def _compute_assay_context_score(row: pd.Series) -> Tuple[float, str, bool, str]:
    base  = ASSAY_CONFIDENCE.get(str(row.get("assay_category","unknown")), ASSAY_CONFIDENCE["unknown"])
    delta = 0.0
    has_dr = bool(row.get("has_dose_response", False))
    if has_dr: delta += ASSAY_CONTEXT_RULES["has_dose_response"]
    doi  = str(row.get("doi","")).strip()
    pmid = str(row.get("pmid","")).strip()
    if   doi  and doi  not in ("","nan","None"): delta += ASSAY_CONTEXT_RULES["has_doi"]
    elif pmid and pmid not in ("","nan","None"): delta += ASSAY_CONTEXT_RULES["has_pmid"]
    bname     = str(row.get("bioassay_name","")).lower()
    assay_cat = str(row.get("assay_category","")).lower()
    assay_type_v2 = "unknown"
    if any(k in bname or k in assay_cat for k in ["confirmatory","confirm","dose"]):
        assay_type_v2 = "confirmatory"; delta += ASSAY_CONTEXT_RULES["is_biochemical"]
    elif any(k in bname for k in ["biochem","enzyme","recombinant","purified"]):
        assay_type_v2 = "biochemical"; delta += ASSAY_CONTEXT_RULES["is_biochemical"]
    elif any(k in bname for k in ["cell","cellular","lysis","transfect"]):
        assay_type_v2 = "cell-based"; delta += ASSAY_CONTEXT_RULES["is_cell_based"]
    elif "hts" in assay_cat or "percent_inhibition" in assay_cat:
        assay_type_v2 = "HTS"; delta += ASSAY_CONTEXT_RULES["is_hts_penalty"]
    elif any(k in bname for k in ["rnai","sirna","shrna"]):
        assay_type_v2 = "HTS"; delta += ASSAY_CONTEXT_RULES["is_rnai"]
    uniprot      = str(row.get("target_uniprot","")).strip()
    organism     = UNIPROT_TO_ORGANISM.get(uniprot, "unknown")
    target_species = "human" if organism=="human" else ("mouse" if organism=="mouse" else "recombinant")
    if target_species != "human": delta += ASSAY_CONTEXT_RULES["is_non_human"]
    score = float(np.clip(base + delta, 0.05, 1.00))
    return score, assay_type_v2, has_dr, target_species

def stage4_annotate(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("\n" + "═" * 60)
    logging.info("STAGE 4 — Assay Annotation")
    logging.info("═" * 60)
    n_in = len(df); df = df.copy()
    df["assay_category"] = df["assay_type"].fillna("unknown")
    df.loc[df["source_id"] == "AID_492970", "assay_category"] = "biochemical_confirmatory"
    for aid in FORMAT_D_AIDS: df.loc[df["source_id"]==aid,"assay_category"] = "HTS"
    for aid in FORMAT_E_AIDS: df.loc[df["source_id"]==aid,"assay_category"] = "panel"
    df["organism"] = df["target_uniprot"].map(UNIPROT_TO_ORGANISM).fillna("recombinant")
    df["is_mouse"] = df["organism"] == "mouse"
    context_results = df.apply(_compute_assay_context_score, axis=1)
    df["assay_confidence_score"] = context_results.apply(lambda x: x[0])
    df["assay_type_v2"]          = context_results.apply(lambda x: x[1])
    df["dose_response_flag"]     = context_results.apply(lambda x: x[2])
    df["target_species"]         = context_results.apply(lambda x: x[3])
    df["assay_context_score"]    = df["assay_confidence_score"]
    df["literature_confidence"] = df.apply(
        lambda r: (1.0 if str(r.get("doi","")) not in ("","nan","None")
        else 0.6 if str(r.get("pmid","")) not in ("","nan","None")
        else 0.0), axis=1)
    text_col = (
        df.get("bioassay_name", pd.Series("",index=df.index)).fillna("").astype(str) + " " +
        df.get("assay_type",    pd.Series("",index=df.index)).fillna("").astype(str) + " " +
        df.get("source_id",     pd.Series("",index=df.index)).fillna("").astype(str)
    )
    td_results = text_col.apply(_detect_time_dependence)
    df["time_dependent"]      = td_results.apply(lambda x: x[0])
    df["preincubation_flag"]  = td_results.apply(lambda x: x[1])
    df["assay_covalent_flag"] = df["time_dependent"]
    if "is_covalent" in df.columns:
        df["covalent_consistent"] = (
            df["is_covalent"] & (df["time_dependent"] | df["preincubation_flag"]))
    else:
        df["covalent_consistent"] = False
    log_stage("4_annotate", n_in, n_in)
    return df

# ══════════════════════════════════════════════════════════════════════
# STAGE 5 — DEDUPLICATION
# ══════════════════════════════════════════════════════════════════════
def stage5_deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("\n" + "═" * 60)
    logging.info("STAGE 5 — Deduplication")
    logging.info("═" * 60)
    n_in = len(df); df = df.copy()
    val_col = df["activity_value_nM"].where(
        df["measurement_type"] != "pct_inhibition", df["pct_inhibition"])
    df["_dedup_val"] = val_col
    n_before = len(df)
    df = df.drop_duplicates(
        subset=["inchikey","source_id","measurement_type","qualifier","_dedup_val"],
        keep="first").copy()
    n_dups = n_before - len(df)
    df.drop(columns=["_dedup_val"], inplace=True)
    df["cross_source_dup"] = (
        df.duplicated(subset=["inchikey","measurement_type","qualifier","activity_value_nM"], keep=False) &
        df["measurement_type"].isin(["IC50","Ki"])
    )
    n_cross = int(df["cross_source_dup"].sum())
    log_stage("5_deduplicate", n_in, len(df),
              f"exact_removed={n_dups:,}, cross_source_flagged={n_cross:,}")
    return df

# ══════════════════════════════════════════════════════════════════════
# STAGE 6 — TIER ASSIGNMENT
# v17.1: validation-tier branch removed; AID 1805620 now flows through
# normal Format-B path into T1/T2/T3 like every other PubChem assay.
# ══════════════════════════════════════════════════════════════════════
def stage6_assign_tiers(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logging.info("\n" + "═" * 60)
    logging.info("STAGE 6 — Tier Assignment")
    logging.info("═" * 60)
    n_in = len(df); df = df.copy()
    ki_df = df[df["measurement_type"]=="Ki"].copy()
    df    = df[df["measurement_type"]!="Ki"].copy()
    df["tier"] = "unassigned"
    df.loc[df["measurement_type"]=="pct_inhibition", "tier"] = "T3"
    mask_t2 = (
        (df["measurement_type"]=="IC50") & df["is_censored"] &
        (df["activity_value_nM"]>0) & df["filter_passed"] &
        ~df["is_mouse"] & (df["tier"]=="unassigned")
    )
    df.loc[mask_t2, "tier"] = "T2"
    mask_t1 = (
        (df["measurement_type"]=="IC50") & (df["qualifier"]=="=") &
        (df["activity_value_nM"]>0) &
        df["pIC50"].between(PACTIVITY_VALID_MIN, PACTIVITY_VALID_MAX) &
        ~df["activity_outlier"] &
        df["filter_passed"] & ~df["is_mouse"] & (df["tier"]=="unassigned") &
        df.get("unit_flag", pd.Series("ok",index=df.index)).isin(["ok","inferred"])
    )
    df.loc[mask_t1, "tier"] = "T1"
    df.loc[df["tier"]=="unassigned","tier"] = "T3"
    tier_summary = df["tier"].value_counts()
    logging.info(_table(
        [(t, f"{tier_summary.get(t,0):,}") for t in ["T1","T2","T3"]],
        ["Tier","Rows"]))
    log_stage("6_tiers", n_in, len(df),
              f"T1={tier_summary.get('T1',0):,}, T2={tier_summary.get('T2',0):,}, "
              f"T3={tier_summary.get('T3',0):,}, Ki={len(ki_df):,}")
    return df, ki_df

# ══════════════════════════════════════════════════════════════════════
# STAGE 7 — INTER-ASSAY BIAS CORRECTION
# ══════════════════════════════════════════════════════════════════════
def stage7_bias_correction(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logging.info("\n" + "═" * 60)
    logging.info("STAGE 7 — Inter-Assay Bias Correction")
    logging.info("═" * 60)
    n_in = len(df); df = df.copy()
    df["pIC50_corrected"] = df["pIC50"]; df["assay_bias"] = 0.0
    eligible = (
        (df["measurement_type"]=="IC50") & (df["qualifier"]=="=") &
        df["pIC50"].between(PACTIVITY_VALID_MIN, PACTIVITY_VALID_MAX) &
        (df["tier"]=="T1")
    )
    sub      = df.loc[eligible,["inchikey","source_id","pIC50"]].copy()
    n_assays = sub.groupby("inchikey")["source_id"].nunique()
    anchors  = n_assays[n_assays >= 2].index
    logging.info(f"  Anchor compounds (≥2 assays): {len(anchors):,}")
    if len(anchors) == 0:
        logging.info("  No anchor compounds — bias correction skipped")
        return df, pd.DataFrame(columns=["source_id","assay_shift","assay_shift_std","n_anchors"])
    sub_a      = sub[sub["inchikey"].isin(anchors)].copy()
    global_med = sub_a.groupby("inchikey")["pIC50"].median().rename("global_median")
    sub_a      = sub_a.join(global_med, on="inchikey")
    sub_a["residual"] = sub_a["pIC50"] - sub_a["global_median"]
    stats = sub_a.groupby("source_id")["residual"].agg(
        assay_shift="median", assay_shift_std="std", n_anchors="count").reset_index()
    shift_map = stats.set_index("source_id")["assay_shift"].to_dict()
    df["assay_bias"] = df["source_id"].map(shift_map).fillna(0.0)
    df.loc[eligible,"pIC50_corrected"] = (
        df.loc[eligible,"pIC50"] - df.loc[eligible,"assay_bias"]
    ).clip(PACTIVITY_VALID_MIN, PACTIVITY_VALID_MAX)
    log_stage("7_bias_correction", n_in, n_in,
              f"anchors={len(anchors):,}, assays_corrected={len(stats):,}")
    return df, stats

# ══════════════════════════════════════════════════════════════════════
# STAGE 8 — REPLICATE AGGREGATION (stereo-aware SMILES selection)
# v17.1 fixes:
#   - removed is_validation from carry-through columns
#   - removed duplicate keys in agg_dict (canonical_smiles,
#     stereo_remediation_status, stereo_isomer_count, stereo_isomer_smiles,
#     stereo_modeling_penalty were each set twice; v17 stereo-aware values
#     won by accident due to dict-literal ordering)
# ══════════════════════════════════════════════════════════════════════
def _compute_outlier_flags(vals: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    n = len(vals)
    is_iqr = pd.Series(False, index=vals.index)
    is_z   = pd.Series(False, index=vals.index)
    score  = pd.Series(0.0,   index=vals.index)
    if n >= 3:
        q1,q3 = vals.quantile(0.25), vals.quantile(0.75); iqr = q3-q1
        is_iqr = (vals < q1 - OUTLIER_IQR_FACTOR*iqr) | (vals > q3 + OUTLIER_IQR_FACTOR*iqr)
        mu,sd  = vals.mean(), vals.std()
        if sd > 0:
            zs    = (vals-mu).abs()/sd; is_z = zs > OUTLIER_ZSCORE_CUT
            iqr_d = np.maximum(0, np.maximum(q1-vals, vals-q3))/(iqr+1e-6)
            score = (0.5*zs.clip(0,5)/5 + 0.5*iqr_d.clip(0,5)/5).clip(0,1)
    return is_iqr, is_z, score

def _stereo_uncertainty_score(stereo_flag: str, n_chiral: int, n_undef: int) -> float:
    if stereo_flag == "achiral":          return 0.0
    if stereo_flag == "defined":          return 0.0
    if stereo_flag == "fully_undefined":  return 1.0
    if stereo_flag == "partial_undefined" and n_chiral > 0:
        return float(n_undef)/float(n_chiral)
    return 0.5

def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    if len(values) == 0: return np.nan
    if len(values) == 1: return float(values[0])
    w   = np.where(np.isfinite(weights) & (weights > 0), weights, 1e-8)
    w   = w / w.sum()
    idx = np.argsort(values); vals_s = values[idx]; w_s = w[idx]
    cum_w = np.cumsum(w_s)
    i_mid = min(np.searchsorted(cum_w, 0.5), len(vals_s)-1)
    return float(vals_s[i_mid])

def _rep_weight(row: pd.Series) -> float:
    ac = float(row.get("assay_confidence_score",0.5) or 0.5)
    sr = SOURCE_RELIABILITY.get(str(row.get("source_db","PubChem")),0.6)
    sd = row.get("sd_reported",np.nan)
    lu = 0.0
    if pd.notna(sd) and sd > 0:
        ic50 = row.get("activity_value_nM",np.nan)
        if pd.notna(ic50) and ic50 > 0:
            lu = min(float(sd)/(np.log(10)*float(ic50)), 1.0)
    uc = float(row.get("unit_confidence_score",0.8) or 0.8)
    return float(np.clip(ac * sr * (1.0-lu) * uc, 1e-6, 1.0))

# Columns carried from a representative replicate row to the aggregated row.
# v17.1: pruned of duplicates and of the v17 stereo columns that get
# explicit overrides below. is_validation removed.
_AGG_CARRY_COLS = [
    "original_smiles", "inchikey_14", "scaffold",
    "stereo_stripped_scaffold", "tautomer_hash", "stereo_flag",
    "n_chiral_centres", "n_undefined_centres", "stereo_uncertainty_score",
    "stereo_defined_flag", "stereo_completeness_score",
    "salt_removed", "steps_applied", "smiles_pre_ph", "protonation_modified",
    "structure_quality_flag", "structure_quality_score",
    "mw", "heavy_atoms", "logP", "tpsa", "hbd", "hba", "rot_bonds",
    "arom_rings", "ring_count", "frac_sp3", "formal_charge",
    "ro5_violations", "lipinski_violations", "ro3_violations",
    "mw_flag_heavy", "flag_mw_extreme", "flag_logp_extreme",
    "flag_hbd_extreme", "flag_hba_extreme", "n_druglike_flags",
    "fsp3_flag", "complexity_score", "natural_product_likeness_flag",
    "is_pains", "is_brenk", "is_aggregator", "substructure_alert_count",
    "is_covalent", "covalent_confidence", "covalent_type", "warhead_count",
    "is_reversible_covalent", "covalent_context_excluded",
    "time_dependent", "preincubation_flag", "assay_covalent_flag",
    "covalent_consistent",
    "activity_task", "assay_type_v2", "dose_response_flag", "target_species",
    "literature_confidence", "activity_unit_original",
    "activity_unit_standardized",
    "unit_confidence_score", "unit_flag", "filter_passed", "assay_category",
    "assay_context_score", "organism", "cross_source_dup", "is_mouse",
    # v17 stereo-resolution columns (set by Stage 2.6, carried verbatim)
    "stereo_resolution_confidence", "stereo_resolved_by",
    "stereo_cross_inchikey", "stereo_cross_source_db",
    "n_ez_specified",
]

def stage8_aggregate(df: pd.DataFrame, output_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logging.info("\n" + "═" * 60)
    logging.info("STAGE 8 — Replicate Aggregation (weighted median)")
    logging.info("═" * 60)
    t1 = df[df["tier"]=="T1"].copy()
    if len(t1) == 0:
        logging.warning("  No T1 rows"); return pd.DataFrame(), pd.DataFrame()
    n_in = len(t1)
    t1["rep_weight"] = t1.apply(_rep_weight, axis=1)
    t1["stereo_uncertainty_score"] = t1.apply(
        lambda r: _stereo_uncertainty_score(
            r.get("stereo_flag","unknown"),
            int(r.get("n_chiral_centres",0) or 0),
            int(r.get("n_undefined_centres",0) or 0)), axis=1)
    t1["pIC50_norm"] = t1.groupby("source_id")["pIC50"].transform(lambda x: x - x.median())
    outlier_records = []
    for (ik,iso), grp in t1.groupby(["inchikey","target_isoform"]):
        vals = grp["pIC50"].dropna()
        if len(vals) < 2: continue
        is_iqr,is_z,score = _compute_outlier_flags(vals)
        for idx in grp.index:
            outlier_records.append({"idx":idx,
                "rep_is_iqr_outlier":    bool(is_iqr.get(idx,False)),
                "rep_is_zscore_outlier": bool(is_z.get(idx,False)),
                "rep_outlier_score":     float(score.get(idx,0.0))})
    if outlier_records:
        odf = pd.DataFrame(outlier_records).set_index("idx")
        for col in odf.columns: t1[col] = odf[col]
    else:
        t1["rep_is_iqr_outlier"]=False; t1["rep_is_zscore_outlier"]=False; t1["rep_outlier_score"]=0.0
    reps_df = t1.copy()
    rep_path = os.path.join(output_dir,"pad_replicates_full.csv")
    reps_df.to_csv(rep_path, index=False)
    logging.info(f"  Saved replicate table ({len(reps_df):,} rows)")

    group_key = ["inchikey","target_isoform","measurement_type"]
    t1_s = t1.sort_values(group_key)
    def _safe_std(x): return float(x.std()) if len(x)>1 else 0.0
    def _safe_mad(x):
        if len(x)<2: return 0.0
        return float(MAD_FACTOR*(x-x.median()).abs().median())
    agg_dict = {}
    for key, grp in t1_s.groupby(group_key):
        ik,iso,mt = key
        vals = grp["pIC50"].dropna()
        wts  = grp.loc[grp["pIC50"].notna(),"rep_weight"].values
        best_smiles = _pick_stereo_representative(grp)
        r    = grp.iloc[0]
        agg_dict[key] = {
            "inchikey": ik, "target_isoform": iso, "measurement_type": mt,
            "pIC50":           _weighted_median(vals.values, wts),
            "pIC50_median":    float(vals.median()) if len(vals) else np.nan,
            "pIC50_mean":      float(vals.mean())   if len(vals) else np.nan,
            "pIC50_corrected": float(grp["pIC50_corrected"].dropna().median()) if "pIC50_corrected" in grp.columns else np.nan,
            "pIC50_std":       _safe_std(vals),
            "pIC50_mad":       _safe_mad(vals),
            "pIC50_min":       float(vals.min()) if len(vals) else np.nan,
            "pIC50_max":       float(vals.max()) if len(vals) else np.nan,
            "n_measurements":  int(len(vals)),
            **{c: r.get(c) for c in _AGG_CARRY_COLS if c in grp.columns},
            "source_list":            ",".join(sorted(set(grp["source_db"].tolist()))),
            "source_count":           int(grp["source_db"].nunique()),
            "assay_id_list":          ",".join(sorted(set(grp["source_id"].tolist()))),
            "source_file":            r.get("source_file",""),
            "assay_confidence_score": float(grp["assay_confidence_score"].mean()) if "assay_confidence_score" in grp.columns else 0.5,
            "n_iqr_outlier_reps":     int(grp.get("rep_is_iqr_outlier",pd.Series(False)).sum()),
            "n_zscore_outlier_reps":  int(grp.get("rep_is_zscore_outlier",pd.Series(False)).sum()),
            "max_outlier_score":      float(grp.get("rep_outlier_score",pd.Series(0.0)).max()),
            "mean_rep_weight":        float(grp["rep_weight"].mean()),
            # Stereo penalty fields — explicit aggregation (max of group)
            "stereo_modeling_penalty": float(
                grp.get("stereo_modeling_penalty",
                        pd.Series(0.0, index=grp.index)).fillna(0.0).max()
            ),
            "stereo_remediation_status": (
                grp["stereo_remediation_status"].mode().iloc[0]
                if "stereo_remediation_status" in grp.columns and len(grp) > 0
                else "resolved"
            ),
            "stereo_isomer_count": int(
                grp.get("stereo_isomer_count",
                        pd.Series(1, index=grp.index)).fillna(1).max()
            ),
            "stereo_isomer_smiles": next(
                (s for s in grp.get("stereo_isomer_smiles",
                                    pd.Series("", index=grp.index)).fillna("")
                 if isinstance(s, str) and s.strip()),
                ""
            ),
            # Stereo-aware canonical_smiles selection (from _pick_stereo_representative)
            "canonical_smiles": best_smiles,
            # Cross-resolution flags (any-of group semantics)
            "stereo_cross_resolved": bool(grp.get("stereo_cross_resolved", pd.Series(False)).any()),
            "stereo_has_ez":         bool(grp.get("stereo_has_ez", pd.Series(False)).any()),
        }
    agg = pd.DataFrame(list(agg_dict.values()))
    agg["pIC50_range"]            = agg["pIC50_max"] - agg["pIC50_min"]
    agg["activity_value_nM"]      = 10**(9 - agg["pIC50"])
    agg["activity_value_corr_nM"] = 10**(9 - agg["pIC50_corrected"])
    agg["high_variance"]          = agg["pIC50_std"] > HIGH_VARIANCE_STD
    agg["label_noise"]            = agg["pIC50_std"] > LABEL_NOISE_STD
    agg["confidence_weight"]      = (1.0/(1.0+agg["pIC50_std"]+STD_EPSILON)).round(4)
    agg["confidence_score"]       = (
        0.40*np.minimum(agg["n_measurements"]/5.0,1.0) +
        0.30*np.minimum(agg["source_count"]/3.0,1.0) +
        0.30*(~agg["label_noise"]).astype(float)
    ).round(4)
    agg["source_weight"] = np.log1p(agg["source_count"])
    # Cross-source consensus scoring
    agg["cross_source_consensus"] = 0.0
    if "source_list" in agg.columns:
        multi_src = agg["source_count"] >= CROSS_SOURCE_MIN_SOURCES
        if multi_src.any():
            std_vals = agg.loc[multi_src, "pIC50_std"].fillna(0.5)
            n_src    = agg.loc[multi_src, "source_count"]
            consensus = (
                np.minimum(n_src / 3.0, 1.0) *
                (1.0 - np.minimum(std_vals / 1.0, 1.0))
            ).clip(0.0, 1.0).round(4)
            agg.loc[multi_src, "cross_source_consensus"] = consensus
    n_consensus = int((agg["cross_source_consensus"] > 0).sum())
    logging.info(f"  Cross-source consensus: {n_consensus:,} compounds with ≥{CROSS_SOURCE_MIN_SOURCES} sources")
    logging.info(_table([
        ("Aggregated compounds",    f"{len(agg):,}"),
        ("From replicates",         f"{n_in:,}"),
        ("High-variance (std>0.5)", f"{int(agg['high_variance'].sum()):,}"),
        ("n_measurements = 1",      f"{int((agg['n_measurements']==1).sum()):,}"),
        ("n_measurements >= 2",     f"{int((agg['n_measurements']>=2).sum()):,}"),
        ("n_measurements >= 3",     f"{int((agg['n_measurements']>=3).sum()):,}"),
    ], ["Metric","Value"]))
    log_stage("8_aggregate", n_in, len(agg))
    return agg, reps_df

# ══════════════════════════════════════════════════════════════════════
# STAGE 9 — HTS NOISE MODELLING
# ══════════════════════════════════════════════════════════════════════
def stage9_hts(df_t3: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logging.info("\n" + "═" * 60)
    logging.info("STAGE 9 — HTS Noise Modelling (T3)")
    logging.info("═" * 60)
    if len(df_t3) == 0: return pd.DataFrame(), pd.DataFrame()
    n_in = len(df_t3); df = df_t3.copy()
    df["z_score"] = np.nan
    for aid, grp in df.groupby("source_id"):
        vals  = grp["pct_inhibition"].dropna()
        if len(vals) == 0: continue
        med   = vals.median()
        mad_s = MAD_FACTOR*(vals-med).abs().median()
        scale = max(mad_s, vals.std(), HTS_NOISE_FLOOR)
        df.loc[grp.index,"z_score"] = (grp["pct_inhibition"]-med)/scale
    df["binary_label"] = np.where(df["pct_inhibition"]>=HTS_ACTIVE_THRESHOLD, 1,
                                  np.where(df["pct_inhibition"].notna(), 0, np.nan))
    df["hts_class"]    = np.where(df["z_score"]>=1.0,"active",
                                  np.where(df["z_score"]<=-1.0,"inactive","borderline"))
    df["hts_confidence"] = np.where(df["z_score"].abs()>=2.0,"high",
                                    np.where(df["z_score"].abs()>=1.0,"medium","low"))
    stats = df.groupby("inchikey").agg(
        n_tested=("source_id","nunique"),
        n_active=("binary_label", lambda x:(x==1).sum())).reset_index()
    stats["hit_rate"] = np.where(stats["n_tested"]>0, stats["n_active"]/stats["n_tested"], 0.0)
    stats["is_frequent_hitter"] = (stats["n_tested"]>=2) & (stats["hit_rate"]>FREQUENT_HITTER_FRAC)
    fh_map = stats.set_index("inchikey")["is_frequent_hitter"].to_dict()
    hr_map = stats.set_index("inchikey")["hit_rate"].to_dict()
    df["is_frequent_hitter"] = df["inchikey"].map(fh_map).fillna(False)
    df["hit_rate"]           = df["inchikey"].map(hr_map).fillna(0.0)
    df["is_promiscuous"]     = df["hit_rate"] > 0.30
    z_comp   = np.minimum(df.get("z_score",pd.Series(0.0,index=df.index)).abs()/3,1.0).fillna(0)
    pct_comp = np.minimum(df.get("pct_inhibition",pd.Series(0.0,index=df.index)).abs()/80,1.0).fillna(0)
    fh_pen   = (~df["is_frequent_hitter"]).astype(float)
    pa_pen   = (~df.get("is_pains",pd.Series(False,index=df.index))).astype(float)
    df["hts_confidence_score"] = (0.35*z_comp+0.35*pct_comp+0.15*fh_pen+0.15*pa_pen).round(4)
    triple   = (df.get("is_pains",False) & df["is_frequent_hitter"] & df["is_promiscuous"])
    is_active = df["binary_label"] == 1
    keep     = ~triple | is_active
    denoised = df[keep].copy()
    n_removed = int((~keep).sum())
    actives   = denoised[denoised["binary_label"]==1].copy()
    inactives = denoised[denoised["binary_label"]==0].copy()
    n_act = len(actives)
    n_target = min(n_act*T3_INACTIVE_RATIO, len(inactives))
    if n_act > 0 and n_target > 0:
        wts     = inactives.get("hts_confidence_score",pd.Series(0.3,index=inactives.index)).fillna(0.3).clip(0.01,1.0)
        wts     = wts/wts.sum()
        sampled = inactives.sample(n=n_target, weights=wts, random_state=RANDOM_SEED)
        balanced = pd.concat([actives,sampled], ignore_index=True)
    else:
        balanced = denoised.copy()
    log_stage("9_hts", n_in, len(denoised),
              f"removed={n_removed:,}, balanced={len(balanced):,}")
    return denoised, balanced

# ══════════════════════════════════════════════════════════════════════
# STAGE 10 — WEIGHTING (stereo penalty ×0.60, no selectivity)
# ══════════════════════════════════════════════════════════════════════
def _compute_label_uncertainty_v2(df: pd.DataFrame) -> pd.Series:
    std_n  = (df.get("pIC50_std",                pd.Series(0.0,index=df.index)).fillna(0.0)/LU_STD_REF).clip(0,1)
    assay  = (1.0-df.get("assay_confidence_score",pd.Series(0.5,index=df.index)).fillna(0.5)).clip(0,1)
    src    = (1.0-df.get("source_reliability_weight",pd.Series(0.6,index=df.index)).fillna(0.6)).clip(0,1)
    unit   = (1.0-df.get("unit_confidence_score",pd.Series(0.8,index=df.index)).fillna(0.8)).clip(0,1)
    ctx    = (1.0-df.get("assay_context_score",  pd.Series(0.5,index=df.index)).fillna(0.5)).clip(0,1)
    return (LU_W_STD*std_n + LU_W_ASSAY*assay + LU_W_SRC*src +
            LU_W_UNIT*unit + LU_W_CONTEXT*ctx).round(4)

def _compute_scaffold_weights(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "stereo_stripped_scaffold" not in df.columns:
        df["scaffold_frequency"]=1; df["is_frequent_scaffold"]=False
        df["scaffold_aware_weight"]=1.0
        df["scaffold_diversity_score"]=1.0
        return df
    freq_map = df["stereo_stripped_scaffold"].fillna("").map(
        df["stereo_stripped_scaffold"].fillna("").value_counts()).fillna(1).astype(int)
    df["scaffold_frequency"]   = freq_map
    df["is_frequent_scaffold"] = freq_map >= FREQUENT_SCAFFOLD_MIN
    scaf_to_id = {s:i for i,s in enumerate(df["stereo_stripped_scaffold"].fillna("").unique())}
    df["scaffold_id"]     = df["stereo_stripped_scaffold"].fillna("").map(scaf_to_id)
    df["scaffold_size"]   = freq_map
    df["series_bias_flag"] = freq_map >= FREQUENT_SCAFFOLD_MIN
    max_freq = max(int(freq_map.max()), 1)
    df["scaffold_diversity_score"] = (1.0 - freq_map/max_freq).clip(0.0, 1.0).round(4)
    w_out = pd.Series(1.0, index=df.index)
    conf  = df.get("confidence_score", pd.Series(0.5,index=df.index)).fillna(0.5)
    scaf_col = df["stereo_stripped_scaffold"].fillna("__no_scaf__")
    for scaf, grp in df.groupby(scaf_col):
        if scaf == "__no_scaf__" or len(grp) <= MAX_SCAFFOLD_CAP: continue
        sorted_idx = conf.loc[grp.index].sort_values(ascending=False).index
        for rank, idx in enumerate(sorted_idx):
            w_out[idx] = 1.0 if rank < MAX_SCAFFOLD_CAP else SCAFFOLD_WEIGHT_DECAY**(rank-MAX_SCAFFOLD_CAP+1)
    df["scaffold_aware_weight"] = w_out.clip(0.01, 1.0).round(4)
    return df

def _compute_t1_novelty(df: pd.DataFrame) -> pd.Series:
    fps = []; valid_idx = []
    for idx, row in df.iterrows():
        smi = row.get("canonical_smiles","")
        mol = Chem.MolFromSmiles(smi) if isinstance(smi,str) else None
        if mol is None: continue
        try:
            fp  = AllChem.GetMorganFingerprintAsBitVect(mol, AD_FP_RADIUS, nBits=AD_FP_BITS)
            arr = np.zeros(AD_FP_BITS, dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr); valid_idx.append(idx)
        except Exception: continue
    if len(fps) < 2: return pd.Series(0.5, index=df.index)
    X    = np.array(fps)
    norm = X.sum(axis=1); dot = X @ X.T
    union = norm[:,None]+norm[None,:]-dot
    tan   = np.divide(dot, union, out=np.zeros_like(dot), where=(union>0))
    np.fill_diagonal(tan, 0.0)
    k_eff = min(AD_KNN_K, tan.shape[1]-1)
    self_sim = np.partition(tan, -k_eff, axis=1)[:,-k_eff:].mean(axis=1)
    result = pd.Series(0.5, index=df.index)
    for i,idx in enumerate(valid_idx): result[idx] = float(self_sim[i])
    return result.round(4)

# ══════════════════════════════════════════════════════════════════════
# STAGE 10.5 — ACTIVITY CLIFF DETECTION
# ══════════════════════════════════════════════════════════════════════
def _detect_activity_cliffs(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logging.info("\n" + "═" * 60)
    logging.info("STAGE 10.5 — Activity Cliff Detection")
    logging.info("═" * 60)
    df = df.copy()
    df["is_activity_cliff"]  = False
    df["n_cliff_partners"]   = 0
    df["cliff_severity"]     = "none"
    df["max_cliff_dpic50"]   = 0.0
    if len(df) > 25000:
        logging.warning("  Skipped: dataset too large for pairwise comparison (>25k compounds)")
        return df, pd.DataFrame()
    if len(df) < 2 or "pIC50" not in df.columns:
        logging.info("  Skipped: insufficient data")
        return df, pd.DataFrame()
    fps = []; valid_idx = []; pic50_vals = []
    ik_vals = []
    for idx, row in df.iterrows():
        smi = row.get("canonical_smiles", "")
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None or pd.isna(row.get("pIC50")):
            continue
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, AD_FP_RADIUS, nBits=AD_FP_BITS)
            arr = np.zeros(AD_FP_BITS, dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)
            valid_idx.append(idx)
            pic50_vals.append(float(row["pIC50"]))
            ik_vals.append(row.get("inchikey", ""))
        except Exception:
            continue
    if len(fps) < 2:
        logging.info("  Skipped: fewer than 2 valid fingerprints")
        return df, pd.DataFrame()
    X = np.array(fps)
    pic50_arr = np.array(pic50_vals)
    norms = X.sum(axis=1)
    dot = X @ X.T
    union = norms[:, None] + norms[None, :] - dot
    tan = np.divide(dot, union, out=np.zeros_like(dot), where=(union > 0))
    np.fill_diagonal(tan, 0.0)
    dpic50 = np.abs(pic50_arr[:, None] - pic50_arr[None, :])
    cliff_mask = (tan >= ACTIVITY_CLIFF_TANIMOTO_MIN) & (dpic50 >= ACTIVITY_CLIFF_DPIC50_MIN)
    cliff_pairs = []
    cliff_counts = np.zeros(len(valid_idx), dtype=int)
    max_dpic50 = np.zeros(len(valid_idx))
    for i in range(len(valid_idx)):
        for j in range(i + 1, len(valid_idx)):
            if cliff_mask[i, j]:
                cliff_counts[i] += 1
                cliff_counts[j] += 1
                d = dpic50[i, j]
                max_dpic50[i] = max(max_dpic50[i], d)
                max_dpic50[j] = max(max_dpic50[j], d)
                severity = ("severe" if d >= ACTIVITY_CLIFF_SEVERITY_BINS[2]
                            else "moderate" if d >= ACTIVITY_CLIFF_SEVERITY_BINS[1]
                            else "mild")
                cliff_pairs.append({
                    "inchikey_1": ik_vals[i],
                    "inchikey_2": ik_vals[j],
                    "tanimoto_similarity": round(float(tan[i, j]), 4),
                    "pIC50_1": round(pic50_vals[i], 3),
                    "pIC50_2": round(pic50_vals[j], 3),
                    "delta_pIC50": round(float(d), 3),
                    "severity": severity,
                })
    for i, idx in enumerate(valid_idx):
        if cliff_counts[i] > 0:
            df.at[idx, "is_activity_cliff"] = True
            df.at[idx, "n_cliff_partners"] = int(cliff_counts[i])
            df.at[idx, "max_cliff_dpic50"] = round(float(max_dpic50[i]), 3)
            d = max_dpic50[i]
            df.at[idx, "cliff_severity"] = (
                "severe" if d >= ACTIVITY_CLIFF_SEVERITY_BINS[2]
                else "moderate" if d >= ACTIVITY_CLIFF_SEVERITY_BINS[1]
                else "mild"
            )
    cliff_df = pd.DataFrame(cliff_pairs)
    n_cliffs = int(df["is_activity_cliff"].sum())
    n_pairs = len(cliff_pairs)
    logging.info(_table([
        ("Cliff compounds",         f"{n_cliffs:,}  ({n_cliffs/max(len(df),1)*100:.1f}%)"),
        ("Cliff pairs",             f"{n_pairs:,}"),
        ("Severe (ΔpIC50≥3.0)",     f"{len([p for p in cliff_pairs if p['severity']=='severe']):,}"),
        ("Moderate (ΔpIC50≥2.0)",   f"{len([p for p in cliff_pairs if p['severity']=='moderate']):,}"),
        ("Mild (ΔpIC50≥1.5)",       f"{len([p for p in cliff_pairs if p['severity']=='mild']):,}"),
    ], ["Metric", "Value"]))
    log_stage("10.5_cliffs", len(df), len(df),
              f"cliff_compounds={n_cliffs:,}, pairs={n_pairs:,}")
    return df, cliff_df

# ══════════════════════════════════════════════════════════════════════
# STAGE 10 — WEIGHTING (stereo_priority_score includes E/Z; v17 weights)
# ══════════════════════════════════════════════════════════════════════
def _compute_stereo_priority_v17(df: pd.DataFrame) -> pd.Series:
    """stereo_priority_score includes E/Z flag for prioritisation."""
    pic50    = df.get("pIC50", pd.Series(0.0, index=df.index)).fillna(0.0)
    pic_norm = (pic50 - pic50.min()) / max(pic50.max() - pic50.min(), 1e-6)
    stereo_incomp = 1.0 - df.get(
        "stereo_completeness_score", pd.Series(0.5, index=df.index)
    ).fillna(0.5)
    lu = df.get(
        "label_uncertainty_score_v2", pd.Series(0.1, index=df.index)
    ).fillna(0.1)
    has_ez   = df.get("stereo_has_ez",    pd.Series(False, index=df.index)).fillna(False)
    ez_bonus = has_ez.astype(float) * 0.10
    priority = (
        0.45 * pic_norm
        + 0.30 * stereo_incomp
        + 0.15 * (1.0 - lu)
        + 0.10 * ez_bonus
    ).clip(0.0, 1.0).round(4)
    return priority

def stage10_weights_v17(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stage 10 — Weighting v17.
    - stereo_modeling_penalty coefficient: 0.60
    - selectivity term: REMOVED (Stage 10.7 dropped from v17)
    - cross_source_consensus bonus: +15% per CROSS_SOURCE_AGREEMENT_BONUS
    - cross_resolved stereo bonus: +10% (proportional to resolution confidence)
    - E/Z stereo: included in stereo_priority_score via stereo_has_ez flag
    """
    logging.info("\n" + "═" * 60)
    logging.info("STAGE 10 — Weighting (stereo penalty ×0.60, no selectivity)")
    logging.info("═" * 60)
    df = df.copy()
    def _src_rel(s):
        if not isinstance(s, str): return 0.60
        if "BindingDB" in s: return SOURCE_RELIABILITY["BindingDB"]
        if "ChEMBL"    in s: return SOURCE_RELIABILITY["ChEMBL"]
        return SOURCE_RELIABILITY["PubChem"]
    df["source_reliability_weight"] = df.get(
        "source_list", pd.Series("PubChem", index=df.index)
    ).apply(_src_rel)
    if "confidence_weight" not in df.columns:
        std = df.get("pIC50_std", pd.Series(0.0, index=df.index))
        df["confidence_weight"] = (1.0 / (1.0 + std + STD_EPSILON)).round(4)
    df["final_weight"] = (
        df["confidence_weight"]
        * df["source_reliability_weight"]
        * df["assay_confidence_score"]
    ).round(4)
    df["label_uncertainty_score_v2"] = _compute_label_uncertainty_v2(df)
    df["label_uncertainty_score"]    = df["label_uncertainty_score_v2"]
    df = _compute_scaffold_weights(df)
    n_down = int((df["scaffold_aware_weight"] < 1.0).sum())
    logging.info(f"  Scaffold-capped compounds: {n_down:,} (cap={MAX_SCAFFOLD_CAP})")
    logging.info("  Computing T1 self-similarity (novelty)...")
    df["t1_self_tanimoto"] = _compute_t1_novelty(df)
    df["t1_novelty_score"] = (1.0 - df["t1_self_tanimoto"]).clip(0, 1).round(4)
    df["stereo_priority_score"] = _compute_stereo_priority_v17(df)
    lu   = df["label_uncertainty_score_v2"]
    fw   = df["final_weight"]
    saw  = df["scaffold_aware_weight"]
    nov  = df["t1_novelty_score"]
    sdiv = df.get("scaffold_diversity_score",
                   pd.Series(0.5, index=df.index)).fillna(0.5)
    comp = df.get("complexity_score",
                   pd.Series(0.3, index=df.index)).fillna(0.3)
    csc  = df.get("cross_source_consensus",
                   pd.Series(0.0, index=df.index)).fillna(0.0)
    smp  = df.get("stereo_modeling_penalty",
                   pd.Series(0.0, index=df.index)).fillna(0.0)
    cross_res_bonus = df.get("stereo_resolution_confidence",
                            pd.Series(0.0, index=df.index)).fillna(0.0) * 0.10
    _raw_weight = (
        fw * saw * (1.0 - lu)
        * (1.0 + 0.20 * nov)
        * (1.0 + 0.10 * sdiv)
        * (1.0 + 0.05 * comp)
        * (1.0 + CROSS_SOURCE_AGREEMENT_BONUS * csc)
        * (1.0 + cross_res_bonus)
        * (1.0 - STEREO_PENALTY_WEIGHT_V17 * smp)
    ).clip(0.0, None)
    df["ml_weight_raw"] = _raw_weight.clip(0.01, 1.5).round(4)
    _w_min = _raw_weight.min()
    _w_max = _raw_weight.max()
    if _w_max > _w_min:
        df["ml_weight"] = (
            0.01 + 0.99 * (_raw_weight - _w_min) / (_w_max - _w_min)
        ).round(4)
    else:
        df["ml_weight"] = pd.Series(0.5, index=df.index)
        logging.warning("  ml_weight: all raw scores identical — set to 0.5")
    logging.info(
        f"  ml_weight normalised to [0.01, 1.0]  "
        f"(raw max={df['ml_weight_raw'].max():.4f})"
    )
    sf   = df.get("stereo_flag",     pd.Series("achiral", index=df.index))
    pic  = df.get("pIC50",           pd.Series(0.0,       index=df.index)).fillna(0.0)
    std  = df.get("pIC50_std",       pd.Series(0.0,       index=df.index)).fillna(0.0)
    recs = pd.Series("use_as_is",    index=df.index)
    is_undef = sf.isin(["fully_undefined", "partial_undefined"])
    recs[is_undef & (pic < 5.0)]              = "low_priority"
    recs[is_undef & (std > 0.3)]              = "enumerate_isomers"
    recs[is_undef & (pic > 7.0) & ~(std > 0.3)] = "flag_for_review"
    recs[is_undef & (pic > 7.0) &  (std > 0.3)] = "enumerate_isomers"
    recs[df.get("stereo_cross_resolved",
                pd.Series(False, index=df.index)).fillna(False)] = "cross_resolved"
    df["stereo_recommendation"] = recs
    n_meas = df.get("n_measurements", pd.Series(1, index=df.index)).fillna(1)
    pstd   = df.get("pIC50_std",       pd.Series(0.5, index=df.index)).fillna(0.5)
    lu_col = df["label_uncertainty_score_v2"]
    df["fidelity_level"] = "T1_standard"
    confirmed_mask = (n_meas >= T1_CONFIRMED_N_MEAS) & (pstd <= T1_CONFIRMED_STD_MAX)
    high_mask      = (
        (n_meas >= T1_HIGH_N_MEAS)
        & (pstd <= T1_HIGH_STD_MAX)
        & (lu_col < T1_HIGH_LU_MAX)
    )
    df.loc[confirmed_mask, "fidelity_level"] = "T1_confirmed"
    df.loc[high_mask,      "fidelity_level"] = "T1_high"
    n_high      = int((df["fidelity_level"] == "T1_high").sum())
    n_confirmed = int((df["fidelity_level"] == "T1_confirmed").sum())
    n_standard  = int((df["fidelity_level"] == "T1_standard").sum())
    n_cross_res = int(df.get("stereo_cross_resolved",
                              pd.Series(False)).fillna(False).sum())
    logging.info(_table([
        ("ml_weight mean",              f"{df['ml_weight'].mean():.3f}"),
        ("label_uncertainty_v2 mean",   f"{lu_col.mean():.3f}"),
        ("T1_high fidelity",            f"{n_high:,}  ({n_high/max(len(df),1)*100:.1f}%)"),
        ("T1_confirmed fidelity",       f"{n_confirmed:,}  ({n_confirmed/max(len(df),1)*100:.1f}%)"),
        ("T1_standard fidelity",        f"{n_standard:,}"),
        ("Stereo cross-resolved",       f"{n_cross_res:,}"),
        ("Frequent-scaffold compounds", f"{int(df['is_frequent_scaffold'].sum()):,}"),
        ("Scaffold diversity mean",     f"{df.get('scaffold_diversity_score',pd.Series(0.5)).mean():.3f}"),
        ("Stereo priority > 0.6",       f"{int((df['stereo_priority_score']>0.6).sum()):,}"),
        ("High stereo uncertainty",     f"{int(is_undef.sum()):,} ({is_undef.mean()*100:.1f}%)"),
    ], ["Metric", "Value"]))
    log_stage("10_weights", len(df), len(df))
    return df

# ══════════════════════════════════════════════════════════════════════
# STAGE 11 — FINAL DATASET OUTPUTS
# v17.1: is_validation removed from output schemas.
# ══════════════════════════════════════════════════════════════════════
_BASE_COLS_V17 = [
    "inchikey", "inchikey_14", "canonical_smiles", "original_smiles", "smiles_pre_ph",
    "scaffold", "stereo_stripped_scaffold", "scaffold_id", "scaffold_size", "tautomer_hash",
    "stereo_flag", "n_chiral_centres", "n_undefined_centres", "stereo_uncertainty_score",
    "stereo_defined_flag", "stereo_completeness_score", "stereo_recommendation",
    "stereo_priority_score",
    "stereo_remediation_status", "stereo_isomer_count", "stereo_modeling_penalty",
    # v17 stereo columns
    "stereo_cross_resolved", "stereo_cross_inchikey", "stereo_cross_source_db",
    "stereo_resolution_confidence", "stereo_resolved_by",
    "stereo_has_ez", "n_ez_specified",
    "salt_removed", "steps_applied", "protonation_modified",
    "structure_quality_flag", "structure_quality_score",
    "mw", "heavy_atoms", "logP", "tpsa", "hbd", "hba", "rot_bonds", "arom_rings",
    "ring_count", "frac_sp3", "formal_charge", "ro5_violations", "lipinski_violations",
    "ro3_violations", "mw_flag_heavy", "n_druglike_flags", "fsp3_flag",
    "complexity_score", "natural_product_likeness_flag",
    "is_pains", "is_brenk", "is_aggregator", "substructure_alert_count",
    "is_covalent", "covalent_confidence", "covalent_type", "warhead_count",
    "is_reversible_covalent", "covalent_context_excluded",
    "target_isoform", "measurement_type", "activity_task",
    "assay_type_v2", "dose_response_flag", "target_species", "literature_confidence",
    "organism", "assay_category", "assay_confidence_score", "assay_context_score",
    "time_dependent", "preincubation_flag", "assay_covalent_flag", "covalent_consistent",
    "activity_unit_original", "activity_unit_standardized",
    "unit_confidence_score", "unit_flag",
    "pIC50", "pIC50_corrected", "pIC50_std", "pIC50_mad", "pIC50_min", "pIC50_max",
    "pIC50_range", "n_measurements", "activity_value_nM", "activity_value_corr_nM",
    "high_variance", "label_noise", "label_uncertainty_score_v2",
    "confidence_weight", "final_weight",
    "scaffold_aware_weight", "scaffold_diversity_score", "ml_weight", "ml_weight_raw",
    "t1_self_tanimoto", "t1_novelty_score", "series_bias_flag",
    "source_reliability_weight", "confidence_score", "source_weight",
    "scaffold_frequency", "is_frequent_scaffold", "fidelity_level",
    "source_list", "source_count", "assay_id_list", "source_file",
    "n_iqr_outlier_reps", "n_zscore_outlier_reps", "max_outlier_score", "mean_rep_weight",
    "cross_source_dup", "filter_passed", "is_mouse",
    "activity_class",
    "is_activity_cliff", "n_cliff_partners", "cliff_severity", "max_cliff_dpic50",
    "cross_source_consensus",
]

_ML_READY_COLS_V17 = [
    "inchikey", "canonical_smiles", "stereo_stripped_scaffold", "scaffold_id", "scaffold_size",
    "stereo_flag", "stereo_defined_flag", "stereo_completeness_score", "stereo_uncertainty_score",
    "stereo_recommendation", "stereo_priority_score", "target_isoform",
    "stereo_remediation_status", "stereo_modeling_penalty",
    "stereo_cross_resolved", "stereo_has_ez",
    "pIC50", "pIC50_corrected", "pIC50_std", "n_measurements", "activity_class",
    "label_uncertainty_score_v2", "high_variance", "label_noise",
    "final_weight", "scaffold_aware_weight", "scaffold_diversity_score",
    "ml_weight", "ml_weight_raw",
    "t1_self_tanimoto", "t1_novelty_score",
    "is_covalent", "covalent_confidence", "assay_type_v2",
    "unit_flag", "unit_confidence_score", "is_pains", "is_brenk",
    "scaffold_frequency", "is_frequent_scaffold", "series_bias_flag", "fidelity_level",
    "complexity_score", "natural_product_likeness_flag", "fsp3_flag",
    "source_list", "source_count", "structure_quality_score",
    "is_activity_cliff", "cliff_severity", "cross_source_consensus",
]

_T2_COLS = [
    "inchikey","inchikey_14","canonical_smiles","scaffold","stereo_stripped_scaffold",
    "stereo_flag","stereo_defined_flag","stereo_completeness_score",
    "mw","heavy_atoms","logP","tpsa","hbd","hba","rot_bonds",
    "is_pains","is_brenk","is_aggregator","is_covalent","covalent_confidence",
    "activity_task","target_isoform","measurement_type","qualifier","organism","assay_category",
    "is_censored","activity_value_nM","original_activity_value","original_unit",
    "activity_lower_bound_nM","activity_upper_bound_nM","pIC50_lower","pIC50_upper",
    "unit_flag","unit_confidence_score","activity_class",
    "source_db","source_id","source_file","filter_passed","fidelity_level",
]

_T3_COLS = [
    "inchikey","inchikey_14","canonical_smiles","stereo_flag","stereo_defined_flag",
    "mw","heavy_atoms","logP","tpsa","hbd","hba",
    "is_pains","is_brenk","is_aggregator","is_covalent","covalent_confidence",
    "activity_task","target_isoform",
    "pct_inhibition","z_score","binary_label","hts_class","hts_confidence","hts_confidence_score",
    "is_frequent_hitter","is_promiscuous","hit_rate","is_weak_label",
    "screening_concentration_uM","pct_inhibition_normalized_flag",
    "is_artifact_high","is_artifact_low","fidelity_level",
    "source_id","source_db","source_file","assay_category","filter_passed",
    "ad_nn_tanimoto","ad_in_domain","ad_mahalanobis_dist",
    "ad_mahal_in_domain","ad_score_combined_v2","ad_reliability_flag",
    "ad_t3_self_score","ad_t3_self_in_domain","ad_t3_self_reliability",
]

_KI_COLS = [
    "inchikey","inchikey_14","canonical_smiles","scaffold","stereo_stripped_scaffold",
    "stereo_flag","n_chiral_centres","n_undefined_centres","stereo_uncertainty_score",
    "mw","heavy_atoms","logP","tpsa","hbd","hba",
    "is_pains","is_brenk","is_covalent","covalent_confidence","covalent_type",
    "activity_task","target_isoform","measurement_type","qualifier","organism","assay_category",
    "pIC50","is_censored","activity_value_nM","original_activity_value","original_unit",
    "pIC50_lower","pIC50_upper","unit_flag","unit_confidence_score","activity_class",
    "source_db","source_id","source_file","filter_passed",
    "ad_nn_tanimoto","ad_in_domain","ad_score_combined_v2","ad_reliability_flag",
]

def _save_csv(df: pd.DataFrame, path: str, cols: Optional[List[str]] = None) -> None:
    df = df.copy()
    if cols:
        seen = set(); deduped = []
        for c in cols:
            if c not in seen: deduped.append(c); seen.add(c)
        for c in deduped:
            if c not in df.columns: df[c] = np.nan
        df = df[[c for c in deduped if c in df.columns]]
    if "inchikey" in df.columns: df = df.sort_values("inchikey")
    df.to_csv(path, index=False)
    logging.info(f"  → {os.path.basename(path)} ({len(df):,} rows)")

def stage11_outputs(t1_agg: pd.DataFrame, t1_strict: pd.DataFrame,
                    t1_confirmed: pd.DataFrame,
                    t2_df: pd.DataFrame, t3_dn: pd.DataFrame, t3_bal: pd.DataFrame,
                    ki_df: pd.DataFrame, output_dir: str, stereo_mode: str = "relaxed") -> None:
    logging.info("\n" + "═" * 60)
    logging.info("STAGE 11 — Final Dataset Outputs")
    logging.info("═" * 60)
    if len(t1_agg) > 0:
        t1_agg = t1_agg.copy()
        t1_agg["activity_class"] = np.where(t1_agg["pIC50"]>=PACTIVITY_THRESHOLD, 1, 0)
        _save_csv(t1_agg,       os.path.join(output_dir,"pad_t1_ic50_aggregated.csv"),    _BASE_COLS_V17)
        _save_csv(t1_strict,    os.path.join(output_dir,"pad_t1_ic50_strict_v17.csv"),    _BASE_COLS_V17)
        _save_csv(t1_agg,       os.path.join(output_dir,"pad_t1_ic50_ml_ready_v17.csv"),  _ML_READY_COLS_V17)
        _save_csv(t1_confirmed, os.path.join(output_dir,"pad_t1_confirmed.csv"),          _BASE_COLS_V17)
        if len(t1_agg) > 0 and "is_covalent" in t1_agg.columns:
            mask_cov    = t1_agg["covalent_confidence"].isin(COVALENT_CONFIDENCE_FOR_SPLIT)
            mask_noncov = ~t1_agg["is_covalent"] | t1_agg["covalent_confidence"].isin(["low","none"])
            _save_csv(t1_agg[mask_noncov], os.path.join(output_dir,"pad_t1_non_covalent.csv"), _BASE_COLS_V17)
            _save_csv(t1_agg[mask_cov],    os.path.join(output_dir,"pad_t1_covalent.csv"),     _BASE_COLS_V17)
            logging.info(_table([
                ("Non-covalent",         f"{mask_noncov.sum():,}"),
                ("Covalent (H+M conf.)", f"{mask_cov.sum():,}"),
            ], ["Covalent split","Rows"]))
        if len(t1_agg) > 0:
            strict_stereo = t1_agg[t1_agg.get("stereo_defined_flag",pd.Series(False))].copy()
            _save_csv(strict_stereo, os.path.join(output_dir,"pad_t1_strict_stereo.csv"), _BASE_COLS_V17)
            logging.info(f"  Stereo: strict={len(strict_stereo):,} / relaxed={len(t1_agg):,}")
        # Classification (T1 + T2 censored)
        cls_parts = []
        if len(t1_agg) > 0:
            t1_cls = t1_agg.copy(); t1_cls["source_tier"]="T1"; cls_parts.append(t1_cls)
        if len(t2_df) > 0:
            t2_ic = t2_df[t2_df.get("measurement_type",pd.Series())=="IC50"].copy()
            if len(t2_ic):
                t2_ic["source_tier"]="T2"; t2_ic["activity_class"]=np.nan
                pl = t2_ic.get("pIC50_lower",pd.Series(np.nan,index=t2_ic.index))
                pu = t2_ic.get("pIC50_upper",pd.Series(np.nan,index=t2_ic.index))
                t2_ic.loc[pl.fillna(-np.inf)>PACTIVITY_THRESHOLD,"activity_class"]=1
                t2_ic.loc[pu.fillna(np.inf)<PACTIVITY_THRESHOLD,"activity_class"]=0
                cls_parts.append(t2_ic)
        if cls_parts:
            cls_df = pd.concat(cls_parts, ignore_index=True, sort=False)
            _save_csv(cls_df, os.path.join(output_dir,"pad_classification_v17.csv"),
                      _BASE_COLS_V17+["source_tier"])
        # Multi-fidelity
        mf_parts = []
        if len(t1_agg) > 0: mf_parts.append(t1_agg.assign(is_weak_label=False))
        if len(t2_df)  > 0:
            t2c = t2_df.copy(); t2c["fidelity_level"]="T2_censored"; t2c["is_weak_label"]=False
            mf_parts.append(t2c)
        if len(t3_dn)  > 0:
            t3c = t3_dn.copy(); t3c["fidelity_level"]="T3_weak"; t3c["is_weak_label"]=True
            mf_parts.append(t3c)
        if mf_parts:
            mf_df = pd.concat(mf_parts, ignore_index=True, sort=False)
            fid_w = {"T1_high":1.0,"T1_confirmed":0.9,"T1_standard":0.8,
                     "T2_censored":0.5,"T3_weak":0.3}
            mf_df = mf_df.assign(
                fidelity_weight=mf_df["fidelity_level"].map(fid_w).fillna(0.3)
            )
            _save_csv(mf_df, os.path.join(output_dir,"pad_multifidelity_v17.csv"),
                      ["inchikey","canonical_smiles","target_isoform",
                       "pIC50","activity_class","binary_label",
                       "fidelity_level","fidelity_weight","is_weak_label",
                       "label_uncertainty_score_v2","ml_weight","scaffold_id",
                       "source_list"])
        _save_csv(t2_df,  os.path.join(output_dir,"pad_t2_censored.csv"),     _T2_COLS)
        # v17.2 fix: pad_t3_hts_balanced.csv was never written prior to this version.
        # The function signature accepted t3_bal and the summary claimed the file
        # existed, but no _save_csv call wrote it. Caught by the v17.2 consistency
        # check. Write it here using the same column schema as t3_dn.
        if len(t3_bal) > 0:
            _save_csv(t3_bal, os.path.join(output_dir,"pad_t3_hts_balanced.csv"), _T3_COLS)
        if len(ki_df) > 0:
            ki_df = ki_df.copy(); ki_df["activity_class"]=np.nan
            mask = (ki_df["qualifier"]=="=") & ki_df["pIC50"].between(PACTIVITY_VALID_MIN,PACTIVITY_VALID_MAX)
            ki_df.loc[mask,"activity_class"]=np.where(ki_df.loc[mask,"pIC50"]>=PACTIVITY_THRESHOLD,1,0)
            _save_csv(ki_df, os.path.join(output_dir,"pad_ki_clean.csv"), _KI_COLS)
    log_stage("11_outputs", 0, 0, f"outputs saved to {output_dir}")

# ══════════════════════════════════════════════════════════════════════
# STAGE 12 — DUAL APPLICABILITY DOMAIN
# ══════════════════════════════════════════════════════════════════════
def _ecfp4_array(df: pd.DataFrame, label: str) -> Tuple[np.ndarray, List[str]]:
    fps: List = []; keys: List = []
    for ik, smi in zip(df.get("inchikey",[]), df.get("canonical_smiles",[])):
        mol = Chem.MolFromSmiles(smi) if isinstance(smi,str) else None
        if mol is None: continue
        try:
            fp  = AllChem.GetMorganFingerprintAsBitVect(mol, AD_FP_RADIUS, nBits=AD_FP_BITS)
            arr = np.zeros(AD_FP_BITS, dtype=np.uint8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr); keys.append(ik)
        except Exception: continue
    logging.info(f"  ECFP4 computed for {label}: {len(fps):,}")
    return np.array(fps, dtype=np.float32), keys

def _tanimoto_matrix(q_fps: np.ndarray, t_fps: np.ndarray) -> np.ndarray:
    t_norm = t_fps.sum(axis=1); q_norm = q_fps.sum(axis=1)
    dot    = q_fps @ t_fps.T
    union  = q_norm[:,None] + t_norm[None,:] - dot
    return np.divide(dot, union, out=np.zeros_like(dot), where=(union>0))

def _compute_ad_core(train_fps: np.ndarray, query_fps: np.ndarray,
                     query_keys: List[str], k: int = AD_KNN_K,
                     tan_cut: float = AD_TANIMOTO_CUT,
                     prefix: str = "ad") -> pd.DataFrame:
    empty = pd.DataFrame({
        "inchikey":                         query_keys,
        f"{prefix}_nn_tanimoto":            np.nan,
        f"{prefix}_in_domain":              False,
        f"{prefix}_local_density":          np.nan,
        f"{prefix}_knn_variance":           np.nan,
        f"{prefix}_mahalanobis_dist":       np.nan,
        f"{prefix}_mahal_in_domain":        False,
        f"{prefix}_score_combined":         0.0,
        f"{prefix}_reliability_flag":       "out_of_domain",
    })
    if len(train_fps)==0 or len(query_fps)==0: return empty
    tan   = _tanimoto_matrix(query_fps, train_fps)
    k_eff = min(k, tan.shape[1])
    top_k_idx    = np.argpartition(tan, -k_eff, axis=1)[:,-k_eff:]
    top_k_scores = np.take_along_axis(tan, top_k_idx, axis=1)
    mean_top_k   = top_k_scores.mean(axis=1)
    var_top_k    = top_k_scores.var(axis=1)
    k_dens    = min(AD_DENSITY_K, tan.shape[1])
    top_dens  = np.partition(tan, -k_dens, axis=1)[:,-k_dens:]
    local_density = (top_dens > 0.0).mean(axis=1) * top_dens.mean(axis=1)
    mahal_in = np.zeros(len(query_fps), dtype=bool)
    mahal_dist = np.full(len(query_fps), np.nan)
    if _SKLEARN_OK and len(train_fps) >= 10:
        try:
            sc  = StandardScaler(); Xt = sc.fit_transform(train_fps)
            Xq  = sc.transform(query_fps)
            cov = EmpiricalCovariance().fit(Xt)
            d_t = cov.mahalanobis(Xt)**0.5; d_q = cov.mahalanobis(Xq)**0.5
            thr = np.percentile(d_t, 97.5)
            mahal_in = d_q <= thr
            mahal_dist = d_q
        except Exception: pass
    tan_comp   = np.minimum(mean_top_k / max(tan_cut, 1e-6), 1.0) * 0.50
    dens_comp  = np.minimum(local_density / max(local_density.mean(), 1e-6), 1.0) * 0.30
    mahal_comp = mahal_in.astype(float) * 0.20
    combined   = (tan_comp + dens_comp + mahal_comp).clip(0, 1)
    in_domain  = mean_top_k >= tan_cut
    def _flag(t, c, v):
        if t < 0.20:   return "out_of_domain"
        if t >= 0.40 and c >= 0.60 and v < 0.05: return "high"
        if t >= 0.30:  return "medium"
        return "low"
    flags = [_flag(t,c,v) for t,c,v in zip(mean_top_k, combined, var_top_k)]
    return pd.DataFrame({
        "inchikey":                   query_keys,
        f"{prefix}_nn_tanimoto":      mean_top_k.round(4),
        f"{prefix}_in_domain":        in_domain,
        f"{prefix}_local_density":    local_density.round(4),
        f"{prefix}_knn_variance":     var_top_k.round(4),
        f"{prefix}_mahalanobis_dist": np.round(mahal_dist, 4),
        f"{prefix}_mahal_in_domain":  mahal_in,
        f"{prefix}_score_combined":   combined.round(4),
        f"{prefix}_reliability_flag": flags,
    })

def _add_ad_to_df(df_target: pd.DataFrame, ad_df: pd.DataFrame) -> pd.DataFrame:
    ad_cols = [c for c in ad_df.columns if c != "inchikey"]
    numeric_cols = [c for c in ad_cols if ad_df[c].dtype not in [bool, object]]
    bool_cols    = [c for c in ad_cols if ad_df[c].dtype == bool]
    other_cols   = [c for c in ad_cols if c not in numeric_cols and c not in bool_cols]
    agg = {"inchikey": "first"}
    for c in numeric_cols: agg[c] = "mean"
    for c in bool_cols:    agg[c] = "any"
    for c in other_cols:   agg[c] = "first"
    ad_dedup = ad_df.groupby("inchikey", as_index=False).agg(
        {c: v for c,v in agg.items() if c in ad_df.columns})
    n_before = len(df_target)
    df_out   = df_target.merge(ad_dedup, on="inchikey", how="left")
    n_after  = len(df_out)
    if n_after != n_before:
        logging.error(f"  AD merge row explosion: {n_before} → {n_after}. Fallback: no merge.")
        for col in ad_dedup.columns:
            if col != "inchikey": df_target[col] = np.nan
        return df_target
    return df_out

def _stratified_t3_sample(df: pd.DataFrame, n: int, seed: int = RANDOM_SEED) -> pd.DataFrame:
    df = df.drop_duplicates("inchikey").copy()
    if len(df) <= n:
        return df
    col = "hts_class" if "hts_class" in df.columns else None
    if col is None:
        return df.sample(n=n, random_state=seed)
    fracs  = df[col].value_counts(normalize=True)
    parts  = []
    for cls, frac in fracs.items():
        cls_df = df[df[col] == cls]
        n_cls  = max(1, round(frac * n))
        parts.append(cls_df.sample(n=min(n_cls, len(cls_df)), random_state=seed))
    sampled = pd.concat(parts).drop_duplicates("inchikey")
    if len(sampled) < n:
        remainder = df[~df["inchikey"].isin(sampled["inchikey"])]
        extra = remainder.sample(
            n=min(n - len(sampled), len(remainder)), random_state=seed)
        sampled = pd.concat([sampled, extra])
    return sampled.head(n).copy()

def stage12_ad_v17(t1_agg: pd.DataFrame, t3_dn: pd.DataFrame,
                   ki_df: pd.DataFrame, output_dir: str) -> dict:
    logging.info("\n" + "═" * 60)
    logging.info("STAGE 12 — Dual Applicability Domain")
    logging.info("             T1-relative AD + T3 self-AD LOO")
    logging.info("═" * 60)
    t3_sub    = _stratified_t3_sample(t3_dn, MAX_T3_SELF_AD_TRAIN) if len(t3_dn) > 0 else pd.DataFrame()
    _t3_ref_ik_set = set(t3_sub["inchikey"].tolist()) if len(t3_sub) > 0 else set()
    ki_unique = ki_df.drop_duplicates("inchikey").copy() if len(ki_df)>0 else pd.DataFrame()
    fps_t1, k_t1 = _ecfp4_array(t1_agg, "T1")
    fps_t3, k_t3 = _ecfp4_array(t3_sub, "T3") if len(t3_sub) else (np.array([]), [])
    fps_ki, k_ki = _ecfp4_array(ki_unique, "Ki") if len(ki_unique) else (np.array([]), [])
    stats = {"n_T1":len(k_t1), "n_T3":len(k_t3), "n_Ki":len(k_ki)}
    if len(fps_t1) > 0 and len(fps_t3) > 0:
        ad_t3_t1rel = _compute_ad_core(fps_t1, fps_t3, k_t3,
                                       k=AD_KNN_K, tan_cut=AD_TANIMOTO_CUT, prefix="ad")
        ad_t3_t1rel = ad_t3_t1rel.rename(columns={
            "ad_score_combined": "ad_score_combined_v2",
        })
        t3_dn = _add_ad_to_df(t3_dn, ad_t3_t1rel)
        stats.update({
            "T3_t1rel_in_domain_pct":    round(float(ad_t3_t1rel["ad_in_domain"].mean()*100),2),
            "T3_t1rel_mean_nn_tanimoto": round(float(ad_t3_t1rel["ad_nn_tanimoto"].mean()),4),
            "T3_t1rel_note": (
                "LOW T1-relative coverage is EXPECTED for HTS vs biochemical data. "
                "Use T3 self-AD for T3 in-domain filtering."
            ),
        })
    if len(fps_t3) >= 20:
        logging.info("  Computing T3 self-AD with proper leave-one-out...")
        t3_norms = fps_t3.sum(axis=1).astype(np.float32)
        t3_dot   = fps_t3 @ fps_t3.T
        t3_union = t3_norms[:, None] + t3_norms[None, :] - t3_dot
        t3_tan   = np.divide(t3_dot, t3_union,
                             out=np.zeros_like(t3_dot, dtype=np.float32),
                             where=(t3_union > 0))
        np.fill_diagonal(t3_tan, -1.0)
        k_eff = min(T3_SELF_AD_KNN_K, t3_tan.shape[1] - 1)
        top_k_idx    = np.argpartition(t3_tan, -k_eff, axis=1)[:, -k_eff:]
        top_k_scores = np.take_along_axis(t3_tan, top_k_idx, axis=1)
        mean_top_k   = top_k_scores.mean(axis=1)
        var_top_k    = top_k_scores.var(axis=1)
        k_dens    = min(AD_DENSITY_K, t3_tan.shape[1] - 1)
        top_dens  = np.partition(t3_tan, -k_dens, axis=1)[:, -k_dens:]
        local_density = (top_dens > 0.0).mean(axis=1) * top_dens.mean(axis=1)
        tan_comp  = np.minimum(mean_top_k / max(T3_SELF_AD_TAN_CUT, 1e-6), 1.0) * 0.60
        dens_comp = np.minimum(local_density / max(local_density.mean(), 1e-6), 1.0) * 0.25
        var_pen   = np.minimum(var_top_k / 0.05, 1.0) * 0.15
        combined  = (tan_comp + dens_comp - var_pen).clip(0, 1)
        in_domain = mean_top_k >= T3_SELF_AD_TAN_CUT
        def _t3_flag(t, c, v):
            if t < 0.15:  return "out_of_domain"
            if t >= 0.35 and c >= 0.55 and v < 0.04: return "high"
            if t >= 0.25: return "medium"
            return "low"
        flags = [_t3_flag(t, c, v) for t, c, v in zip(mean_top_k, combined, var_top_k)]
        ad_t3_self_ref = pd.DataFrame({
            "inchikey":               k_t3,
            "ad_t3_self_score":       combined.round(4),
            "ad_t3_self_in_domain":   in_domain,
            "ad_t3_self_reliability": flags,
        })
        logging.info(f"  Scoring all {len(t3_dn):,} T3 compounds against "
                     f"{len(fps_t3):,}-compound reference set...")
        fps_t3_all, k_t3_all = _ecfp4_array(t3_dn, "T3_all")
        if len(fps_t3_all) > 0:
            tan_all = _tanimoto_matrix(fps_t3_all, fps_t3)
            _ref_idx_lookup = {ik: i for i, ik in enumerate(k_t3)}
            for row_i, ik in enumerate(k_t3_all):
                if ik in _ref_idx_lookup:
                    tan_all[row_i, _ref_idx_lookup[ik]] = -1.0
            k_eff_all   = min(T3_SELF_AD_KNN_K, tan_all.shape[1])
            top_k_all   = np.argpartition(tan_all, -k_eff_all, axis=1)[:, -k_eff_all:]
            scores_all  = np.take_along_axis(tan_all, top_k_all, axis=1)
            mean_all    = scores_all.mean(axis=1)
            var_all     = scores_all.var(axis=1)
            dens_all    = (scores_all > 0.0).mean(axis=1) * scores_all.mean(axis=1)
            tan_c_all   = np.minimum(mean_all / max(T3_SELF_AD_TAN_CUT, 1e-6), 1.0) * 0.60
            dens_c_all  = np.minimum(
                dens_all / max(float(dens_all.mean()), 1e-6), 1.0) * 0.25
            var_p_all   = np.minimum(var_all / 0.05, 1.0) * 0.15
            combined_all = (tan_c_all + dens_c_all - var_p_all).clip(0, 1)
            in_dom_all   = mean_all >= T3_SELF_AD_TAN_CUT
            def _t3_flag_all(t, c, v, in_ref):
                if t < 0.15:  return "out_of_domain"
                base = "high" if (t >= 0.35 and c >= 0.55 and v < 0.04) else (
                    "medium" if t >= 0.25 else "low")
                return base if in_ref else f"{base}_vs_reference"
            flags_all = [
                _t3_flag_all(t, c, v, ik in _t3_ref_ik_set)
                for ik, t, c, v in zip(k_t3_all, mean_all, combined_all, var_all)
            ]
            ad_t3_self_all = pd.DataFrame({
                "inchikey":               k_t3_all,
                "ad_t3_self_score":       combined_all.round(4),
                "ad_t3_self_in_domain":   in_dom_all,
                "ad_t3_self_reliability": flags_all,
            })
        else:
            ad_t3_self_all = ad_t3_self_ref
        t3_dn = _add_ad_to_df(t3_dn, ad_t3_self_all)
        if "ad_t3_self_score" in t3_dn.columns:
            t3_indomain = t3_dn[t3_dn["ad_t3_self_score"] >= 0.30].copy()
        else:
            t3_indomain = t3_dn[t3_dn.get("ad_score_combined_v2", pd.Series(0.0)) >= 0.30].copy()
        _save_csv(t3_dn,       os.path.join(output_dir, "pad_t3_hts_denoised.csv"),  _T3_COLS)
        _save_csv(t3_indomain, os.path.join(output_dir, "pad_t3_hts_indomain.csv"),  _T3_COLS)
        logging.info(f"  T3 self-AD in-domain (≥0.30): {len(t3_indomain):,}/{len(t3_dn):,}")
        n_selfad_in = int(in_domain.sum())
        stats.update({
            "T3_self_in_domain_pct":    round(float(n_selfad_in / max(len(k_t3), 1) * 100), 2),
            "T3_self_mean_score":       round(float(combined.mean()), 4),
            "T3_self_mean_nn_tanimoto": round(float(mean_top_k.mean()), 4),
            "T3_indomain_for_ML":       len(t3_indomain),
            "T3_self_ad_method":        "leave-one-out kNN",
        })
    else:
        _save_csv(t3_dn, os.path.join(output_dir, "pad_t3_hts_denoised.csv"), _T3_COLS)
        _save_csv(pd.DataFrame(), os.path.join(output_dir, "pad_t3_hts_indomain.csv"), _T3_COLS)
        logging.warning("  T3 self-AD skipped: fewer than 20 unique compounds")
    if len(fps_t1) > 0 and len(fps_ki) > 0:
        ad_ki = _compute_ad_core(fps_t1, fps_ki, k_ki,
                                 k=AD_KNN_K, tan_cut=AD_TANIMOTO_CUT, prefix="ad")
        ad_ki = ad_ki.rename(columns={"ad_score_combined":"ad_score_combined_v2"})
        ki_df = _add_ad_to_df(ki_df, ad_ki)
        _save_csv(ki_df, os.path.join(output_dir,"pad_ki_clean.csv"), _KI_COLS)
        stats.update({
            "Ki_in_domain_pct":    round(float(ad_ki["ad_in_domain"].mean()*100),2),
            "Ki_mean_nn_tanimoto": round(float(ad_ki["ad_nn_tanimoto"].mean()),4),
            "Ki_mean_combined":    round(float(ad_ki["ad_score_combined_v2"].mean()),4),
        })
    if "ad_reliability_flag" in t3_dn.columns and len(t3_dn)>0:
        r_dist = t3_dn.groupby("ad_reliability_flag").size()
        logging.info("  T3 T1-relative AD reliability:")
        logging.info(_table([(k,f"{v:,}",f"{v/len(t3_dn)*100:.1f}%")
                             for k,v in r_dist.items()],
                            ["Reliability","Count","Pct"]))
    with open(os.path.join(output_dir,"ad_stats_v17.json"),"w") as f:
        json.dump(stats, f, indent=2)
    logging.info(_table([(k,str(v)) for k,v in stats.items()], ["AD Metric","Value"]))
    log_stage("12_ad_v17", 0, 0, "dual AD: T1-relative + T3-self LOO")
    return stats

# ══════════════════════════════════════════════════════════════════════
# STAGE 13 — QC (stereo-focused)
# v17.1: validation-leakage check removed.
# ══════════════════════════════════════════════════════════════════════
def stage13_qc_v17(t1: pd.DataFrame, ki: pd.DataFrame, t2: pd.DataFrame,
                   t3: pd.DataFrame, bias_df=None) -> tuple:
    logging.info("\n" + "═" * 60)
    logging.info("STAGE 13 — QC (stereo-focused)")
    logging.info("═" * 60)
    errors:   list = []
    warnings: list = []

    # ── Structural integrity ───────────────────────────────────────────
    for name, d in [("T1", t1), ("Ki", ki), ("T2", t2)]:
        if len(d) == 0:
            continue
        if "canonical_smiles" in d.columns:
            bad = d["canonical_smiles"].apply(
                lambda s: Chem.MolFromSmiles(s) is None if isinstance(s, str) else True
            )
            if bad.any():
                errors.append(f"[{name}] {bad.sum():,} invalid SMILES")
        if "inchikey" in d.columns:
            bad = ~d["inchikey"].str.match(
                r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$", na=False
            )
            if bad.any():
                errors.append(f"[{name}] {bad.sum():,} malformed InChIKeys")

    # ── pIC50 range ────────────────────────────────────────────────────
    if len(t1) > 0 and "pIC50" in t1.columns:
        out = ~t1["pIC50"].between(PACTIVITY_VALID_MIN, PACTIVITY_VALID_MAX)
        if out.any():
            errors.append(f"[T1] {out.sum():,} pIC50 outside valid range")

    # ── Type safety ────────────────────────────────────────────────────
    if len(t1) > 0 and "measurement_type" in t1.columns:
        if "Ki" in t1["measurement_type"].values:
            errors.append("[T1] Ki values in IC50 dataset — TYPE VIOLATION")
        n_pct = int((t1["measurement_type"] == "pct_inhibition").sum())
        if n_pct:
            errors.append(
                f"[T1] {n_pct:,} pct_inhibition rows — STRICT SEPARATION VIOLATED"
            )
        else:
            logging.info("  ✓ pct_inhibition strictly excluded from T1 regression")

    # ── Duplicate InChIKey ─────────────────────────────────────────────
    if len(t1) > 0:
        dups = t1.groupby(["inchikey", "target_isoform"]).size()
        bad  = dups[dups > 1]
        if len(bad):
            errors.append(f"[T1] {len(bad):,} duplicate InChIKey+isoform pairs")
        else:
            logging.info("  ✓ No duplicate InChIKey+isoform in T1")

    # NOTE: validation-leakage check removed in v17.1 — there is no
    # held-out validation set in PAD4-Bench v1. See paper §6.3.

    # ── Units ──────────────────────────────────────────────────────────
    if len(t1) > 0 and "unit_flag" in t1.columns:
        bad_units = t1[t1["unit_flag"].isin(["suspicious", "unknown"])]
        if len(bad_units):
            warnings.append(
                f"{len(bad_units):,} T1 records with suspicious/unknown units"
            )
        else:
            logging.info("  ✓ Unit validation: all T1 units OK or inferred")

    # ── Label uncertainty ──────────────────────────────────────────────
    if len(t1) > 0 and "label_uncertainty_score_v2" in t1.columns:
        lu = t1["label_uncertainty_score_v2"].dropna()
        if (lu < 0).any() or (lu > 1).any():
            errors.append("[T1] label_uncertainty_score_v2 out of [0,1]")
        else:
            logging.info(f"  ✓ LU v2: mean={lu.mean():.3f}, median={lu.median():.3f}")
        if (lu > 0.6).mean() > 0.30:
            warnings.append(
                f"{(lu>0.6).mean():.0%} T1 have LU v2 > 0.6 — consider filtering"
            )

    # ── STEREO QC ──────────────────────────────────────────────────────
    if len(t1) > 0 and "stereo_defined_flag" in t1.columns:
        pct_def = t1["stereo_defined_flag"].mean() * 100
        logging.info(f"  ✓ Stereo defined: {pct_def:.1f}% of T1")
        if pct_def < 60:
            warnings.append(
                f"Only {pct_def:.1f}% T1 have defined stereo (target ≥60%). "
                "Prioritise pad_t1_strict_stereo.csv for primary model. "
                "Use stereo_priority_score to guide experimental resolution."
            )
    if len(t1) > 0 and "stereo_cross_resolved" in t1.columns:
        n_cross = int(t1["stereo_cross_resolved"].fillna(False).sum())
        logging.info(
            f"  ✓ Stereo cross-source resolved: {n_cross:,} T1 compounds "
            f"({n_cross/max(len(t1),1)*100:.1f}%)"
        )
    if len(t1) > 0 and "stereo_has_ez" in t1.columns:
        n_ez = int(t1["stereo_has_ez"].fillna(False).sum())
        logging.info(
            f"  ✓ E/Z geometric stereo: {n_ez:,} T1 compounds "
            f"({n_ez/max(len(t1),1)*100:.1f}%)"
        )
    if len(t1) > 0 and "stereo_modeling_penalty" in t1.columns:
        smp        = t1["stereo_modeling_penalty"].fillna(0.0)
        mean_pen   = float(smp.mean())
        high_pen   = int((smp > 0.40).sum())
        logging.info(
            f"  ✓ Stereo modeling penalty: mean={mean_pen:.3f}, "
            f"high (>0.40)={high_pen:,} ({high_pen/max(len(t1),1)*100:.1f}%)"
        )
        if mean_pen > 0.20:
            warnings.append(
                f"Mean stereo modeling penalty = {mean_pen:.3f} (threshold 0.20). "
                f"{high_pen:,} compounds have penalty >0.40. "
                "Consider resolving stereo for top-priority compounds."
            )
    if len(t1) > 0 and "stereo_remediation_status" in t1.columns:
        status_dist    = t1["stereo_remediation_status"].value_counts()
        n_unresolvable = int(status_dist.get("unresolvable", 0))
        n_complex      = int(status_dist.get("complex", 0))
        n_cross_res    = int(status_dist.get("cross_resolved", 0))
        mean_penalty   = float(t1.get("stereo_modeling_penalty",
                                       pd.Series(0.0)).mean())
        logging.info(
            f"  ✓ Stereo remediation: "
            f"resolved={status_dist.get('resolved',0):,}, "
            f"cross_resolved={n_cross_res:,}, "
            f"enumerable={status_dist.get('enumerable',0):,}, "
            f"complex={n_complex:,}, "
            f"unresolvable={n_unresolvable:,}"
        )
        logging.info(f"    Mean stereo modeling penalty: {mean_penalty:.3f}")
        if n_complex > len(t1) * 0.10:
            warnings.append(
                f"{n_complex:,} T1 compounds ({n_complex/len(t1)*100:.1f}%) "
                "have >3 undefined stereocenters — "
                "use pad_t1_strict_stereo.csv for final benchmarks"
            )

    # ── Scaffold imbalance ─────────────────────────────────────────────
    if len(t1) > 0 and "scaffold_frequency" in t1.columns:
        freq_pct = t1["is_frequent_scaffold"].mean() * 100
        logging.info(f"  ✓ Frequent-scaffold compounds: {freq_pct:.1f}%")
        if freq_pct > 40:
            warnings.append(
                f"Scaffold imbalance: {freq_pct:.1f}% in frequent scaffolds — "
                "ml_weight capping + scaffold_diversity_score applied"
            )

    # ── Fidelity ───────────────────────────────────────────────────────
    if len(t1) > 0 and "fidelity_level" in t1.columns:
        n_high      = int((t1["fidelity_level"] == "T1_high").sum())
        n_confirmed = int((t1["fidelity_level"] == "T1_confirmed").sum())
        n_standard  = int((t1["fidelity_level"] == "T1_standard").sum())
        logging.info(
            f"  ✓ Fidelity: T1_high={n_high:,}, "
            f"T1_confirmed={n_confirmed:,}, T1_standard={n_standard:,}"
        )
        if n_high < 50:
            warnings.append(
                f"T1_high only {n_high:,} compounds — few triplicates. "
                f"Use T1_confirmed ({n_confirmed:,}) for replicate-quality training."
            )

    # ── Activity cliffs ────────────────────────────────────────────────
    if len(t1) > 0 and "is_activity_cliff" in t1.columns:
        n_cliffs  = int(t1["is_activity_cliff"].sum())
        cliff_pct = n_cliffs / max(len(t1), 1) * 100
        logging.info(f"  ✓ Activity cliffs: {n_cliffs:,} compounds ({cliff_pct:.1f}%)")
        if cliff_pct > 30:
            warnings.append(
                f"High activity cliff rate ({cliff_pct:.1f}%) — "
                "use cliff-aware splitting"
            )

    # ── Cross-source consensus ─────────────────────────────────────────
    if len(t1) > 0 and "cross_source_consensus" in t1.columns:
        n_consensus = int((t1["cross_source_consensus"] > 0).sum())
        logging.info(
            f"  ✓ Cross-source consensus: {n_consensus:,} compounds with ≥2 sources"
        )

    # ── T3 AD checks ──────────────────────────────────────────────────
    if len(t3) > 0 and "ad_t3_self_score" in t3.columns:
        selfad_pct = (t3["ad_t3_self_score"] >= 0.30).mean() * 100
        logging.info(f"  ✓ T3 self-AD in-domain: {selfad_pct:.1f}%")
        if selfad_pct < 30:
            warnings.append(
                f"T3 self-AD coverage = {selfad_pct:.1f}% — "
                "HTS library is structurally very diverse"
            )
    if len(t3) > 0 and "ad_t3_self_in_domain" in t3.columns:
        selfad_pct = t3["ad_t3_self_in_domain"].mean() * 100
        if selfad_pct > 98:
            warnings.append(
                f"T3 self-AD in-domain = {selfad_pct:.1f}% — "
                "check leave-one-out implementation"
            )
        else:
            logging.info(
                f"  ✓ T3 self-AD in-domain: {selfad_pct:.1f}% (leave-one-out kNN)"
            )

    # ── Tooling ────────────────────────────────────────────────────────
    if not _BRENK_OK:
        warnings.append("Brenk catalog unavailable — is_brenk=False")
    if not _DIMORPHITE_OK:
        warnings.append("dimorphite-dl not installed — using RDKit pH fallback")
    if not _SKLEARN_OK:
        warnings.append("scikit-learn not installed — Mahalanobis AD disabled")
    if not _FIND_POTENTIAL_STEREO_OK:
        warnings.append(
            "FindPotentialStereo unavailable — using legacy stereo census. "
            "Upgrade RDKit ≥ 2020 for E/Z stereo detection."
        )

    # ── SMARTS ─────────────────────────────────────────────────────────
    n_all  = (len(WARHEAD_SMARTS_HIGH) + len(WARHEAD_SMARTS_MEDIUM)
              + len(WARHEAD_SMARTS_LOW) + len(CONTEXT_EXCLUSION_SMARTS))
    n_comp = (len(_WARHEAD_HIGH) + len(_WARHEAD_MEDIUM)
              + len(_WARHEAD_LOW) + len(_CONTEXT_EXCL))
    if n_comp < n_all:
        errors.append(f"SMARTS: {n_comp}/{n_all} compiled — {n_all-n_comp} failed")
    else:
        logging.info(f"  ✓ SMARTS: all {n_comp} compiled")

    # ── pIC50 distribution ─────────────────────────────────────────────
    if len(t1) > 0 and "pIC50" in t1.columns:
        med     = t1["pIC50"].median()
        std_val = t1["pIC50"].std()
        if not (5.0 <= med <= 8.0):
            warnings.append(f"pIC50 median={med:.2f} (expected 5–8)")
        if std_val > 2.5:
            warnings.append(f"pIC50 std={std_val:.2f} (expected <2.5)")

    status = "PASSED" if not errors else "FAILED"
    logging.info(_table([
        ("Hard errors",   f"{len(errors):,}"),
        ("Soft warnings", f"{len(warnings):,}"),
        ("Status",        status),
    ], ["QC", "Result"]))
    for e in errors:   logging.error(f"  ✗ {e}")
    for w in warnings: logging.warning(f"  ⚠ {w}")
    log_stage("13_qc_v17", 0, 0,
              f"{status}: {len(errors)} errors, {len(warnings)} warnings")
    return errors, warnings

# ══════════════════════════════════════════════════════════════════════
# REPRODUCIBILITY & SUMMARY
# ══════════════════════════════════════════════════════════════════════
def _md5_file(fp: str) -> str:
    h = hashlib.md5()
    try:
        with open(fp,"rb") as f:
            for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
        return h.hexdigest()
    except Exception: return "unavailable"

def save_dataset_summary(output_dir: str, t1: pd.DataFrame, ki: pd.DataFrame,
                         t2: pd.DataFrame, t3_dn: pd.DataFrame, t3_bal: pd.DataFrame,
                         errors: List[str], warnings_list: List[str],
                         ad_stats: dict) -> None:
    import rdkit
    def _quick_stats(df, col="pIC50"):
        s = df.get(col, pd.Series()).dropna()
        if len(s)==0: return {}
        return {"n":int(len(s)), "mean":round(float(s.mean()),3),
                "std":round(float(s.std()),3), "min":round(float(s.min()),3),
                "median":round(float(s.median()),3), "max":round(float(s.max()),3)}
    n_confirmed = int((t1.get("fidelity_level",pd.Series())=="T1_confirmed").sum()) if len(t1) else 0
    summary = {
        "generated":        datetime.now().isoformat(),
        "pipeline_version": PIPELINE_VERSION,
        "environment": {
            "rdkit": rdkit.__version__, "pandas": pd.__version__,
            "dimorphite_dl": _DIMORPHITE_OK, "sklearn": _SKLEARN_OK,
            "find_potential_stereo": _FIND_POTENTIAL_STEREO_OK,
        },
        "dataset_sizes": {
            "T1_IC50":      len(t1),   "T1_Ki":       len(ki),
            "T2_censored":  len(t2),   "T3_denoised":  len(t3_dn),
            "T3_balanced":  len(t3_bal),
        },
        "T1_pIC50":               _quick_stats(t1,"pIC50"),
        "T1_ml_weight":           _quick_stats(t1,"ml_weight"),
        "T1_label_uncertainty":   _quick_stats(t1,"label_uncertainty_score_v2"),
        "T1_stereo_priority":     _quick_stats(t1,"stereo_priority_score"),
        "T1_scaffold_diversity":  _quick_stats(t1,"scaffold_diversity_score"),
        "T1_complexity":          _quick_stats(t1,"complexity_score"),
        "T1_covalent": {
            "high":   int((t1.get("covalent_confidence",pd.Series())=="high").sum()),
            "medium": int((t1.get("covalent_confidence",pd.Series())=="medium").sum()),
            "low":    int((t1.get("covalent_confidence",pd.Series())=="low").sum()),
        },
        "T1_stereo": {
            "defined_pct":           round(float(t1.get("stereo_defined_flag",pd.Series(False)).mean()*100),1),
            "fully_undefined_pct":   round(float((t1.get("stereo_flag",pd.Series())=="fully_undefined").mean()*100),1),
            "remediation_resolved":  int((t1.get("stereo_remediation_status",pd.Series())=="resolved").sum()),
            "remediation_enumerable":int((t1.get("stereo_remediation_status",pd.Series())=="enumerable").sum()),
            "remediation_complex":   int((t1.get("stereo_remediation_status",pd.Series())=="complex").sum()),
            "remediation_unresolvable": int((t1.get("stereo_remediation_status",pd.Series())=="unresolvable").sum()),
            "mean_modeling_penalty":  round(float(t1.get("stereo_modeling_penalty",pd.Series(0.0)).mean()),3),
            "cross_resolved":         int(t1.get("stereo_cross_resolved", pd.Series(False)).sum()),
            "ez_present":             int(t1.get("stereo_has_ez", pd.Series(False)).sum()),
        },
        "T1_fidelity": {
            "T1_high":      int((t1.get("fidelity_level",pd.Series())=="T1_high").sum()),
            "T1_confirmed": n_confirmed,
            "T1_standard":  int((t1.get("fidelity_level",pd.Series())=="T1_standard").sum()),
        },
        "T1_scaffold_bias": {
            "frequent_scaffold_pct": round(float(t1.get("is_frequent_scaffold",pd.Series(False)).mean()*100),1),
            "n_unique_scaffolds":    int(t1.get("stereo_stripped_scaffold",pd.Series()).nunique()),
        },
        "applicability_domain": ad_stats,
        "unit_quality": {
            "ok_pct":         round(float((t1.get("unit_flag",pd.Series("ok"))=="ok").mean()*100),1),
            "inferred_pct":   round(float((t1.get("unit_flag",pd.Series())=="inferred").mean()*100),1),
            "suspicious_pct": round(float((t1.get("unit_flag",pd.Series())=="suspicious").mean()*100),1),
        },
        "qc_errors":   errors,
        "qc_warnings": warnings_list,
        "T1_activity_cliffs": {
            "n_cliff_compounds": int(t1.get("is_activity_cliff", pd.Series(False)).sum()),
            "n_cliff_pairs": "see pad_activity_cliffs.csv",
        },
        "T1_cross_source_consensus": {
            "n_consensus_compounds": int((t1.get("cross_source_consensus", pd.Series(0.0)) > 0).sum()),
            "mean_consensus": round(float(t1.get("cross_source_consensus", pd.Series(0.0)).mean()), 3),
        },
        "ml_recommendations": {
            "primary_regression_file":      "pad_t1_non_covalent.csv",
            "confirmed_subset_file":        "pad_t1_confirmed.csv",
            "covalent_regression_file":     "pad_t1_covalent.csv",
            "classification_file":          "pad_classification_v17.csv",
            "multifidelity_file":           "pad_multifidelity_v17.csv",
            "strict_stereo_file":           "pad_t1_strict_stereo.csv",
            "weight_column":                "ml_weight",
            "label_column_regression":      "pIC50",
            "label_column_classification":  "activity_class",
            "split_key":                    "scaffold_id",
            "do_not_mix_T1_T3":             True,
            "T3_in_domain_file":            "pad_t3_hts_indomain.csv",
            "T3_in_domain_uses":            "ad_t3_self_score (T3 self-AD, not T1-relative)",
            "stereo_priority_column":       "stereo_priority_score",
            "activity_cliff_file":          "pad_activity_cliffs.csv",
        },
    }
    path = os.path.join(output_dir,"dataset_summary.json")
    with open(path,"w") as f: json.dump(summary, f, indent=2, default=str)
    logging.info("  Saved dataset_summary.json")

def save_config(input_dir: str, output_dir: str, manifest: List[dict],
                extra: dict = None) -> dict:
    import rdkit
    config = {
        "generated":        datetime.now().isoformat(),
        "pipeline_version": PIPELINE_VERSION,
        "random_seed":      RANDOM_SEED,
        "environment": {
            "rdkit": rdkit.__version__, "pandas": pd.__version__,
            "numpy": np.__version__, "python": sys.version.split()[0],
            "sklearn": _SKLEARN_OK, "dimorphite_dl": _DIMORPHITE_OK,
            "brenk_catalog": _BRENK_OK, "find_potential_stereo": _FIND_POTENTIAL_STEREO_OK,
            "stereo_api": _STEREO_API,   # "new" / "legacy" / "unavailable"
        },
        "thresholds": {
            "pIC50_active":           PACTIVITY_THRESHOLD,
            "pIC50_valid_range":      [PACTIVITY_VALID_MIN, PACTIVITY_VALID_MAX],
            "high_variance_std":      HIGH_VARIANCE_STD,
            "label_noise_std":        LABEL_NOISE_STD,
            "frequent_scaffold_min":  FREQUENT_SCAFFOLD_MIN,
            "max_scaffold_cap":       MAX_SCAFFOLD_CAP,
            "ad_tanimoto_cut":        AD_TANIMOTO_CUT,
            "ad_knn_k":               AD_KNN_K,
            "t3_self_ad_tan_cut":     T3_SELF_AD_TAN_CUT,
            "t3_self_ad_knn_k":       T3_SELF_AD_KNN_K,
            "t1_high_n_meas":         T1_HIGH_N_MEAS,
            "t1_confirmed_n_meas":    T1_CONFIRMED_N_MEAS,
            "t1_high_lu_max":         T1_HIGH_LU_MAX,
            "covalent_split_tiers":   list(COVALENT_CONFIDENCE_FOR_SPLIT),
            "activity_cliff_tanimoto_min": ACTIVITY_CLIFF_TANIMOTO_MIN,
            "activity_cliff_dpic50_min":   ACTIVITY_CLIFF_DPIC50_MIN,
            "cross_source_min_sources":    CROSS_SOURCE_MIN_SOURCES,
            "stereo_penalty_weight_v17":   STEREO_PENALTY_WEIGHT_V17,
            "stereo_cross_resolve":        STEREO_CROSS_RESOLVE,
            "stereo_enum_max_isomers":     STEREO_ENUM_MAX_ISOMERS,
        },
        "label_uncertainty_v2_weights": {
            "LU_W_STD": LU_W_STD, "LU_W_ASSAY": LU_W_ASSAY, "LU_W_SRC": LU_W_SRC,
            "LU_W_UNIT": LU_W_UNIT, "LU_W_CONTEXT": LU_W_CONTEXT,
        },
        # v17.1: input_md5 keyed by absolute path so duplicate filenames in
        # different subdirectories (pubchem/ vs bindingdb/ etc.) don't collide.
        "input_md5": {m["filename"]: _md5_file(m.get("path", os.path.join(input_dir, m["filename"])))
                      for m in manifest},
        "input_dir": input_dir, "output_dir": output_dir,
        **(extra or {}),
    }
    with open(os.path.join(output_dir,"pipeline_config.json"),"w") as f:
        json.dump(config, f, indent=2)
    logging.info("  Saved pipeline_config.json")
    return config

# ══════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════
class PAD4Pipeline:
    """PAD4 Inhibitor Data Curation Pipeline."""
    def __init__(self, input_dir: str, output_dir: Optional[str] = None,
                 log_level: str = "INFO", n_workers: int = 1,
                 stereo_mode: str = "relaxed"):
        self.input_dir   = input_dir
        self.output_dir  = output_dir or os.path.join(input_dir,"curated_v17")
        self.n_workers   = n_workers
        self.stereo_mode = stereo_mode
        os.makedirs(self.output_dir, exist_ok=True)
        setup_logging(log_level)
        # State
        self.manifest:  List[dict]   = []
        self.n_mirror:  int          = 0
        self.df:        pd.DataFrame = pd.DataFrame()
        self.ki_df:     pd.DataFrame = pd.DataFrame()
        self.bias_df:   pd.DataFrame = pd.DataFrame()
        self.t1_agg:    pd.DataFrame = pd.DataFrame()
        self.t1_strict: pd.DataFrame = pd.DataFrame()
        self.t1_confirmed: pd.DataFrame = pd.DataFrame()
        self.t2_df:     pd.DataFrame = pd.DataFrame()
        self.t3_df:     pd.DataFrame = pd.DataFrame()
        self.t3_dn:     pd.DataFrame = pd.DataFrame()
        self.t3_bal:    pd.DataFrame = pd.DataFrame()
        self.ad_stats:  dict         = {}
        self.cliff_df:  pd.DataFrame = pd.DataFrame()
        self.errors:    List[str]    = []
        self.warnings:  List[str]    = []

    def run(self) -> int:
        global _STAGE_LOG
        _STAGE_LOG = []  # Reset stage log for this run
        # Seed RNGs deterministically
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

        _log_tooling_status()
        logging.info("="*60)
        logging.info(f"PAD4 DATA CURATION PIPELINE  v{PIPELINE_VERSION}")
        logging.info(f"  random_seed = {RANDOM_SEED}")
        logging.info(f"  input_dir   = {self.input_dir}")
        logging.info(f"  output_dir  = {self.output_dir}")
        logging.info("="*60)

        raw_df, self.manifest, self.n_mirror = stage1_load(self.input_dir)
        self.df = stage2_standardize(raw_df, n_workers=self.n_workers)
        self.df = stage2_5_covalent(self.df)
        self.df = stage2_6_stereo_resolution(self.df)
        self.df = stage2_7_unit_normalization(self.df)
        self.df = stage2_8_complexity(self.df)
        self.df = stage3_activity(self.df)
        self.df = stage3_5_pct_inhibition(self.df)
        self.df = stage2_9_stereo_remediation(self.df, n_workers=self.n_workers)
        self.df = stage4_annotate(self.df)
        self.df = stage5_deduplicate(self.df)
        self.df, self.ki_df   = stage6_assign_tiers(self.df)
        self.df, self.bias_df = stage7_bias_correction(self.df)
        self.t1_agg, _ = stage8_aggregate(self.df, self.output_dir)

        self.t2_df              = self.df[self.df["tier"]=="T2"].copy()
        self.t3_df              = self.df[self.df["tier"]=="T3"].copy()
        self.t3_dn, self.t3_bal = stage9_hts(self.t3_df)

        # Carry T3 weak-label columns
        for col in ["is_weak_label","fidelity_level","screening_concentration_uM",
                    "pct_inhibition_normalized_flag"]:
            if col in self.df.columns and len(self.t3_dn)>0 and col not in self.t3_dn.columns:
                merge_src = self.df[["inchikey",col]].drop_duplicates("inchikey")
                self.t3_dn = self.t3_dn.merge(merge_src, on="inchikey", how="left")
        if len(self.ki_df)>0:
            if "pIC50_std" not in self.ki_df.columns: self.ki_df["pIC50_std"]=0.0
            self.ki_df["confidence_weight"] = (
                1.0/(1.0+self.ki_df["pIC50_std"]+STD_EPSILON)).round(4)
            for col in ["unit_confidence_score","unit_flag","activity_unit_original",
                        "activity_unit_standardized"]:
                if col not in self.ki_df.columns:
                    self.ki_df[col] = 1.0 if "score" in col else ("nM" if "unit" in col else "ok")

        self.t1_agg = stage10_weights_v17(self.t1_agg)
        self.t1_agg, self.cliff_df = _detect_activity_cliffs(self.t1_agg)
        if len(self.cliff_df) > 0:
            self.cliff_df.to_csv(
                os.path.join(self.output_dir, "pad_activity_cliffs.csv"), index=False)
            logging.info(f"  Saved pad_activity_cliffs.csv ({len(self.cliff_df):,} pairs)")

        self.t1_strict    = self._build_strict()
        self.t1_confirmed = self._build_confirmed()
        stage11_outputs(self.t1_agg, self.t1_strict, self.t1_confirmed,
                        self.t2_df, self.t3_dn, self.t3_bal, self.ki_df,
                        self.output_dir, stereo_mode=self.stereo_mode)
        if len(self.bias_df)>0:
            self.bias_df.to_csv(os.path.join(self.output_dir,"pad_assay_bias_report.csv"),index=False)
        self.ad_stats = stage12_ad_v17(
            self.t1_agg, self.t3_dn, self.ki_df, self.output_dir)
        self.errors, self.warnings = stage13_qc_v17(
            self.t1_agg, self.ki_df, self.t2_df, self.t3_df, self.bias_df)

        save_config(self.input_dir, self.output_dir, self.manifest,
                    extra={"n_workers":self.n_workers,"stereo_mode":self.stereo_mode})
        save_dataset_summary(self.output_dir, self.t1_agg, self.ki_df,
                             self.t2_df, self.t3_dn, self.t3_bal,
                             self.errors, self.warnings, self.ad_stats)
        qc = {"generated": datetime.now().isoformat(), "pipeline_version": PIPELINE_VERSION,
              "dataset_sizes": {"T1":len(self.t1_agg),"Ki":len(self.ki_df),
                                "T2":len(self.t2_df),"T3_dn":len(self.t3_dn),
                                "T1_confirmed":len(self.t1_confirmed),
                                "activity_cliff_pairs":len(self.cliff_df)},
              "qc_errors": self.errors, "qc_warnings": self.warnings,
              "stage_log": _STAGE_LOG}
        with open(os.path.join(self.output_dir,"pad_qc_report.json"),"w") as f:
            json.dump(qc, f, indent=2, default=str)
        self._verify_output_consistency()
        self._print_summary()
        return 0 if not self.errors else 1

    def _verify_output_consistency(self) -> None:
        """
        v17.2: cross-check on-disk row counts against in-memory counts,
        AND verify every CSV declared in the summary table actually exists.

        Reviewer-flagged: v17.1 had a 9-row discrepancy between Stage 11 and
        the final summary for pad_t1_non_covalent.csv (different non-covalent
        masks). This check caught that on its first run. It also caught a
        latent v17.0 bug: pad_t3_hts_balanced.csv was never actually written
        (the function signature accepted t3_bal but no _save_csv call wrote
        it), even though the summary claimed it existed.
        """
        logging.info("\n" + "═" * 60)
        logging.info("OUTPUT CONSISTENCY CHECK")
        logging.info("═" * 60)
        is_cov   = self.t1_agg.get("is_covalent",          pd.Series(False, index=self.t1_agg.index)).fillna(False)
        cov_conf = self.t1_agg.get("covalent_confidence",  pd.Series("none", index=self.t1_agg.index))
        stereo_def = self.t1_agg.get("stereo_defined_flag", pd.Series(False, index=self.t1_agg.index)).fillna(False)
        mask_noncov = (~is_cov) | cov_conf.isin(["low", "none"])
        mask_cov_hm = cov_conf.isin(COVALENT_CONFIDENCE_FOR_SPLIT)

        # Files with known expected row counts (strict check)
        expected_with_counts = [
            ("pad_t1_ic50_aggregated.csv",   len(self.t1_agg)),
            ("pad_t1_ic50_strict_v17.csv",   len(self.t1_strict)),
            ("pad_t1_ic50_ml_ready_v17.csv", len(self.t1_agg)),
            ("pad_t1_confirmed.csv",         len(self.t1_confirmed)),
            ("pad_t1_non_covalent.csv",      int(mask_noncov.sum())),
            ("pad_t1_covalent.csv",          int(mask_cov_hm.sum())),
            ("pad_t1_strict_stereo.csv",     int(stereo_def.sum())),
            ("pad_t2_censored.csv",          len(self.t2_df)),
            ("pad_t3_hts_denoised.csv",      len(self.t3_dn)),
            ("pad_t3_hts_balanced.csv",      len(self.t3_bal)),
            ("pad_t3_hts_indomain.csv",      len(self.t3_dn)),  # all T3, scored
            ("pad_ki_clean.csv",             len(self.ki_df)),
        ]
        # Files where we don't track an exact expected count — just check
        # that they exist and have at least one row (for non-empty datasets).
        existence_only = [
            "pad_replicates_full.csv",
            "pad_classification_v17.csv",
            "pad_multifidelity_v17.csv",
            "pad_activity_cliffs.csv",
        ]

        rows = []
        n_mismatch = 0
        for fname, n_expected in expected_with_counts:
            fp = os.path.join(self.output_dir, fname)
            if not os.path.exists(fp):
                rows.append((fname, str(n_expected), "(missing)", "✗"))
                n_mismatch += 1
                continue
            try:
                with open(fp, "r") as f:
                    n_actual = sum(1 for _ in f) - 1
            except Exception as e:
                rows.append((fname, str(n_expected), f"(err: {e})", "✗"))
                n_mismatch += 1
                continue
            mark = "✓" if n_actual == n_expected else "✗"
            if n_actual != n_expected:
                n_mismatch += 1
            rows.append((fname, f"{n_expected:,}", f"{n_actual:,}", mark))

        # Files with existence-only checks
        for fname in existence_only:
            fp = os.path.join(self.output_dir, fname)
            if not os.path.exists(fp):
                rows.append((fname, "(exists)", "(missing)", "✗"))
                n_mismatch += 1
                continue
            try:
                with open(fp, "r") as f:
                    n_actual = sum(1 for _ in f) - 1
                rows.append((fname, "(exists)", f"{n_actual:,}", "✓"))
            except Exception as e:
                rows.append((fname, "(exists)", f"(err: {e})", "✗"))
                n_mismatch += 1

        # JSON files — pure existence check
        for fname in ["dataset_summary.json", "pipeline_config.json",
                      "ad_stats_v17.json", "pad_qc_report.json"]:
            fp = os.path.join(self.output_dir, fname)
            if os.path.exists(fp):
                rows.append((fname, "(exists)", "ok", "✓"))
            else:
                rows.append((fname, "(exists)", "(missing)", "✗"))
                n_mismatch += 1

        logging.info(_table(rows, ["File", "Expected", "On disk", "OK"]))
        if n_mismatch:
            self.warnings.append(
                f"Output consistency: {n_mismatch} file(s) failed verification — "
                f"investigate before using these outputs."
            )
            logging.warning(
                f"  ⚠ {n_mismatch} file(s) failed verification (see table above)"
            )
        else:
            logging.info("  ✓ All output files present and row counts match in-memory tier sizes")

    def _build_strict(self) -> pd.DataFrame:
        if len(self.t1_agg)==0: return pd.DataFrame()
        ok_cats = {"biochemical_confirmatory","biochemical_single_point","binding","confirmatory"}
        mask = (
            (self.t1_agg["measurement_type"]=="IC50") &
            (~self.t1_agg.get("high_variance",pd.Series(False,index=self.t1_agg.index))) &
            (self.t1_agg["assay_category"].isin(ok_cats)) &
            (self.t1_agg.get("unit_flag",pd.Series("ok",index=self.t1_agg.index)).isin(["ok"]))
        )
        strict = self.t1_agg[mask].copy()
        logging.info(f"  T1 strict: {len(strict):,}/{len(self.t1_agg):,} compounds")
        return strict

    def _build_confirmed(self) -> pd.DataFrame:
        """T1_confirmed: n_measurements ≥ 2 AND pIC50_std ≤ 0.50."""
        if len(self.t1_agg)==0: return pd.DataFrame()
        mask = self.t1_agg["fidelity_level"].isin(["T1_high","T1_confirmed"])
        confirmed = self.t1_agg[mask].copy()
        logging.info(f"  T1 confirmed: {len(confirmed):,}/{len(self.t1_agg):,} compounds (n≥2, std≤0.5)")
        return confirmed

    def _print_summary(self) -> None:
        status    = "✓ PASSED" if not self.errors else f"✗ FAILED ({len(self.errors)} errors)"
        # v17.2 fix: use the same mask Stage 11 uses to write pad_t1_non_covalent.csv
        # (LOW-confidence covalents are written to the non-covalent split). The
        # earlier `~is_covalent` count differed from the actual file row count by
        # ~9 rows (the LOW-covalent compounds).
        if len(self.t1_agg) and "is_covalent" in self.t1_agg.columns:
            is_cov = self.t1_agg["is_covalent"].fillna(False)
            cov_conf = self.t1_agg.get("covalent_confidence", pd.Series("none", index=self.t1_agg.index))
            mask_noncov = (~is_cov) | cov_conf.isin(["low", "none"])
            mask_cov_hm = cov_conf.isin(COVALENT_CONFIDENCE_FOR_SPLIT)
            n_non_cov = int(mask_noncov.sum())
            n_cov_hm  = int(mask_cov_hm.sum())
            n_strict_stereo = int(self.t1_agg.get("stereo_defined_flag", pd.Series(False)).fillna(False).sum())
        else:
            n_non_cov = n_cov_hm = n_strict_stereo = 0
        n_t1_high = int((self.t1_agg.get("fidelity_level",pd.Series())=="T1_high").sum()) if len(self.t1_agg) else 0
        n_t1_conf = int((self.t1_agg.get("fidelity_level",pd.Series())=="T1_confirmed").sum()) if len(self.t1_agg) else 0
        logging.info("\n"+"═"*60)
        logging.info(f"PIPELINE COMPLETE  v{PIPELINE_VERSION}")
        logging.info("═"*60)
        n_cliffs = int(self.t1_agg.get("is_activity_cliff", pd.Series(False)).sum()) if len(self.t1_agg) else 0
        rows = [
            ("pad_t1_ic50_aggregated.csv",    f"{len(self.t1_agg):>7,}"),
            ("pad_t1_ic50_strict_v17.csv",    f"{len(self.t1_strict):>7,}"),
            ("pad_t1_ic50_ml_ready_v17.csv",  f"{len(self.t1_agg):>7,}"),
            ("pad_t1_confirmed.csv",           f"{len(self.t1_confirmed):>7,}  [n≥2, std≤0.5]"),
            ("pad_t1_non_covalent.csv",        f"{n_non_cov:>7,}"),
            ("pad_t1_covalent.csv",            f"{n_cov_hm:>7,}"),
            ("pad_t1_strict_stereo.csv",       f"{n_strict_stereo:>7,}"),
            ("pad_classification_v17.csv",     "          ✓"),
            ("pad_multifidelity_v17.csv",      "          ✓"),
            ("pad_t2_censored.csv",            f"{len(self.t2_df):>7,}"),
            ("pad_t3_hts_denoised.csv",        f"{len(self.t3_dn):>7,}"),
            ("pad_t3_hts_balanced.csv",        f"{len(self.t3_bal):>7,}"),
            ("pad_t3_hts_indomain.csv",        "          ✓"),
            ("pad_ki_clean.csv",               f"{len(self.ki_df):>7,}"),
            ("pad_replicates_full.csv",        "          ✓"),
            ("ad_stats_v17.json",              "          ✓"),
            ("dataset_summary.json",           "          ✓"),
            ("pipeline_config.json",           "          ✓"),
            ("pad_activity_cliffs.csv",         f"          ✓  [{len(self.cliff_df):,} pairs]"),
            ("pad_qc_report.json",             "          ✓"),
        ]
        logging.info(_table(rows, ["Output","Rows/Status"]))
        logging.info(f"\nQC: {status}")
        logging.info(f"  T1_high: {n_t1_high:,}  T1_confirmed: {n_t1_conf:,}  Total T1: {len(self.t1_agg):,}")
        logging.info(f"  Activity cliffs: {n_cliffs:,}")
        logging.info(f"  pH method: {'Dimorphite-DL' if _DIMORPHITE_OK else 'RDKit fallback'}")
        logging.info(f"  Stereo: {'FindPotentialStereo (E/Z+tetrahedral)' if _FIND_POTENTIAL_STEREO_OK else 'legacy stereo census'}")
        logging.info(f"  Output: {self.output_dir}")

# ══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════
def _find_repo_root() -> Optional[Path]:
    """
    Walk up from this file's location looking for a directory that contains
    `data/raw`. Returns None if not found within 4 levels.

    This makes path defaults independent of where the user invokes the
    script from — `python scripts/foo.py` and `python ../scripts/foo.py`
    both resolve the same way.
    """
    here = Path(__file__).resolve().parent
    for parent in [here, *here.parents][:5]:
        if (parent / "data" / "raw").exists():
            return parent
    return None

def _default_input_dir() -> str:
    """
    Resolve a sensible default input directory.

    Search order:
      1. <repo_root>/data/raw  (repo root inferred from __file__)
      2. <cwd>/data/raw        (legacy CWD-based behavior)
      3. <cwd>                 (final fallback)
    The CLI accepts --input_dir to override.
    """
    repo = _find_repo_root()
    if repo is not None:
        return str(repo / "data" / "raw")
    cwd = Path.cwd()
    for candidate in [cwd / "data" / "raw", cwd]:
        if candidate.exists():
            return str(candidate)
    return str(cwd)

def _default_output_dir() -> str:
    """
    Resolve a sensible default output directory (`<repo_root>/data/processed`).
    Falls back to CWD-based logic, then `<input>/curated_v17`.
    """
    repo = _find_repo_root()
    if repo is not None:
        return str(repo / "data" / "processed")
    cwd = Path.cwd()
    proc = cwd / "data" / "processed"
    if proc.parent.exists():
        return str(proc)
    return str(cwd / "curated_v17")

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            f"PAD4 Inhibitor Data Curation Pipeline v{PIPELINE_VERSION}\n"
            "v17.1: cleaned v17.0 — validation-set scaffolding removed,\n"
            "       duplicate functions deduplicated, Stage 2.6 consolidated,\n"
            "       repo-aware paths, deterministic seeding.\n"
            "Install extras: pip install dimorphite-dl scikit-learn"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input_dir",   type=str, default=_default_input_dir(),
                        help="Directory containing raw data (recurses into pubchem/, bindingdb/, chembl/)")
    parser.add_argument("--output_dir",  type=str, default=None,
                        help="Output directory (default: data/processed if it exists, else <input>/curated_v17)")
    parser.add_argument("--log_level",   type=str, default="INFO",
                        choices=["DEBUG","INFO","WARNING","ERROR"])
    parser.add_argument("--n_workers",   type=int, default=1,
                        help="Parallel workers for structure standardization")
    parser.add_argument("--stereo_mode", type=str, default="relaxed",
                        choices=["relaxed","strict"],
                        help="relaxed: retain undefined stereo | strict: drop undefined")
    args = parser.parse_args()
    output_dir = args.output_dir or _default_output_dir()
    pipeline = PAD4Pipeline(
        input_dir   = args.input_dir,
        output_dir  = output_dir,
        log_level   = args.log_level,
        n_workers   = args.n_workers,
        stereo_mode = args.stereo_mode,
    )
    return pipeline.run()

if __name__ == "__main__":
    sys.exit(main())