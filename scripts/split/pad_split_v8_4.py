#!/usr/bin/env python3
"""
PAD4 Splitting v8.4 — Cliff-Detection + Coverage-Accounting Fix
================================================================
Patches over v8.3. Fixes four bugs that prevented the cliff-aware split
from actually using cliff information, plus two correctness issues found
in v8.3 review.

Changes from v8.3 → v8.4
-------------------------
  [FIX-11] _detect_cliff_id_columns(): added recognition for the
           cliff-file column naming used in this project's preprocessing
           pipeline: (inchikey_1, inchikey_2). Previously, files with
           these column names silently fell back to scaffold split with
           zero usable pairs, yielding cliff_test_coverage_pct = 0.

  [FIX-12] split_cliff_aware(): cliff IDs are now truncated to the first
           14 characters before lookup. The cliff-pairs file uses full
           27-character InChIKeys (e.g. AAUHSICUDJELSM-FWLHTVFYSA-N) but
           T1 uses 14-character prefixes (AAUHSICUDJELSM). Without this
           truncation, zero pairs map even when column detection works.

  [FIX-13] split_cliff_aware(): cliff_test_coverage_pct is now recomputed
           AFTER the IK14 post-hoc dedup step. In v8.3 the IK14 dedup
           could move cliff-derived test compounds to train, but the
           reported coverage was the pre-dedup count, inflating the
           number that goes into the paper.

  [FIX-14] generate_scaffold_cv_folds(): tiebreak between equally-empty
           folds is now randomised. Previously largest-first scaffold
           order combined with deterministic argmin systematically
           placed the largest scaffolds in fold 0.

  [FIX-15] split_scaffold_capped(): singleton stratification was using
           np.linspace(...).astype(int) which produces duplicate indices
           when target ≈ pool size. Switched to np.unique() de-dup and
           proper accounting so no singleton is silently lost.

  [FIX-16] split_cliff_aware(): added an explicit "fell back because of
           empty/unmappable cliff file" log line distinct from "no file
           provided", so silent fallback is visible in the run log.

Cliff-aware coverage expectations on small datasets
---------------------------------------------------
With N=2845 compounds, test_budget = ceil(0.15 × 2845) = 427, and at
most ~300 unique compounds participate in cliff pairs (lower-activity
member of ~half the pairs voted to test). Realistic
cliff_test_coverage_pct on this dataset ranges 30–40%, NOT ≥50%.
Docstring updated to reflect this honestly.

Changes from v8.2 → v8.3 (preserved)
-------------------------------------
  See v8.3 changelog. All v8.3 features retained:
  - split_cliff_aware (sixth benchmark split)
  - generate_scaffold_cv_folds
  - FIX-6 through FIX-10

Usage
-----
  cd /home/nidhal/PAD4_BENCH
  python scripts/pad_split_v8_4.py --mode both --force \\
      --cliff_pairs_file data/processed/pad_activity_cliffs.csv

  (--force is required because v8.3 left a frozen BENCHMARK_HASH.txt.)
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import math
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    from rdkit.DataStructs import ConvertToNumpyArray
    from rdkit.ML.Cluster import Butina
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    log.warning("RDKit not found — similarity split will be skipped")

# ══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════
PIPELINE_VERSION = "v8.4"

BUTINA_N_LIMIT = 10_000
MAX_IMBALANCE_DRIFT_PP = 20.0

SPLIT_NAMES = ["scaffold", "random", "similarity", "confirmed", "lead_opt", "cliff_aware"]
SUBSETS = ["train", "val", "test"]

SIMILARITY_CUTOFFS = [0.40, 0.50, 0.55, 0.60, 0.65, 0.35, 0.30, 0.25]
MIN_VAL_FRAC = 0.08

EXPECTED_MATCH   = {"random"}
EXPECTED_DIFFER  = {"scaffold", "similarity"}
SOFT_DIFFER      = {"lead_opt", "confirmed", "cliff_aware"}

DISJOINT_SPLITS  = frozenset({"scaffold", "confirmed"})

# ══════════════════════════════════════════════════════════════════════
# REPO-AWARE PATHS
# ══════════════════════════════════════════════════════════════════════
def _find_repo_root() -> Optional[Path]:
    here = Path(__file__).resolve().parent
    for parent in [here, *here.parents][:5]:
        if (parent / "data" / "processed").exists() or (parent / "data" / "raw").exists():
            return parent
    return None

def _default_processed_dir() -> str:
    repo = _find_repo_root()
    if repo is not None:
        return str(repo / "data" / "processed")
    return str(Path.cwd())

def _default_output_dir() -> str:
    repo = _find_repo_root()
    if repo is not None:
        return str(repo / "data" / "splits")
    return str(Path.cwd() / "splits_v8_4")

# ══════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════
def hash_df(df: pd.DataFrame) -> str:
    df_sorted = df.reindex(sorted(df.columns), axis=1).reset_index(drop=True)
    h = pd.util.hash_pandas_object(df_sorted, index=True)
    return hashlib.md5(h.values.tobytes()).hexdigest()

def hash_config(cfg: dict) -> str:
    return hashlib.md5(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()

def md5_file(path: Path, chunk: int = 65536) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()

def get_inchikey14(df: pd.DataFrame) -> Optional[pd.Series]:
    for col in ["inchikey_14", "InChIKey_14"]:
        if col in df.columns:
            return df[col].astype(str).str[:14]
    for col in ["inchikey", "InChIKey"]:
        if col in df.columns:
            return df[col].astype(str).str[:14]
    return None

def norm_scaffold(val):
    """Normalise scaffold IDs to strings without '.0' float artifacts."""
    if isinstance(val, pd.Series):
        s = val.astype(str)
        s = s.str.replace(r"\.0$", "", regex=True)
        return s
    s = str(val)
    return s[:-2] if s.endswith(".0") else s

def get_ecfp4(smi: str, n_bits: int = 2048):
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=2, nBits=n_bits, useChirality=True
    )

def fps_to_numpy(fps: list, n_bits: int = 2048) -> np.ndarray:
    arr = np.zeros((len(fps), n_bits), dtype=np.uint8)
    for i, fp in enumerate(fps):
        if fp is not None:
            ConvertToNumpyArray(fp, arr[i])
    return arr

def featurise(df: pd.DataFrame, smiles_col: str, label: str,
              n_bits: int = 2048) -> Tuple[pd.DataFrame, list, np.ndarray]:
    fps, mask = [], []
    for smi in df[smiles_col]:
        fp = get_ecfp4(smi, n_bits)
        fps.append(fp)
        mask.append(fp is not None)
    n_bad = sum(1 for v in mask if not v)
    valid_df = df[mask].copy().reset_index(drop=True)
    valid_fps = [fp for fp, v in zip(fps, mask) if v]
    arr = fps_to_numpy(valid_fps, n_bits)
    if n_bad:
        log.warning(f"  {label}: {n_bad} invalid SMILES dropped")
    log.info(f"  {label}: {len(valid_df):,} featurised (chirality-aware ECFP4)")
    return valid_df, valid_fps, arr

def tanimoto_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a.astype(np.float32), b.astype(np.float32)
    dot = a @ b.T
    sum_a = a.sum(1, keepdims=True)
    sum_b = b.sum(1, keepdims=True).T
    denom = sum_a + sum_b - dot
    denom = np.where(denom == 0, 1e-9, denom)
    return (dot / denom).astype(np.float32)

def nn_sim_batch(q: np.ndarray, r: np.ndarray,
                 batch: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    n = len(q)
    mx = np.zeros(n, dtype=np.float32)
    idx = np.zeros(n, dtype=np.int32)
    for s in range(0, n, batch):
        e = min(s + batch, n)
        sim = tanimoto_matrix(q[s:e], r)
        mx[s:e] = sim.max(1)
        idx[s:e] = sim.argmax(1)
    return mx, idx

def cross_sim_stats(test_arr: np.ndarray, train_arr: np.ndarray) -> dict:
    if not len(test_arr) or not len(train_arr):
        return {}
    mx, _ = nn_sim_batch(test_arr, train_arr)
    return {
        "max_nn_tanimoto": float(mx.max()),
        "mean_nn_tanimoto": float(mx.mean()),
        "pct_above_0.70": round(float((mx > 0.70).mean() * 100), 1),
        "pct_above_0.65": round(float((mx > 0.65).mean() * 100), 1),
        "pct_below_0.40": round(float((mx < 0.40).mean() * 100), 1),
        "difficulty_score": round(float(1.0 - mx.mean()), 3),
    }

@contextlib.contextmanager
def _timed(label: str):
    """Context manager that logs wall-clock time for each split."""
    t0 = time.time()
    yield
    log.info(f"  [{label}] completed in {time.time()-t0:.1f}s")

# ══════════════════════════════════════════════════════════════════════
# DEDUPLICATION
# ══════════════════════════════════════════════════════════════════════
_STEREO_PREF = {"defined": 0, "achiral": 1, "undefined": 2}

def _stereo_rank(v) -> int:
    if isinstance(v, str):
        return _STEREO_PREF.get(v.lower().strip(), 3)
    return 3

def dedupe_by_inchikey14(
    df: pd.DataFrame,
    label: str,
    id_col: str = "inchikey_14",
    stereo_col: str = "stereo_flag",
    weight_col: str = "ml_weight",
    max_loss_frac: float = 0.25,
) -> pd.DataFrame:
    if id_col not in df.columns:
        log.warning(f"  {label}: no {id_col} column; skipping dedup")
        return df
    n_in = len(df)
    if n_in == 0:
        return df
    df = df.copy()
    df["_stereo_rank"] = df[stereo_col].apply(_stereo_rank) if stereo_col in df.columns else 3
    df["_weight"]      = df[weight_col].fillna(0.0) if weight_col in df.columns else 0.0
    df["_row_idx"]     = np.arange(n_in)
    df_sorted = df.sort_values(
        by=["_stereo_rank", "_weight", "_row_idx"],
        ascending=[True, False, True],
    )
    df_out = (df_sorted
              .drop_duplicates(subset=[id_col], keep="first")
              .drop(columns=["_stereo_rank", "_weight", "_row_idx"])
              .sort_values(by=id_col)
              .reset_index(drop=True))
    n_out     = len(df_out)
    n_dropped = n_in - n_out
    drop_frac = n_dropped / max(1, n_in)
    if n_dropped == 0:
        log.info(f"  {label}: dedup -> no duplicates ({n_in} rows kept)")
    else:
        log.info(f"  {label}: dedup -> removed {n_dropped} rows ({100*drop_frac:.1f}%), kept {n_out}")
    if drop_frac > max_loss_frac:
        raise RuntimeError(
            f"{label}: dedup would remove {100*drop_frac:.1f}% of rows "
            f"(> {100*max_loss_frac:.0f}% threshold). Aborting."
        )
    return df_out

# ══════════════════════════════════════════════════════════════════════
# SCAFFOLD STATS
# ══════════════════════════════════════════════════════════════════════
def scaffold_stats(df: pd.DataFrame, label: str, scaffold_col: str) -> dict:
    if scaffold_col not in df.columns:
        return {"label": label, "error": "no scaffold column"}
    counts = df[scaffold_col].value_counts()
    top_2 = counts.iloc[:2].sum() if len(counts) >= 2 else counts.sum()
    return {
        "label": label,
        "n_compounds": len(df),
        "n_scaffolds": int(len(counts)),
        "n_singletons": int((counts == 1).sum()),
        "pct_singletons": round(100 * (counts == 1).mean(), 1),
        "largest_scaffold_size": int(counts.iloc[0]) if len(counts) else 0,
        "top2_coverage": int(top_2),
        "top2_coverage_pct": round(100 * top_2 / len(df), 1) if len(df) else 0.0,
        "top5_sizes": counts.head(5).tolist(),
    }

def log_scaffold_health(stats_d: dict):
    if "error" in stats_d:
        log.warning(f"  {stats_d['label']}: {stats_d['error']}")
        return
    s = stats_d
    log.info(
        f"  {s['label']}: n={s['n_compounds']:,} scaffolds={s['n_scaffolds']} "
        f"singletons={s['pct_singletons']:.0f}% top2={s['top2_coverage']} "
        f"({s['top2_coverage_pct']:.0f}%) max_size={s['largest_scaffold_size']}"
    )

# ══════════════════════════════════════════════════════════════════════
# LEAKAGE AUDIT
# ══════════════════════════════════════════════════════════════════════
def audit_split(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    split_name: str,
    scaffold_col: str = "scaffold_id",
    activity_col: str = "pIC50",
    allow_scaffold_overlap: bool = False,
    strict_inchikey: bool = True,
) -> dict:
    report = {
        "split": split_name,
        "hashes": {
            "train": hash_df(train), "val": hash_df(val), "test": hash_df(test),
        },
        "errors": [],
    }
    if scaffold_col in train.columns:
        tr_sc = set(train[scaffold_col].dropna())
        te_sc = set(test[scaffold_col].dropna())
        va_sc = set(val[scaffold_col].dropna())
        tt_ov = tr_sc & te_sc
        tv_ov = tr_sc & va_sc
        report["scaffold_train_test_overlap"] = len(tt_ov)
        report["scaffold_train_val_overlap"]  = len(tv_ov)
        if tt_ov and not allow_scaffold_overlap:
            msg = f"SCAFFOLD LEAKAGE in {split_name}: {len(tt_ov)} in train∩test"
            report["errors"].append(msg)
            raise RuntimeError(msg)
        elif tt_ov and allow_scaffold_overlap:
            log.info(f"  [OK] {split_name}: {len(tt_ov)} scaffolds shared (expected)")
        else:
            log.info(f"  [OK] {split_name}: no scaffold leakage")

    ik14_tr = get_inchikey14(train)
    ik14_te = get_inchikey14(test)
    ik14_va = get_inchikey14(val)
    if ik14_tr is not None and ik14_te is not None:
        tt_ik = set(ik14_tr) & set(ik14_te)
        tv_ik = set(ik14_tr) & (set(ik14_va) if ik14_va is not None else set())
        report["inchikey14_train_test_overlap"] = len(tt_ik)
        report["inchikey14_train_val_overlap"]  = len(tv_ik)
        if tt_ik:
            msg = f"INCHIKEY-14 LEAKAGE in {split_name}: {len(tt_ik)} compounds"
            if strict_inchikey:
                report["errors"].append(msg)
                raise RuntimeError(msg)
            else:
                log.warning(f"  {msg} (tolerated)")
        else:
            log.info(f"  [OK] {split_name}: no InChIKey-14 leakage")

    if activity_col in train.columns:
        tr_act = float((train[activity_col] >= 6.0).mean() * 100)
        te_act = float((test[activity_col] >= 6.0).mean() * 100)
        va_act = float((val[activity_col] >= 6.0).mean() * 100)
        drift_te = abs(tr_act - te_act)
        report["imbalance"] = {
            "train_pct_active": round(tr_act, 1),
            "val_pct_active":   round(va_act, 1),
            "test_pct_active":  round(te_act, 1),
            "train_test_drift_pp": round(drift_te, 1),
        }
        if drift_te > MAX_IMBALANCE_DRIFT_PP:
            log.warning(f"  [!!] {split_name}: activity drift {drift_te:.1f}pp")
        else:
            log.info(f"  [OK] {split_name}: activity drift {drift_te:.1f}pp")
    return report

def ks_check(tr: pd.DataFrame, ot: pd.DataFrame, split_type: str,
             act: str = "pIC50", label: str = "") -> dict:
    if act not in tr.columns:
        return {}
    stat, p = stats.ks_2samp(tr[act].dropna(), ot[act].dropna())
    result = {"ks_stat": round(float(stat), 4), "ks_p": float(p)}

    if split_type in EXPECTED_MATCH:
        if p < 0.05:
            msg, unexp = "unexpected distribution shift", True
            log.warning(f"  KS train vs {label}: stat={stat:.3f} p={p:.2e} — {msg}")
        else:
            msg, unexp = "distributions match (expected)", False
            log.info(f"  KS train vs {label}: stat={stat:.3f} p={p:.2f} — {msg}")

    elif split_type in EXPECTED_DIFFER:
        if p < 0.05:
            msg, unexp = "distributions differ (expected)", False
            log.info(f"  KS train vs {label}: stat={stat:.3f} p={p:.2e} — {msg}")
        else:
            msg, unexp = "unexpected distribution similarity", True
            log.warning(f"  KS train vs {label}: stat={stat:.3f} p={p:.2f} — {msg}")

    elif split_type == "confirmed":
        if stat < 0.15:
            msg, unexp = "mild distribution shift (acceptable)", False
            log.info(f"  KS train vs {label}: stat={stat:.3f} p={p:.2e} — {msg}")
        else:
            msg, unexp = "elevated shift (review confirmed-set scaffold allocation)", True
            log.warning(f"  KS train vs {label}: stat={stat:.3f} p={p:.2e} — {msg}")

    elif split_type in SOFT_DIFFER:
        if stat < 0.15:
            msg, unexp = "mild distribution shift (acceptable)", False
            log.info(f"  KS train vs {label}: stat={stat:.3f} p={p:.2e} — {msg}")
        elif stat < 0.25:
            msg, unexp = "moderate shift (review)", False
            log.warning(f"  KS train vs {label}: stat={stat:.3f} p={p:.2e} — {msg}")
        else:
            msg, unexp = "strong shift (verify stratification)", True
            log.warning(f"  KS train vs {label}: stat={stat:.3f} p={p:.2e} — {msg}")

    else:
        msg   = "distributions differ" if p < 0.05 else "distributions match"
        unexp = False
        log.info(f"  KS train vs {label}: stat={stat:.3f} p={p:.2e} — {msg}")

    result["note"]       = msg
    result["unexpected"] = unexp
    return result

# ══════════════════════════════════════════════════════════════════════
# SCAFFOLD SPLIT
# ══════════════════════════════════════════════════════════════════════
def split_scaffold_capped(
    df: pd.DataFrame,
    test_frac: float = 0.15,
    val_frac: float = 0.10,
    seed: int = 42,
    scaffold_col: str = "scaffold_id",
    cap_frac: float = 0.08,
    activity_col: str = "pIC50",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    df  = df.copy().reset_index(drop=True)
    n   = len(df)
    if scaffold_col not in df.columns:
        raise ValueError(f"'{scaffold_col}' not in data")

    test_budget   = int(math.ceil(test_frac * n))
    val_budget    = int(math.ceil(val_frac * n))
    scaffold_cap  = max(1, int(math.ceil(cap_frac * test_budget)))
    log.info(f"  n={n} test_budget={test_budget} val_budget={val_budget} cap={scaffold_cap}")

    scaf_to_idx: Dict[str, List[int]] = defaultdict(list)
    for i, s in enumerate(df[scaffold_col]):
        scaf_to_idx[str(s)].append(i)

    multi_scafs    = [s for s, ixs in scaf_to_idx.items() if len(ixs) > 1]
    singleton_scafs = [s for s, ixs in scaf_to_idx.items() if len(ixs) == 1]
    multi_sorted   = sorted(multi_scafs, key=lambda s: len(scaf_to_idx[s]))

    oversized = [s for s in multi_sorted if len(scaf_to_idx[s]) > scaffold_cap]
    eligible  = [s for s in multi_sorted if len(scaf_to_idx[s]) <= scaffold_cap]
    n_oversized_cmpds = sum(len(scaf_to_idx[s]) for s in oversized)
    log.info(
        f"  {len(multi_scafs)} multi scaffolds, {len(singleton_scafs)} singletons | "
        f"{len(oversized)} oversized scaffolds ({n_oversized_cmpds} compounds) forced to train"
    )

    test_idx:  List[int] = []
    val_idx:   List[int] = []
    train_idx: List[int] = []

    singleton_test_reserve  = int(math.ceil(0.10 * test_budget))
    multi_test_target       = test_budget - singleton_test_reserve

    if eligible:
        eligible_sizes = np.array([len(scaf_to_idx[s]) for s in eligible])
        try:
            bin_edges = np.quantile(eligible_sizes, [0, 0.25, 0.5, 0.75, 1.0])
            bin_edges = np.unique(bin_edges)
            bin_idx   = np.digitize(eligible_sizes, bin_edges[1:-1])
            n_bins    = len(bin_edges) - 1 if len(bin_edges) > 1 else 1
        except Exception:
            bin_idx = np.zeros(len(eligible), dtype=int)
            n_bins  = 1

        bin_to_scafs: Dict[int, List[str]] = defaultdict(list)
        for s, b in zip(eligible, bin_idx):
            bin_to_scafs[int(b)].append(s)
        for b in bin_to_scafs:
            rng.shuffle(bin_to_scafs[b])

        bin_compositions = {
            int(b): {
                "n_scafs":     len(scafs),
                "n_compounds": int(sum(len(scaf_to_idx[s]) for s in scafs)),
                "size_range":  (
                    int(min(len(scaf_to_idx[s]) for s in scafs)),
                    int(max(len(scaf_to_idx[s]) for s in scafs)),
                ) if scafs else (0, 0),
            }
            for b, scafs in bin_to_scafs.items()
        }
        log.info(f"  scaffold-size bins: {bin_compositions}")

        total_eligible_compounds = sum(c["n_compounds"] for c in bin_compositions.values())
        bin_test_budgets = {}
        for b, comp in bin_compositions.items():
            frac = comp["n_compounds"] / max(1, total_eligible_compounds)
            bin_test_budgets[b] = int(round(multi_test_target * frac))

        for b in sorted(bin_to_scafs.keys()):
            filled_in_bin = 0
            budget_b = bin_test_budgets.get(b, 0)
            for scaf in bin_to_scafs[b]:
                size = len(scaf_to_idx[scaf])
                if size > scaffold_cap:
                    continue
                if filled_in_bin >= budget_b:
                    break
                if filled_in_bin + size > budget_b + scaffold_cap:
                    continue
                test_idx.extend(scaf_to_idx[scaf])
                filled_in_bin += size

    assigned_to_test = {
        scaf for scaf in multi_sorted
        if all(i in set(test_idx) for i in scaf_to_idx[scaf])
    }

    for scaf in multi_sorted:
        if scaf in assigned_to_test:
            continue
        size = len(scaf_to_idx[scaf])
        if len(val_idx) >= val_budget:
            break
        if size > scaffold_cap:
            continue
        if len(val_idx) + size > val_budget:
            continue
        val_idx.extend(scaf_to_idx[scaf])

    assigned_to_val = {
        scaf for scaf in multi_sorted
        if all(i in set(val_idx) for i in scaf_to_idx[scaf])
        and scaf not in assigned_to_test
    }

    for scaf in multi_sorted:
        if scaf in assigned_to_test or scaf in assigned_to_val:
            continue
        train_idx.extend(scaf_to_idx[scaf])

    singleton_idxs = [scaf_to_idx[s][0] for s in singleton_scafs]
    if singleton_idxs and activity_col in df.columns:
        singleton_idxs_sorted = sorted(singleton_idxs, key=lambda i: df.at[i, activity_col])
        remaining_test_budget = max(0, test_budget - len(test_idx))
        remaining_val_budget  = max(0, val_budget  - len(val_idx))

        total_singleton_to_test = min(remaining_test_budget, len(singleton_idxs_sorted))
        total_singleton_to_val  = min(
            remaining_val_budget, len(singleton_idxs_sorted) - total_singleton_to_test
        )

        # [FIX-15] dedupe linspace indices so no singleton is silently lost
        if total_singleton_to_test > 0:
            raw_idx     = np.unique(
                np.linspace(0, len(singleton_idxs_sorted) - 1,
                            total_singleton_to_test).astype(int)
            )
            test_sample = [singleton_idxs_sorted[i] for i in raw_idx]
            test_idx.extend(test_sample)
            remaining   = [i for i in singleton_idxs_sorted if i not in set(test_sample)]
        else:
            remaining = singleton_idxs_sorted

        if total_singleton_to_val > 0 and remaining:
            raw_idx    = np.unique(
                np.linspace(0, len(remaining) - 1,
                            total_singleton_to_val).astype(int)
            )
            val_sample = [remaining[i] for i in raw_idx]
            val_idx.extend(val_sample)
            remaining  = [i for i in remaining if i not in set(val_sample)]

        train_idx.extend(remaining)
        log.info(
            f"  singletons -> test:{total_singleton_to_test} "
            f"val:{total_singleton_to_val} train:{len(remaining)}"
        )
    else:
        train_idx.extend(singleton_idxs)

    train_set, val_set, test_set = set(train_idx), set(val_idx), set(test_idx)
    assert not (train_set & test_set), "train/test overlap"
    assert not (train_set & val_set),  "train/val overlap"
    assert not (val_set   & test_set), "val/test overlap"
    assigned = train_set | val_set | test_set
    missing  = set(range(n)) - assigned
    if missing:
        log.warning(f"  {len(missing)} unassigned rows; adding to train")
        train_idx.extend(sorted(missing))

    tr_scafs = set(df.iloc[train_idx][scaffold_col])
    te_scafs = set(df.iloc[test_idx][scaffold_col])
    va_scafs = set(df.iloc[val_idx][scaffold_col])
    leaks_tt = tr_scafs & te_scafs
    leaks_tv = tr_scafs & va_scafs
    if leaks_tt or leaks_tv:
        raise RuntimeError(
            f"Internal scaffold leakage (tt={len(leaks_tt)}, tv={len(leaks_tv)})"
        )

    return (
        df.iloc[sorted(train_idx)].copy(),
        df.iloc[sorted(val_idx)].copy(),
        df.iloc[sorted(test_idx)].copy(),
    )

# ══════════════════════════════════════════════════════════════════════
# SIMILARITY SPLIT
# ══════════════════════════════════════════════════════════════════════
def _split_similarity_once(
    df: pd.DataFrame,
    smiles_col: str,
    cutoff: float,
    test_frac: float,
    val_frac: float,
    seed: int,
    scaffold_col: str,
    cap_frac: float,
    quiet: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    if not HAS_RDKIT:
        raise RuntimeError("RDKit required")

    valid_df, valid_fps, valid_arr = featurise(df, smiles_col, "Similarity")
    n = len(valid_df)
    if n > BUTINA_N_LIMIT:
        raise RuntimeError(
            f"Similarity split: N={n:,} exceeds hard limit {BUTINA_N_LIMIT:,}."
        )

    test_budget   = int(math.ceil(test_frac * n))
    val_budget    = int(math.ceil(val_frac * n))
    cluster_cap   = max(1, int(math.ceil(cap_frac * test_budget)))
    if not quiet:
        log.info(
            f"  n={n} test_budget={test_budget} val_budget={val_budget} "
            f"cluster_cap={cluster_cap}"
        )
        log.info(f"  Butina clustering at cutoff={cutoff:.2f}")

    dists = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(valid_fps[i], valid_fps[:i])
        dists.extend([1.0 - s for s in sims])

    clusters = Butina.ClusterData(dists, n, 1.0 - cutoff, isDistData=True)
    n_cl                  = len(clusters)
    n_sin                 = sum(1 for c in clusters if len(c) == 1)
    largest_cluster_size  = len(clusters[0]) if clusters else 0
    largest_cluster_frac  = largest_cluster_size / n if n > 0 else 0.0

    if not quiet:
        log.info(
            f"  {n_cl:,} clusters | largest={largest_cluster_size} | "
            f"singletons={n_sin} ({100*n_sin/max(1,n_cl):.0f}%)"
        )

    rng = np.random.default_rng(seed)
    clusters_list = list(clusters)
    rng.shuffle(clusters_list)
    clusters_asc = sorted(clusters_list, key=len)

    test_idx:  List[int] = []
    val_idx:   List[int] = []
    train_idx: List[int] = []

    for cl in clusters_asc:
        idxs = list(cl)
        size = len(idxs)
        if len(test_idx) < test_budget:
            if size > cluster_cap:
                train_idx.extend(idxs)
            elif len(test_idx) + size > test_budget:
                if len(val_idx) + size <= val_budget:
                    val_idx.extend(idxs)
                else:
                    train_idx.extend(idxs)
            else:
                test_idx.extend(idxs)
        elif len(val_idx) < val_budget and size <= cluster_cap:
            if len(val_idx) + size <= val_budget:
                val_idx.extend(idxs)
            else:
                train_idx.extend(idxs)
        else:
            train_idx.extend(idxs)

    tr_arr   = valid_arr[sorted(train_idx)]
    te_arr   = valid_arr[sorted(test_idx)]
    sim_stats = cross_sim_stats(te_arr, tr_arr)

    mx = sim_stats.get("max_nn_tanimoto",  0.0)
    mn = sim_stats.get("mean_nn_tanimoto", 0.0)
    if not quiet:
        log.info(
            f"  cross-split Tanimoto: max={mx:.3f} mean={mn:.3f} "
            f"difficulty={sim_stats.get('difficulty_score', 0):.3f}"
        )

    sim_stats.update({
        "butina_cutoff":            cutoff,
        "n_clusters":               n_cl,
        "n_singletons":             n_sin,
        "largest_cluster_size":     largest_cluster_size,
        "largest_cluster_fraction": round(largest_cluster_frac, 4),
        "cluster_cap":              cluster_cap,
        "paper_note": (
            f"Butina at Tanimoto={cutoff:.2f} → {n_cl} clusters. "
            f"Test from smallest clusters (hard OOD), cap={cluster_cap}. "
            f"Mean NN Tanimoto to train: {mn:.3f}."
        ),
    })

    return (
        valid_df.iloc[sorted(train_idx)].copy(),
        valid_df.iloc[sorted(val_idx)].copy(),
        valid_df.iloc[sorted(test_idx)].copy(),
        sim_stats,
    )


def split_similarity_adaptive(
    df: pd.DataFrame,
    smiles_col: str = "canonical_smiles",
    cutoffs: Optional[List[float]] = None,
    test_frac: float = 0.15,
    val_frac: float = 0.10,
    seed: int = 42,
    scaffold_col: str = "scaffold_id",
    cap_frac: float = 0.10,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    if cutoffs is None:
        cutoffs = SIMILARITY_CUTOFFS

    best_result = None
    best_score  = -999999
    best_cutoff = None
    best_stats  = None
    tried       = []

    for cutoff in cutoffs:
        tried.append(cutoff)
        is_first = (cutoff == cutoffs[0])
        if not is_first:
            log.info(f"  Retrying at {cutoff:.2f}...")

        try:
            tr, va, te, sim_stats = _split_similarity_once(
                df, smiles_col, cutoff, test_frac, val_frac, seed,
                scaffold_col, cap_frac, quiet=not is_first,
            )
        except Exception as e:
            log.warning(f"  Similarity cutoff {cutoff:.2f} failed: {e}")
            continue

        n_total     = len(tr) + len(va) + len(te)
        test_budget = int(math.ceil(test_frac * n_total))
        largest_frac = sim_stats.get("largest_cluster_fraction", 0)
        n_cl         = sim_stats.get("n_clusters", 0)
        val_size     = len(va)
        test_size    = len(te)

        failures = []
        if largest_frac > 0.50:
            failures.append(f"largest cluster {largest_frac*100:.1f}%")
        if n_cl < 100:
            failures.append(f"only {n_cl} clusters")
        if val_size == 0:
            failures.append(f"val={val_size}")
        if test_size < 0.8 * test_budget:
            failures.append(f"test={test_size} < {0.8*test_budget:.0f}")

        if not failures:
            if not is_first:
                log.info(f"  Similarity cutoff {cutoff:.2f} accepted.")
            sim_stats["selected_cutoff"] = cutoff
            sim_stats["adaptive_tried"]  = tried
            sim_stats["degenerate"]      = False
            return tr, va, te, sim_stats

        log.warning(f"  Similarity cutoff {cutoff:.2f} rejected:")
        for f in failures:
            log.warning(f"    {f}")

        score  = 0
        score -= largest_frac * 200
        score -= max(0, 100 - n_cl)
        score -= 100 if val_size == 0 else 0
        score -= max(0, test_budget - test_size) * 2

        if score > best_score:
            best_score  = score
            best_result = (tr, va, te)
            best_cutoff = cutoff
            best_stats  = sim_stats.copy()
            best_stats["rejection_reasons"] = failures

    if best_result is None:
        raise RuntimeError("Similarity split failed at all cutoffs")

    log.warning(f"  Keeping best available cutoff={best_cutoff:.2f} (DEGENERATE)")
    tr, va, te = best_result
    best_stats["selected_cutoff"]     = best_cutoff
    best_stats["adaptive_tried"]      = tried
    best_stats["degenerate"]          = True
    best_stats["degenerate_reasons"]  = best_stats.get("rejection_reasons", [])
    best_stats["paper_note"] = (
        best_stats.get("paper_note", "") +
        " DEGENERATE: dominant chemotype prevented a clean OOD partition; "
        "report as auxiliary stress-test only, not co-equal with primary splits."
    )
    return tr, va, te, best_stats

# ══════════════════════════════════════════════════════════════════════
# RANDOM SPLIT
# ══════════════════════════════════════════════════════════════════════
def split_random(
    df: pd.DataFrame,
    test_frac: float = 0.15,
    val_frac: float = 0.10,
    seed: int = 42,
    activity_col: str = "pIC50",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    df  = df.copy().reset_index(drop=True)
    n   = len(df)

    if activity_col not in df.columns:
        all_idx = rng.permutation(n)
        n_test  = int(n * test_frac)
        n_val   = int(n * val_frac)
        return (
            df.iloc[sorted(all_idx[n_test + n_val:])].copy(),
            df.iloc[sorted(all_idx[n_test:n_test + n_val])].copy(),
            df.iloc[sorted(all_idx[:n_test])].copy(),
        )

    df["_q"] = pd.qcut(df[activity_col], q=5, labels=False, duplicates="drop")
    test_idx  = []
    val_idx   = []
    train_idx = []

    for q in sorted(df["_q"].dropna().unique()):
        qp = df[df["_q"] == q].index.tolist()
        rng.shuffle(qp)
        n_q       = len(qp)
        n_test_q  = int(round(n_q * test_frac))
        n_val_q   = int(round(n_q * val_frac))
        test_idx.extend(qp[:n_test_q])
        val_idx.extend(qp[n_test_q:n_test_q + n_val_q])
        train_idx.extend(qp[n_test_q + n_val_q:])

    assigned = set(test_idx) | set(val_idx) | set(train_idx)
    orphans  = [i for i in range(n) if i not in assigned]
    if orphans:
        log.warning(f"  split_random: {len(orphans)} orphan rows (NaN quantile) -> train")
        train_idx.extend(orphans)

    df = df.drop(columns=["_q"])
    return (
        df.iloc[sorted(train_idx)].copy(),
        df.iloc[sorted(val_idx)].copy(),
        df.iloc[sorted(test_idx)].copy(),
    )

# ══════════════════════════════════════════════════════════════════════
# CONFIRMED & LEAD OPT
# ══════════════════════════════════════════════════════════════════════
def split_confirmed(
    confirmed: pd.DataFrame,
    test_frac: float = 0.15,
    val_frac: float = 0.10,
    seed: int = 42,
    scaffold_col: str = "scaffold_id",
    cap_frac: float = 0.08,
    activity_col: str = "pIC50",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return split_scaffold_capped(
        confirmed, test_frac=test_frac, val_frac=val_frac, seed=seed,
        scaffold_col=scaffold_col, cap_frac=cap_frac, activity_col=activity_col,
    )

def split_lead_opt(
    df: pd.DataFrame,
    test_frac: float = 0.15,
    val_frac: float = 0.10,
    seed: int = 42,
    scaffold_col: str = "scaffold_id",
    min_scaffold_size: int = 4,
    activity_col: str = "pIC50",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    df  = df.copy().reset_index(drop=True)
    if scaffold_col not in df.columns:
        raise ValueError(f"'{scaffold_col}' not in data")

    ik14_series = get_inchikey14(df)
    has_ik14    = ik14_series is not None

    scaf_to_idx: Dict[str, List[int]] = defaultdict(list)
    for i, s in enumerate(df[scaffold_col]):
        scaf_to_idx[str(s)].append(i)

    eligible_scafs       = [s for s, ixs in scaf_to_idx.items() if len(ixs) >= min_scaffold_size]
    small_scafs          = [s for s, ixs in scaf_to_idx.items() if len(ixs) < min_scaffold_size]
    n_eligible_compounds = sum(len(scaf_to_idx[s]) for s in eligible_scafs)

    log.info(
        f"  lead_opt: {len(eligible_scafs)} eligible scaffolds "
        f"(size >= {min_scaffold_size}, total {n_eligible_compounds} compounds); "
        f"{len(small_scafs)} small -> train"
    )

    train_idx: List[int] = []
    val_idx:   List[int] = []
    test_idx:  List[int] = []

    for scaf in eligible_scafs:
        idxs = scaf_to_idx[scaf]
        n_s  = len(idxs)
        if activity_col in df.columns:
            acts        = [df.at[i, activity_col] for i in idxs]
            order       = np.argsort(acts)
            idxs_sorted = [idxs[o] for o in order]
        else:
            idxs_sorted = list(idxs)
            rng.shuffle(idxs_sorted)

        n_test = max(1, int(round(n_s * test_frac)))
        n_val  = max(0, int(round(n_s * val_frac)))
        if n_test + n_val >= n_s:
            n_test = max(1, n_s - 2)
            n_val  = max(0, n_s - n_test - 1)

        step_test = max(1, n_s // n_test)
        test_pick = idxs_sorted[::step_test][:n_test]
        remaining = [i for i in idxs_sorted if i not in set(test_pick)]

        if n_val > 0 and remaining:
            step_val  = max(1, len(remaining) // n_val)
            val_pick  = remaining[::step_val][:n_val]
            remaining = [i for i in remaining if i not in set(val_pick)]
        else:
            val_pick = []

        test_idx.extend(test_pick)
        val_idx.extend(val_pick)
        train_idx.extend(remaining)

    for scaf in small_scafs:
        train_idx.extend(scaf_to_idx[scaf])

    if has_ik14:
        ik14     = ik14_series.astype(str).values
        tr_keys  = set(ik14[i] for i in train_idx)
        te_over  = [i for i in test_idx if ik14[i] in tr_keys]
        va_over  = [i for i in val_idx  if ik14[i] in tr_keys]
        if te_over:
            log.warning(f"  lead_opt: {len(te_over)} IK14 dups in test -> train")
            test_idx  = [i for i in test_idx if i not in set(te_over)]
            train_idx.extend(te_over)
        if va_over:
            log.warning(f"  lead_opt: {len(va_over)} IK14 dups in val -> train")
            val_idx   = [i for i in val_idx if i not in set(va_over)]
            train_idx.extend(va_over)

    tr_set, va_set, te_set = set(train_idx), set(val_idx), set(test_idx)
    assert not (tr_set & te_set)
    assert not (tr_set & va_set)
    assert not (va_set & te_set)

    tr_scafs = set(df.iloc[train_idx][scaffold_col])
    te_scafs = set(df.iloc[test_idx][scaffold_col])
    shared   = tr_scafs & te_scafs
    log.info(f"  lead_opt: {len(shared)} scaffolds shared train↔test (BY DESIGN)")

    meta = {
        "split_type":                    "lead_optimization",
        "min_scaffold_size":             min_scaffold_size,
        "n_eligible_scaffolds":          len(eligible_scafs),
        "n_eligible_compounds":          n_eligible_compounds,
        "n_small_scaffolds_to_train":    len(small_scafs),
        "n_shared_scaffolds_train_test": len(shared),
        "paper_note": (
            f"Lead-opt split: within-scaffold sampling for scaffolds with "
            f">= {min_scaffold_size} compounds. Scaffold overlap is BY DESIGN. "
            f"InChIKey-14 disjointness strictly enforced."
        ),
    }
    return (
        df.iloc[sorted(train_idx)].copy(),
        df.iloc[sorted(val_idx)].copy(),
        df.iloc[sorted(test_idx)].copy(),
        meta,
    )

# ══════════════════════════════════════════════════════════════════════
# CLIFF-AWARE SPLIT
# ══════════════════════════════════════════════════════════════════════
def _detect_cliff_id_columns(
    cliff_df: pd.DataFrame,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Auto-detect which columns in cliff_df hold the two compound IDs.

    [FIX-11]  Added (inchikey_1, inchikey_2) — the convention used by
              this project's preprocessing pipeline (v17.x).
    """
    candidates = [
        # PAD4 v17.x preprocessing convention
        ("inchikey_1",      "inchikey_2"),
        # Generic conventions
        ("id_a",            "id_b"),
        ("inchikey_14_a",   "inchikey_14_b"),
        ("ik14_a",          "ik14_b"),
        ("compound_a",      "compound_b"),
        ("compound_id_a",   "compound_id_b"),
        ("inchikey_a",      "inchikey_b"),
    ]
    for ca, cb in candidates:
        if ca in cliff_df.columns and cb in cliff_df.columns:
            return ca, cb
    return None, None


def split_cliff_aware(
    df: pd.DataFrame,
    cliff_pairs_df: Optional[pd.DataFrame] = None,
    test_frac: float = 0.15,
    val_frac: float = 0.10,
    seed: int = 42,
    scaffold_col: str = "stereo_stripped_scaffold",
    activity_col: str = "pIC50",
    id_col: str = "inchikey_14",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Cliff-aware split (v8.4).

    Philosophy
    ----------
    Activity cliffs are pairs of structurally similar compounds
    (Tanimoto ≥ 0.8 by ECFP4) with a large activity gap (ΔpIC50 ≥ 1.5).
    They expose a failure mode unique to molecular ML: the model has a
    training analogue at high similarity but must still predict a very
    different label. Standard splits systematically under-represent this
    scenario in test because scaffold-assignment groups similar compounds
    together.

    This split deliberately reverses that: the LESS-active member of each
    cliff pair is voted into test, the more-active member into train.
    Conflicts (compound in multiple pairs) are resolved by majority
    vote; ties broken by activity (lower → test).

    Coverage expectations
    ---------------------
    The fraction of test compounds that are genuine cliff members
    (cliff_test_coverage_pct) depends on three things:

      1. Number of unique compounds in the cliff pairs file
      2. Test budget (test_frac × N)
      3. How votes resolve when a compound appears in multiple pairs

    On the PAD4 dataset (N=2845, 356 pairs, ~301 cliff compounds),
    realistic coverage is 30–45%. On larger or cliff-richer datasets it
    can reach 60–80%. Coverage below 20% is reported but should be
    treated as a degraded benchmark.

    [FIX-12]  cliff_pairs IDs are truncated to id_col_length (14 chars by
              default) so 27-character full InChIKeys from the cliff
              file map to 14-character InChIKey prefixes in T1.
    [FIX-13]  cliff_test_coverage_pct is recomputed AFTER the IK14
              dedup step so the reported number matches the saved CSV.
    [FIX-16]  Explicit log line distinguishes "no file" from "file
              present but unmappable".
    """
    rng = np.random.default_rng(seed)
    df  = df.copy().reset_index(drop=True)
    n   = len(df)

    if id_col not in df.columns:
        raise ValueError(f"id_col '{id_col}' not in dataframe")

    # Determine the canonical ID length from the dataframe itself.
    # T1 uses 14 chars; if the dataframe stores full InChIKeys we adapt.
    sample_id = str(df[id_col].iloc[0]) if len(df) else ""
    id_len    = len(sample_id) if sample_id else 14
    if id_len not in (14, 27):
        log.warning(
            f"  cliff_aware: unusual id_col length ({id_len}); "
            f"defaulting to 14-char truncation"
        )
        id_len = 14

    test_budget = int(math.ceil(test_frac * n))
    val_budget  = int(math.ceil(val_frac  * n))

    # Build lookup using the canonical length of the dataframe's IDs
    id_to_idx: Dict[str, int] = {
        str(v)[:id_len]: i for i, v in enumerate(df[id_col])
    }
    act_values: Optional[np.ndarray] = (
        df[activity_col].values if activity_col in df.columns else None
    )

    votes: Dict[int, int]  = defaultdict(int)
    n_pairs_loaded  = 0
    n_pairs_usable  = 0

    id_a_col = id_b_col = None
    fallback  = False
    fallback_reason = None

    if cliff_pairs_df is not None and len(cliff_pairs_df) > 0:
        id_a_col, id_b_col = _detect_cliff_id_columns(cliff_pairs_df)
        if id_a_col is None:
            log.warning(
                "  cliff_aware: cannot identify cliff pair ID columns "
                f"(found: {list(cliff_pairs_df.columns[:6])}). "
                "Falling back to scaffold split."
            )
            fallback = True
            fallback_reason = "column_detection_failed"
        else:
            n_pairs_loaded = len(cliff_pairs_df)
            log.info(
                f"  cliff_aware: detected ID columns: "
                f"'{id_a_col}' / '{id_b_col}'"
            )
            for _, row in cliff_pairs_df.iterrows():
                # [FIX-12] truncate to dataframe's ID length
                ida = str(row[id_a_col])[:id_len]
                idb = str(row[id_b_col])[:id_len]
                idx_a = id_to_idx.get(ida)
                idx_b = id_to_idx.get(idb)
                if idx_a is None or idx_b is None:
                    continue
                n_pairs_usable += 1
                if act_values is not None:
                    if act_values[idx_a] <= act_values[idx_b]:
                        votes[idx_a] += 1
                        votes[idx_b] -= 1
                    else:
                        votes[idx_b] += 1
                        votes[idx_a] -= 1
                else:
                    votes[idx_a] += 1
                    votes[idx_b] -= 1
            # [FIX-16] flag the case where the file was present but produced no matches
            if n_pairs_usable == 0:
                log.warning(
                    f"  cliff_aware: {n_pairs_loaded} pairs loaded but ZERO mapped to "
                    f"compounds in the dataframe (id_col='{id_col}', id_len={id_len}). "
                    "Check that the cliff file uses InChIKeys compatible with the dataset."
                )
                fallback = True
                fallback_reason = "zero_pairs_mapped"
    else:
        log.warning("  cliff_aware: no cliff pairs provided. Falling back to scaffold split.")
        fallback = True
        fallback_reason = "no_file_provided"

    if fallback:
        log.warning(
            f"  cliff_aware: running scaffold split as substitute "
            f"(reason: {fallback_reason})"
        )
        tr, va, te = split_scaffold_capped(
            df, test_frac=test_frac, val_frac=val_frac, seed=seed,
            scaffold_col=scaffold_col, activity_col=activity_col,
        )
        meta = {
            "split_type":              "cliff_aware_fallback_scaffold",
            "fallback_reason":         fallback_reason,
            "n_cliff_pairs_loaded":    n_pairs_loaded,
            "n_cliff_pairs_usable":    0,
            "cliff_test_count":        0,
            "cliff_test_pct":          0.0,
            "cliff_test_coverage_pct": 0.0,
            "paper_note":              "No usable cliff pairs; scaffold split used as fallback.",
        }
        return tr, va, te, meta

    log.info(
        f"  cliff_aware: {n_pairs_loaded} pairs loaded, "
        f"{n_pairs_usable} usable (both compounds in dataset)"
    )

    # Resolve votes
    cliff_test_idx:  List[int] = []
    cliff_train_idx: List[int] = []
    ambiguous_idx:   List[int] = []

    for idx, v in votes.items():
        if v > 0:
            cliff_test_idx.append(idx)
        elif v < 0:
            cliff_train_idx.append(idx)
        else:
            if act_values is not None and act_values[idx] < float(np.median(act_values)):
                cliff_test_idx.append(idx)
            else:
                ambiguous_idx.append(idx)

    cliff_test_set   = set(cliff_test_idx)
    cliff_train_idx  = [i for i in cliff_train_idx if i not in cliff_test_set]
    cliff_train_set  = set(cliff_train_idx)

    log.info(
        f"  cliff_aware: vote result → test_candidates={len(cliff_test_idx)}, "
        f"train_anchors={len(cliff_train_idx)}, ambiguous={len(ambiguous_idx)}"
    )

    # Cap test cliff set at budget
    if len(cliff_test_idx) > test_budget:
        if act_values is not None:
            cliff_test_idx.sort(key=lambda i: act_values[i])
        else:
            rng.shuffle(cliff_test_idx)
        cliff_test_idx = cliff_test_idx[:test_budget]
        cliff_test_set = set(cliff_test_idx)
        log.info(f"  cliff_aware: test capped at budget ({test_budget})")

    # Fill remaining test budget with non-cliff compounds
    assigned_cliff   = cliff_test_set | cliff_train_set
    free_idx         = [i for i in range(n) if i not in assigned_cliff]
    remaining_budget = max(0, test_budget - len(cliff_test_idx))

    extra_test: List[int] = []
    if remaining_budget > 0 and free_idx:
        free_df = df.iloc[sorted(free_idx)].copy()
        if activity_col in free_df.columns and len(free_df) > 5:
            free_df["_q"] = pd.qcut(
                free_df[activity_col], q=min(5, len(free_df)), labels=False, duplicates="drop"
            )
            for q in sorted(free_df["_q"].dropna().unique()):
                qp    = free_df[free_df["_q"] == q].index.tolist()
                n_q   = max(0, int(round(remaining_budget * len(qp) / len(free_df))))
                picks = list(rng.choice(qp, size=min(n_q, len(qp)), replace=False))
                extra_test.extend(picks)
            if len(extra_test) < remaining_budget:
                extra_set = set(extra_test)
                top_up    = [i for i in free_idx if i not in extra_set]
                rng.shuffle(top_up)
                extra_test.extend(top_up[:remaining_budget - len(extra_test)])
        else:
            free_perm = list(free_idx)
            rng.shuffle(free_perm)
            extra_test = free_perm[:remaining_budget]

    all_test_idx = cliff_test_idx + extra_test[:remaining_budget]
    test_set     = set(all_test_idx)

    # Fill val from non-test, non-cliff-anchor pool
    val_pool = [i for i in range(n) if i not in test_set and i not in cliff_train_set]
    val_idx: List[int] = []

    if val_pool:
        vp_df = df.iloc[sorted(val_pool)].copy()
        if activity_col in vp_df.columns and len(vp_df) > 5:
            vp_df["_q"] = pd.qcut(
                vp_df[activity_col], q=min(5, len(vp_df)), labels=False, duplicates="drop"
            )
            for q in sorted(vp_df["_q"].dropna().unique()):
                qp   = vp_df[vp_df["_q"] == q].index.tolist()
                n_q  = max(0, int(round(val_budget * len(qp) / len(vp_df))))
                picks = list(rng.choice(qp, size=min(n_q, len(qp)), replace=False))
                val_idx.extend(picks)
            if len(val_idx) < val_budget:
                val_set_now = set(val_idx)
                top_up      = [i for i in val_pool if i not in val_set_now]
                rng.shuffle(top_up)
                val_idx.extend(top_up[:val_budget - len(val_idx)])
        else:
            rng.shuffle(val_pool)
            val_idx = val_pool[:val_budget]

    val_idx  = val_idx[:val_budget]
    val_set  = set(val_idx)

    train_idx = [i for i in range(n) if i not in test_set and i not in val_set]

    # IK14 disjointness enforcement (post-hoc safety net)
    ik14 = get_inchikey14(df)
    if ik14 is not None:
        ik14_vals = ik14.values
        tr_keys   = set(ik14_vals[i] for i in train_idx)
        te_dups   = [i for i in all_test_idx if ik14_vals[i] in tr_keys]
        if te_dups:
            log.warning(
                f"  cliff_aware: {len(te_dups)} IK14 dups between test and train -> moved to train"
            )
            te_dup_set   = set(te_dups)
            all_test_idx = [i for i in all_test_idx if i not in te_dup_set]
            train_idx.extend(te_dups)
            test_set     = set(all_test_idx)

    # Final assertions
    tr_set = set(train_idx)
    va_set = set(val_idx)
    te_set = set(all_test_idx)
    assert not (tr_set & te_set), "cliff_aware: train/test IK14 overlap"
    assert not (tr_set & va_set), "cliff_aware: train/val overlap"
    assert not (va_set & te_set), "cliff_aware: val/test overlap"

    # [FIX-13] Recompute coverage AFTER the IK14 dedup step
    final_cliff_test = [i for i in cliff_test_idx if i in te_set]
    n_cliff_in_test  = len(final_cliff_test)
    cliff_test_cov   = round(100 * n_cliff_in_test / max(1, len(all_test_idx)), 1)

    # Log actual scaffold overlap (expected and reported, not an error)
    if scaffold_col in df.columns:
        sc_tr = set(df.iloc[train_idx][scaffold_col])
        sc_te = set(df.iloc[all_test_idx][scaffold_col])
        shared_sc = sc_tr & sc_te
        log.info(
            f"  cliff_aware: {len(shared_sc)} scaffolds shared train↔test (BY DESIGN, "
            f"cliff pairs share scaffolds)"
        )

    log.info(
        f"  cliff_aware: {len(train_idx):,} train | {len(val_idx):,} val | "
        f"{len(all_test_idx):,} test  "
        f"(cliff-derived={n_cliff_in_test}, {cliff_test_cov}% of test)"
    )

    meta = {
        "split_type":              "cliff_aware",
        "n_cliff_pairs_loaded":    n_pairs_loaded,
        "n_cliff_pairs_usable":    n_pairs_usable,
        "cliff_test_count":        n_cliff_in_test,
        "cliff_test_coverage_pct": cliff_test_cov,
        "n_train":                 len(train_idx),
        "n_val":                   len(val_idx),
        "n_test":                  len(all_test_idx),
        "paper_note": (
            f"Cliff-aware split: for each activity cliff pair (Tanimoto>=0.8, "
            f"ΔpIC50>=1.5), the less-active compound is voted into test. "
            f"{n_cliff_in_test} of {len(all_test_idx)} test compounds ({cliff_test_cov}%) "
            f"are genuine cliff members. Scaffold overlap is BY DESIGN. "
            f"IK14 disjointness strictly enforced."
        ),
    }
    return (
        df.iloc[sorted(train_idx)].copy(),
        df.iloc[sorted(val_idx)].copy(),
        df.iloc[sorted(all_test_idx)].copy(),
        meta,
    )

# ══════════════════════════════════════════════════════════════════════
# SCAFFOLD k-FOLD CV GENERATOR
# ══════════════════════════════════════════════════════════════════════
def generate_scaffold_cv_folds(
    df: pd.DataFrame,
    n_folds: int = 5,
    seed: int = 42,
    scaffold_col: str = "stereo_stripped_scaffold",
    activity_col: str = "pIC50",
    out_dir: Optional[Path] = None,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Scaffold-stratified k-fold cross-validation generator.

    [FIX-14] When multiple folds tie for "most under-budget", the tie
             is broken randomly instead of always picking the lowest k.
             Previously the largest scaffolds systematically went to
             fold 0.
    """
    rng = np.random.default_rng(seed)
    df  = df.copy().reset_index(drop=True)
    n   = len(df)

    if scaffold_col not in df.columns:
        raise ValueError(f"'{scaffold_col}' not in data for CV folds")

    scaf_to_idx: Dict[str, List[int]] = defaultdict(list)
    for i, s in enumerate(df[scaffold_col]):
        scaf_to_idx[str(s)].append(i)

    multi_scafs    = [(s, ixs) for s, ixs in scaf_to_idx.items() if len(ixs) > 1]
    singleton_idxs = [ixs[0] for s, ixs in scaf_to_idx.items() if len(ixs) == 1]

    multi_scafs.sort(key=lambda x: -len(x[1]))
    targets      = {k: n // n_folds for k in range(n_folds)}
    for k in range(n % n_folds):
        targets[k] += 1

    fold_idx: Dict[int, List[int]] = {k: [] for k in range(n_folds)}
    counts   = {k: 0 for k in range(n_folds)}

    for scaf, idxs in multi_scafs:
        # [FIX-14] random tiebreak between equally-empty folds
        deficits = {k: targets[k] - counts[k] for k in range(n_folds)}
        max_def  = max(deficits.values())
        tied_ks  = [k for k, d in deficits.items() if d == max_def]
        best_k   = int(rng.choice(tied_ks))
        fold_idx[best_k].extend(idxs)
        counts[best_k] += len(idxs)

    rng.shuffle(singleton_idxs)
    for i, idx in enumerate(singleton_idxs):
        k = i % n_folds
        fold_idx[k].append(idx)
        counts[k] += 1

    log.info(f"  CV folds (n={n_folds}): sizes = {[len(fold_idx[k]) for k in range(n_folds)]}")

    folds: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
    for val_k in range(n_folds):
        val_indices   = sorted(fold_idx[val_k])
        train_indices = sorted(
            idx for k in range(n_folds) if k != val_k for idx in fold_idx[k]
        )
        val_df   = df.iloc[val_indices].copy().assign(cv_fold=val_k, cv_split="val")
        train_df = df.iloc[train_indices].copy().assign(cv_fold=val_k, cv_split="train")
        folds.append((train_df, val_df))

        if out_dir is not None:
            fold_dir = out_dir / f"fold_{val_k}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            train_df.to_csv(fold_dir / "train.csv", index=False)
            val_df.to_csv(  fold_dir / "val.csv",   index=False)

    return folds

# ══════════════════════════════════════════════════════════════════════
# SAVE & SUMMARISE
# ══════════════════════════════════════════════════════════════════════
def save_split(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
               out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    train.assign(split="train").to_csv(out_dir / "train.csv",       index=False)
    val.assign(  split="val"  ).to_csv(out_dir / "val.csv",         index=False)
    test.assign( split="test" ).to_csv(out_dir / "test_locked.csv", index=False)
    test.assign( split="test" ).to_csv(out_dir / "test.csv",        index=False)

def subset_summary(df: pd.DataFrame, label: str, scaffold_col: str,
                   activity_col: str) -> dict:
    s = {"label": label, "n": len(df)}
    if scaffold_col in df.columns:
        s["n_scaffolds"]    = int(df[scaffold_col].nunique())
        counts              = df[scaffold_col].value_counts()
        s["pct_singletons"] = round(100 * (counts == 1).mean(), 1) if len(counts) else 0.0
    if activity_col in df.columns:
        s.update({
            f"{activity_col}_mean": round(float(df[activity_col].mean()), 3),
            f"{activity_col}_std":  round(float(df[activity_col].std()),  3),
            "pct_active":           round(float((df[activity_col] >= 6.0).mean() * 100), 1),
        })
    return s

# ══════════════════════════════════════════════════════════════════════
# HEALTH SCORE
# ══════════════════════════════════════════════════════════════════════
def compute_health_score(
    split_type: str,
    audit_report: dict,
    ks_result: dict,
    n_val: int,
    sim_stats: Optional[dict] = None,
    scaffold_diversity_ratio: Optional[float] = None,
    cliff_meta: Optional[dict] = None,
) -> Tuple[int, str]:
    score = 100

    if audit_report.get("errors"):
        score -= 100

    drift = audit_report.get("imbalance", {}).get("train_test_drift_pp", 0)
    if drift > 35:   score -= 30
    elif drift > 20: score -= 20
    elif drift > 10: score -= 10
    elif drift > 5:  score -= 4
    elif drift > 2:  score -= 2

    if sim_stats:
        largest_frac = sim_stats.get("largest_cluster_fraction", 0)
        if largest_frac > 0.60: score -= 40
        elif largest_frac > 0.40: score -= 20

    if n_val == 0:
        score -= 25

    if ks_result:
        stat = ks_result.get("ks_stat", 0)
        p    = ks_result.get("ks_p",    1.0)

        if split_type in EXPECTED_MATCH:
            if p < 0.01: score -= 8
            elif p < 0.05: score -= 4

        elif split_type in EXPECTED_DIFFER:
            if p >= 0.05: score -= 15
            elif stat > 0.40: score -= 10

        elif split_type == "confirmed":
            if stat >= 0.20: score -= 10
            elif stat >= 0.10: score -= 3

        elif split_type in SOFT_DIFFER:
            if stat >= 0.30: score -= 20
            elif stat >= 0.20: score -= 12
            elif stat >= 0.15: score -= 5

    if scaffold_diversity_ratio is not None:
        if scaffold_diversity_ratio < 0.30: score -= 5
        elif scaffold_diversity_ratio < 0.50: score -= 2

    if split_type == "cliff_aware" and cliff_meta is not None:
        cov = cliff_meta.get("cliff_test_coverage_pct", 0)
        if cov < 20:
            score -= 20
        elif cov < 40:
            score -= 8
        elif cov >= 60:
            score += 5

    score = max(0, score)
    if score >= 90:   label = "EXCELLENT"
    elif score >= 75: label = "GOOD"
    elif score >= 60: label = "ACCEPTABLE"
    elif score >= 40: label = "WEAK"
    else:             label = "DEGENERATE"

    return score, label

def diversity_ratio(train: pd.DataFrame, test: pd.DataFrame,
                    scaffold_col: str) -> Optional[float]:
    if scaffold_col not in train.columns or scaffold_col not in test.columns:
        return None
    if len(train) == 0 or len(test) == 0:
        return None
    tr_counts = train[scaffold_col].value_counts()
    te_counts = test[scaffold_col].value_counts()
    tr_sing   = float((tr_counts == 1).mean()) if len(tr_counts) else 0.0
    te_sing   = float((te_counts == 1).mean()) if len(te_counts) else 0.0
    if tr_sing < 1e-6:
        return None
    return round(te_sing / tr_sing, 3)

# ══════════════════════════════════════════════════════════════════════
# REGRESSION RUNNER
# ══════════════════════════════════════════════════════════════════════
def _record_diagnostics(
    split_name: str,
    tr: pd.DataFrame, va: pd.DataFrame, te: pd.DataFrame,
    audit: dict, ks_res: dict, scaffold_col: str,
    health: Tuple[int, str],
    sim_stats: Optional[dict] = None,
    div_ratio: Optional[float] = None,
    cliff_meta: Optional[dict] = None,
) -> dict:
    h, hl = health
    d = {
        "n_train": len(tr), "n_val": len(va), "n_test": len(te),
        "n_scaffolds_train": int(tr[scaffold_col].nunique()) if scaffold_col in tr.columns else None,
        "n_scaffolds_test":  int(te[scaffold_col].nunique()) if scaffold_col in te.columns else None,
        "activity_drift_pp": audit.get("imbalance", {}).get("train_test_drift_pp"),
        "ks_stat":           ks_res.get("ks_stat"),
        "ks_p":              ks_res.get("ks_p"),
        "ks_note":           ks_res.get("note"),
        "scaffold_diversity_ratio":  div_ratio,
        "largest_cluster_fraction":  sim_stats.get("largest_cluster_fraction") if sim_stats else None,
        "selected_cutoff":           sim_stats.get("selected_cutoff") if sim_stats else None,
        "degenerate":                sim_stats.get("degenerate", False) if sim_stats else False,
        "health_score": h,
        "health_label": hl,
    }
    if cliff_meta is not None:
        d["cliff_test_count"]        = cliff_meta.get("cliff_test_count")
        d["cliff_test_coverage_pct"] = cliff_meta.get("cliff_test_coverage_pct")
    return d


def run_regression_splits(t1: pd.DataFrame, confirmed: pd.DataFrame, args,
                          out: Path, seed: int) -> Tuple[dict, dict, dict, dict]:
    reg_out = out / "regression"
    reg_out.mkdir(parents=True, exist_ok=True)
    audit_kw  = dict(scaffold_col=args.scaffold_col, activity_col=args.activity_col)
    all_stats:       dict = {}
    all_audits:      dict = {}
    all_health:      dict = {}
    all_diagnostics: dict = {}

    # 1. SCAFFOLD
    log.info("\n" + "=" * 68)
    log.info(f"  [1/6] SCAFFOLD (seed={seed}, cap_frac={args.scaffold_cap_frac})")
    log.info("=" * 68)
    with _timed("scaffold"):
        tr, va, te = split_scaffold_capped(
            t1, test_frac=args.test_frac, val_frac=args.val_frac, seed=seed,
            scaffold_col=args.scaffold_col, cap_frac=args.scaffold_cap_frac,
            activity_col=args.activity_col,
        )
        log.info(f"  {len(tr):,} train | {len(va):,} val | {len(te):,} test")
        sh_tr = scaffold_stats(tr, "train_sc", args.scaffold_col)
        sh_te = scaffold_stats(te, "test_sc",  args.scaffold_col)
        log_scaffold_health(sh_tr); log_scaffold_health(sh_te)
        all_audits["scaffold"] = audit_split(tr, va, te, "scaffold", **audit_kw, strict_inchikey=True)
        ks_res = ks_check(tr, te, "scaffold", args.activity_col, "test")
        save_split(tr, va, te, reg_out / "scaffold")
        div    = diversity_ratio(tr, te, args.scaffold_col)
        if div is not None:
            log.info(f"  scaffold-diversity ratio: {div:.3f}")
        health = compute_health_score("scaffold", all_audits["scaffold"], ks_res, len(va),
                                       scaffold_diversity_ratio=div)
        all_stats["scaffold"]       = {
            "train": subset_summary(tr, "train", args.scaffold_col, args.activity_col),
            "val":   subset_summary(va, "val",   args.scaffold_col, args.activity_col),
            "test":  subset_summary(te, "test",  args.scaffold_col, args.activity_col),
        }
        all_health["scaffold"]      = {"train": sh_tr, "test": sh_te}
        all_diagnostics["scaffold"] = _record_diagnostics(
            "scaffold", tr, va, te, all_audits["scaffold"], ks_res, args.scaffold_col, health, div_ratio=div,
        )

    # 2. RANDOM
    log.info("\n" + "=" * 68)
    log.info(f"  [2/6] RANDOM (seed={seed}, IID baseline)")
    log.info("=" * 68)
    with _timed("random"):
        tr, va, te = split_random(
            t1, test_frac=args.test_frac, val_frac=args.val_frac, seed=seed,
            activity_col=args.activity_col,
        )
        log.info(f"  {len(tr):,} train | {len(va):,} val | {len(te):,} test")
        sh_tr = scaffold_stats(tr, "train_rnd", args.scaffold_col)
        sh_te = scaffold_stats(te, "test_rnd",  args.scaffold_col)
        log_scaffold_health(sh_tr); log_scaffold_health(sh_te)
        all_audits["random"] = audit_split(
            tr, va, te, "random", **audit_kw, allow_scaffold_overlap=True, strict_inchikey=True,
        )
        ks_res = ks_check(tr, te, "random", args.activity_col, "test")
        save_split(tr, va, te, reg_out / "random")
        div    = diversity_ratio(tr, te, args.scaffold_col)
        if div is not None:
            log.info(f"  scaffold-diversity ratio: {div:.3f}")
        health = compute_health_score("random", all_audits["random"], ks_res, len(va),
                                       scaffold_diversity_ratio=div)
        all_stats["random"]       = {
            "train": subset_summary(tr, "train", args.scaffold_col, args.activity_col),
            "val":   subset_summary(va, "val",   args.scaffold_col, args.activity_col),
            "test":  subset_summary(te, "test",  args.scaffold_col, args.activity_col),
        }
        all_health["random"]      = {"train": sh_tr, "test": sh_te}
        all_diagnostics["random"] = _record_diagnostics(
            "random", tr, va, te, all_audits["random"], ks_res, args.scaffold_col, health, div_ratio=div,
        )

    # 3. SIMILARITY
    log.info("\n" + "=" * 68)
    log.info(f"  [3/6] SIMILARITY (adaptive Butina)")
    log.info("=" * 68)
    if args.skip_similarity or not HAS_RDKIT:
        log.info("  skipped")
        all_stats["similarity"]       = {"skipped": True}
        all_diagnostics["similarity"] = {"skipped": True, "health_score": 0, "health_label": "DEGENERATE"}
    else:
        with _timed("similarity"):
            try:
                tr, va, te, sim_s = split_similarity_adaptive(
                    t1, args.smiles_col,
                    test_frac=args.test_frac, val_frac=args.val_frac, seed=seed,
                    scaffold_col=args.scaffold_col, cap_frac=args.similarity_cap_frac,
                )
                log.info(f"  {len(tr):,} train | {len(va):,} val | {len(te):,} test")
                if sim_s.get("degenerate"):
                    log.warning(f"  [!!] Similarity split is DEGENERATE (cutoff={sim_s.get('selected_cutoff')})")
                sh_tr = scaffold_stats(tr, "train_sim", args.scaffold_col)
                sh_te = scaffold_stats(te, "test_sim",  args.scaffold_col)
                log_scaffold_health(sh_tr); log_scaffold_health(sh_te)
                all_audits["similarity"] = audit_split(
                    tr, va, te, "similarity", **audit_kw, allow_scaffold_overlap=True, strict_inchikey=True,
                )
                ks_res = ks_check(tr, te, "similarity", args.activity_col, "test")
                save_split(tr, va, te, reg_out / "similarity")
                div    = diversity_ratio(tr, te, args.scaffold_col)
                if div is not None:
                    log.info(f"  scaffold-diversity ratio: {div:.3f}")
                health = compute_health_score("similarity", all_audits["similarity"], ks_res,
                                               len(va), sim_s, scaffold_diversity_ratio=div)
                all_stats["similarity"]       = {
                    "train":        subset_summary(tr, "train", args.scaffold_col, args.activity_col),
                    "val":          subset_summary(va, "val",   args.scaffold_col, args.activity_col),
                    "test":         subset_summary(te, "test",  args.scaffold_col, args.activity_col),
                    "butina_stats": sim_s,
                }
                all_health["similarity"]      = {"train": sh_tr, "test": sh_te}
                all_diagnostics["similarity"] = _record_diagnostics(
                    "similarity", tr, va, te, all_audits["similarity"], ks_res,
                    args.scaffold_col, health, sim_stats=sim_s, div_ratio=div,
                )
            except Exception as e:
                log.error(f"  Similarity split failed: {e}")
                all_stats["similarity"]       = {"error": str(e)}
                all_diagnostics["similarity"] = {"error": str(e), "health_score": 0, "health_label": "DEGENERATE"}

    # 4. CONFIRMED
    log.info("\n" + "=" * 68)
    log.info(f"  [4/6] CONFIRMED (independent scaffold-capped)")
    log.info("=" * 68)
    with _timed("confirmed"):
        try:
            tr, va, te = split_confirmed(
                confirmed, test_frac=args.test_frac, val_frac=args.val_frac,
                seed=seed, scaffold_col=args.scaffold_col,
                cap_frac=args.scaffold_cap_frac, activity_col=args.activity_col,
            )
            log.info(f"  {len(tr):,} train | {len(va):,} val | {len(te):,} test")
            sh_tr = scaffold_stats(tr, "train_cf", args.scaffold_col)
            sh_te = scaffold_stats(te, "test_cf",  args.scaffold_col)
            log_scaffold_health(sh_tr); log_scaffold_health(sh_te)
            all_audits["confirmed"] = audit_split(
                tr, va, te, "confirmed", **audit_kw, strict_inchikey=True,
            )
            ks_res = ks_check(tr, te, "confirmed", args.activity_col, "test")
            save_split(tr, va, te, reg_out / "confirmed")
            div    = diversity_ratio(tr, te, args.scaffold_col)
            if div is not None:
                log.info(f"  scaffold-diversity ratio: {div:.3f}")
            health = compute_health_score("confirmed", all_audits["confirmed"], ks_res, len(va),
                                           scaffold_diversity_ratio=div)
            all_stats["confirmed"]       = {
                "train": subset_summary(tr, "train", args.scaffold_col, args.activity_col),
                "val":   subset_summary(va, "val",   args.scaffold_col, args.activity_col),
                "test":  subset_summary(te, "test",  args.scaffold_col, args.activity_col),
            }
            all_health["confirmed"]      = {"train": sh_tr, "test": sh_te}
            all_diagnostics["confirmed"] = _record_diagnostics(
                "confirmed", tr, va, te, all_audits["confirmed"], ks_res, args.scaffold_col, health, div_ratio=div,
            )
        except Exception as e:
            log.error(f"  Confirmed split failed: {e}")
            all_stats["confirmed"]       = {"error": str(e)}
            all_diagnostics["confirmed"] = {"error": str(e), "health_score": 0, "health_label": "DEGENERATE"}

    # 5. LEAD OPT
    log.info("\n" + "=" * 68)
    log.info(f"  [5/6] LEAD_OPT (min_scaffold_size={args.lead_opt_min_size})")
    log.info("=" * 68)
    with _timed("lead_opt"):
        try:
            tr, va, te, lo_meta = split_lead_opt(
                t1, test_frac=args.test_frac, val_frac=args.val_frac, seed=seed,
                scaffold_col=args.scaffold_col,
                min_scaffold_size=args.lead_opt_min_size,
                activity_col=args.activity_col,
            )
            log.info(f"  {len(tr):,} train | {len(va):,} val | {len(te):,} test")
            sh_tr = scaffold_stats(tr, "train_lo", args.scaffold_col)
            sh_te = scaffold_stats(te, "test_lo",  args.scaffold_col)
            log_scaffold_health(sh_tr); log_scaffold_health(sh_te)
            all_audits["lead_opt"] = audit_split(
                tr, va, te, "lead_opt", **audit_kw, allow_scaffold_overlap=True, strict_inchikey=True,
            )
            ks_res = ks_check(tr, te, "lead_opt", args.activity_col, "test")
            save_split(tr, va, te, reg_out / "lead_opt")
            div    = diversity_ratio(tr, te, args.scaffold_col)
            if div is not None:
                log.info(f"  scaffold-diversity ratio: {div:.3f}")
            health = compute_health_score("lead_opt", all_audits["lead_opt"], ks_res, len(va),
                                           scaffold_diversity_ratio=div)
            all_stats["lead_opt"]       = {
                "train":    subset_summary(tr, "train", args.scaffold_col, args.activity_col),
                "val":      subset_summary(va, "val",   args.scaffold_col, args.activity_col),
                "test":     subset_summary(te, "test",  args.scaffold_col, args.activity_col),
                "metadata": lo_meta,
            }
            all_health["lead_opt"]      = {"train": sh_tr, "test": sh_te}
            all_diagnostics["lead_opt"] = _record_diagnostics(
                "lead_opt", tr, va, te, all_audits["lead_opt"], ks_res, args.scaffold_col, health, div_ratio=div,
            )
        except Exception as e:
            log.error(f"  lead_opt split failed: {e}")
            all_stats["lead_opt"]       = {"error": str(e)}
            all_diagnostics["lead_opt"] = {"error": str(e), "health_score": 0, "health_label": "DEGENERATE"}

    # 6. CLIFF-AWARE
    log.info("\n" + "=" * 68)
    log.info(f"  [6/6] CLIFF-AWARE (less-active cliff member → test)")
    log.info("=" * 68)

    if args.skip_cliff_aware:
        log.info("  cliff_aware split: SKIPPED (--skip_cliff_aware)")
        all_stats["cliff_aware"]       = {"skipped": True}
        all_diagnostics["cliff_aware"] = {"skipped": True, "health_score": 0, "health_label": "DEGENERATE"}
    else:
        with _timed("cliff_aware"):
            cliff_pairs_df: Optional[pd.DataFrame] = None
            if args.cliff_pairs_file:
                cp = Path(args.cliff_pairs_file)
                if cp.exists():
                    cliff_pairs_df = pd.read_csv(cp)
                    log.info(f"  Loaded {len(cliff_pairs_df)} cliff pairs from {cp}")
                else:
                    log.warning(f"  cliff_pairs_file not found: {cp} — running fallback")
            else:
                auto_path = Path(args.t1_file).parent / "pad_activity_cliffs.csv"
                if auto_path.exists():
                    cliff_pairs_df = pd.read_csv(auto_path)
                    log.info(f"  Auto-loaded {len(cliff_pairs_df)} cliff pairs from {auto_path}")
                else:
                    log.warning("  No cliff_pairs_file; set --cliff_pairs_file or place "
                                 "pad_activity_cliffs.csv in processed dir")

            try:
                tr, va, te, ca_meta = split_cliff_aware(
                    t1,
                    cliff_pairs_df=cliff_pairs_df,
                    test_frac=args.test_frac,
                    val_frac=args.val_frac,
                    seed=seed,
                    scaffold_col=args.scaffold_col,
                    activity_col=args.activity_col,
                    id_col=args.id_col,
                )
                log.info(f"  {len(tr):,} train | {len(va):,} val | {len(te):,} test")
                sh_tr = scaffold_stats(tr, "train_ca", args.scaffold_col)
                sh_te = scaffold_stats(te, "test_ca",  args.scaffold_col)
                log_scaffold_health(sh_tr); log_scaffold_health(sh_te)
                all_audits["cliff_aware"] = audit_split(
                    tr, va, te, "cliff_aware", **audit_kw,
                    allow_scaffold_overlap=True, strict_inchikey=True,
                )
                ks_res = ks_check(tr, te, "cliff_aware", args.activity_col, "test")
                save_split(tr, va, te, reg_out / "cliff_aware")
                div    = diversity_ratio(tr, te, args.scaffold_col)
                if div is not None:
                    log.info(f"  scaffold-diversity ratio: {div:.3f}")
                health = compute_health_score(
                    "cliff_aware", all_audits["cliff_aware"], ks_res, len(va),
                    scaffold_diversity_ratio=div, cliff_meta=ca_meta,
                )
                all_stats["cliff_aware"]       = {
                    "train":    subset_summary(tr, "train", args.scaffold_col, args.activity_col),
                    "val":      subset_summary(va, "val",   args.scaffold_col, args.activity_col),
                    "test":     subset_summary(te, "test",  args.scaffold_col, args.activity_col),
                    "metadata": ca_meta,
                }
                all_health["cliff_aware"]      = {"train": sh_tr, "test": sh_te}
                all_diagnostics["cliff_aware"] = _record_diagnostics(
                    "cliff_aware", tr, va, te, all_audits["cliff_aware"], ks_res,
                    args.scaffold_col, health, div_ratio=div, cliff_meta=ca_meta,
                )
            except Exception as e:
                log.error(f"  cliff_aware split failed: {e}")
                import traceback; traceback.print_exc()
                all_stats["cliff_aware"]       = {"error": str(e)}
                all_diagnostics["cliff_aware"] = {"error": str(e), "health_score": 0, "health_label": "DEGENERATE"}

    # Optional CV folds
    if getattr(args, "cv_folds", 0) > 1:
        log.info(f"\n  Generating {args.cv_folds}-fold scaffold CV...")
        cv_dir = reg_out / "cv_folds"
        cv_dir.mkdir(parents=True, exist_ok=True)
        generate_scaffold_cv_folds(
            t1, n_folds=args.cv_folds, seed=seed,
            scaffold_col=args.scaffold_col, activity_col=args.activity_col,
            out_dir=cv_dir,
        )
        log.info(f"  CV folds written to {cv_dir}")

    return all_stats, all_audits, all_health, all_diagnostics

# ══════════════════════════════════════════════════════════════════════
# CLASSIFICATION ALIGNMENT
# ══════════════════════════════════════════════════════════════════════
def load_regression_split_assignments(
    regression_splits_dir: Path,
    scaffold_col: str = "scaffold_id",
    id_col: str = "inchikey_14",
) -> Dict[str, Dict[str, str]]:
    assignments: Dict[str, Dict[str, str]] = {}
    for split_name in SPLIT_NAMES:
        split_dir = regression_splits_dir / split_name
        if not split_dir.exists():
            log.warning(f"  {split_name}: directory missing, skipping")
            continue

        split_assignments: Dict[str, str] = {}
        conflicts = 0
        for subset in ("train", "val", "test"):
            if subset == "test":
                path = split_dir / "test_locked.csv"
                if not path.exists():
                    path = split_dir / "test.csv"
            else:
                path = split_dir / f"{subset}.csv"
            if not path.exists():
                continue

            df_s = pd.read_csv(path)
            if scaffold_col not in df_s.columns:
                log.error(f"  {split_name}/{subset}: no {scaffold_col}")
                continue

            for sc in norm_scaffold(df_s[scaffold_col].dropna()).unique():
                if sc in split_assignments and split_assignments[sc] != subset:
                    conflicts += 1
                    if split_name in DISJOINT_SPLITS:
                        raise RuntimeError(
                            f"Regression {split_name} split has scaffold '{sc}' "
                            f"in both {split_assignments[sc]} and {subset}. "
                            f"Disjoint splits must not share scaffolds."
                        )
                    continue
                split_assignments[sc] = subset

        if conflicts:
            log.info(f"  {split_name}: {conflicts} scaffold overlaps (expected for non-disjoint splits)")
        log.info(f"  {split_name}: {len(split_assignments)} scaffolds mapped")
        assignments[split_name] = split_assignments
    return assignments

def load_regression_id_assignments(
    regression_splits_dir: Path,
    id_col: str = "inchikey_14",
) -> Dict[str, Dict[str, str]]:
    assignments: Dict[str, Dict[str, str]] = {}
    for split_name in SPLIT_NAMES:
        split_dir = regression_splits_dir / split_name
        if not split_dir.exists():
            continue
        id_map: Dict[str, str] = {}
        for subset in ("train", "val", "test"):
            if subset == "test":
                path = split_dir / "test_locked.csv"
                if not path.exists():
                    path = split_dir / "test.csv"
            else:
                path = split_dir / f"{subset}.csv"
            if not path.exists():
                continue
            df_s = pd.read_csv(path)
            if id_col not in df_s.columns:
                continue
            for cid in df_s[id_col].dropna().astype(str).unique():
                if cid not in id_map:
                    id_map[cid] = subset
        assignments[split_name] = id_map
    return assignments

def assert_scaffold_id_consistency(
    reg_dir: Path,
    cls_df: pd.DataFrame,
    id_col: str,
    scaffold_col: str,
    sample_split: str = "scaffold",
) -> None:
    reg_train_path = reg_dir / sample_split / "train.csv"
    if not reg_train_path.exists():
        log.warning(f"  Cannot verify scaffold consistency: {reg_train_path} not found")
        return

    try:
        reg = pd.read_csv(reg_train_path, usecols=[id_col, scaffold_col])
    except (ValueError, KeyError) as e:
        log.warning(f"  Cannot verify scaffold consistency: {e}")
        return

    reg[id_col]       = reg[id_col].astype(str)
    reg[scaffold_col] = norm_scaffold(reg[scaffold_col].astype(str))

    cls_view = cls_df[[id_col, scaffold_col]].copy()
    cls_view[id_col]       = cls_view[id_col].astype(str)
    cls_view[scaffold_col] = norm_scaffold(cls_view[scaffold_col].astype(str))

    shared = reg.merge(cls_view, on=id_col, suffixes=("_reg", "_cls"))
    if len(shared) == 0:
        log.warning(
            "  Scaffold consistency check: no shared compounds between "
            "regression train and classification — alignment will rely "
            "on scaffold-string match only"
        )
        return

    mismatches = (shared[f"{scaffold_col}_reg"] != shared[f"{scaffold_col}_cls"]).sum()
    if mismatches > 0:
        bad      = shared[shared[f"{scaffold_col}_reg"] != shared[f"{scaffold_col}_cls"]].head(3)
        examples = bad.to_dict("records")
        raise RuntimeError(
            f"\n\n  Scaffold ID mismatch: {mismatches}/{len(shared)} shared "
            f"compounds have different '{scaffold_col}' in regression vs classification.\n"
            f"  Examples: {examples}\n\n"
            f"  This almost certainly means '{scaffold_col}' is a per-file integer ID, "
            f"not a canonical scaffold key.\n"
            f"  Fix: use --scaffold_col stereo_stripped_scaffold"
        )

    log.info(f"  Scaffold ID consistency verified: {len(shared)} shared compounds, all matching")

def route_split_vectorized(
    cls_df: pd.DataFrame,
    split_name: str,
    scaffold_assignments: Dict[str, str],
    id_assignments: Dict[str, str],
    scaffold_col: str,
    id_col: str,
) -> pd.Series:
    sc       = norm_scaffold(cls_df[scaffold_col])
    by_scaffold = sc.map(scaffold_assignments)

    if split_name in DISJOINT_SPLITS:
        return by_scaffold.fillna("unassigned").astype(object)

    cid    = cls_df[id_col].astype(str)
    by_id  = cid.map(id_assignments)
    return by_id.fillna(by_scaffold).fillna("unassigned").astype(object)

def distribute_unassigned(
    df_unassigned: pd.DataFrame,
    n_train_target: int,
    n_val_target: int,
    n_test_target: int,
    scaffold_col: str,
    seed: int = 42,
    stratify_col: Optional[str] = None,
) -> pd.Series:
    rng    = np.random.default_rng(seed)
    result = pd.Series("train", index=df_unassigned.index, dtype=object)
    if len(df_unassigned) == 0:
        return result

    sc_norm    = norm_scaffold(df_unassigned[scaffold_col])
    scaf_sizes = sc_norm.value_counts().to_dict()
    unique_scafs = list(scaf_sizes.keys())

    if stratify_col and stratify_col in df_unassigned.columns:
        strat_means = (
            df_unassigned.assign(_sc=sc_norm)
            .groupby("_sc")[stratify_col]
            .apply(lambda s: float(pd.to_numeric(s, errors="coerce").fillna(0).mean()))
            .to_dict()
        )
        unique_scafs.sort(key=lambda s: (-scaf_sizes[s], strat_means.get(s, 0.5)))
    else:
        rng.shuffle(unique_scafs)
        unique_scafs.sort(key=lambda s: -scaf_sizes[s])

    targets = {"train": n_train_target, "val": n_val_target, "test": n_test_target}
    counts  = {"train": 0, "val": 0, "test": 0}

    assignments_by_scaffold: Dict[str, str] = {}
    for sc in unique_scafs:
        sz      = scaf_sizes[sc]
        deficits = {b: targets[b] - counts[b] for b in counts}
        order   = ["test", "val", "train"]
        best    = max(order, key=lambda b: deficits[b])
        assignments_by_scaffold[sc] = best
        counts[best] += sz

    result = sc_norm.map(assignments_by_scaffold).fillna("train").astype(object)
    result.index = df_unassigned.index
    return result

def align_one_split(
    cls_df: pd.DataFrame,
    split_name: str,
    scaffold_assignments: Dict[str, str],
    id_assignments: Dict[str, str],
    out_dir: Path,
    reg_dir: Path,
    scaffold_col: str,
    id_col: str,
    seed: int,
    stratify_col: Optional[str] = None,
) -> dict:
    n = len(cls_df)

    subset_col      = route_split_vectorized(
        cls_df, split_name, scaffold_assignments, id_assignments,
        scaffold_col, id_col,
    )
    unassigned_mask = subset_col == "unassigned"
    n_unassigned    = int(unassigned_mask.sum())
    routed_frac     = (n - n_unassigned) / max(1, n)
    log.info(
        f"  {split_name}: {n - n_unassigned}/{n} routed ({100*routed_frac:.1f}%), "
        f"{n_unassigned} to distribute"
    )

    floored = False
    if n_unassigned > 0:
        aligned_counts = Counter(subset_col[~unassigned_mask])
        tr_cnt = aligned_counts.get("train", 0)
        va_cnt = aligned_counts.get("val",   0)
        te_cnt = aligned_counts.get("test",  0)

        total_aligned = tr_cnt + va_cnt + te_cnt
        if total_aligned > 0:
            tr_frac = tr_cnt / total_aligned
            va_frac = va_cnt / total_aligned
            te_frac = te_cnt / total_aligned
        else:
            tr_frac, va_frac, te_frac = 0.75, 0.10, 0.15

        if va_frac < MIN_VAL_FRAC:
            deficit  = MIN_VAL_FRAC - va_frac
            log.info(f"  {split_name}: val_frac={va_frac:.3f} below floor; adjusting")
            va_frac  = MIN_VAL_FRAC
            tr_frac  = max(0.0, tr_frac - deficit)
            floored  = True

        log.info(
            f"  {split_name}: redistribution fractions "
            f"train={tr_frac:.3f} val={va_frac:.3f} test={te_frac:.3f}"
        )

        n_test_target  = int(round(n_unassigned * te_frac))
        n_val_target   = int(round(n_unassigned * va_frac))
        n_train_target = n_unassigned - n_test_target - n_val_target

        unassigned_df = cls_df[unassigned_mask]
        distributed   = distribute_unassigned(
            unassigned_df,
            n_train_target=n_train_target,
            n_val_target=n_val_target,
            n_test_target=n_test_target,
            scaffold_col=scaffold_col,
            seed=seed,
            stratify_col=stratify_col,
        )
        subset_col = subset_col.where(~unassigned_mask, distributed)

    out_dir.mkdir(parents=True, exist_ok=True)
    counts_out = Counter(subset_col)

    for sub in ("train", "val", "test"):
        df_sub = cls_df[subset_col == sub].copy()
        df_sub = df_sub.assign(split=sub)
        df_sub.to_csv(out_dir / f"{sub}.csv", index=False)
        if sub == "test":
            df_sub.to_csv(out_dir / "test_locked.csv", index=False)

    total = sum(counts_out[s] for s in ("train", "val", "test"))
    if total > 0:
        actual_test_pct = 100 * counts_out["test"] / total
        actual_val_pct  = 100 * counts_out["val"]  / total
        log.info(
            f"  {split_name}: train={counts_out['train']} "
            f"val={counts_out['val']} ({actual_val_pct:.1f}%) "
            f"test={counts_out['test']} ({actual_test_pct:.1f}%)"
        )

    audit = {
        "split":              split_name,
        "n_total":            n,
        "n_aligned":          n - n_unassigned,
        "n_distributed":      n_unassigned,
        "routed_pct":         round(100 * routed_frac, 1),
        "redistributed_pct":  round(100 * (1 - routed_frac), 1),
        "val_floor_applied":  floored,
        "counts":             dict(counts_out),
        "pct_train":          round(100 * counts_out["train"] / max(1, total), 1),
        "pct_val":            round(100 * counts_out["val"]   / max(1, total), 1),
        "pct_test":           round(100 * counts_out["test"]  / max(1, total), 1),
    }

    if split_name in DISJOINT_SPLITS:
        tr_scafs = set(norm_scaffold(cls_df[subset_col == "train"][scaffold_col]))
        te_scafs = set(norm_scaffold(cls_df[subset_col == "test"][scaffold_col]))
        leaks    = tr_scafs & te_scafs
        if leaks:
            log.error(f"  {split_name}: SCAFFOLD LEAKAGE: {len(leaks)} shared")
            audit["leakage_scaffolds"] = sorted(list(leaks))[:20]
        else:
            log.info(f"  {split_name}: scaffold-disjoint verified")

    if split_name in DISJOINT_SPLITS:
        reg_train_path = reg_dir / split_name / "train.csv"
        if reg_train_path.exists():
            try:
                reg_train_ids = set(
                    pd.read_csv(reg_train_path, usecols=[id_col])[id_col].astype(str)
                )
                cls_test_ids = set(cls_df.loc[subset_col == "test", id_col].astype(str))
                cls_val_ids  = set(cls_df.loc[subset_col == "val",  id_col].astype(str))
                leak_test    = reg_train_ids & cls_test_ids
                leak_val     = reg_train_ids & cls_val_ids
                audit["cross_dataset_leakage_ik14"] = {
                    "cls_test_in_reg_train": len(leak_test),
                    "cls_val_in_reg_train":  len(leak_val),
                    "stacking_safe":         len(leak_test) == 0 and len(leak_val) == 0,
                }
                if leak_test or leak_val:
                    log.warning(
                        f"  {split_name}: cross-dataset IK14 leakage — "
                        f"cls_test∩reg_train={len(leak_test)}, "
                        f"cls_val∩reg_train={len(leak_val)}"
                    )
                else:
                    log.info(f"  {split_name}: stacking-safe")
            except (ValueError, KeyError) as e:
                log.warning(f"  {split_name}: cross-dataset audit skipped: {e}")

    return audit

def run_classification_alignment(cls_df: pd.DataFrame, args, out_dir: Path,
                                 seed: int) -> dict:
    if args.regression_splits_dir:
        reg_dir = Path(args.regression_splits_dir)
    else:
        reg_dir = out_dir / "regression"
    if not reg_dir.exists():
        raise ValueError(f"Regression splits dir not found: {reg_dir}")

    cls_out = out_dir / "classification"
    cls_out.mkdir(parents=True, exist_ok=True)

    log.info("\n" + "=" * 68)
    log.info("  CLASSIFICATION ALIGNMENT")
    log.info("=" * 68)
    log.info(f"  Reading regression splits from: {reg_dir}")
    log.info(f"  Writing classification splits to: {cls_out}")

    scaffold_col = args.scaffold_col
    id_col       = args.id_col

    if scaffold_col not in cls_df.columns:
        raise ValueError(f"No {scaffold_col} in classification file")
    if id_col not in cls_df.columns:
        raise ValueError(f"No {id_col} in classification file")

    log.info("\nVerifying scaffold-ID consistency across regression/classification:")
    assert_scaffold_id_consistency(reg_dir, cls_df, id_col, scaffold_col)

    log.info("\nLoading regression split assignments:")
    scaffold_maps = load_regression_split_assignments(reg_dir, scaffold_col=scaffold_col, id_col=id_col)
    id_maps       = load_regression_id_assignments(reg_dir, id_col=id_col)

    sample_cls = set(norm_scaffold(cls_df[scaffold_col].dropna()))
    for split_name in SPLIT_NAMES:
        if split_name not in scaffold_maps:
            continue
        sample_reg = list(scaffold_maps[split_name].keys())[:20]
        matches    = sum(1 for s in sample_reg if s in sample_cls)
        if matches == 0 and len(scaffold_maps[split_name]) > 0:
            raise ValueError(
                f"ZERO scaffold matches for {split_name} split. "
                f"Are you using the same scaffold column? "
                f"You passed: --scaffold_col {scaffold_col}"
            )

    log.info("\nAligning classification to regression splits:")
    all_audits = {}
    for split_name in SPLIT_NAMES:
        if split_name not in scaffold_maps:
            log.info(f"  {split_name}: skipped (no regression split on disk)")
            continue
        log.info(f"\n--- {split_name} ---")
        split_out = cls_out / split_name
        audit     = align_one_split(
            cls_df=cls_df,
            split_name=split_name,
            scaffold_assignments=scaffold_maps[split_name],
            id_assignments=id_maps.get(split_name, {}),
            out_dir=split_out,
            reg_dir=reg_dir,
            scaffold_col=scaffold_col,
            id_col=id_col,
            seed=seed,
            stratify_col=args.class_stratify_col,
        )
        all_audits[split_name] = audit

    master = {
        "generated":             pd.Timestamp.now().isoformat(),
        "pipeline_version":      PIPELINE_VERSION,
        "classification_file":   str(args.classification_file),
        "regression_splits_dir": str(reg_dir),
        "n_classification_rows": len(cls_df),
        "seed":                  seed,
        "splits":                all_audits,
    }
    with open(out_dir / "alignment_audit.json", "w") as f:
        json.dump(master, f, indent=2, default=str)
    log.info(f"\nAudit: {out_dir / 'alignment_audit.json'}")
    return all_audits

# ══════════════════════════════════════════════════════════════════════
# MANIFEST & FREEZE HELPERS
# ══════════════════════════════════════════════════════════════════════
def write_output_manifest(out: Path, cfg_hash: str) -> Path:
    manifest = {
        "generated":        pd.Timestamp.now().isoformat(),
        "pipeline_version": PIPELINE_VERSION,
        "config_hash":      cfg_hash,
        "root":             str(out.resolve()),
        "files":            [],
    }
    for csv_path in sorted(out.rglob("*.csv")):
        try:
            n_rows = sum(1 for _ in open(csv_path)) - 1
        except Exception:
            n_rows = None
        manifest["files"].append({
            "path":  str(csv_path.relative_to(out)),
            "rows":  n_rows,
            "bytes": csv_path.stat().st_size,
            "md5":   md5_file(csv_path),
        })
    path = out / "splits_manifest.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return path

def write_benchmark_hash(out: Path, cfg_hash: str, cfg: dict) -> Path:
    path = out / "BENCHMARK_HASH.txt"
    with open(path, "w") as f:
        f.write(f"{cfg_hash}\n")
        f.write(f"pipeline_version: {PIPELINE_VERSION}\n")
        f.write(f"generated: {pd.Timestamp.now().isoformat()}\n")
        f.write(f"seed: {cfg.get('seed')}\n")
        f.write(f"test_frac: {cfg.get('test_frac')}\n")
        f.write(f"val_frac: {cfg.get('val_frac')}\n")
        f.write(f"\nTo freeze this benchmark:\n")
        f.write(f"  git add data/splits/\n")
        f.write(f"  git commit -m 'freeze: pad4bench splits {cfg_hash[:8]}'\n")
        f.write(f"  git tag pad4bench-{PIPELINE_VERSION}-splits\n")
    return path

def check_freeze_collision(out: Path, force: bool) -> None:
    hash_path = out / "BENCHMARK_HASH.txt"
    if hash_path.exists() and not force:
        existing = hash_path.read_text().splitlines()[0].strip()
        raise RuntimeError(
            f"\n  Frozen benchmark already exists at {out}\n"
            f"  Existing hash: {existing}\n"
            f"  Pass --force to override."
        )

# ══════════════════════════════════════════════════════════════════════
# WRITE MASTER SUMMARY & DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════
def write_master_summary(out: Path, cfg: dict, cfg_hash: str, t1: pd.DataFrame,
                         all_stats: dict, all_audits: dict, all_health: dict,
                         all_diagnostics: dict):
    log.info("\n" + "=" * 68)
    log.info("  SPLIT HEALTH SUMMARY")
    log.info("=" * 68)
    log.info(f"  {'split':<14} {'subset':<7} {'n':>6} {'scaff':>6} {'sing%':>6} {'top2':>6}")
    log.info("  " + "-" * 60)
    for split, sub_d in all_health.items():
        for sub_name, s in sub_d.items():
            if "error" in s:
                continue
            log.info(
                f"  {split:<14} {sub_name:<7} {s['n_compounds']:>6} "
                f"{s['n_scaffolds']:>6} {s['pct_singletons']:>5.0f}% "
                f"{s['top2_coverage']:>6}"
            )

    log.info("\n" + "=" * 68)
    log.info("  BENCHMARK HEALTH SCORES")
    log.info("=" * 68)
    for split_name in SPLIT_NAMES:
        if split_name in all_diagnostics and "health_score" in all_diagnostics[split_name]:
            d = all_diagnostics[split_name]
            if "error" in d:
                log.info(f"  {split_name:<14} ERROR")
            else:
                extra = ""
                if d.get("scaffold_diversity_ratio") is not None:
                    extra += f"  div_ratio={d['scaffold_diversity_ratio']:.2f}"
                if d.get("cliff_test_coverage_pct") is not None:
                    extra += f"  cliff_test={d['cliff_test_coverage_pct']:.0f}%"
                log.info(
                    f"  {split_name:<14} score={d['health_score']:>3} "
                    f"{d['health_label']:<11}{extra}"
                )

    master = {
        "generated":        pd.Timestamp.now().isoformat(),
        "pipeline_version": PIPELINE_VERSION,
        "config_hash":      cfg_hash,
        "config":           cfg,
        "seed":             cfg.get("seed"),
        "data_hash_t1":     hash_df(t1),
        "splits":           all_stats,
        "leakage_audits":   all_audits,
        "scaffold_health":  all_health,
        "diagnostics":      all_diagnostics,
    }
    with open(out / "master_audit.json", "w") as f:
        json.dump(master, f, indent=2, default=str)

    diag_out = {}
    for split_name in SPLIT_NAMES:
        if split_name in all_diagnostics:
            d = all_diagnostics[split_name]
            diag_out[split_name] = {k: v for k, v in d.items() if k not in ("error", "skipped")}
    with open(out / "split_diagnostics.json", "w") as f:
        json.dump(diag_out, f, indent=2, default=str)

    rows = []
    for method, s in all_stats.items():
        if "error" in s or "skipped" in s:
            continue
        tr = s.get("train", {}); te = s.get("test", {}); va = s.get("val", {})
        row = {
            "method":              method,
            "n_train":             tr.get("n"),
            "n_val":               va.get("n"),
            "n_test":              te.get("n"),
            "n_scaffolds_train":   tr.get("n_scaffolds"),
            "n_scaffolds_test":    te.get("n_scaffolds"),
            "test_pic50_mean":     te.get("pIC50_mean"),
            "test_pic50_std":      te.get("pIC50_std"),
            "test_pct_active":     te.get("pct_active"),
            "health_score":        all_diagnostics.get(method, {}).get("health_score"),
            "health_label":        all_diagnostics.get(method, {}).get("health_label"),
        }
        if method == "cliff_aware" and "metadata" in s:
            row["cliff_test_count"]        = s["metadata"].get("cliff_test_count")
            row["cliff_test_coverage_pct"] = s["metadata"].get("cliff_test_coverage_pct")
        rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(out / "splits_summary.csv", index=False)

def print_benchmark_recommendations(diagnostics: dict):
    log.info("\n" + "=" * 68)
    log.info("  BENCHMARK RECOMMENDATIONS")
    log.info("=" * 68)
    log.info("  Primary benchmark splits (report co-equally):")
    primary = ["random", "scaffold", "confirmed", "lead_opt"]
    for split_name in primary:
        if split_name in diagnostics and "error" not in diagnostics[split_name]:
            score = diagnostics[split_name].get("health_score", "?")
            label = diagnostics[split_name].get("health_label", "?")
            log.info(f"    ✓ {split_name:<14} score={score} {label}")
        else:
            log.info(f"    ! {split_name:<14} unavailable")

    log.info("\n  Auxiliary stress-tests (report separately):")
    for aux in ["similarity", "cliff_aware"]:
        if aux in diagnostics and "error" not in diagnostics[aux]:
            score = diagnostics[aux].get("health_score", 0)
            label = diagnostics[aux].get("health_label", "?")
            notes = ""
            if aux == "similarity" and diagnostics[aux].get("degenerate"):
                notes = " (DEGENERATE — dominant chemotype)"
            if aux == "cliff_aware":
                cov = diagnostics[aux].get("cliff_test_coverage_pct", 0)
                notes = f" (cliff_test_coverage={cov:.0f}%)"
            log.info(f"    ~ {aux:<14} score={score} {label}{notes}")
        else:
            log.info(f"    ! {aux:<14} unavailable")

# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=f"PAD4 Splitting {PIPELINE_VERSION} — Cliff-Detection + Coverage-Accounting Fix",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", choices=["regression", "classification", "both"],
                   default="both", help="Execution mode")
    p.add_argument("--output_dir", default=_default_output_dir())

    p.add_argument("--t1_file",
                   default=str(Path(_default_processed_dir()) / "pad_t1_non_covalent.csv"))
    p.add_argument("--confirmed_file",
                   default=str(Path(_default_processed_dir()) / "pad_t1_confirmed.csv"))
    p.add_argument("--classification_file",
                   default=str(Path(_default_processed_dir()) / "pad_classification_v17.csv"))
    p.add_argument("--regression_splits_dir", default="",
                   help="Override path for existing regression splits (classification mode only).")

    p.add_argument("--cliff_pairs_file", default="",
                   help="CSV with cliff pairs. Auto-detects column conventions "
                        "(inchikey_1/inchikey_2, id_a/id_b, etc). "
                        "Auto-discovered as pad_activity_cliffs.csv in processed dir "
                        "if not set.")

    p.add_argument("--smiles_col",       default="canonical_smiles")
    p.add_argument("--scaffold_col",     default="stereo_stripped_scaffold",
                   help="Content-derived scaffold column (NOT a per-file integer).")
    p.add_argument("--id_col",           default="inchikey_14")
    p.add_argument("--activity_col",     default="pIC50")
    p.add_argument("--class_stratify_col", default="activity_class")

    p.add_argument("--test_frac",           type=float, default=0.15)
    p.add_argument("--val_frac",            type=float, default=0.10)
    p.add_argument("--scaffold_cap_frac",   type=float, default=0.08)
    p.add_argument("--similarity_cap_frac", type=float, default=0.10)
    p.add_argument("--lead_opt_min_size",   type=int,   default=4)
    p.add_argument("--butina_cutoff",       type=float, default=0.40)
    p.add_argument("--seed",                type=int,   default=42)

    p.add_argument("--cv_folds", type=int, default=0,
                   help="If > 1, also generate scaffold-stratified k-fold CV "
                        "under regression/cv_folds/. 0 = disabled.")

    p.add_argument("--skip_similarity",     action="store_true")
    p.add_argument("--skip_cliff_aware",    action="store_true",
                   help="Skip the cliff-aware split entirely.")
    p.add_argument("--skip_regression",     action="store_true")
    p.add_argument("--skip_classification", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing frozen benchmark.")
    return p

def main():
    p    = build_parser()
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg      = {k: v for k, v in vars(args).items() if k not in ("output_dir", "force")}
    cfg_hash = hash_config(cfg)

    log.info("=" * 68)
    log.info(f"  PAD4 SPLITTING {PIPELINE_VERSION}  (cliff-detection + coverage-accounting fix)")
    log.info("=" * 68)
    log.info(f"  config hash:   {cfg_hash}")
    log.info(f"  mode:          {args.mode}")
    log.info(f"  scaffold_col:  {args.scaffold_col}")
    log.info(f"  output:        {out.resolve()}")
    log.info(f"  splits:        {SPLIT_NAMES}")

    try:
        check_freeze_collision(out, args.force)
    except RuntimeError as e:
        log.error(str(e))
        sys.exit(2)

    all_diagnostics = {}

    if args.mode in ("regression", "both") and not args.skip_regression:
        log.info("\n>>> REGRESSION MODE <<<\n")
        t1        = pd.read_csv(args.t1_file)
        confirmed = pd.read_csv(args.confirmed_file)
        log.info(f"  T1: {len(t1):,} compounds, {t1[args.scaffold_col].nunique():,} scaffolds")
        log.info(f"  Confirmed: {len(confirmed):,} compounds")

        log.info("\n  Deduplicating InChIKey-14 groups:")
        t1        = dedupe_by_inchikey14(t1,        "T1",        id_col=args.id_col)
        confirmed = dedupe_by_inchikey14(confirmed, "Confirmed", id_col=args.id_col)

        all_stats, all_audits, all_health, all_diagnostics = run_regression_splits(
            t1, confirmed, args, out, args.seed
        )
        write_master_summary(out, cfg, cfg_hash, t1, all_stats, all_audits, all_health, all_diagnostics)
        log.info("  Regression splits complete.")

    if args.mode in ("classification", "both") and not args.skip_classification:
        log.info("\n>>> CLASSIFICATION MODE <<<\n")
        cls_path = Path(args.classification_file)
        if not cls_path.exists():
            raise FileNotFoundError(f"Classification file not found: {cls_path}")

        cls_df = pd.read_csv(cls_path)
        log.info(f"  Read {len(cls_df):,} classification rows")

        log.info("\n  Deduplicating InChIKey-14 groups:")
        cls_df = dedupe_by_inchikey14(cls_df, "Classification", id_col=args.id_col)

        if args.mode == "classification":
            log.warning(
                "  Running in classification-only mode: benchmark recommendations "
                "will not be printed (no regression diagnostics available). "
                "Run --mode both for a full report."
            )

        run_classification_alignment(cls_df, args, out, args.seed)
        log.info("  Classification alignment complete.")

    if all_diagnostics:
        print_benchmark_recommendations(all_diagnostics)

    log.info("\n" + "=" * 68)
    log.info("  WRITING MANIFEST + BENCHMARK HASH")
    log.info("=" * 68)
    manifest_path = write_output_manifest(out, cfg_hash)
    hash_path     = write_benchmark_hash(out, cfg_hash, cfg)
    log.info(f"  manifest: {manifest_path}")
    log.info(f"  hash:     {hash_path}")

    log.info("\n" + "=" * 68)
    log.info(f"  DONE. output: {out.resolve()}")
    log.info(f"  To freeze: git tag pad4bench-{PIPELINE_VERSION}-splits")
    log.info("=" * 68)

if __name__ == "__main__":
    main()