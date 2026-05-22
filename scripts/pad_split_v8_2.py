#!/usr/bin/env python3
"""
PAD4 Splitting v8.2 — Patched Frozen Benchmark Release
=======================================================
Patches over v8.1. Adds a single but critical I/O fix: regression and
classification splits now write to separate subdirectories, so they
don't overwrite each other on disk.

Changes from v8.1 → v8.2
-------------------------
  [FIX-5] Regression and classification splits are now written to
          separate subtrees:
              data/splits/
              ├── regression/
              │   ├── scaffold/   { train.csv, val.csv, test.csv, ... }
              │   ├── random/     { ... }
              │   ├── ...
              ├── classification/
              │   ├── scaffold/   { ... }
              │   ├── ...
              ├── master_audit.json
              ├── alignment_audit.json
              ├── splits_manifest.json
              └── BENCHMARK_HASH.txt
          v8.1 wrote both to e.g. data/splits/scaffold/train.csv, so
          whichever task ran second silently overwrote the first
          (classification ran second → regression CSVs were gone).
          The split *logic* is unchanged; only output paths and the
          classification-alignment reader changed.

  [INFRA] `run_classification_alignment` now reads regression splits
          from `<output_dir>/regression/` by default, matching the new
          layout. A user-provided `--regression_splits_dir` still
          overrides this for the case where someone wants to align
          classification against an externally-provided regression
          benchmark.

Changes from v8.0 → v8.1 (preserved)
-------------------------------------
  [FIX-1] distribute_unassigned: replaced the order-sensitive
          "shuffle + linear fraction check" with a deterministic
          largest-first greedy fill that picks the most under-budget
          bucket at each step. In v8.0, the first scaffold examined
          always landed in `test` regardless of size, then the budget
          logic ran away. On the confirmed-classification split this
          inflated the test set to 27% of data vs the intended 15%.
          The new algorithm respects targets to within one scaffold
          and is invariant to scaffold ordering noise.

  [FIX-2] Routing vectorized. v8.0 used .apply(axis=1) and then mutated
          the resulting Series in-place; this is slow (~100× slower)
          and breaks under pandas 3.x copy-on-write semantics. Now uses
          pure .map() / .fillna() with no row dispatch and no mutation.

  [FIX-3] Default scaffold column changed from "scaffold_id" (a per-file
          integer in many cleaning pipelines, which silently mismatches
          across files) to "stereo_stripped_scaffold" (a canonical SMILES
          string, which is the join key v6_aligned originally used).
          Plus a new assertion `assert_scaffold_id_consistency()` runs at
          the start of classification alignment: for compounds shared
          between regression train and classification, scaffold IDs MUST
          be identical, or the script aborts. This catches per-file
          integer-ID drift before it corrupts the benchmark.

  [FIX-4] Cross-dataset leakage audit added. For scaffold and confirmed
          (the disjoint splits), the audit now also checks
          classification_test ∩ regression_train at the InChIKey-14
          level. Even if you don't stack today, this number tells you
          whether stacking is viable later, and surfaces silent leakage
          if the scaffold-ID join is ever wrong.

Everything else from v8.0 is preserved verbatim.

Usage
-----
  cd /home/nidhal/PAD4_BENCH
  python scripts/pad_split_v8_1.py --mode both
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
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
PIPELINE_VERSION = "v8.2"

BUTINA_N_LIMIT = 10_000
MAX_IMBALANCE_DRIFT_PP = 20.0
SPLIT_NAMES = ["scaffold", "random", "similarity", "confirmed", "lead_opt"]
SUBSETS = ["train", "val", "test"]

SIMILARITY_CUTOFFS = [0.40, 0.50, 0.55, 0.60, 0.65, 0.35, 0.30, 0.25]
MIN_VAL_FRAC = 0.08

EXPECTED_MATCH = {"random"}
EXPECTED_DIFFER = {"scaffold", "similarity"}
SOFT_DIFFER = {"lead_opt", "confirmed"}

DISJOINT_SPLITS = frozenset({"scaffold", "confirmed"})

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
    return str(Path.cwd() / "splits_v8_1")

# ══════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════
def hash_df(df: pd.DataFrame) -> str:
    # Sort columns for stability across pipeline edits that reorder them.
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
    df["_weight"] = df[weight_col].fillna(0.0) if weight_col in df.columns else 0.0
    df["_row_idx"] = np.arange(n_in)
    df_sorted = df.sort_values(
        by=["_stereo_rank", "_weight", "_row_idx"],
        ascending=[True, False, True],
    )
    df_out = df_sorted.drop_duplicates(subset=[id_col], keep="first") \
        .drop(columns=["_stereo_rank", "_weight", "_row_idx"]) \
        .sort_values(by=id_col) \
        .reset_index(drop=True)
    n_out = len(df_out)
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
        report["scaffold_train_val_overlap"] = len(tv_ov)
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
        report["inchikey14_train_val_overlap"] = len(tv_ik)
        if tt_ik:
            msg = (f"INCHIKEY-14 LEAKAGE in {split_name}: {len(tt_ik)} compounds")
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
            "val_pct_active": round(va_act, 1),
            "test_pct_active": round(te_act, 1),
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
            msg = "unexpected distribution shift"
            log.warning(f"  KS train vs {label}: stat={stat:.3f} p={p:.2e} — {msg}")
            result["note"] = msg
            result["unexpected"] = True
        else:
            msg = "distributions match (expected)"
            log.info(f"  KS train vs {label}: stat={stat:.3f} p={p:.2f} — {msg}")
            result["note"] = msg
            result["unexpected"] = False

    elif split_type in EXPECTED_DIFFER:
        if p < 0.05:
            msg = "distributions differ (expected)"
            log.info(f"  KS train vs {label}: stat={stat:.3f} p={p:.2e} — {msg}")
            result["note"] = msg
            result["unexpected"] = False
        else:
            msg = "unexpected distribution similarity"
            log.warning(f"  KS train vs {label}: stat={stat:.3f} p={p:.2f} — {msg}")
            result["note"] = msg
            result["unexpected"] = True

    elif split_type == "confirmed":
        if stat < 0.15:
            msg = "mild distribution shift (acceptable for scaffold-capped confirmed split)"
            log.info(f"  KS train vs {label}: stat={stat:.3f} p={p:.2e} — {msg}")
            result["note"] = msg
            result["unexpected"] = False
        else:
            msg = "elevated shift (review confirmed-set scaffold allocation)"
            log.warning(f"  KS train vs {label}: stat={stat:.3f} p={p:.2e} — {msg}")
            result["note"] = msg
            result["unexpected"] = True

    elif split_type in SOFT_DIFFER:  # lead_opt
        if stat < 0.15:
            msg = "mild distribution shift (acceptable)"
            log.info(f"  KS train vs {label}: stat={stat:.3f} p={p:.2e} — {msg}")
            result["note"] = msg
            result["unexpected"] = False
        elif stat < 0.25:
            msg = "moderate shift (review)"
            log.warning(f"  KS train vs {label}: stat={stat:.3f} p={p:.2e} — {msg}")
            result["note"] = msg
            result["unexpected"] = False
        else:
            msg = "strong shift (verify stratification)"
            log.warning(f"  KS train vs {label}: stat={stat:.3f} p={p:.2e} — {msg}")
            result["note"] = msg
            result["unexpected"] = True

    else:
        msg = "distributions differ" if p < 0.05 else "distributions match"
        log.info(f"  KS train vs {label}: stat={stat:.3f} p={p:.2e} — {msg}")
        result["note"] = msg
        result["unexpected"] = False

    return result

# ══════════════════════════════════════════════════════════════════════
# SCAFFOLD SPLIT (unchanged from v8.0)
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
    df = df.copy().reset_index(drop=True)
    n = len(df)
    if scaffold_col not in df.columns:
        raise ValueError(f"'{scaffold_col}' not in data")

    test_budget = int(math.ceil(test_frac * n))
    val_budget = int(math.ceil(val_frac * n))
    scaffold_cap = max(1, int(math.ceil(cap_frac * test_budget)))
    log.info(f"  n={n} test_budget={test_budget} val_budget={val_budget} cap={scaffold_cap}")

    scaf_to_idx: Dict[str, List[int]] = defaultdict(list)
    for i, s in enumerate(df[scaffold_col]):
        scaf_to_idx[str(s)].append(i)

    multi_scafs = [s for s, ixs in scaf_to_idx.items() if len(ixs) > 1]
    singleton_scafs = [s for s, ixs in scaf_to_idx.items() if len(ixs) == 1]
    multi_sorted = sorted(multi_scafs, key=lambda s: len(scaf_to_idx[s]))
    log.info(f"  {len(multi_scafs)} multi scaffolds, {len(singleton_scafs)} singletons")

    test_idx: List[int] = []
    val_idx: List[int] = []
    train_idx: List[int] = []

    singleton_test_reserve = int(math.ceil(0.10 * test_budget))
    multi_test_target = test_budget - singleton_test_reserve

    eligible = [s for s in multi_sorted if len(scaf_to_idx[s]) <= scaffold_cap]
    oversized = [s for s in multi_sorted if len(scaf_to_idx[s]) > scaffold_cap]

    if eligible:
        eligible_sizes = np.array([len(scaf_to_idx[s]) for s in eligible])
        try:
            bin_edges = np.quantile(eligible_sizes, [0, 0.25, 0.5, 0.75, 1.0])
            bin_edges = np.unique(bin_edges)
            bin_idx = np.digitize(eligible_sizes, bin_edges[1:-1])
            n_bins = len(bin_edges) - 1 if len(bin_edges) > 1 else 1
        except Exception:
            bin_idx = np.zeros(len(eligible), dtype=int)
            n_bins = 1

        bin_to_scafs: Dict[int, List[str]] = defaultdict(list)
        for s, b in zip(eligible, bin_idx):
            bin_to_scafs[int(b)].append(s)
        for b in bin_to_scafs:
            rng.shuffle(bin_to_scafs[b])

        bin_compositions = {
            int(b): {
                "n_scafs": len(scafs),
                "n_compounds": int(sum(len(scaf_to_idx[s]) for s in scafs)),
                "size_range": (
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

    assigned_to_test = {scaf for scaf in multi_sorted
                        if all(i in test_idx for i in scaf_to_idx[scaf])}

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

    assigned_to_val = {scaf for scaf in multi_sorted
                       if all(i in val_idx for i in scaf_to_idx[scaf])
                       and scaf not in assigned_to_test}

    for scaf in multi_sorted:
        if scaf in assigned_to_test or scaf in assigned_to_val:
            continue
        train_idx.extend(scaf_to_idx[scaf])

    singleton_idxs = [scaf_to_idx[s][0] for s in singleton_scafs]
    if singleton_idxs and activity_col in df.columns:
        singleton_idxs_sorted = sorted(
            singleton_idxs, key=lambda i: df.at[i, activity_col]
        )
        remaining_test_budget = max(0, test_budget - len(test_idx))
        remaining_val_budget = max(0, val_budget - len(val_idx))

        total_singleton_to_test = min(remaining_test_budget, len(singleton_idxs_sorted))
        total_singleton_to_val = min(
            remaining_val_budget, len(singleton_idxs_sorted) - total_singleton_to_test
        )

        if total_singleton_to_test > 0:
            indices = np.linspace(
                0, len(singleton_idxs_sorted) - 1, total_singleton_to_test
            ).astype(int)
            test_sample = [singleton_idxs_sorted[i] for i in indices]
            test_idx.extend(test_sample)
            remaining = [i for i in singleton_idxs_sorted if i not in set(test_sample)]
        else:
            remaining = singleton_idxs_sorted

        if total_singleton_to_val > 0 and remaining:
            indices = np.linspace(
                0, len(remaining) - 1, total_singleton_to_val
            ).astype(int)
            val_sample = [remaining[i] for i in indices]
            val_idx.extend(val_sample)
            remaining = [i for i in remaining if i not in set(val_sample)]

        train_idx.extend(remaining)
        log.info(f"  singletons -> test:{total_singleton_to_test} val:{total_singleton_to_val} train:{len(remaining)}")
    else:
        train_idx.extend(singleton_idxs)

    train_set, val_set, test_set = set(train_idx), set(val_idx), set(test_idx)
    assert not (train_set & test_set), "train/test overlap"
    assert not (train_set & val_set), "train/val overlap"
    assert not (val_set & test_set), "val/test overlap"
    assigned = train_set | val_set | test_set
    missing = set(range(n)) - assigned
    if missing:
        log.warning(f"  {len(missing)} unassigned; adding to train")
        train_idx.extend(sorted(missing))

    tr_scafs = set(df.iloc[train_idx][scaffold_col])
    te_scafs = set(df.iloc[test_idx][scaffold_col])
    va_scafs = set(df.iloc[val_idx][scaffold_col])
    leaks_tt = tr_scafs & te_scafs
    leaks_tv = tr_scafs & va_scafs
    if leaks_tt or leaks_tv:
        raise RuntimeError(f"Internal scaffold leakage (tt={len(leaks_tt)}, tv={len(leaks_tv)})")

    return (
        df.iloc[sorted(train_idx)].copy(),
        df.iloc[sorted(val_idx)].copy(),
        df.iloc[sorted(test_idx)].copy(),
    )

# ══════════════════════════════════════════════════════════════════════
# SIMILARITY SPLIT (unchanged from v8.0)
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

    test_budget = int(math.ceil(test_frac * n))
    val_budget = int(math.ceil(val_frac * n))
    cluster_cap = max(1, int(math.ceil(cap_frac * test_budget)))
    if not quiet:
        log.info(f"  n={n} test_budget={test_budget} val_budget={val_budget} cluster_cap={cluster_cap}")
        log.info(f"  Butina clustering at cutoff={cutoff:.2f}")

    dists = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(valid_fps[i], valid_fps[:i])
        dists.extend([1.0 - s for s in sims])

    clusters = Butina.ClusterData(dists, n, 1.0 - cutoff, isDistData=True)
    n_cl = len(clusters)
    n_sin = sum(1 for c in clusters if len(c) == 1)
    largest_cluster_size = len(clusters[0]) if clusters else 0
    largest_cluster_fraction = largest_cluster_size / n if n > 0 else 0.0

    if not quiet:
        log.info(f"  {n_cl:,} clusters | largest={largest_cluster_size} | singletons={n_sin} ({100*n_sin/max(1,n_cl):.0f}%)")

    clusters_asc = sorted(clusters, key=len)
    test_idx: List[int] = []
    val_idx: List[int] = []
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

    tr_arr = valid_arr[sorted(train_idx)]
    te_arr = valid_arr[sorted(test_idx)]
    sim_stats = cross_sim_stats(te_arr, tr_arr)

    mx = sim_stats.get("max_nn_tanimoto", 0.0)
    mn = sim_stats.get("mean_nn_tanimoto", 0.0)
    if not quiet:
        log.info(f"  cross-split Tanimoto: max={mx:.3f} mean={mn:.3f} difficulty={sim_stats.get('difficulty_score', 0):.3f}")

    sim_stats.update({
        "butina_cutoff": cutoff,
        "n_clusters": n_cl,
        "n_singletons": n_sin,
        "largest_cluster_size": largest_cluster_size,
        "largest_cluster_fraction": round(largest_cluster_fraction, 4),
        "cluster_cap": cluster_cap,
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
    best_score = -999999
    best_cutoff = None
    best_stats = None
    tried = []

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

        n_total = len(tr) + len(va) + len(te)
        test_budget = int(math.ceil(test_frac * n_total))

        largest_frac = sim_stats.get("largest_cluster_fraction", 0)
        n_cl = sim_stats.get("n_clusters", 0)
        val_size = len(va)
        test_size = len(te)

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
            sim_stats["adaptive_tried"] = tried
            sim_stats["degenerate"] = False
            return tr, va, te, sim_stats

        log.warning(f"  Similarity cutoff {cutoff:.2f} rejected:")
        for f in failures:
            log.warning(f"    {f}")

        score = 0
        score -= largest_frac * 200
        score -= max(0, 100 - n_cl)
        score -= 100 if val_size == 0 else 0
        score -= max(0, test_budget - test_size) * 2

        if score > best_score:
            best_score = score
            best_result = (tr, va, te)
            best_cutoff = cutoff
            best_stats = sim_stats.copy()
            best_stats["rejection_reasons"] = failures

    if best_result is None:
        raise RuntimeError("Similarity split failed at all cutoffs")

    log.warning(f"  Keeping best available cutoff={best_cutoff:.2f} (DEGENERATE)")
    tr, va, te = best_result
    best_stats["selected_cutoff"] = best_cutoff
    best_stats["adaptive_tried"] = tried
    best_stats["degenerate"] = True
    best_stats["degenerate_reasons"] = best_stats.get("rejection_reasons", [])
    best_stats["paper_note"] = (
        best_stats.get("paper_note", "") +
        " DEGENERATE: dominant chemotype prevented a clean OOD partition; "
        "report as auxiliary stress-test only, not co-equal with primary splits."
    )
    return tr, va, te, best_stats

# ══════════════════════════════════════════════════════════════════════
# RANDOM SPLIT (unchanged from v8.0)
# ══════════════════════════════════════════════════════════════════════
def split_random(
    df: pd.DataFrame,
    test_frac: float = 0.15,
    val_frac: float = 0.10,
    seed: int = 42,
    activity_col: str = "pIC50",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    df = df.copy().reset_index(drop=True)
    n = len(df)

    if activity_col not in df.columns:
        all_idx = rng.permutation(n)
        n_test = int(n * test_frac)
        n_val = int(n * val_frac)
        return (
            df.iloc[sorted(all_idx[n_test + n_val:])].copy(),
            df.iloc[sorted(all_idx[n_test:n_test + n_val])].copy(),
            df.iloc[sorted(all_idx[:n_test])].copy(),
        )

    df["_q"] = pd.qcut(df[activity_col], q=5, labels=False, duplicates="drop")
    test_idx, val_idx, train_idx = [], [], []
    for q in sorted(df["_q"].dropna().unique()):
        qp = df[df["_q"] == q].index.tolist()
        rng.shuffle(qp)
        n_q = len(qp)
        n_test_q = int(round(n_q * test_frac))
        n_val_q = int(round(n_q * val_frac))
        test_idx.extend(qp[:n_test_q])
        val_idx.extend(qp[n_test_q:n_test_q + n_val_q])
        train_idx.extend(qp[n_test_q + n_val_q:])
    df = df.drop(columns=["_q"])

    return (
        df.iloc[sorted(train_idx)].copy(),
        df.iloc[sorted(val_idx)].copy(),
        df.iloc[sorted(test_idx)].copy(),
    )

# ══════════════════════════════════════════════════════════════════════
# CONFIRMED & LEAD OPT (unchanged from v8.0)
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
    df = df.copy().reset_index(drop=True)
    if scaffold_col not in df.columns:
        raise ValueError(f"'{scaffold_col}' not in data")

    ik14_series = get_inchikey14(df)
    has_ik14 = ik14_series is not None

    scaf_to_idx: Dict[str, List[int]] = defaultdict(list)
    for i, s in enumerate(df[scaffold_col]):
        scaf_to_idx[str(s)].append(i)

    eligible_scafs = [s for s, ixs in scaf_to_idx.items() if len(ixs) >= min_scaffold_size]
    small_scafs = [s for s, ixs in scaf_to_idx.items() if len(ixs) < min_scaffold_size]
    n_eligible_compounds = sum(len(scaf_to_idx[s]) for s in eligible_scafs)

    log.info(
        f"  lead_opt: {len(eligible_scafs)} eligible scaffolds "
        f"(size >= {min_scaffold_size}, total {n_eligible_compounds} compounds); "
        f"{len(small_scafs)} small -> train"
    )

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for scaf in eligible_scafs:
        idxs = scaf_to_idx[scaf]
        n = len(idxs)
        if activity_col in df.columns:
            acts = [df.at[i, activity_col] for i in idxs]
            order = np.argsort(acts)
            idxs_sorted = [idxs[o] for o in order]
        else:
            idxs_sorted = list(idxs)
            rng.shuffle(idxs_sorted)

        n_test = max(1, int(round(n * test_frac)))
        n_val = max(0, int(round(n * val_frac)))
        if n_test + n_val >= n:
            n_test = max(1, n - 2)
            n_val = max(0, n - n_test - 1)

        step_test = max(1, n // n_test)
        test_pick = idxs_sorted[::step_test][:n_test]
        remaining = [i for i in idxs_sorted if i not in set(test_pick)]

        if n_val > 0 and remaining:
            step_val = max(1, len(remaining) // n_val)
            val_pick = remaining[::step_val][:n_val]
            remaining = [i for i in remaining if i not in set(val_pick)]
        else:
            val_pick = []

        test_idx.extend(test_pick)
        val_idx.extend(val_pick)
        train_idx.extend(remaining)

    for scaf in small_scafs:
        train_idx.extend(scaf_to_idx[scaf])

    if has_ik14:
        ik14 = ik14_series.astype(str).values
        tr_keys = set(ik14[i] for i in train_idx)
        te_overlap = [i for i in test_idx if ik14[i] in tr_keys]
        va_overlap = [i for i in val_idx if ik14[i] in tr_keys]
        if te_overlap:
            log.warning(f"  lead_opt: {len(te_overlap)} IK14 dups in test -> train")
            test_idx = [i for i in test_idx if i not in set(te_overlap)]
            train_idx.extend(te_overlap)
        if va_overlap:
            log.warning(f"  lead_opt: {len(va_overlap)} IK14 dups in val -> train")
            val_idx = [i for i in val_idx if i not in set(va_overlap)]
            train_idx.extend(va_overlap)

    tr_set, va_set, te_set = set(train_idx), set(val_idx), set(test_idx)
    assert not (tr_set & te_set)
    assert not (tr_set & va_set)
    assert not (va_set & te_set)

    tr_scafs = set(df.iloc[train_idx][scaffold_col])
    te_scafs = set(df.iloc[test_idx][scaffold_col])
    shared = tr_scafs & te_scafs
    log.info(f"  lead_opt: {len(shared)} scaffolds shared train↔test (BY DESIGN)")

    meta = {
        "split_type": "lead_optimization",
        "min_scaffold_size": min_scaffold_size,
        "n_eligible_scaffolds": len(eligible_scafs),
        "n_eligible_compounds": n_eligible_compounds,
        "n_small_scaffolds_to_train": len(small_scafs),
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
# SAVE & SUMMARISE
# ══════════════════════════════════════════════════════════════════════
def save_split(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
               out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    train.assign(split="train").to_csv(out_dir / "train.csv", index=False)
    val.assign(split="val").to_csv(out_dir / "val.csv", index=False)
    test.assign(split="test").to_csv(out_dir / "test_locked.csv", index=False)
    test.assign(split="test").to_csv(out_dir / "test.csv", index=False)

def subset_summary(df: pd.DataFrame, label: str, scaffold_col: str,
                   activity_col: str) -> dict:
    s = {"label": label, "n": len(df)}
    if scaffold_col in df.columns:
        s["n_scaffolds"] = int(df[scaffold_col].nunique())
        counts = df[scaffold_col].value_counts()
        s["pct_singletons"] = round(100 * (counts == 1).mean(), 1) if len(counts) else 0.0
    if activity_col in df.columns:
        s.update({
            f"{activity_col}_mean": round(float(df[activity_col].mean()), 3),
            f"{activity_col}_std": round(float(df[activity_col].std()), 3),
            "pct_active": round(float((df[activity_col] >= 6.0).mean() * 100), 1),
        })
    return s

# ══════════════════════════════════════════════════════════════════════
# HEALTH SCORE (unchanged from v8.0)
# ══════════════════════════════════════════════════════════════════════
def compute_health_score(
    split_type: str,
    audit_report: dict,
    ks_result: dict,
    n_val: int,
    sim_stats: Optional[dict] = None,
    scaffold_diversity_ratio: Optional[float] = None,
) -> Tuple[int, str]:
    score = 100

    if audit_report.get("errors"):
        score -= 100

    drift = audit_report.get("imbalance", {}).get("train_test_drift_pp", 0)
    if drift > 35:
        score -= 30
    elif drift > 20:
        score -= 20
    elif drift > 10:
        score -= 10
    elif drift > 5:
        score -= 4
    elif drift > 2:
        score -= 2

    if sim_stats:
        largest_frac = sim_stats.get("largest_cluster_fraction", 0)
        if largest_frac > 0.60:
            score -= 40
        elif largest_frac > 0.40:
            score -= 20

    if n_val == 0:
        score -= 25

    if ks_result:
        stat = ks_result.get("ks_stat", 0)
        p = ks_result.get("ks_p", 1.0)

        if split_type in EXPECTED_MATCH:
            if p < 0.01:
                score -= 8
            elif p < 0.05:
                score -= 4
        elif split_type in EXPECTED_DIFFER:
            if p >= 0.05:
                score -= 15
            elif stat > 0.40:
                score -= 10
        elif split_type == "confirmed":
            if stat >= 0.20:
                score -= 10
            elif stat >= 0.10:
                score -= 3
        elif split_type in SOFT_DIFFER:
            if stat >= 0.30:
                score -= 20
            elif stat >= 0.20:
                score -= 12
            elif stat >= 0.15:
                score -= 5

    if scaffold_diversity_ratio is not None:
        if scaffold_diversity_ratio < 0.30:
            score -= 5
        elif scaffold_diversity_ratio < 0.50:
            score -= 2

    score = max(0, score)
    if score >= 90:
        label = "EXCELLENT"
    elif score >= 75:
        label = "GOOD"
    elif score >= 60:
        label = "ACCEPTABLE"
    elif score >= 40:
        label = "WEAK"
    else:
        label = "DEGENERATE"

    return score, label

def diversity_ratio(train: pd.DataFrame, test: pd.DataFrame,
                    scaffold_col: str) -> Optional[float]:
    if scaffold_col not in train.columns or scaffold_col not in test.columns:
        return None
    if len(train) == 0 or len(test) == 0:
        return None
    tr_counts = train[scaffold_col].value_counts()
    te_counts = test[scaffold_col].value_counts()
    tr_sing = float((tr_counts == 1).mean()) if len(tr_counts) else 0.0
    te_sing = float((te_counts == 1).mean()) if len(te_counts) else 0.0
    if tr_sing < 1e-6:
        return None
    return round(te_sing / tr_sing, 3)

# ══════════════════════════════════════════════════════════════════════
# REGRESSION RUNNER (unchanged from v8.0 — pass through scaffold_col)
# ══════════════════════════════════════════════════════════════════════
def _record_diagnostics(
    split_name: str,
    tr: pd.DataFrame, va: pd.DataFrame, te: pd.DataFrame,
    audit: dict, ks_res: dict, scaffold_col: str,
    health: Tuple[int, str], sim_stats: Optional[dict] = None,
    div_ratio: Optional[float] = None,
) -> dict:
    h, hl = health
    return {
        "n_train": len(tr), "n_val": len(va), "n_test": len(te),
        "n_scaffolds_train": int(tr[scaffold_col].nunique()) if scaffold_col in tr.columns else None,
        "n_scaffolds_test": int(te[scaffold_col].nunique()) if scaffold_col in te.columns else None,
        "activity_drift_pp": audit.get("imbalance", {}).get("train_test_drift_pp"),
        "ks_stat": ks_res.get("ks_stat"),
        "ks_p": ks_res.get("ks_p"),
        "ks_note": ks_res.get("note"),
        "scaffold_diversity_ratio": div_ratio,
        "largest_cluster_fraction": sim_stats.get("largest_cluster_fraction") if sim_stats else None,
        "selected_cutoff": sim_stats.get("selected_cutoff") if sim_stats else None,
        "degenerate": sim_stats.get("degenerate", False) if sim_stats else False,
        "health_score": h,
        "health_label": hl,
    }

def run_regression_splits(t1: pd.DataFrame, confirmed: pd.DataFrame, args,
                          out: Path, seed: int) -> Tuple[dict, dict, dict, dict]:
    # FIX-5: regression splits go to <out>/regression/, not <out>/
    reg_out = out / "regression"
    reg_out.mkdir(parents=True, exist_ok=True)
    audit_kw = dict(scaffold_col=args.scaffold_col, activity_col=args.activity_col)
    all_stats: dict = {}
    all_audits: dict = {}
    all_health: dict = {}
    all_diagnostics: dict = {}

    # 1. SCAFFOLD
    log.info("\n" + "=" * 68)
    log.info(f"  [1/5] SCAFFOLD (seed={seed}, cap_frac={args.scaffold_cap_frac})")
    log.info("=" * 68)
    tr, va, te = split_scaffold_capped(
        t1, test_frac=args.test_frac, val_frac=args.val_frac, seed=seed,
        scaffold_col=args.scaffold_col, cap_frac=args.scaffold_cap_frac,
        activity_col=args.activity_col,
    )
    log.info(f"  {len(tr):,} train | {len(va):,} val | {len(te):,} test")
    sh_tr = scaffold_stats(tr, "train_sc", args.scaffold_col)
    sh_te = scaffold_stats(te, "test_sc", args.scaffold_col)
    log_scaffold_health(sh_tr); log_scaffold_health(sh_te)
    all_audits["scaffold"] = audit_split(tr, va, te, "scaffold", **audit_kw, strict_inchikey=True)
    ks_res = ks_check(tr, te, "scaffold", args.activity_col, "test")
    save_split(tr, va, te, reg_out / "scaffold")
    div = diversity_ratio(tr, te, args.scaffold_col)
    if div is not None:
        log.info(f"  scaffold-diversity ratio (test/train singleton-pct): {div:.3f}")
    health = compute_health_score("scaffold", all_audits["scaffold"], ks_res, len(va),
                                   scaffold_diversity_ratio=div)
    all_stats["scaffold"] = {
        "train": subset_summary(tr, "train", args.scaffold_col, args.activity_col),
        "val": subset_summary(va, "val", args.scaffold_col, args.activity_col),
        "test": subset_summary(te, "test", args.scaffold_col, args.activity_col),
    }
    all_health["scaffold"] = {"train": sh_tr, "test": sh_te}
    all_diagnostics["scaffold"] = _record_diagnostics(
        "scaffold", tr, va, te, all_audits["scaffold"], ks_res,
        args.scaffold_col, health, div_ratio=div,
    )

    # 2. RANDOM
    log.info("\n" + "=" * 68)
    log.info(f"  [2/5] RANDOM (seed={seed}, IID baseline)")
    log.info("=" * 68)
    tr, va, te = split_random(
        t1, test_frac=args.test_frac, val_frac=args.val_frac, seed=seed,
        activity_col=args.activity_col,
    )
    log.info(f"  {len(tr):,} train | {len(va):,} val | {len(te):,} test")
    sh_tr = scaffold_stats(tr, "train_rnd", args.scaffold_col)
    sh_te = scaffold_stats(te, "test_rnd", args.scaffold_col)
    log_scaffold_health(sh_tr); log_scaffold_health(sh_te)
    all_audits["random"] = audit_split(
        tr, va, te, "random", **audit_kw,
        allow_scaffold_overlap=True, strict_inchikey=True,
    )
    ks_res = ks_check(tr, te, "random", args.activity_col, "test")
    save_split(tr, va, te, reg_out / "random")
    div = diversity_ratio(tr, te, args.scaffold_col)
    if div is not None:
        log.info(f"  scaffold-diversity ratio (test/train singleton-pct): {div:.3f}")
    health = compute_health_score("random", all_audits["random"], ks_res, len(va),
                                   scaffold_diversity_ratio=div)
    all_stats["random"] = {
        "train": subset_summary(tr, "train", args.scaffold_col, args.activity_col),
        "val": subset_summary(va, "val", args.scaffold_col, args.activity_col),
        "test": subset_summary(te, "test", args.scaffold_col, args.activity_col),
    }
    all_health["random"] = {"train": sh_tr, "test": sh_te}
    all_diagnostics["random"] = _record_diagnostics(
        "random", tr, va, te, all_audits["random"], ks_res,
        args.scaffold_col, health, div_ratio=div,
    )

    # 3. SIMILARITY
    log.info("\n" + "=" * 68)
    log.info(f"  [3/5] SIMILARITY (adaptive Butina, v8 bidirectional)")
    log.info("=" * 68)
    if args.skip_similarity or not HAS_RDKIT:
        log.info("  skipped")
        all_stats["similarity"] = {"skipped": True}
        all_diagnostics["similarity"] = {"skipped": True, "health_score": 0, "health_label": "DEGENERATE"}
    else:
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
            sh_te = scaffold_stats(te, "test_sim", args.scaffold_col)
            log_scaffold_health(sh_tr); log_scaffold_health(sh_te)
            all_audits["similarity"] = audit_split(
                tr, va, te, "similarity", **audit_kw,
                allow_scaffold_overlap=True, strict_inchikey=True,
            )
            ks_res = ks_check(tr, te, "similarity", args.activity_col, "test")
            save_split(tr, va, te, reg_out / "similarity")
            div = diversity_ratio(tr, te, args.scaffold_col)
            if div is not None:
                log.info(f"  scaffold-diversity ratio (test/train singleton-pct): {div:.3f}")
            health = compute_health_score("similarity", all_audits["similarity"], ks_res,
                                           len(va), sim_s, scaffold_diversity_ratio=div)
            all_stats["similarity"] = {
                "train": subset_summary(tr, "train", args.scaffold_col, args.activity_col),
                "val": subset_summary(va, "val", args.scaffold_col, args.activity_col),
                "test": subset_summary(te, "test", args.scaffold_col, args.activity_col),
                "butina_stats": sim_s,
            }
            all_health["similarity"] = {"train": sh_tr, "test": sh_te}
            all_diagnostics["similarity"] = _record_diagnostics(
                "similarity", tr, va, te, all_audits["similarity"], ks_res,
                args.scaffold_col, health, sim_stats=sim_s, div_ratio=div,
            )
        except Exception as e:
            log.error(f"  Similarity split failed: {e}")
            all_stats["similarity"] = {"error": str(e)}
            all_diagnostics["similarity"] = {"error": str(e), "health_score": 0, "health_label": "DEGENERATE"}

    # 4. CONFIRMED
    log.info("\n" + "=" * 68)
    log.info(f"  [4/5] CONFIRMED (independent scaffold-capped)")
    log.info("=" * 68)
    try:
        tr, va, te = split_confirmed(
            confirmed, test_frac=args.test_frac, val_frac=args.val_frac,
            seed=seed, scaffold_col=args.scaffold_col,
            cap_frac=args.scaffold_cap_frac, activity_col=args.activity_col,
        )
        log.info(f"  {len(tr):,} train | {len(va):,} val | {len(te):,} test")
        sh_tr = scaffold_stats(tr, "train_cf", args.scaffold_col)
        sh_te = scaffold_stats(te, "test_cf", args.scaffold_col)
        log_scaffold_health(sh_tr); log_scaffold_health(sh_te)
        all_audits["confirmed"] = audit_split(
            tr, va, te, "confirmed", **audit_kw, strict_inchikey=True,
        )
        ks_res = ks_check(tr, te, "confirmed", args.activity_col, "test")
        save_split(tr, va, te, reg_out / "confirmed")
        div = diversity_ratio(tr, te, args.scaffold_col)
        if div is not None:
            log.info(f"  scaffold-diversity ratio (test/train singleton-pct): {div:.3f}")
        health = compute_health_score("confirmed", all_audits["confirmed"], ks_res, len(va),
                                       scaffold_diversity_ratio=div)
        all_stats["confirmed"] = {
            "train": subset_summary(tr, "train", args.scaffold_col, args.activity_col),
            "val": subset_summary(va, "val", args.scaffold_col, args.activity_col),
            "test": subset_summary(te, "test", args.scaffold_col, args.activity_col),
        }
        all_health["confirmed"] = {"train": sh_tr, "test": sh_te}
        all_diagnostics["confirmed"] = _record_diagnostics(
            "confirmed", tr, va, te, all_audits["confirmed"], ks_res,
            args.scaffold_col, health, div_ratio=div,
        )
    except Exception as e:
        log.error(f"  Confirmed split failed: {e}")
        all_stats["confirmed"] = {"error": str(e)}
        all_diagnostics["confirmed"] = {"error": str(e), "health_score": 0, "health_label": "DEGENERATE"}

    # 5. LEAD OPT
    log.info("\n" + "=" * 68)
    log.info(f"  [5/5] LEAD_OPT (min_scaffold_size={args.lead_opt_min_size})")
    log.info("=" * 68)
    try:
        tr, va, te, lo_meta = split_lead_opt(
            t1, test_frac=args.test_frac, val_frac=args.val_frac, seed=seed,
            scaffold_col=args.scaffold_col,
            min_scaffold_size=args.lead_opt_min_size,
            activity_col=args.activity_col,
        )
        log.info(f"  {len(tr):,} train | {len(va):,} val | {len(te):,} test")
        sh_tr = scaffold_stats(tr, "train_lo", args.scaffold_col)
        sh_te = scaffold_stats(te, "test_lo", args.scaffold_col)
        log_scaffold_health(sh_tr); log_scaffold_health(sh_te)
        all_audits["lead_opt"] = audit_split(
            tr, va, te, "lead_opt", **audit_kw,
            allow_scaffold_overlap=True, strict_inchikey=True,
        )
        ks_res = ks_check(tr, te, "lead_opt", args.activity_col, "test")
        save_split(tr, va, te, reg_out / "lead_opt")
        div = diversity_ratio(tr, te, args.scaffold_col)
        if div is not None:
            log.info(f"  scaffold-diversity ratio (test/train singleton-pct): {div:.3f}")
        health = compute_health_score("lead_opt", all_audits["lead_opt"], ks_res, len(va),
                                       scaffold_diversity_ratio=div)
        all_stats["lead_opt"] = {
            "train": subset_summary(tr, "train", args.scaffold_col, args.activity_col),
            "val": subset_summary(va, "val", args.scaffold_col, args.activity_col),
            "test": subset_summary(te, "test", args.scaffold_col, args.activity_col),
            "metadata": lo_meta,
        }
        all_health["lead_opt"] = {"train": sh_tr, "test": sh_te}
        all_diagnostics["lead_opt"] = _record_diagnostics(
            "lead_opt", tr, va, te, all_audits["lead_opt"], ks_res,
            args.scaffold_col, health, div_ratio=div,
        )
    except Exception as e:
        log.error(f"  lead_opt split failed: {e}")
        all_stats["lead_opt"] = {"error": str(e)}
        all_diagnostics["lead_opt"] = {"error": str(e), "health_score": 0, "health_label": "DEGENERATE"}

    return all_stats, all_audits, all_health, all_diagnostics

# ══════════════════════════════════════════════════════════════════════
# CLASSIFICATION ALIGNMENT  (v8.1: vectorized routing + cross-dataset
# audits + scaffold-ID consistency assertion + corrected redistribution)
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

            df = pd.read_csv(path)
            if scaffold_col not in df.columns:
                log.error(f"  {split_name}/{subset}: no {scaffold_col}")
                continue

            for sc in norm_scaffold(df[scaffold_col].dropna()).unique():
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
            log.info(f"  {split_name}: {conflicts} scaffold overlaps (expected for random/similarity/lead_opt)")
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
            df = pd.read_csv(path)
            if id_col not in df.columns:
                continue
            for cid in df[id_col].dropna().astype(str).unique():
                if cid not in id_map:
                    id_map[cid] = subset
        assignments[split_name] = id_map
    return assignments

# ── FIX-3: cross-file scaffold-ID consistency assertion ──────────────
def assert_scaffold_id_consistency(
    reg_dir: Path,
    cls_df: pd.DataFrame,
    id_col: str,
    scaffold_col: str,
    sample_split: str = "scaffold",
) -> None:
    """
    For compounds shared between regression train and classification, the
    scaffold IDs must be identical. If they're not, the join key is broken
    (most commonly because scaffold_col is a per-file integer ID rather
    than a canonical SMILES). This is the silent failure mode that gives
    valid-looking-but-wrong splits, so we fail loudly.
    """
    reg_train_path = reg_dir / sample_split / "train.csv"
    if not reg_train_path.exists():
        log.warning(f"  Cannot verify scaffold consistency: {reg_train_path} not found")
        return

    try:
        reg = pd.read_csv(reg_train_path, usecols=[id_col, scaffold_col])
    except (ValueError, KeyError) as e:
        log.warning(f"  Cannot verify scaffold consistency: {e}")
        return

    reg[id_col] = reg[id_col].astype(str)
    reg[scaffold_col] = norm_scaffold(reg[scaffold_col].astype(str))

    cls_view = cls_df[[id_col, scaffold_col]].copy()
    cls_view[id_col] = cls_view[id_col].astype(str)
    cls_view[scaffold_col] = norm_scaffold(cls_view[scaffold_col].astype(str))

    shared = reg.merge(cls_view, on=id_col, suffixes=("_reg", "_cls"))
    if len(shared) == 0:
        log.warning("  Scaffold consistency check: no shared compounds between "
                    "regression train and classification — alignment will rely "
                    "on scaffold-string match only")
        return

    mismatches = (shared[f"{scaffold_col}_reg"] != shared[f"{scaffold_col}_cls"]).sum()
    if mismatches > 0:
        # Show a few examples for debugging
        bad = shared[shared[f"{scaffold_col}_reg"] != shared[f"{scaffold_col}_cls"]].head(3)
        examples = bad.to_dict("records")
        raise RuntimeError(
            f"\n\n  Scaffold ID mismatch: {mismatches}/{len(shared)} shared "
            f"compounds have different '{scaffold_col}' in regression vs "
            f"classification.\n"
            f"  Examples: {examples}\n\n"
            f"  This almost certainly means '{scaffold_col}' is a per-file "
            f"integer ID, not a canonical scaffold key. The IDs were assigned "
            f"independently in each CSV and don't match across files.\n"
            f"  Fix: use --scaffold_col stereo_stripped_scaffold (a SMILES "
            f"string), or any column that is content-derived rather than "
            f"position-derived."
        )

    log.info(f"  Scaffold ID consistency verified: {len(shared)} shared "
             f"compounds, all matching")

# ── FIX-2: vectorized routing ────────────────────────────────────────
def route_split_vectorized(
    cls_df: pd.DataFrame,
    split_name: str,
    scaffold_assignments: Dict[str, str],
    id_assignments: Dict[str, str],
    scaffold_col: str,
    id_col: str,
) -> pd.Series:
    """
    Return a Series of 'train'|'val'|'test'|'unassigned' aligned to cls_df.

    For disjoint splits (scaffold, confirmed): scaffold-only routing.
    For overlap-allowed splits: ID match first, scaffold fallback.

    No row-by-row .apply() and no in-place mutation. Result is a fresh
    Series, safe under pandas copy-on-write.
    """
    sc = norm_scaffold(cls_df[scaffold_col])
    by_scaffold = sc.map(scaffold_assignments)

    if split_name in DISJOINT_SPLITS:
        return by_scaffold.fillna("unassigned").astype(object)

    cid = cls_df[id_col].astype(str)
    by_id = cid.map(id_assignments)
    return by_id.fillna(by_scaffold).fillna("unassigned").astype(object)

# ── FIX-1: corrected largest-first greedy distribution ───────────────
def distribute_unassigned(
    df_unassigned: pd.DataFrame,
    n_train_target: int,
    n_val_target: int,
    n_test_target: int,
    scaffold_col: str,
    seed: int = 42,
    stratify_col: Optional[str] = None,
) -> pd.Series:
    """
    Assign novel-scaffold compounds to train/val/test buckets, keeping
    each scaffold intact (scaffold-disjoint distribution).

    Algorithm: largest-first greedy. At each step, the next scaffold is
    placed in whichever bucket has the largest remaining capacity
    (target - current). Smallest scaffolds end up last and fine-tune the
    balance. This respects targets to within roughly one scaffold size,
    and is invariant to scaffold-name ordering.

    v8.0 had a bug here: it computed `current_test_frac` BEFORE adding
    the current scaffold, so the first scaffold always landed in test,
    and subsequent fraction checks ran away from the target.
    """
    rng = np.random.default_rng(seed)
    result = pd.Series("train", index=df_unassigned.index, dtype=object)
    if len(df_unassigned) == 0:
        return result

    # Use vectorized groupby instead of per-scaffold .loc filtering
    sc_norm = norm_scaffold(df_unassigned[scaffold_col])
    scaf_sizes = sc_norm.value_counts().to_dict()
    unique_scafs = list(scaf_sizes.keys())

    # Stratification: sort scaffolds by activity-class mean so high-activity
    # and low-activity scaffolds get spread across buckets, not clumped.
    if stratify_col and stratify_col in df_unassigned.columns:
        strat_means = (df_unassigned.assign(_sc=sc_norm)
                                    .groupby("_sc")[stratify_col]
                                    .apply(lambda s: float(pd.to_numeric(s, errors="coerce").fillna(0).mean()))
                                    .to_dict())
        # Largest-first within stratification: sort by (size desc, activity)
        unique_scafs.sort(key=lambda s: (-scaf_sizes[s], strat_means.get(s, 0.5)))
    else:
        # Pure largest-first; ties broken by shuffled order
        rng.shuffle(unique_scafs)
        unique_scafs.sort(key=lambda s: -scaf_sizes[s])

    # Greedy fill: each scaffold goes to the most under-budget bucket.
    targets = {"train": n_train_target, "val": n_val_target, "test": n_test_target}
    counts = {"train": 0, "val": 0, "test": 0}

    assignments_by_scaffold: Dict[str, str] = {}
    for sc in unique_scafs:
        sz = scaf_sizes[sc]
        # Bucket with largest remaining capacity (may be negative if over)
        deficits = {b: targets[b] - counts[b] for b in counts}
        # Prefer test/val before train when deficits tie, so they fill first
        order = ["test", "val", "train"]
        best = max(order, key=lambda b: deficits[b])
        assignments_by_scaffold[sc] = best
        counts[best] += sz

    # Map back to rows
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

    # FIX-2: vectorized routing
    subset_col = route_split_vectorized(
        cls_df, split_name, scaffold_assignments, id_assignments,
        scaffold_col, id_col,
    )

    unassigned_mask = subset_col == "unassigned"
    n_unassigned = int(unassigned_mask.sum())
    routed_frac = (n - n_unassigned) / max(1, n)
    log.info(f"  {split_name}: {n - n_unassigned}/{n} routed ({100*routed_frac:.1f}%), "
             f"{n_unassigned} to distribute")

    floored = False
    if n_unassigned > 0:
        aligned_counts = Counter(subset_col[~unassigned_mask])
        tr_cnt = aligned_counts.get("train", 0)
        va_cnt = aligned_counts.get("val", 0)
        te_cnt = aligned_counts.get("test", 0)

        total_aligned = tr_cnt + va_cnt + te_cnt
        if total_aligned > 0:
            tr_frac = tr_cnt / total_aligned
            va_frac = va_cnt / total_aligned
            te_frac = te_cnt / total_aligned
        else:
            tr_frac, va_frac, te_frac = 0.75, 0.10, 0.15

        if va_frac < MIN_VAL_FRAC:
            deficit = MIN_VAL_FRAC - va_frac
            log.info(f"  {split_name}: val_frac={va_frac:.3f} below floor "
                     f"({MIN_VAL_FRAC}); taking {deficit:.3f} from train")
            va_frac = MIN_VAL_FRAC
            tr_frac = max(0.0, tr_frac - deficit)
            floored = True

        log.info(f"  {split_name}: redistribution fractions "
                 f"train={tr_frac:.3f} val={va_frac:.3f} test={te_frac:.3f}")

        n_test_target = int(round(n_unassigned * te_frac))
        n_val_target = int(round(n_unassigned * va_frac))
        n_train_target = n_unassigned - n_test_target - n_val_target

        unassigned_df = cls_df[unassigned_mask]
        # FIX-1: corrected greedy distribution
        distributed = distribute_unassigned(
            unassigned_df,
            n_train_target=n_train_target,
            n_val_target=n_val_target,
            n_test_target=n_test_target,
            scaffold_col=scaffold_col,
            seed=seed,
            stratify_col=stratify_col,
        )
        # Vectorized merge — no in-place mutation
        subset_col = subset_col.where(~unassigned_mask, distributed)

    out_dir.mkdir(parents=True, exist_ok=True)
    counts_out = Counter(subset_col)

    for sub in ("train", "val", "test"):
        df_sub = cls_df[subset_col == sub].copy()
        df_sub = df_sub.assign(split=sub)
        df_sub.to_csv(out_dir / f"{sub}.csv", index=False)
        if sub == "test":
            df_sub.to_csv(out_dir / "test_locked.csv", index=False)

    # Verify proportions look right (catches silent budget bugs)
    total = sum(counts_out[s] for s in ("train", "val", "test"))
    if total > 0:
        actual_test_pct = 100 * counts_out["test"] / total
        actual_val_pct = 100 * counts_out["val"] / total
        log.info(f"  {split_name}: train={counts_out['train']} "
                 f"val={counts_out['val']} ({actual_val_pct:.1f}%) "
                 f"test={counts_out['test']} ({actual_test_pct:.1f}%)")
    else:
        log.info(f"  {split_name}: train={counts_out['train']} val={counts_out['val']} test={counts_out['test']}")

    audit = {
        "split": split_name,
        "n_total": n,
        "n_aligned": n - n_unassigned,
        "n_distributed": n_unassigned,
        "routed_pct": round(100 * routed_frac, 1),
        "redistributed_pct": round(100 * (1 - routed_frac), 1),
        "val_floor_applied": floored,
        "counts": dict(counts_out),
        "pct_train": round(100 * counts_out["train"] / max(1, total), 1),
        "pct_val": round(100 * counts_out["val"] / max(1, total), 1),
        "pct_test": round(100 * counts_out["test"] / max(1, total), 1),
    }

    # Within-classification scaffold disjointness check (disjoint splits only)
    if split_name in DISJOINT_SPLITS:
        tr_scafs = set(norm_scaffold(cls_df[subset_col == "train"][scaffold_col]))
        te_scafs = set(norm_scaffold(cls_df[subset_col == "test"][scaffold_col]))
        leaks = tr_scafs & te_scafs
        if leaks:
            log.error(f"  {split_name}: SCAFFOLD LEAKAGE: {len(leaks)} shared")
            audit["leakage_scaffolds"] = sorted(list(leaks))[:20]
        else:
            log.info(f"  {split_name}: scaffold-disjoint verified")

    # FIX-4: cross-dataset InChIKey-14 leakage audit
    # For stacked models, classification_test must be disjoint from
    # regression_train at the compound level.
    if split_name in DISJOINT_SPLITS:
        reg_train_path = reg_dir / split_name / "train.csv"
        if reg_train_path.exists():
            try:
                reg_train_ids = set(
                    pd.read_csv(reg_train_path, usecols=[id_col])[id_col].astype(str)
                )
                cls_test_ids = set(cls_df.loc[subset_col == "test", id_col].astype(str))
                cls_val_ids = set(cls_df.loc[subset_col == "val", id_col].astype(str))
                leak_test = reg_train_ids & cls_test_ids
                leak_val = reg_train_ids & cls_val_ids
                audit["cross_dataset_leakage_ik14"] = {
                    "cls_test_in_reg_train": len(leak_test),
                    "cls_val_in_reg_train": len(leak_val),
                    "stacking_safe": len(leak_test) == 0 and len(leak_val) == 0,
                }
                if leak_test or leak_val:
                    log.warning(
                        f"  {split_name}: cross-dataset IK14 leakage — "
                        f"cls_test∩reg_train={len(leak_test)}, "
                        f"cls_val∩reg_train={len(leak_val)} "
                        f"(matters if stacking classifier→regressor)"
                    )
                else:
                    log.info(f"  {split_name}: stacking-safe (no cls_test/val ∩ reg_train at IK14 level)")
            except (ValueError, KeyError) as e:
                log.warning(f"  {split_name}: cross-dataset audit skipped: {e}")

    return audit

def run_classification_alignment(cls_df: pd.DataFrame, args, out_dir: Path,
                                 seed: int) -> dict:
    # FIX-5: regression splits live under <out_dir>/regression/. If the
    # user supplied --regression_splits_dir explicitly, honor that
    # verbatim (it's an external benchmark, not our own subdir).
    if args.regression_splits_dir:
        reg_dir = Path(args.regression_splits_dir)
    else:
        reg_dir = out_dir / "regression"
    if not reg_dir.exists():
        raise ValueError(f"Regression splits dir not found: {reg_dir}")

    # Classification splits go to <out_dir>/classification/.
    cls_out = out_dir / "classification"
    cls_out.mkdir(parents=True, exist_ok=True)

    log.info("\n" + "=" * 68)
    log.info("  CLASSIFICATION ALIGNMENT")
    log.info("=" * 68)
    log.info(f"  Reading regression splits from: {reg_dir}")
    log.info(f"  Writing classification splits to: {cls_out}")

    scaffold_col = args.scaffold_col
    id_col = args.id_col

    if scaffold_col not in cls_df.columns:
        raise ValueError(f"No {scaffold_col} in classification file")
    if id_col not in cls_df.columns:
        raise ValueError(f"No {id_col} in classification file")

    # FIX-3: verify scaffold IDs mean the same thing across files
    log.info("\nVerifying scaffold-ID consistency across regression/classification:")
    assert_scaffold_id_consistency(reg_dir, cls_df, id_col, scaffold_col)

    log.info("\nLoading regression split assignments:")
    scaffold_maps = load_regression_split_assignments(
        reg_dir, scaffold_col=scaffold_col, id_col=id_col
    )
    id_maps = load_regression_id_assignments(reg_dir, id_col=id_col)

    # Smoke test: ensure scaffold IDs actually overlap; if zero overlap,
    # either scaffold_col is wrong or the files come from different
    # cleaning runs. Fail rather than silently produce garbage.
    sample_cls = set(norm_scaffold(cls_df[scaffold_col].dropna()))
    for split_name in SPLIT_NAMES:
        if split_name not in scaffold_maps:
            continue
        sample_reg = list(scaffold_maps[split_name].keys())[:20]
        matches = sum(1 for s in sample_reg if s in sample_cls)
        if matches == 0 and len(scaffold_maps[split_name]) > 0:
            raise ValueError(
                f"ZERO scaffold matches for {split_name} split. "
                f"Regression scaffolds (sample): {sample_reg[:3]}. "
                f"Classification scaffolds (sample): {list(sample_cls)[:3]}. "
                f"Are you using the same scaffold column? "
                f"You passed: --scaffold_col {scaffold_col}"
            )

    log.info("\nAligning classification to regression splits:")
    all_audits = {}
    for split_name in SPLIT_NAMES:
        if split_name not in scaffold_maps:
            log.info(f"  {split_name}: skipped")
            continue
        log.info(f"\n--- {split_name} ---")
        split_out = cls_out / split_name
        audit = align_one_split(
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
        "generated": pd.Timestamp.now().isoformat(),
        "pipeline_version": PIPELINE_VERSION,
        "classification_file": str(args.classification_file),
        "regression_splits_dir": str(reg_dir),
        "n_classification_rows": len(cls_df),
        "seed": seed,
        "splits": all_audits,
    }
    with open(out_dir / "alignment_audit.json", "w") as f:
        json.dump(master, f, indent=2, default=str)
    log.info(f"\nAudit: {out_dir / 'alignment_audit.json'}")
    return all_audits

# ══════════════════════════════════════════════════════════════════════
# MANIFEST & FREEZE HELPERS (unchanged from v8.0)
# ══════════════════════════════════════════════════════════════════════
def write_output_manifest(out: Path, cfg_hash: str) -> Path:
    manifest = {
        "generated": pd.Timestamp.now().isoformat(),
        "pipeline_version": PIPELINE_VERSION,
        "config_hash": cfg_hash,
        "root": str(out.resolve()),
        "files": [],
    }
    for csv_path in sorted(out.rglob("*.csv")):
        try:
            n_rows = sum(1 for _ in open(csv_path)) - 1
        except Exception:
            n_rows = None
        manifest["files"].append({
            "path": str(csv_path.relative_to(out)),
            "rows": n_rows,
            "bytes": csv_path.stat().st_size,
            "md5": md5_file(csv_path),
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
        existing = hash_path.read_text().splitlines()[0].strip() if hash_path.exists() else "?"
        raise RuntimeError(
            f"\n  Frozen benchmark already exists at {out}\n"
            f"  Existing hash: {existing}\n"
            f"  Refusing to overwrite. Pass --force to override.\n"
            f"  If you intend to update the benchmark, also bump PIPELINE_VERSION."
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
    log.info(f"  {'split':<12} {'subset':<7} {'n':>6} {'scaff':>6} "
             f"{'sing%':>6} {'top2':>6}")
    log.info("  " + "-" * 58)
    for split, sub_d in all_health.items():
        for sub_name, s in sub_d.items():
            if "error" in s:
                continue
            log.info(
                f"  {split:<12} {sub_name:<7} {s['n_compounds']:>6} "
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
                log.info(f"  {split_name:<12} ERROR")
            else:
                extra = ""
                if d.get("scaffold_diversity_ratio") is not None:
                    extra = f"  div_ratio={d['scaffold_diversity_ratio']:.2f}"
                log.info(f"  {split_name:<12} score={d['health_score']:>3} "
                         f"{d['health_label']:<11}{extra}")

    master = {
        "generated": pd.Timestamp.now().isoformat(),
        "pipeline_version": PIPELINE_VERSION,
        "config_hash": cfg_hash,
        "config": cfg,
        "seed": cfg.get("seed"),
        "data_hash_t1": hash_df(t1),
        "splits": all_stats,
        "leakage_audits": all_audits,
        "scaffold_health": all_health,
        "diagnostics": all_diagnostics,
    }
    with open(out / "master_audit.json", "w") as f:
        json.dump(master, f, indent=2, default=str)

    diag_out = {}
    for split_name in SPLIT_NAMES:
        if split_name in all_diagnostics:
            d = all_diagnostics[split_name]
            diag_out[split_name] = {
                k: v for k, v in d.items()
                if k not in ("error", "skipped")
            }
    with open(out / "split_diagnostics.json", "w") as f:
        json.dump(diag_out, f, indent=2, default=str)

    rows = []
    for method, s in all_stats.items():
        if "error" in s or "skipped" in s:
            continue
        tr = s.get("train", {}); te = s.get("test", {}); va = s.get("val", {})
        rows.append({
            "method": method,
            "n_train": tr.get("n"), "n_val": va.get("n"), "n_test": te.get("n"),
            "n_scaffolds_train": tr.get("n_scaffolds"),
            "n_scaffolds_test": te.get("n_scaffolds"),
            "test_pic50_mean": te.get("pIC50_mean"),
            "test_pic50_std": te.get("pIC50_std"),
            "test_pct_active": te.get("pct_active"),
            "health_score": all_diagnostics.get(method, {}).get("health_score"),
            "health_label": all_diagnostics.get(method, {}).get("health_label"),
        })
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
            log.info(f"    ✓ {split_name:<12} score={score} {label}")
        else:
            log.info(f"    ! {split_name:<12} unavailable")

    log.info("\n  Auxiliary stress-test (report separately, not co-equally):")
    if "similarity" in diagnostics and "error" not in diagnostics["similarity"]:
        score = diagnostics["similarity"].get("health_score", 0)
        label = diagnostics["similarity"].get("health_label", "?")
        if diagnostics["similarity"].get("degenerate"):
            log.info(f"    ! similarity   score={score} {label} (DEGENERATE — "
                     f"dominant chemotype; report as stress-test only)")
        else:
            log.info(f"    ~ similarity   score={score} {label}")
    else:
        log.info(f"    ! similarity   unavailable")

# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=f"PAD4 Splitting {PIPELINE_VERSION} — Patched Frozen Benchmark Release",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", choices=["regression", "classification", "both"],
                   default="both", help="Execution mode")
    p.add_argument("--output_dir", default=_default_output_dir(),
                   help="Output directory for all splits")

    p.add_argument("--t1_file",
                   default=str(Path(_default_processed_dir()) / "pad_t1_non_covalent.csv"),
                   help="Main regression T1 CSV")
    p.add_argument("--confirmed_file",
                   default=str(Path(_default_processed_dir()) / "pad_t1_confirmed.csv"),
                   help="Confirmed subset CSV")

    p.add_argument("--classification_file",
                   default=str(Path(_default_processed_dir()) / "pad_classification_v17.csv"),
                   help="Classification CSV to align")
    p.add_argument("--regression_splits_dir", default="",
                   help="Existing regression splits dir. If empty (default), "
                        "uses <output_dir>/regression/ automatically. Override "
                        "only when aligning classification against externally-"
                        "provided regression splits.")

    # FIX-3: default scaffold_col to content-derived key, not file-local integer
    p.add_argument("--smiles_col", default="canonical_smiles")
    p.add_argument("--scaffold_col", default="stereo_stripped_scaffold",
                   help="Scaffold column for cross-file alignment. Must be "
                        "content-derived (e.g. stereo-canonical SMILES). "
                        "DO NOT use 'scaffold_id' if it's a per-file integer.")
    p.add_argument("--id_col", default="inchikey_14")
    p.add_argument("--activity_col", default="pIC50")
    p.add_argument("--class_stratify_col", default="activity_class",
                   help="Column for class-stratified distribution of unassigned compounds")

    p.add_argument("--test_frac", type=float, default=0.15)
    p.add_argument("--val_frac", type=float, default=0.10)
    p.add_argument("--scaffold_cap_frac", type=float, default=0.08)
    p.add_argument("--similarity_cap_frac", type=float, default=0.10)
    p.add_argument("--lead_opt_min_size", type=int, default=4)

    p.add_argument("--butina_cutoff", type=float, default=0.40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip_similarity", action="store_true")
    p.add_argument("--skip_regression", action="store_true")
    p.add_argument("--skip_classification", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="Overwrite an existing frozen benchmark (use with care)")
    return p

def main():
    p = build_parser()
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = {k: v for k, v in vars(args).items() if k not in ("output_dir", "force")}
    cfg_hash = hash_config(cfg)

    log.info("=" * 68)
    log.info(f"  PAD4 SPLITTING {PIPELINE_VERSION}  (patched frozen benchmark)")
    log.info("=" * 68)
    log.info(f"  config hash: {cfg_hash}")
    log.info(f"  mode: {args.mode}")
    log.info(f"  scaffold_col: {args.scaffold_col}")
    log.info(f"  output: {out.resolve()}")

    try:
        check_freeze_collision(out, args.force)
    except RuntimeError as e:
        log.error(str(e))
        sys.exit(2)

    all_diagnostics = {}

    if args.mode in ("regression", "both") and not args.skip_regression:
        log.info("\n>>> REGRESSION MODE <<<\n")
        t1 = pd.read_csv(args.t1_file)
        confirmed = pd.read_csv(args.confirmed_file)
        log.info(f"  T1: {len(t1):,} compounds, {t1[args.scaffold_col].nunique():,} scaffolds")
        log.info(f"  Confirmed: {len(confirmed):,} compounds")

        log.info("\n  Deduplicating InChIKey-14 groups:")
        t1 = dedupe_by_inchikey14(t1, "T1", id_col=args.id_col)
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

        run_classification_alignment(cls_df, args, out, args.seed)
        log.info("  Classification alignment complete.")

    if all_diagnostics:
        print_benchmark_recommendations(all_diagnostics)

    log.info("\n" + "=" * 68)
    log.info("  WRITING MANIFEST + BENCHMARK HASH")
    log.info("=" * 68)
    manifest_path = write_output_manifest(out, cfg_hash)
    hash_path = write_benchmark_hash(out, cfg_hash, cfg)
    log.info(f"  manifest: {manifest_path}")
    log.info(f"  hash:     {hash_path}")

    log.info("\n" + "=" * 68)
    log.info(f"  DONE. output: {out.resolve()}")
    log.info(f"  To freeze: git tag pad4bench-{PIPELINE_VERSION}-splits")
    log.info("=" * 68)

if __name__ == "__main__":
    main()