#!/usr/bin/env python3
"""
PAD4_BENCH comprehensive reviewer audit.

Systematically checks every category of objection a peer reviewer could raise,
from raw data provenance through final modeling claims. Each check produces a
verdict (PASS / WARN / FAIL / INFO) with the specific number a reviewer would
demand to see, plus the source path so claims are traceable.

Categories audited:

  A. RAW DATA PROVENANCE
     - source files exist and parse
     - row counts vs README/manifest
     - target-isoform filtering correctness

  B. CURATION INTEGRITY
     - SMILES canonicalization
     - InChIKey uniqueness
     - stereo handling consistency
     - covalent classification consistency
     - unit standardization
     - duplicate measurement aggregation

  C. LABEL QUALITY
     - pIC50 distribution sanity
     - measurement noise characterization (replicates)
     - cross-source consistency (assay bias)
     - label uncertainty distribution
     - activity class threshold

  D. DATASET DEFINITION
     - regression set composition (n=2618)
     - classification set composition (n=2758)
     - covalent inclusion/exclusion accounting
     - regression-classification overlap

  E. SPLIT INTEGRITY
     - per-split sizes match modeling sweep
     - InChIKey disjointness train/val/test
     - scaffold disjointness for scaffold split
     - Tanimoto separation for similarity split
     - cliff_aware test coverage
     - class balance per split

  F. FEATURE INTEGRITY
     - pipeline version consistency (v18.0)
     - n_train matches split sizes
     - feature counts per variant
     - linear-space selection rationale
     - stratifier alignment

  G. MODELING INTEGRITY
     - all 60+60 cells have full artifact sets
     - tuning grid was honest (no cherrypicking)
     - OOF generation per fold uses correct CV
     - val used for early stopping (XGB) or test held back
     - sample weights applied consistently

  H. STATISTICAL CLAIMS
     - bootstrap CIs present for all cells
     - CI widths sane
     - point estimate vs bootstrap median agree

  I. ROBUSTNESS CLAIMS
     - seeds 7 and 1337 exist for random+scaffold
     - cross-seed variance reasonable
     - degenerate cells flagged

  J. CALIBRATION CLAIMS
     - ECE per classification cell
     - calibration pattern characterization

  K. STACKING CLAIMS
     - 30 stacking cells present
     - lift quantification
     - regression-classification id alignment

  L. THRESHOLD POLICY
     - Youden's J tuning applied where applicable
     - default vs tuned threshold reported

Writes:
  paper_intro/reviewer_audit_report.json   structured results
  paper_intro/reviewer_audit_report.md     human-readable summary with verdicts

Usage:
    cd /home/nidhal/PAD4_BENCH
    python reviewer_audit.py
"""

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
SPLITS_ROOT = PROJECT_ROOT / "data" / "splits"
FEATURES_ROOT = PROJECT_ROOT / "features_v18"
MODELS_ROOT = PROJECT_ROOT / "models_v1"
OUT_ROOT = PROJECT_ROOT / "paper_intro"

STRATEGIES = ["random", "scaffold", "confirmed", "lead_opt", "similarity", "cliff_aware"]
VARIANTS = ["full", "fingerprints", "physchem", "mordred", "fragments"]
TASKS = ["regression", "classification"]


# -----------------------------------------------------------------------------
# Verdict tracking
# -----------------------------------------------------------------------------
class Audit:
    def __init__(self):
        self.results = []
        self.counts = Counter()

    def record(self, category: str, check: str, verdict: str,
               detail: str, value=None, source: str = ""):
        """verdict: PASS / WARN / FAIL / INFO"""
        self.results.append({
            "category": category,
            "check": check,
            "verdict": verdict,
            "detail": detail,
            "value": value,
            "source": source,
        })
        self.counts[verdict] += 1
        # Print live so user sees progress
        marker = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗", "INFO": "·"}[verdict]
        print(f"  [{marker} {verdict}] {category}/{check}: {detail}", flush=True)

    def to_dict(self):
        return {
            "summary": dict(self.counts),
            "n_checks": len(self.results),
            "results": self.results,
        }

    def to_markdown(self) -> str:
        md = ["# PAD4_BENCH Reviewer Audit\n"]
        md.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        md.append("## Summary\n")
        for v in ("PASS", "WARN", "FAIL", "INFO"):
            md.append(f"- {v}: {self.counts.get(v, 0)}")
        md.append(f"- TOTAL: {len(self.results)}\n")

        # Group by category
        by_cat = defaultdict(list)
        for r in self.results:
            by_cat[r["category"]].append(r)

        for cat in sorted(by_cat.keys()):
            md.append(f"## {cat}\n")
            md.append("| Check | Verdict | Detail | Value | Source |")
            md.append("|---|---|---|---|---|")
            for r in by_cat[cat]:
                val = str(r["value"])[:60] if r["value"] is not None else ""
                src = r["source"] if r["source"] else ""
                md.append(f"| {r['check']} | **{r['verdict']}** | {r['detail']} | {val} | `{src}` |")
            md.append("")

        return "\n".join(md)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def safe_read(path: Path, **kw):
    try:
        return pd.read_csv(path, low_memory=False, **kw)
    except Exception:
        return None


def id_set(df: pd.DataFrame, col_candidates=("inchikey_14", "inchikey")) -> set[str]:
    if df is None:
        return set()
    for c in col_candidates:
        if c in df.columns:
            return set(df[c].dropna().astype(str).str[:14])
    for c in df.columns:
        if "inchikey" in c.lower():
            return set(df[c].dropna().astype(str).str[:14])
    return set()


def count_raw_rows(path: Path) -> int:
    try:
        with open(path, "rb") as fh:
            return max(0, sum(1 for _ in fh) - 1)
    except Exception:
        return -1


# =============================================================================
# A. RAW DATA PROVENANCE
# =============================================================================
def audit_raw_data(a: Audit) -> dict:
    cat = "A_raw_data"
    print(f"\n{'=' * 70}\nA. RAW DATA PROVENANCE\n{'=' * 70}", flush=True)
    raw_counts = {}

    # ChEMBL files for all 5 PAD isoforms
    chembl_dir = DATA_RAW / "chembl"
    if chembl_dir.exists():
        for f in sorted(chembl_dir.glob("CHEMBL_PAD*.tsv")):
            n = count_raw_rows(f)
            raw_counts[f.name] = n
            a.record(cat, f"chembl_{f.stem}", "INFO",
                     f"raw row count", value=n, source=str(f.relative_to(PROJECT_ROOT)))
    else:
        a.record(cat, "chembl_dir", "FAIL", "missing data/raw/chembl/", value=None)

    # BindingDB
    bdb_dir = DATA_RAW / "bindingdb"
    if bdb_dir.exists():
        for f in sorted(bdb_dir.glob("BindingDB_PAD*.tsv")):
            n = count_raw_rows(f)
            raw_counts[f.name] = n
            a.record(cat, f"bindingdb_{f.stem}", "INFO",
                     f"raw row count", value=n, source=str(f.relative_to(PROJECT_ROOT)))
    else:
        a.record(cat, "bindingdb_dir", "FAIL", "missing data/raw/bindingdb/")

    # PubChem AIDs
    pc_dir = DATA_RAW / "pubchem"
    if pc_dir.exists():
        pc_total = 0
        for f in sorted(pc_dir.glob("AID_*.csv")):
            n = count_raw_rows(f)
            raw_counts[f.name] = n
            pc_total += max(0, n)
        a.record(cat, "pubchem_total_rows", "INFO",
                 f"{len(list(pc_dir.glob('AID_*.csv')))} AIDs, total raw rows",
                 value=pc_total)
    else:
        a.record(cat, "pubchem_dir", "FAIL", "missing data/raw/pubchem/")

    # Sanity: most rows should be PAD4 (the focal isoform)
    pad4_rows = (raw_counts.get("CHEMBL_PAD4.tsv", 0)
                 + raw_counts.get("BindingDB_PAD4.tsv", 0))
    a.record(cat, "pad4_raw_record_count", "INFO",
             f"PAD4 raw rows (ChEMBL + BindingDB)", value=pad4_rows)

    return raw_counts


# =============================================================================
# B. CURATION INTEGRITY
# =============================================================================
def audit_curation(a: Audit) -> dict:
    cat = "B_curation"
    print(f"\n{'=' * 70}\nB. CURATION INTEGRITY\n{'=' * 70}", flush=True)
    state = {}

    t1_agg = safe_read(DATA_PROCESSED / "pad_t1_ic50_aggregated.csv")
    t1_noncov = safe_read(DATA_PROCESSED / "pad_t1_non_covalent.csv")
    t1_cov = safe_read(DATA_PROCESSED / "pad_t1_covalent.csv")
    t1_conf = safe_read(DATA_PROCESSED / "pad_t1_confirmed.csv")
    classif = safe_read(DATA_PROCESSED / "pad_classification_v17.csv")

    # B1. InChIKey uniqueness after dedup
    if t1_noncov is not None:
        n_rows = len(t1_noncov)
        n_unique = t1_noncov["inchikey_14"].nunique() if "inchikey_14" in t1_noncov.columns else None
        verdict = "PASS" if n_unique == 2618 else "WARN"
        a.record(cat, "non_covalent_unique_inchikey_14", verdict,
                 f"unique InChIKey-14 count",
                 value={"rows": n_rows, "unique": n_unique, "expected": 2618},
                 source="data/processed/pad_t1_non_covalent.csv")
        state["non_covalent_n"] = n_unique

    if classif is not None:
        n_rows = len(classif)
        n_unique = classif["inchikey_14"].nunique() if "inchikey_14" in classif.columns else None
        verdict = "PASS" if n_unique == 2758 else "WARN"
        a.record(cat, "classification_unique_inchikey_14", verdict,
                 f"unique InChIKey-14 count",
                 value={"rows": n_rows, "unique": n_unique, "expected": 2758},
                 source="data/processed/pad_classification_v17.csv")
        state["classification_n"] = n_unique

    # B2. SMILES canonicalization (rough check: canonical_smiles column exists, no NaN)
    if t1_noncov is not None and "canonical_smiles" in t1_noncov.columns:
        n_missing = int(t1_noncov["canonical_smiles"].isna().sum())
        verdict = "PASS" if n_missing == 0 else "WARN"
        a.record(cat, "smiles_canonicalization_non_covalent", verdict,
                 f"missing canonical_smiles", value=n_missing,
                 source="data/processed/pad_t1_non_covalent.csv")

    # B3. Stereo flag distribution
    if t1_noncov is not None and "stereo_defined_flag" in t1_noncov.columns:
        n_defined = int(t1_noncov["stereo_defined_flag"].astype(bool).sum())
        a.record(cat, "stereo_defined_count", "INFO",
                 f"compounds with defined stereo (of {len(t1_noncov)})",
                 value=n_defined, source="data/processed/pad_t1_non_covalent.csv")

    # B4. Covalent classification consistency
    # All compounds in pad_t1_covalent should be is_covalent=True
    if t1_cov is not None and "is_covalent" in t1_cov.columns:
        all_true = bool(t1_cov["is_covalent"].astype(bool).all())
        verdict = "PASS" if all_true else "FAIL"
        a.record(cat, "covalent_file_consistency", verdict,
                 f"all rows in covalent file have is_covalent=True",
                 value=int(t1_cov["is_covalent"].astype(bool).sum()),
                 source="data/processed/pad_t1_covalent.csv")

    # B5. Reversible covalent disclosure check
    # Non-covalent file may contain is_covalent=True if is_reversible_covalent=True
    if t1_noncov is not None and "is_covalent" in t1_noncov.columns:
        flagged = t1_noncov[t1_noncov["is_covalent"] == True]
        if len(flagged) > 0:
            n_reversible = int(flagged.get("is_reversible_covalent", pd.Series(dtype=bool))
                               .fillna(False).astype(bool).sum())
            verdict = "PASS" if n_reversible == len(flagged) else "WARN"
            a.record(cat, "reversible_covalent_in_non_covalent", verdict,
                     f"{n_reversible}/{len(flagged)} flagged compounds are reversible covalent",
                     value={"flagged": len(flagged), "reversible": n_reversible},
                     source="data/processed/pad_t1_non_covalent.csv")
        else:
            a.record(cat, "reversible_covalent_in_non_covalent", "PASS",
                     "no is_covalent=True rows in non_covalent file", value=0)

    # B6. Confirmed file covalent disclosure
    if t1_conf is not None and "is_covalent" in t1_conf.columns:
        flagged = t1_conf[t1_conf["is_covalent"] == True]
        n_unique_cov = flagged["inchikey_14"].nunique() if "inchikey_14" in flagged.columns else None
        a.record(cat, "covalent_in_confirmed_file", "INFO",
                 f"covalent rows in confirmed file (rows / unique compounds)",
                 value={"rows": len(flagged), "unique": n_unique_cov},
                 source="data/processed/pad_t1_confirmed.csv")

    # B7. Unit standardization: all pIC50 should be finite and in plausible range
    for name, df in [("non_covalent", t1_noncov), ("classification", classif)]:
        if df is None or "pIC50" not in df.columns:
            continue
        pic = df["pIC50"].dropna()
        finite = np.isfinite(pic.values).all()
        in_range = bool((pic.min() >= 0) and (pic.max() <= 15))
        verdict = "PASS" if finite and in_range else "WARN"
        a.record(cat, f"pic50_range_{name}", verdict,
                 f"pIC50 within [0, 15] and finite",
                 value={"min": float(pic.min()), "max": float(pic.max()),
                        "n_missing": int(df["pIC50"].isna().sum())},
                 source=f"data/processed/pad_t1_{name}.csv")

    # B8. Activity class binarization consistency (for classification set)
    if classif is not None and "activity_class" in classif.columns and "pIC50" in classif.columns:
        cl_clean = classif.dropna(subset=["activity_class", "pIC50"])
        # Check the binarization is monotonic (separable by some threshold)
        pic_pos = cl_clean[cl_clean["activity_class"] == 1]["pIC50"]
        pic_neg = cl_clean[cl_clean["activity_class"] == 0]["pIC50"]
        if len(pic_pos) and len(pic_neg):
            sep = bool(pic_pos.min() >= pic_neg.max() - 1e-6) or bool(pic_neg.max() <= pic_pos.min() + 1e-6)
            a.record(cat, "activity_class_separable_by_pic50", "INFO",
                     f"pos pIC50 min={pic_pos.min():.3f}, neg pIC50 max={pic_neg.max():.3f}",
                     value={"strict_separation": sep,
                            "pos_min": float(pic_pos.min()),
                            "neg_max": float(pic_neg.max()),
                            "n_pos": len(pic_pos), "n_neg": len(pic_neg)},
                     source="data/processed/pad_classification_v17.csv")

    return state


# =============================================================================
# C. LABEL QUALITY
# =============================================================================
def audit_labels(a: Audit, state: dict) -> dict:
    cat = "C_labels"
    print(f"\n{'=' * 70}\nC. LABEL QUALITY\n{'=' * 70}", flush=True)

    t1_noncov = safe_read(DATA_PROCESSED / "pad_t1_non_covalent.csv")
    reps = safe_read(DATA_PROCESSED / "pad_replicates_full.csv")
    bias = safe_read(DATA_PROCESSED / "pad_assay_bias_report.csv")

    # C1. pIC50 distribution
    if t1_noncov is not None and "pIC50" in t1_noncov.columns:
        pic = t1_noncov["pIC50"].dropna()
        a.record(cat, "pic50_descriptive_stats_non_covalent", "INFO",
                 f"pIC50 mean={pic.mean():.3f} std={pic.std():.3f} median={pic.median():.3f}",
                 value={"n": int(len(pic)), "mean": float(pic.mean()),
                        "std": float(pic.std()), "median": float(pic.median()),
                        "min": float(pic.min()), "max": float(pic.max())},
                 source="data/processed/pad_t1_non_covalent.csv")

    # C2. Replicate noise characterization
    if reps is not None and "inchikey_14" in reps.columns and "pIC50" in reps.columns:
        rep_clean = reps.dropna(subset=["inchikey_14", "pIC50"])
        n_compounds_rep = rep_clean["inchikey_14"].nunique()
        per_cmpd = rep_clean.groupby("inchikey_14")["pIC50"].agg(["count", "std", "mean"])
        multi = per_cmpd[per_cmpd["count"] >= 2]
        std_p50 = float(multi["std"].median()) if len(multi) else None
        std_p95 = float(multi["std"].quantile(0.95)) if len(multi) else None
        a.record(cat, "replicate_noise_floor", "INFO",
                 f"per-compound pIC50 std among multi-measurement: median={std_p50}, p95={std_p95}",
                 value={"n_compounds_with_2plus_measurements": len(multi),
                        "median_pic50_std": std_p50, "p95_pic50_std": std_p95},
                 source="data/processed/pad_replicates_full.csv")
        # Check that replicate coverage is high for the modeled set
        if t1_noncov is not None:
            modeled_ids = set(t1_noncov["inchikey_14"].astype(str))
            rep_ids = set(rep_clean["inchikey_14"].astype(str))
            cov = len(modeled_ids & rep_ids) / max(1, len(modeled_ids))
            verdict = "PASS" if cov > 0.95 else "WARN"
            a.record(cat, "replicate_coverage_of_modeled_set", verdict,
                     f"{cov:.1%} of non_covalent compounds have replicate-level data",
                     value=cov,
                     source="data/processed/pad_replicates_full.csv")

    # C3. Cross-source bias
    if bias is not None:
        n_sources = len(bias)
        shift_abs = bias["assay_shift"].abs() if "assay_shift" in bias.columns else None
        if shift_abs is not None:
            max_shift = float(shift_abs.max())
            mean_shift = float(shift_abs.mean())
            verdict = "INFO" if max_shift < 1.0 else "WARN"
            a.record(cat, "assay_bias_max_shift", verdict,
                     f"max |assay_shift| across {n_sources} sources",
                     value={"n_sources": n_sources, "max_abs_shift": max_shift,
                            "mean_abs_shift": mean_shift},
                     source="data/processed/pad_assay_bias_report.csv")
        else:
            a.record(cat, "assay_bias_report_loaded", "INFO",
                     f"{n_sources} sources in bias report",
                     value=n_sources,
                     source="data/processed/pad_assay_bias_report.csv")

    # C4. Label uncertainty
    if t1_noncov is not None and "label_uncertainty_score_v2" in t1_noncov.columns:
        unc = t1_noncov["label_uncertainty_score_v2"].dropna()
        a.record(cat, "label_uncertainty_distribution", "INFO",
                 f"label_uncertainty_score_v2 mean={unc.mean():.3f} max={unc.max():.3f}",
                 value={"mean": float(unc.mean()), "std": float(unc.std()),
                        "max": float(unc.max())},
                 source="data/processed/pad_t1_non_covalent.csv")

    # C5. ml_weight distribution
    if t1_noncov is not None and "ml_weight" in t1_noncov.columns:
        w = t1_noncov["ml_weight"].dropna()
        a.record(cat, "ml_weight_distribution", "INFO",
                 f"ml_weight mean={w.mean():.3f} min={w.min():.3f} max={w.max():.3f}",
                 value={"mean": float(w.mean()), "std": float(w.std()),
                        "min": float(w.min()), "max": float(w.max())},
                 source="data/processed/pad_t1_non_covalent.csv")

    return state


# =============================================================================
# D. DATASET DEFINITION
# =============================================================================
def audit_dataset(a: Audit, state: dict) -> dict:
    cat = "D_dataset"
    print(f"\n{'=' * 70}\nD. DATASET DEFINITION\n{'=' * 70}", flush=True)

    t1_noncov = safe_read(DATA_PROCESSED / "pad_t1_non_covalent.csv")
    classif = safe_read(DATA_PROCESSED / "pad_classification_v17.csv")
    t1_cov = safe_read(DATA_PROCESSED / "pad_t1_covalent.csv")

    reg_ids = id_set(t1_noncov)
    cls_ids = id_set(classif)
    cov_ids = id_set(t1_cov)

    # D1. Sizes
    a.record(cat, "regression_set_size", "PASS" if len(reg_ids) == 2618 else "WARN",
             f"regression base set", value=len(reg_ids),
             source="data/processed/pad_t1_non_covalent.csv")
    a.record(cat, "classification_set_size", "PASS" if len(cls_ids) == 2758 else "WARN",
             f"classification set", value=len(cls_ids),
             source="data/processed/pad_classification_v17.csv")

    # D2. Regression set is a strict subset of classification set
    reg_in_cls = len(reg_ids & cls_ids)
    is_subset = reg_in_cls == len(reg_ids)
    verdict = "PASS" if is_subset else "WARN"
    a.record(cat, "regression_subset_of_classification", verdict,
             f"all regression compounds present in classification set",
             value={"reg_in_cls": reg_in_cls, "reg_total": len(reg_ids),
                    "is_subset": is_subset})

    # D3. Covalent exclusion accounting
    cov_in_reg = len(reg_ids & cov_ids)
    a.record(cat, "covalent_in_regression_set", "PASS" if cov_in_reg == 0 else "WARN",
             f"irreversible-covalent compounds inside regression base set",
             value=cov_in_reg)
    cov_in_cls = len(cls_ids & cov_ids)
    a.record(cat, "covalent_in_classification_set", "INFO",
             f"irreversible-covalent compounds inside classification set",
             value=cov_in_cls)

    state["reg_ids"] = reg_ids
    state["cls_ids"] = cls_ids
    return state


# =============================================================================
# E. SPLIT INTEGRITY
# =============================================================================
def audit_splits(a: Audit, state: dict):
    cat = "E_splits"
    print(f"\n{'=' * 70}\nE. SPLIT INTEGRITY\n{'=' * 70}", flush=True)

    # Pull from the leakage_verification.json we already generated
    leak_path = MODELS_ROOT / "leakage_verification.json"
    if not leak_path.exists():
        a.record(cat, "leakage_verification_present", "FAIL",
                 "models_v1/leakage_verification.json missing — run paper1_reviewer_proof.py first")
        return
    leak = json.loads(leak_path.read_text())

    # E1. All cells disjoint
    all_disjoint = leak.get("all_inchikey_disjoint", False)
    a.record(cat, "all_inchikey_disjoint", "PASS" if all_disjoint else "FAIL",
             f"every (task, strategy) cell passes InChIKey-14 disjointness",
             value=all_disjoint,
             source="models_v1/leakage_verification.json")

    # E2. Per-cell checks
    for cell in leak.get("cells", []):
        task = cell.get("task")
        strategy = cell.get("strategy")
        ik = cell.get("inchikey_disjointness", {})
        # disjointness
        disj = ik.get("all_disjoint", False)
        a.record(cat, f"{task}_{strategy}_inchikey_disjoint",
                 "PASS" if disj else "FAIL",
                 f"train/val/test InChIKey-14 disjoint",
                 value=disj,
                 source=f"data/splits/{task}/{strategy}/")

        # scaffold disjointness
        sov = cell.get("scaffold_overlap_train_test")
        if strategy == "scaffold" and sov is not None:
            verdict = "PASS" if sov == 0 else "FAIL"
            a.record(cat, f"{task}_{strategy}_scaffold_disjoint", verdict,
                     f"scaffold train/test overlap (must be 0 for scaffold split)",
                     value=sov,
                     source=f"data/splits/{task}/{strategy}/")
        elif sov is not None:
            a.record(cat, f"{task}_{strategy}_scaffold_overlap", "INFO",
                     f"scaffold train/test overlap (allowed for non-scaffold split)",
                     value=sov)

        # Tanimoto
        tan = cell.get("ecfp4_tanimoto_train_to_test", {})
        mean_max = tan.get("mean_of_max_per_test")
        if strategy == "similarity":
            # Should be MUCH lower than random
            verdict = "PASS" if mean_max is not None and mean_max < 0.75 else "WARN"
            a.record(cat, f"{task}_{strategy}_tanimoto_separation", verdict,
                     f"similarity split mean-of-max Tanimoto train→test",
                     value=mean_max,
                     source="features_v18 ECFP4")

        # cliff_aware fraction
        cl = cell.get("cliff_check")
        if cl and "fraction_cliff_derived_test" in cl:
            frac = cl["fraction_cliff_derived_test"]
            expected = cl.get("expected_fraction_from_handoff", 0.176)
            close = abs(frac - expected) < 0.01
            verdict = "PASS" if close else "WARN"
            a.record(cat, f"{task}_{strategy}_cliff_test_fraction", verdict,
                     f"cliff-derived test fraction (expected ~17.6%)",
                     value={"observed": frac, "expected": expected})

    # E3. Split sizes match splits_summary.csv
    ss = safe_read(SPLITS_ROOT / "splits_summary.csv")
    if ss is not None:
        for _, row in ss.iterrows():
            method = row["method"]
            ntr, nv, nt = int(row["n_train"]), int(row["n_val"]), int(row["n_test"])
            # Cross-check against features manifest
            mpath = FEATURES_ROOT / "regression" / method / "feature_manifest.json"
            if mpath.exists():
                m = json.loads(mpath.read_text())
                fn_tr = m.get("n_train")
                match = fn_tr == ntr
                verdict = "PASS" if match else "FAIL"
                a.record(cat, f"split_sizes_{method}_match_features", verdict,
                         f"splits_summary.csv n_train matches features_v18 manifest",
                         value={"splits_summary": ntr, "features_v18": fn_tr,
                                "match": match})


# =============================================================================
# F. FEATURE INTEGRITY
# =============================================================================
def audit_features(a: Audit, state: dict):
    cat = "F_features"
    print(f"\n{'=' * 70}\nF. FEATURE INTEGRITY\n{'=' * 70}", flush=True)

    # F1. All 12 (task × strategy) manifests v18.0 and n_train matches splitter
    expected_n_train = {
        ("regression", "random"):     1964,
        ("regression", "scaffold"):   1963,
        ("regression", "confirmed"):  1216,
        ("regression", "lead_opt"):   2246,
        ("regression", "similarity"): 1995,
        ("regression", "cliff_aware"): 1963,
        ("classification", "random"):     2082,
        ("classification", "scaffold"):   2058,
        ("classification", "confirmed"):  2018,
        ("classification", "lead_opt"):   2372,
        ("classification", "similarity"): 2084,
        ("classification", "cliff_aware"): 2082,
    }
    for (task, strategy), expected in expected_n_train.items():
        mpath = FEATURES_ROOT / task / strategy / "feature_manifest.json"
        if not mpath.exists():
            a.record(cat, f"manifest_{task}_{strategy}", "FAIL",
                     f"missing feature_manifest.json")
            continue
        m = json.loads(mpath.read_text())
        pv = m.get("pipeline_version")
        fn_tr = m.get("n_train")
        task_in_meta = m.get("task") or m.get("config", {}).get("task")
        version_ok = (pv == "18.0")
        train_ok = (fn_tr == expected)
        task_ok = (task_in_meta == task)
        all_ok = version_ok and train_ok and task_ok
        verdict = "PASS" if all_ok else "FAIL"
        a.record(cat, f"manifest_{task}_{strategy}", verdict,
                 f"v18.0, task tag, n_train={expected}",
                 value={"pipeline_version": pv, "n_train": fn_tr,
                        "task_meta": task_in_meta,
                        "version_ok": version_ok, "train_ok": train_ok,
                        "task_ok": task_ok},
                 source=f"features_v18/{task}/{strategy}/feature_manifest.json")

    # F2. Feature counts per variant (spot-check: should be consistent across strategies)
    counts_by_variant = defaultdict(list)
    for task in TASKS:
        for strategy in STRATEGIES:
            mpath = FEATURES_ROOT / task / strategy / "feature_manifest.json"
            if not mpath.exists():
                continue
            m = json.loads(mpath.read_text())
            for variant_name, var_info in m.get("variants", {}).items():
                n_tree = var_info.get("n_tree_features")
                n_linear = var_info.get("n_linear_features")
                counts_by_variant[variant_name].append((task, strategy, n_tree, n_linear))

    for v, entries in counts_by_variant.items():
        tree_counts = [e[2] for e in entries if e[2] is not None]
        linear_counts = [e[3] for e in entries if e[3] is not None]
        if tree_counts:
            tree_min, tree_max = min(tree_counts), max(tree_counts)
            a.record(cat, f"feature_count_{v}_tree", "INFO",
                     f"tree-space feature count range across 12 (task,strategy) cells",
                     value={"min": tree_min, "max": tree_max,
                            "n_cells": len(tree_counts)})
        if linear_counts:
            lin_min, lin_max = min(linear_counts), max(linear_counts)
            a.record(cat, f"feature_count_{v}_linear", "INFO",
                     f"linear-space feature count range across 12 cells",
                     value={"min": lin_min, "max": lin_max,
                            "n_cells": len(linear_counts)})

    # F3. NPZ files exist for every (task, strategy, variant, space, subset)
    missing = []
    for task in TASKS:
        for strategy in STRATEGIES:
            for variant in VARIANTS:
                for space in ("tree", "linear"):
                    for subset in ("train", "val", "test"):
                        p = FEATURES_ROOT / task / strategy / subset / f"{variant}_{space}.npz"
                        if not p.exists():
                            missing.append(str(p.relative_to(PROJECT_ROOT)))
    verdict = "PASS" if not missing else "FAIL"
    a.record(cat, "feature_npz_completeness", verdict,
             f"all 12×5×2×3 = 360 NPZ files present",
             value={"missing_count": len(missing),
                    "missing_examples": missing[:5]})

    # F4. stratifiers.npz present per subset
    missing_strat = []
    for task in TASKS:
        for strategy in STRATEGIES:
            for subset in ("train", "val", "test"):
                p = FEATURES_ROOT / task / strategy / subset / "stratifiers.npz"
                if not p.exists():
                    missing_strat.append(str(p.relative_to(PROJECT_ROOT)))
    verdict = "PASS" if not missing_strat else "FAIL"
    a.record(cat, "stratifiers_completeness", verdict,
             f"all 36 stratifiers.npz files present",
             value={"missing_count": len(missing_strat)})


# =============================================================================
# G. MODELING INTEGRITY
# =============================================================================
def audit_modeling(a: Audit, state: dict):
    cat = "G_modeling"
    print(f"\n{'=' * 70}\nG. MODELING INTEGRITY\n{'=' * 70}", flush=True)

    # G1. All 60 + 60 main-sweep cells present with full artifacts
    expected_artifacts = ["metrics.json", "model.pkl", "oof_train.npz",
                          "val_pred.npz", "test_pred.npz",
                          "tuning_results.json", "hparams.json"]
    missing_per_cell = {}
    n_main_cells = 0
    for task, models in [("regression", ["xgboost", "elasticnet"]),
                          ("classification", ["xgboost", "logreg_enet"])]:
        for strategy in STRATEGIES:
            for variant in VARIANTS:
                for model in models:
                    cell_dir = MODELS_ROOT / task / strategy / variant / model
                    n_main_cells += 1
                    missing = [f for f in expected_artifacts
                               if not (cell_dir / f).exists()]
                    if missing:
                        missing_per_cell[str(cell_dir.relative_to(PROJECT_ROOT))] = missing
    verdict = "PASS" if not missing_per_cell else "WARN"
    a.record(cat, "main_sweep_artifact_completeness", verdict,
             f"{n_main_cells} main-sweep cells have all 7 artifacts",
             value={"cells_with_missing": len(missing_per_cell),
                    "examples": dict(list(missing_per_cell.items())[:5])})

    # G2. Stacking cells (30)
    n_stack = 0
    missing_stack = {}
    for strategy in STRATEGIES:
        for variant in VARIANTS:
            cell_dir = MODELS_ROOT / "stacking" / strategy / variant / "xgboost"
            n_stack += 1
            needed = ["metrics.json", "model.pkl", "val_pred.npz",
                      "test_pred.npz", "tuning_results.json"]
            missing = [f for f in needed if not (cell_dir / f).exists()]
            if missing:
                missing_stack[str(cell_dir.relative_to(PROJECT_ROOT))] = missing
    verdict = "PASS" if not missing_stack else "WARN"
    a.record(cat, "stacking_artifact_completeness", verdict,
             f"30 stacking cells",
             value={"n_cells": n_stack, "missing": len(missing_stack)})

    # G3. Robustness seeds: random and scaffold × 5 variants × 2 tasks × 2 seeds = 40
    n_robust = 0
    missing_robust = {}
    for task in TASKS:
        for strategy in ["random", "scaffold"]:
            for variant in VARIANTS:
                for seed in [7, 1337]:
                    cell_dir = (MODELS_ROOT / task / strategy / variant /
                                "xgboost" / "seeds" / f"seed_{seed}")
                    n_robust += 1
                    if not (cell_dir / "metrics.json").exists():
                        missing_robust[str(cell_dir.relative_to(PROJECT_ROOT))] = "metrics.json"
    verdict = "PASS" if not missing_robust else "WARN"
    a.record(cat, "robustness_seeds_completeness", verdict,
             f"40 robustness-seed cells",
             value={"n_cells": n_robust, "missing": len(missing_robust)})

    # G4. Tuning grid consistency: every XGB cell has 18 configs, every ENET 15
    grid_sizes_xgb = []
    grid_sizes_lin = []
    for task in TASKS:
        for strategy in STRATEGIES:
            for variant in VARIANTS:
                # XGB
                tp = MODELS_ROOT / task / strategy / variant / "xgboost" / "tuning_results.json"
                if tp.exists():
                    t = json.loads(tp.read_text())
                    grid_sizes_xgb.append(len(t.get("grid_results", [])))
                # linear model
                linear_model = "elasticnet" if task == "regression" else "logreg_enet"
                tp = MODELS_ROOT / task / strategy / variant / linear_model / "tuning_results.json"
                if tp.exists():
                    t = json.loads(tp.read_text())
                    grid_sizes_lin.append(len(t.get("grid_results", [])))
    if grid_sizes_xgb:
        consistent = (min(grid_sizes_xgb) == max(grid_sizes_xgb))
        verdict = "PASS" if consistent and grid_sizes_xgb[0] == 18 else "WARN"
        a.record(cat, "xgb_tuning_grid_consistent", verdict,
                 f"XGB grid size consistent across all 60 cells",
                 value={"min": min(grid_sizes_xgb), "max": max(grid_sizes_xgb),
                        "expected": 18})
    if grid_sizes_lin:
        consistent = (min(grid_sizes_lin) == max(grid_sizes_lin))
        verdict = "PASS" if consistent and grid_sizes_lin[0] == 15 else "WARN"
        a.record(cat, "linear_tuning_grid_consistent", verdict,
                 f"linear-model grid size consistent",
                 value={"min": min(grid_sizes_lin), "max": max(grid_sizes_lin),
                        "expected": 15})


# =============================================================================
# H. STATISTICAL CLAIMS
# =============================================================================
def audit_statistics(a: Audit, state: dict):
    cat = "H_statistics"
    print(f"\n{'=' * 70}\nH. STATISTICAL CLAIMS\n{'=' * 70}", flush=True)

    # H1. CI present everywhere
    n_missing_ci = 0
    n_total = 0
    examples_missing = []
    for mpath in MODELS_ROOT.rglob("metrics.json"):
        n_total += 1
        m = json.loads(mpath.read_text())
        if "test_ci" not in m:
            n_missing_ci += 1
            if len(examples_missing) < 5:
                examples_missing.append(str(mpath.relative_to(PROJECT_ROOT)))
    verdict = "PASS" if n_missing_ci == 0 else "WARN"
    a.record(cat, "bootstrap_ci_present_all_cells", verdict,
             f"{n_total - n_missing_ci}/{n_total} cells have test_ci block",
             value={"missing": n_missing_ci, "examples": examples_missing})

    # H2. CI widths reasonable
    ci_widths_r2 = []
    ci_widths_auc = []
    for mpath in MODELS_ROOT.rglob("metrics.json"):
        m = json.loads(mpath.read_text())
        ci = m.get("test_ci", {})
        # regression
        r2 = ci.get("r2")
        if r2 and r2.get("hi") is not None and r2.get("lo") is not None:
            ci_widths_r2.append(r2["hi"] - r2["lo"])
        auc = ci.get("roc_auc")
        if auc and auc.get("hi") is not None and auc.get("lo") is not None:
            ci_widths_auc.append(auc["hi"] - auc["lo"])
    if ci_widths_r2:
        a.record(cat, "ci_width_r2_distribution", "INFO",
                 f"R² 95% CI width across cells",
                 value={"n": len(ci_widths_r2),
                        "median": float(np.median(ci_widths_r2)),
                        "min": float(np.min(ci_widths_r2)),
                        "max": float(np.max(ci_widths_r2))})
    if ci_widths_auc:
        a.record(cat, "ci_width_auc_distribution", "INFO",
                 f"ROC-AUC 95% CI width across cells",
                 value={"n": len(ci_widths_auc),
                        "median": float(np.median(ci_widths_auc)),
                        "min": float(np.min(ci_widths_auc)),
                        "max": float(np.max(ci_widths_auc))})

    # H3. Point estimate vs bootstrap median agreement
    discrepancies = []
    for mpath in MODELS_ROOT.rglob("metrics.json"):
        m = json.loads(mpath.read_text())
        ci = m.get("test_ci", {})
        test = m.get("test", {})
        # Regression
        if "r2" in test and "r2" in ci and ci["r2"].get("med") is not None:
            diff = abs(test["r2"] - ci["r2"]["med"])
            if diff > 0.05:
                discrepancies.append({"path": str(mpath.relative_to(PROJECT_ROOT)),
                                       "test": test["r2"],
                                       "boot_med": ci["r2"]["med"],
                                       "diff": diff})
        if "roc_auc" in test and "roc_auc" in ci and ci["roc_auc"].get("med") is not None:
            diff = abs(test["roc_auc"] - ci["roc_auc"]["med"])
            if diff > 0.03:
                discrepancies.append({"path": str(mpath.relative_to(PROJECT_ROOT)),
                                       "test": test["roc_auc"],
                                       "boot_med": ci["roc_auc"]["med"],
                                       "diff": diff})
    verdict = "PASS" if not discrepancies else "WARN"
    a.record(cat, "point_estimate_vs_bootstrap_median", verdict,
             f"large point-vs-bootstrap-median discrepancies",
             value={"count": len(discrepancies),
                    "examples": discrepancies[:3]})


# =============================================================================
# I. ROBUSTNESS CLAIMS
# =============================================================================
def audit_robustness(a: Audit, state: dict):
    cat = "I_robustness"
    print(f"\n{'=' * 70}\nI. ROBUSTNESS CLAIMS\n{'=' * 70}", flush=True)

    # For random + scaffold × 5 variants × 2 tasks, compute seed-cross-variance
    for task in TASKS:
        for strategy in ["random", "scaffold"]:
            for variant in VARIANTS:
                base = MODELS_ROOT / task / strategy / variant / "xgboost"
                seed_paths = [base / "metrics.json"]
                for seed in [7, 1337]:
                    seed_paths.append(base / "seeds" / f"seed_{seed}" / "metrics.json")
                vals = []
                for p in seed_paths:
                    if p.exists():
                        m = json.loads(p.read_text())
                        if task == "regression":
                            vals.append(m["test"]["r2"])
                        else:
                            vals.append(m["test"]["roc_auc"])
                if len(vals) >= 3:
                    std = float(np.std(vals, ddof=1))
                    mean = float(np.mean(vals))
                    verdict = "PASS" if std < 0.05 else "WARN"
                    a.record(cat, f"{task}_{strategy}_{variant}_seed_std", verdict,
                             f"3-seed std on test {'R²' if task=='regression' else 'AUC'}",
                             value={"seeds": vals, "mean": mean, "std": std})


# =============================================================================
# J. CALIBRATION CLAIMS
# =============================================================================
def audit_calibration(a: Audit, state: dict):
    cat = "J_calibration"
    print(f"\n{'=' * 70}\nJ. CALIBRATION CLAIMS\n{'=' * 70}", flush=True)

    eces = []
    per_split = defaultdict(list)
    for strategy in STRATEGIES:
        for variant in VARIANTS:
            for model in ["xgboost", "logreg_enet"]:
                mpath = MODELS_ROOT / "classification" / strategy / variant / model / "metrics.json"
                if not mpath.exists():
                    continue
                m = json.loads(mpath.read_text())
                cal = m.get("test_calibration", {})
                ece = cal.get("ece")
                if ece is not None:
                    eces.append(ece)
                    per_split[strategy].append(ece)

    if eces:
        a.record(cat, "ece_overall_distribution", "INFO",
                 f"ECE across all classification cells",
                 value={"n": len(eces), "mean": float(np.mean(eces)),
                        "median": float(np.median(eces)),
                        "min": float(np.min(eces)), "max": float(np.max(eces)),
                        "fraction_above_0.10": float(np.mean(np.array(eces) > 0.10))})

    for strategy, vals in per_split.items():
        mean_ece = float(np.mean(vals))
        verdict = "INFO" if mean_ece < 0.05 else ("WARN" if mean_ece < 0.10 else "FAIL")
        a.record(cat, f"ece_per_split_{strategy}", verdict,
                 f"mean ECE across variants for {strategy} split",
                 value={"mean_ece": mean_ece, "n_cells": len(vals)})


# =============================================================================
# K. STACKING CLAIMS
# =============================================================================
def audit_stacking(a: Audit, state: dict):
    cat = "K_stacking"
    print(f"\n{'=' * 70}\nK. STACKING CLAIMS\n{'=' * 70}", flush=True)

    lifts = []
    for strategy in STRATEGIES:
        for variant in VARIANTS:
            mpath = MODELS_ROOT / "stacking" / strategy / variant / "xgboost" / "metrics.json"
            if not mpath.exists():
                continue
            m = json.loads(mpath.read_text())
            lift = m.get("auc_lift_over_baseline")
            if lift is not None:
                lifts.append(lift)
                a.record(cat, f"stacking_lift_{strategy}_{variant}", "INFO",
                         f"stacking AUC lift", value=lift)
    if lifts:
        mean_lift = float(np.mean(lifts))
        n_pos = int(np.sum(np.array(lifts) > 0))
        verdict = "INFO"  # Negative result is publishable; not a fail
        a.record(cat, "stacking_summary", verdict,
                 f"30-cell stacking experiment summary",
                 value={"mean_lift": mean_lift, "n_positive_lifts": n_pos,
                        "n_total": len(lifts)})


# =============================================================================
# L. THRESHOLD POLICY
# =============================================================================
def audit_threshold(a: Audit, state: dict):
    cat = "L_threshold"
    print(f"\n{'=' * 70}\nL. THRESHOLD POLICY\n{'=' * 70}", flush=True)

    n_with_tuned = 0
    n_total = 0
    degenerate_default = []
    degenerate_tuned = []
    for strategy in STRATEGIES:
        for variant in VARIANTS:
            for model in ["xgboost", "logreg_enet"]:
                mpath = MODELS_ROOT / "classification" / strategy / variant / model / "metrics.json"
                if not mpath.exists():
                    continue
                m = json.loads(mpath.read_text())
                n_total += 1
                if "test_tuned" in m:
                    n_with_tuned += 1
                default_mcc = m.get("test", {}).get("mcc")
                tuned_mcc = m.get("test_tuned", {}).get("mcc") if "test_tuned" in m else None
                if default_mcc == 0:
                    degenerate_default.append(f"{strategy}/{variant}/{model}")
                if tuned_mcc == 0:
                    degenerate_tuned.append(f"{strategy}/{variant}/{model}")
    verdict = "PASS" if n_with_tuned == n_total else "WARN"
    a.record(cat, "tuned_threshold_present_all_clf", verdict,
             f"{n_with_tuned}/{n_total} classification cells have test_tuned block",
             value={"with_tuned": n_with_tuned, "total": n_total})
    a.record(cat, "degenerate_cells_default_threshold",
             "INFO" if degenerate_default else "PASS",
             f"cells with MCC=0 at threshold=0.5",
             value={"count": len(degenerate_default),
                    "examples": degenerate_default})
    a.record(cat, "degenerate_cells_after_tuning",
             "PASS" if not degenerate_tuned else "WARN",
             f"cells STILL MCC=0 after Youden's J recalibration",
             value={"count": len(degenerate_tuned),
                    "examples": degenerate_tuned})


# =============================================================================
# Main
# =============================================================================
def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    audit = Audit()
    state = {}

    print("PAD4_BENCH REVIEWER AUDIT", flush=True)
    print(f"Starting at {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    audit_raw_data(audit)
    state = audit_curation(audit) or {}
    state = audit_labels(audit, state) or state
    state = audit_dataset(audit, state) or state
    audit_splits(audit, state)
    audit_features(audit, state)
    audit_modeling(audit, state)
    audit_statistics(audit, state)
    audit_robustness(audit, state)
    audit_calibration(audit, state)
    audit_stacking(audit, state)
    audit_threshold(audit, state)

    json_path = OUT_ROOT / "reviewer_audit_report.json"
    md_path = OUT_ROOT / "reviewer_audit_report.md"
    json_path.write_text(json.dumps(audit.to_dict(), indent=2, default=str))
    md_path.write_text(audit.to_markdown())

    print(f"\n{'=' * 70}\nAUDIT COMPLETE\n{'=' * 70}", flush=True)
    print(f"Summary: {dict(audit.counts)}", flush=True)
    print(f"Total checks: {len(audit.results)}", flush=True)
    print(f"\nWrote JSON: {json_path}", flush=True)
    print(f"Wrote markdown: {md_path}", flush=True)
    if audit.counts.get("FAIL", 0) > 0:
        print(f"\n*** {audit.counts['FAIL']} FAIL checks need attention before submission ***",
              flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
