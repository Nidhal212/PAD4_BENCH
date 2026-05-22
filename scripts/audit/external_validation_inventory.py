#!/usr/bin/env python3
"""
PAD4_BENCH external validation inventory.

Before deciding whether to add an external validation experiment to Paper 1,
this script inspects every data file to determine what could potentially
serve as:

  (a) TRUE external validation (independent data, never seen by the pipeline)
  (b) CURATION-sensitivity validation (alternative curation of same data)
  (c) USELESS for Paper 1 validation (different label space, mechanism, etc.)

For each candidate file:
  - load it
  - count unique compounds
  - check overlap with the modeled regression set (n=2,618)
  - check overlap with the modeled classification set (n=2,758)
  - determine if compounds outside the modeled set carry usable labels
  - report what kind of validation each file could enable

For temporal-validation potential, also checks if any file carries date
metadata that could be used for a post-cutoff holdout.

Writes:
  paper_intro/external_validation_inventory.{json,md}
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_RAW       = PROJECT_ROOT / "data" / "raw"
OUT_ROOT       = PROJECT_ROOT / "paper_intro"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def safe_read(path: Path, **kw):
    try:
        return pd.read_csv(path, low_memory=False, **kw)
    except Exception as e:
        log(f"  FAIL reading {path.name}: {e}")
        return None


def id_set(df: pd.DataFrame) -> set[str]:
    if df is None:
        return set()
    for col in ("inchikey_14",):
        if col in df.columns:
            return set(df[col].dropna().astype(str))
    if "inchikey" in df.columns:
        return set(df["inchikey"].dropna().astype(str).str[:14])
    for c in df.columns:
        if "inchikey" in c.lower():
            return set(df[c].dropna().astype(str).str[:14])
    return set()


def check_date_columns(df: pd.DataFrame) -> list[str]:
    """Return columns that look like dates (for temporal-holdout potential)."""
    date_cols = []
    if df is None:
        return date_cols
    for c in df.columns:
        cl = c.lower()
        if any(k in cl for k in ("date", "year", "doi", "pmid",
                                   "publication", "submitted")):
            date_cols.append(c)
    return date_cols


def check_label_compatibility(df: pd.DataFrame) -> dict:
    """Check if the dataframe carries usable pIC50 (regression target) or
    activity_class (classification target)."""
    out = {"has_pIC50": False, "has_activity_class": False,
           "pIC50_n_nonnull": 0, "activity_class_n_nonnull": 0,
           "pIC50_range": None, "activity_class_distribution": None}
    if df is None:
        return out
    if "pIC50" in df.columns:
        out["has_pIC50"] = True
        nonnull = df["pIC50"].dropna()
        out["pIC50_n_nonnull"] = int(len(nonnull))
        if len(nonnull) > 0:
            out["pIC50_range"] = [float(nonnull.min()), float(nonnull.max())]
    if "activity_class" in df.columns:
        out["has_activity_class"] = True
        nonnull = df["activity_class"].dropna()
        out["activity_class_n_nonnull"] = int(len(nonnull))
        if len(nonnull) > 0:
            vc = nonnull.value_counts().to_dict()
            out["activity_class_distribution"] = {str(k): int(v)
                                                   for k, v in vc.items()}
    return out


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    report = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
              "candidates": []}

    log("=" * 70)
    log("PAD4_BENCH external validation inventory")
    log("=" * 70)

    # Load the modeled sets (regression base and classification base)
    log("\nLoading modeled (training universe) sets...")
    reg_df = safe_read(DATA_PROCESSED / "pad_t1_non_covalent.csv")
    cls_df = safe_read(DATA_PROCESSED / "pad_classification_v17.csv")
    reg_ids = id_set(reg_df)
    cls_ids = id_set(cls_df)
    log(f"  regression set:     {len(reg_ids):,} unique InChIKey-14")
    log(f"  classification set: {len(cls_ids):,} unique InChIKey-14")
    log(f"  union:              {len(reg_ids | cls_ids):,} unique InChIKey-14")

    modeled_ids = reg_ids | cls_ids
    report["modeled_universe"] = {
        "regression_n": len(reg_ids),
        "classification_n": len(cls_ids),
        "union_n": len(modeled_ids),
    }

    # =========================================================================
    # Inspect every candidate file
    # =========================================================================
    candidates = [
        # (file, role hypothesis, comments)
        ("pad_t1_covalent.csv",
         "irreversible covalents",
         "different mechanism — not fair external test"),
        ("pad_t1_ic50_aggregated.csv",
         "pre-covalent-filter T1",
         "same data, broader curation"),
        ("pad_t1_ic50_strict_v17.csv",
         "stricter T1 curation",
         "same data, stricter curation — curation-sensitivity candidate"),
        ("pad_t1_ic50_ml_ready_v17.csv",
         "alternative ML-prep T1",
         "same data, alternative prep — curation-sensitivity candidate"),
        ("pad_t1_strict_stereo.csv",
         "stereo-strict T1",
         "same data, stereo-strict filter"),
        ("pad_t2_censored.csv",
         "T2 censored measurements",
         "different label space (censored) — Paper 3 territory"),
        ("pad_t3_hts_balanced.csv",
         "T3 HTS balanced",
         "different label space (binary HTS) — Paper 3 territory"),
        ("pad_t3_hts_denoised.csv",
         "T3 HTS denoised",
         "different label space (binary HTS) — Paper 3 territory"),
        ("pad_t3_hts_indomain.csv",
         "T3 HTS in-domain",
         "different label space (binary HTS) — Paper 3 territory"),
        ("pad_ki_clean.csv",
         "Ki binding affinity",
         "different physical quantity — cross-assay-type, Paper 3"),
        ("pad_multifidelity_v17.csv",
         "multi-fidelity composite",
         "mixed tiers, Paper 3"),
        ("pad_t1_confirmed.csv",
         "confirmed-quality T1",
         "already a Paper 1 split, NOT external"),
        ("pad_classification_v17_backup.csv",
         "backup of classification set",
         "duplicate file, NOT external"),
    ]

    log("\nChecking each candidate file...")
    log("=" * 70)
    for fname, role, comment in candidates:
        path = DATA_PROCESSED / fname
        log(f"\n  File: {fname}")
        log(f"  Hypothesis: {role}")
        if not path.exists():
            log(f"  MISSING — skipping")
            report["candidates"].append({"file": fname, "exists": False})
            continue

        df = safe_read(path)
        if df is None:
            report["candidates"].append({"file": fname, "exists": True,
                                         "loaded": False})
            continue

        ids_here = id_set(df)
        n_here = len(ids_here)
        in_modeled = ids_here & modeled_ids
        outside_modeled = ids_here - modeled_ids
        outside_reg = ids_here - reg_ids
        outside_cls = ids_here - cls_ids

        # Check label compatibility
        labels = check_label_compatibility(df)
        date_cols = check_date_columns(df)

        # Reasoning: can this file provide external/curation validation?
        verdict = "NOT USEFUL"
        verdict_reason = ""

        if n_here == 0:
            verdict = "EMPTY"
            verdict_reason = "no compounds"
        elif len(outside_modeled) == 0:
            verdict = "SUBSET"
            verdict_reason = ("all compounds already in modeled universe — "
                              "would be data leakage to use as 'external'")
        elif len(outside_modeled) < 10:
            verdict = "TOO SMALL"
            verdict_reason = (f"only {len(outside_modeled)} compounds outside "
                              f"modeled universe; insufficient for validation")
        elif not labels["has_pIC50"] and not labels["has_activity_class"]:
            verdict = "INCOMPATIBLE LABELS"
            verdict_reason = ("no pIC50 or activity_class columns; cannot "
                              "evaluate models trained on these targets")
        elif fname == "pad_t1_covalent.csv":
            verdict = "MECHANISTICALLY DIFFERENT"
            verdict_reason = ("irreversible covalent inhibitors are a different "
                              "mechanism; using as external test conflates "
                              "model error with mechanism difference")
        elif "t3_hts" in fname:
            verdict = "DIFFERENT LABEL SPACE"
            verdict_reason = ("T3 HTS provides binary single-concentration hits, "
                              "not pIC50 — cannot evaluate regression; "
                              "binary labels are noisy and Paper-3-territory")
        elif "t2_censored" in fname:
            verdict = "DIFFERENT LABEL SPACE"
            verdict_reason = "censored measurements need specialized evaluation"
        elif "ki" in fname:
            verdict = "DIFFERENT ASSAY"
            verdict_reason = ("Ki binding affinity is a different physical "
                              "quantity from IC50; cross-assay-type transfer "
                              "is a separate research question")
        elif "multifidelity" in fname:
            verdict = "COMPOSITE"
            verdict_reason = "mixed tiers; not a single external set"
        elif "backup" in fname:
            verdict = "DUPLICATE"
            verdict_reason = "backup of already-modeled file"
        elif fname == "pad_t1_confirmed.csv":
            verdict = "ALREADY USED"
            verdict_reason = "already powers the Confirmed split"
        elif fname in ("pad_t1_ic50_strict_v17.csv",
                        "pad_t1_ic50_ml_ready_v17.csv",
                        "pad_t1_strict_stereo.csv",
                        "pad_t1_ic50_aggregated.csv"):
            n_strict_unique = len(outside_modeled)
            if n_strict_unique >= 10 and labels["has_pIC50"]:
                verdict = "CURATION SENSITIVITY"
                verdict_reason = (f"alternative curation; {n_strict_unique} "
                                  f"compounds outside modeled universe with "
                                  f"pIC50 labels could be used as a "
                                  f"curation-sensitivity check (NOT true "
                                  f"external — same raw sources)")
            else:
                verdict = "INSUFFICIENT"
                verdict_reason = (f"only {n_strict_unique} compounds outside, "
                                  f"or labels not usable")

        log(f"  n unique compounds:      {n_here:,}")
        log(f"  in modeled universe:     {len(in_modeled):,}")
        log(f"  OUTSIDE modeled universe: {len(outside_modeled):,}")
        log(f"  outside reg set:         {len(outside_reg):,}")
        log(f"  outside cls set:         {len(outside_cls):,}")
        log(f"  has pIC50:               {labels['has_pIC50']}  "
            f"(n_nonnull={labels['pIC50_n_nonnull']})")
        log(f"  has activity_class:      {labels['has_activity_class']}  "
            f"(n_nonnull={labels['activity_class_n_nonnull']})")
        log(f"  date/source columns:     {date_cols if date_cols else 'none'}")
        log(f"  VERDICT: {verdict}")
        log(f"  reasoning: {verdict_reason}")

        # If this is a curation-sensitivity candidate, also list a few example
        # InChIKeys outside the modeled set so the user can spot-check.
        examples = list(outside_modeled)[:5] if outside_modeled else []

        report["candidates"].append({
            "file": fname,
            "exists": True,
            "hypothesis_role": role,
            "comment": comment,
            "n_unique_compounds": n_here,
            "in_modeled_universe": len(in_modeled),
            "outside_modeled_universe": len(outside_modeled),
            "outside_regression_set": len(outside_reg),
            "outside_classification_set": len(outside_cls),
            "has_pIC50": labels["has_pIC50"],
            "pIC50_n_nonnull": labels["pIC50_n_nonnull"],
            "pIC50_range": labels["pIC50_range"],
            "has_activity_class": labels["has_activity_class"],
            "activity_class_n_nonnull": labels["activity_class_n_nonnull"],
            "activity_class_distribution": labels["activity_class_distribution"],
            "date_columns": date_cols,
            "verdict": verdict,
            "reasoning": verdict_reason,
            "example_ids_outside_modeled": examples,
        })

    # =========================================================================
    # Raw data temporal-holdout potential
    # =========================================================================
    log("\n" + "=" * 70)
    log("Raw data temporal-holdout potential")
    log("=" * 70)
    raw_dirs = ["chembl", "bindingdb", "pubchem"]
    raw_info = {}
    for sub in raw_dirs:
        d = DATA_RAW / sub
        if not d.exists():
            continue
        # Just peek at the first file to see what columns exist
        files = sorted(d.glob("*.tsv")) + sorted(d.glob("*.csv"))
        if not files:
            continue
        first = files[0]
        try:
            # Sniff separator
            sep = "\t" if first.suffix == ".tsv" else ","
            df0 = pd.read_csv(first, sep=sep, nrows=5, low_memory=False)
            date_cols = check_date_columns(df0)
            raw_info[sub] = {
                "file_inspected": first.name,
                "all_columns": list(df0.columns),
                "date_or_source_columns": date_cols,
            }
            log(f"\n  {sub}/{first.name}:")
            log(f"    date/source cols: {date_cols if date_cols else 'none detected in column names'}")
            log(f"    total cols: {len(df0.columns)}")
        except Exception as e:
            log(f"  FAIL inspecting {first}: {e}")
            raw_info[sub] = {"error": str(e)}

    report["raw_data_temporal_potential"] = raw_info

    # =========================================================================
    # Summary recommendation
    # =========================================================================
    log("\n" + "=" * 70)
    log("SUMMARY: usable validation options for Paper 1")
    log("=" * 70)

    options = []
    for c in report["candidates"]:
        if c.get("verdict") == "CURATION SENSITIVITY":
            options.append((c["file"], c["outside_modeled_universe"],
                             c.get("pIC50_n_nonnull", 0)))

    if options:
        log("\nCuration-sensitivity candidates (NOT true external validation):")
        for fname, n_out, n_pic in options:
            log(f"  {fname}: {n_out} compounds outside modeled set, "
                f"{n_pic} with pIC50")
    else:
        log("\nNo curation-sensitivity candidates with sufficient compounds.")

    # True external requires raw-data temporal split (out of scope without
    # date-stamped raw data)
    has_temporal_dates = any(
        v.get("date_or_source_columns") for v in raw_info.values()
        if isinstance(v, dict)
    )
    log(f"\nTrue temporal-holdout potential in raw data: "
        f"{'POSSIBLE' if has_temporal_dates else 'NOT IMMEDIATELY ACCESSIBLE'}")
    if has_temporal_dates:
        log("  → would require re-pulling raw + re-curating + re-modeling")

    log("\nRECOMMENDED PATH:")
    if options:
        best = max(options, key=lambda x: x[2])
        log(f"  Run a curation-sensitivity check using {best[0]}")
        log(f"  ({best[1]} candidate compounds, {best[2]} with pIC50 labels)")
        log("  → small supplementary panel, no claim of external validation")
    else:
        log("  No suitable validation candidates on disk.")
        log("  → use Limitations paragraph in Discussion, do not add validation")

    # Save reports
    json_path = OUT_ROOT / "external_validation_inventory.json"
    md_path = OUT_ROOT / "external_validation_inventory.md"
    json_path.write_text(json.dumps(report, indent=2, default=str))

    # Markdown summary
    md = ["# External validation inventory — Paper 1\n"]
    md.append(f"Generated: {report['generated']}\n")
    md.append("## Modeled universe (training data)\n")
    md.append(f"- regression base: {report['modeled_universe']['regression_n']:,} compounds")
    md.append(f"- classification base: {report['modeled_universe']['classification_n']:,} compounds")
    md.append(f"- union: {report['modeled_universe']['union_n']:,} compounds\n")
    md.append("## File-by-file verdict\n")
    md.append("| File | Unique | Outside modeled | Has pIC50 | Verdict |")
    md.append("|---|---|---|---|---|")
    for c in report["candidates"]:
        if not c.get("exists", False):
            continue
        md.append(f"| {c['file']} | {c.get('n_unique_compounds', '?'):,} | "
                  f"{c.get('outside_modeled_universe', '?')} | "
                  f"{c.get('has_pIC50', '?')} | "
                  f"**{c.get('verdict', '?')}** |")
    md.append("\n## Detailed reasoning per file\n")
    for c in report["candidates"]:
        if not c.get("exists", False):
            continue
        md.append(f"### {c['file']}")
        md.append(f"- **Verdict**: {c.get('verdict', '?')}")
        md.append(f"- Reasoning: {c.get('reasoning', '?')}")
        md.append(f"- Unique compounds: {c.get('n_unique_compounds', '?'):,}")
        md.append(f"- Outside modeled universe: {c.get('outside_modeled_universe', '?')}")
        md.append("")

    md_path.write_text("\n".join(md))
    log(f"\nWrote: {json_path.relative_to(PROJECT_ROOT)}")
    log(f"Wrote: {md_path.relative_to(PROJECT_ROOT)}")
    log("\nDONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
