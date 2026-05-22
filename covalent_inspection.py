#!/usr/bin/env python3
"""
Covalent compound metadata inspection.

The introspection script revealed:
  - 9 compounds with is_covalent=True inside pad_t1_non_covalent.csv
  - 20 unique covalent compounds inside the 'confirmed' regression split
    (derived from pad_t1_confirmed.csv, which has 33 covalent rows)

This script pulls the covalent-related metadata for those specific compounds
so we can decide:
  (a) Are they reversible covalents intentionally kept? -> document, don't refilter
  (b) Are they irreversible covalents that slipped through? -> consider refiltering

Writes:
  paper_intro/covalent_inspection.csv  -- one row per flagged compound
  paper_intro/covalent_inspection.md   -- human-readable summary
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path("/home/nidhal/PAD4_BENCH")
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUT_ROOT = PROJECT_ROOT / "paper_intro"

COVALENT_META_COLS = [
    "inchikey_14", "inchikey", "canonical_smiles",
    "is_covalent", "covalent_confidence", "covalent_type",
    "warhead_count", "is_reversible_covalent",
    "covalent_consistent", "covalent_context_excluded",
    "assay_covalent_flag",
    "target_isoform", "measurement_type", "assay_type_v2",
    "pIC50", "pIC50_corrected", "pIC50_std", "n_measurements",
    "ml_weight", "source_list", "source_count",
]


def select_cols(df: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in COVALENT_META_COLS if c in df.columns]
    return df[keep]


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("Loading files ...")
    t1_noncov = pd.read_csv(DATA_PROCESSED / "pad_t1_non_covalent.csv",
                            low_memory=False)
    t1_confirmed = pd.read_csv(DATA_PROCESSED / "pad_t1_confirmed.csv",
                               low_memory=False)
    t1_covalent = pd.read_csv(DATA_PROCESSED / "pad_t1_covalent.csv",
                              low_memory=False)
    print(f"  non_covalent: {len(t1_noncov)} rows")
    print(f"  confirmed:    {len(t1_confirmed)} rows")
    print(f"  covalent:     {len(t1_covalent)} rows (reference)")

    # ---------------------------------------------------------------
    # The 9 is_covalent=True compounds inside pad_t1_non_covalent
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Group A: is_covalent=True inside pad_t1_non_covalent.csv")
    print("=" * 70)
    flagged_noncov = t1_noncov[t1_noncov["is_covalent"] == True].copy()
    print(f"  rows: {len(flagged_noncov)}")
    print(f"  unique InChIKey-14: {flagged_noncov['inchikey_14'].nunique()}")

    df_a = select_cols(flagged_noncov)
    df_a.insert(0, "source_file", "pad_t1_non_covalent")
    print("\nFull rows:")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 240)
    pd.set_option("display.max_colwidth", 50)
    print(df_a.to_string(index=False))

    # ---------------------------------------------------------------
    # The is_covalent=True compounds inside pad_t1_confirmed
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Group B: is_covalent=True inside pad_t1_confirmed.csv")
    print("=" * 70)
    flagged_confirmed = t1_confirmed[t1_confirmed["is_covalent"] == True].copy()
    print(f"  rows: {len(flagged_confirmed)}")
    print(f"  unique InChIKey-14: {flagged_confirmed['inchikey_14'].nunique()}")

    df_b = select_cols(flagged_confirmed)
    df_b.insert(0, "source_file", "pad_t1_confirmed")
    print("\nFull rows:")
    print(df_b.to_string(index=False))

    # ---------------------------------------------------------------
    # Compare to the "real" covalent file for reference
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Reference: covalent_confidence distribution in pad_t1_covalent.csv")
    print("=" * 70)
    if "covalent_confidence" in t1_covalent.columns:
        print(t1_covalent["covalent_confidence"].value_counts(dropna=False).to_string())
    if "covalent_type" in t1_covalent.columns:
        print("\ncovalent_type distribution:")
        print(t1_covalent["covalent_type"].value_counts(dropna=False).to_string())
    if "is_reversible_covalent" in t1_covalent.columns:
        print("\nis_reversible_covalent distribution:")
        print(t1_covalent["is_reversible_covalent"].value_counts(dropna=False).to_string())

    # ---------------------------------------------------------------
    # Summary of the flagged compounds' classifications
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Group A summary (non_covalent flagged):")
    print("=" * 70)
    for col in ("covalent_confidence", "covalent_type",
                "is_reversible_covalent", "covalent_consistent",
                "covalent_context_excluded", "assay_covalent_flag"):
        if col in flagged_noncov.columns:
            print(f"\n  {col}:")
            print("    " + flagged_noncov[col].value_counts(dropna=False)
                  .to_string().replace("\n", "\n    "))

    print("\n" + "=" * 70)
    print("Group B summary (confirmed flagged):")
    print("=" * 70)
    for col in ("covalent_confidence", "covalent_type",
                "is_reversible_covalent", "covalent_consistent",
                "covalent_context_excluded", "assay_covalent_flag"):
        if col in flagged_confirmed.columns:
            print(f"\n  {col}:")
            print("    " + flagged_confirmed[col].value_counts(dropna=False)
                  .to_string().replace("\n", "\n    "))

    # ---------------------------------------------------------------
    # Overlap: are the 9 in non_covalent a subset of the 20 in confirmed?
    # ---------------------------------------------------------------
    a_ids = set(flagged_noncov["inchikey_14"].astype(str))
    b_ids = set(flagged_confirmed["inchikey_14"].astype(str))
    print("\n" + "=" * 70)
    print("Overlap between Group A and Group B")
    print("=" * 70)
    print(f"  A only: {len(a_ids - b_ids)}")
    print(f"  B only: {len(b_ids - a_ids)}")
    print(f"  both:   {len(a_ids & b_ids)}")
    if a_ids - b_ids:
        print(f"  A-only IDs: {sorted(a_ids - b_ids)}")
    if b_ids - a_ids:
        print(f"  B-only IDs: {sorted(b_ids - a_ids)}")

    # ---------------------------------------------------------------
    # Save combined CSV
    # ---------------------------------------------------------------
    combined = pd.concat([df_a, df_b], ignore_index=True)
    csv_path = OUT_ROOT / "covalent_inspection.csv"
    combined.to_csv(csv_path, index=False)
    print(f"\nWrote combined CSV to {csv_path}")

    # ---------------------------------------------------------------
    # Markdown decision aid
    # ---------------------------------------------------------------
    md = ["# Covalent compounds inside Paper 1 modeling sets\n"]
    md.append("## Group A — 9 is_covalent=True rows inside pad_t1_non_covalent.csv\n")
    md.append(f"- Total rows: {len(flagged_noncov)}")
    md.append(f"- Unique compounds: {flagged_noncov['inchikey_14'].nunique()}\n")
    if "is_reversible_covalent" in flagged_noncov.columns:
        rev_count = int(flagged_noncov["is_reversible_covalent"].fillna(False)
                        .astype(bool).sum())
        md.append(f"- is_reversible_covalent=True: **{rev_count} of {len(flagged_noncov)}**")
    if "covalent_confidence" in flagged_noncov.columns:
        md.append(f"- covalent_confidence values: {dict(flagged_noncov['covalent_confidence'].value_counts(dropna=False))}")
    if "covalent_context_excluded" in flagged_noncov.columns:
        md.append(f"- covalent_context_excluded values: {dict(flagged_noncov['covalent_context_excluded'].value_counts(dropna=False))}\n")

    md.append("## Group B — covalent rows inside pad_t1_confirmed.csv\n")
    md.append(f"- Total rows: {len(flagged_confirmed)}")
    md.append(f"- Unique compounds: {flagged_confirmed['inchikey_14'].nunique()}\n")
    if "is_reversible_covalent" in flagged_confirmed.columns:
        rev_count = int(flagged_confirmed["is_reversible_covalent"].fillna(False)
                        .astype(bool).sum())
        md.append(f"- is_reversible_covalent=True: **{rev_count} of {len(flagged_confirmed)}**")
    if "covalent_confidence" in flagged_confirmed.columns:
        md.append(f"- covalent_confidence values: {dict(flagged_confirmed['covalent_confidence'].value_counts(dropna=False))}")
    if "covalent_context_excluded" in flagged_confirmed.columns:
        md.append(f"- covalent_context_excluded values: {dict(flagged_confirmed['covalent_context_excluded'].value_counts(dropna=False))}\n")

    md.append("## Decision framework\n")
    md.append("- If most/all are reversible covalent: keep as-is, add one disclosure sentence.")
    md.append("- If most are irreversible: consider refiltering (small but real overhead).")
    md.append("- If covalent_context_excluded=True predominantly: the curator explicitly")
    md.append("  decided to include these despite covalent flag (e.g. assay context made")
    md.append("  the covalent mechanism irrelevant). Document and move on.\n")

    md_path = OUT_ROOT / "covalent_inspection.md"
    md_path.write_text("\n".join(md))
    print(f"Wrote markdown summary to {md_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
