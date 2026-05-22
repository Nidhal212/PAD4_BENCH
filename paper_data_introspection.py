#!/usr/bin/env python3
"""
PAD4_BENCH data introspection.

Answers the scope questions needed before writing the paper data figures:

  Q1 - Confirmed split provenance: is it derived from pad_t1_non_covalent
       (covalent-free) or from pad_t1_confirmed (may include covalents)?
       Result tells us whether "Paper 1 excludes covalents" needs nuance.

  Q2 - Replicate data: does pad_replicates_full cover the modeled compounds?
       If yes, we can produce a noise-floor figure (D10 supplementary).

Plus several other things worth knowing for the paper:
  - Exact compound counts at every curation stage.
  - Source contribution (ChEMBL / BindingDB / PubChem) to the modeled sets.
  - Overlap between regression set (2,618 non-covalent) and classification
    set (2,758) -- the stacking experiment's foundation.
  - InChIKey-14 vs full-InChIKey consistency.
  - Whether the activity_cliffs file's compounds are all in the modeled set.

Writes:
  paper_intro/data_intro_report.json   -- structured results
  paper_intro/data_intro_report.md     -- human-readable summary

Usage:
    cd /home/nidhal/PAD4_BENCH
    python paper_data_introspection.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path("/home/nidhal/PAD4_BENCH")
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
SPLITS_ROOT = PROJECT_ROOT / "data" / "splits"
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
OUT_ROOT = PROJECT_ROOT / "paper_intro"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def safe_read(path: Path, **kwargs) -> pd.DataFrame | None:
    if not path.exists():
        print(f"  MISSING: {path}", flush=True)
        return None
    try:
        df = pd.read_csv(path, low_memory=False, **kwargs)
        return df
    except Exception as e:
        print(f"  FAIL reading {path}: {e}", flush=True)
        return None


def id_set(df: pd.DataFrame, prefer: str = "inchikey_14") -> set[str]:
    """Return the InChIKey-14 set from a dataframe, falling back gracefully."""
    if df is None:
        return set()
    if prefer in df.columns:
        return set(df[prefer].dropna().astype(str))
    if "inchikey" in df.columns:
        return set(df["inchikey"].dropna().astype(str).str[:14])
    # Try any column that looks like an InChIKey
    for c in df.columns:
        if "inchikey" in c.lower():
            return set(df[c].dropna().astype(str).str[:14])
    return set()


def show(name: str, value) -> None:
    print(f"  {name}: {value}", flush=True)


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}", flush=True)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    report: dict = {}

    # ============== Load all relevant CSVs ==============
    section("Loading processed CSVs")
    t1_aggregated = safe_read(DATA_PROCESSED / "pad_t1_ic50_aggregated.csv")
    t1_noncov     = safe_read(DATA_PROCESSED / "pad_t1_non_covalent.csv")
    t1_covalent   = safe_read(DATA_PROCESSED / "pad_t1_covalent.csv")
    t1_confirmed  = safe_read(DATA_PROCESSED / "pad_t1_confirmed.csv")
    t1_ml_ready   = safe_read(DATA_PROCESSED / "pad_t1_ic50_ml_ready_v17.csv")
    t1_strict     = safe_read(DATA_PROCESSED / "pad_t1_ic50_strict_v17.csv")
    classif       = safe_read(DATA_PROCESSED / "pad_classification_v17.csv")
    cliffs        = safe_read(DATA_PROCESSED / "pad_activity_cliffs.csv")
    replicates    = safe_read(DATA_PROCESSED / "pad_replicates_full.csv")
    multifid      = safe_read(DATA_PROCESSED / "pad_multifidelity_v17.csv")
    t2_cens       = safe_read(DATA_PROCESSED / "pad_t2_censored.csv")
    t3_balanced   = safe_read(DATA_PROCESSED / "pad_t3_hts_balanced.csv")
    t3_denoised   = safe_read(DATA_PROCESSED / "pad_t3_hts_denoised.csv")
    t3_indomain   = safe_read(DATA_PROCESSED / "pad_t3_hts_indomain.csv")
    ki_clean      = safe_read(DATA_PROCESSED / "pad_ki_clean.csv")
    assay_bias    = safe_read(DATA_PROCESSED / "pad_assay_bias_report.csv")

    sizes = {
        "t1_aggregated":   None if t1_aggregated is None else len(t1_aggregated),
        "t1_non_covalent": None if t1_noncov is None else len(t1_noncov),
        "t1_covalent":     None if t1_covalent is None else len(t1_covalent),
        "t1_confirmed":    None if t1_confirmed is None else len(t1_confirmed),
        "t1_ml_ready":     None if t1_ml_ready is None else len(t1_ml_ready),
        "t1_strict":       None if t1_strict is None else len(t1_strict),
        "classification":  None if classif is None else len(classif),
        "activity_cliffs": None if cliffs is None else len(cliffs),
        "replicates_full": None if replicates is None else len(replicates),
        "multifidelity":   None if multifid is None else len(multifid),
        "t2_censored":     None if t2_cens is None else len(t2_cens),
        "t3_balanced":     None if t3_balanced is None else len(t3_balanced),
        "t3_denoised":     None if t3_denoised is None else len(t3_denoised),
        "t3_indomain":     None if t3_indomain is None else len(t3_indomain),
        "ki_clean":        None if ki_clean is None else len(ki_clean),
        "assay_bias_rows": None if assay_bias is None else len(assay_bias),
    }
    section("File sizes (rows)")
    for k, v in sizes.items():
        show(k, v)
    report["file_sizes"] = sizes

    # ============== ID sets ==============
    section("Building InChIKey-14 sets")
    sets = {}
    for name, df in [
        ("t1_aggregated", t1_aggregated),
        ("t1_non_covalent", t1_noncov),
        ("t1_covalent", t1_covalent),
        ("t1_confirmed", t1_confirmed),
        ("t1_ml_ready", t1_ml_ready),
        ("t1_strict", t1_strict),
        ("classification", classif),
        ("multifidelity", multifid),
    ]:
        sets[name] = id_set(df)
        show(f"unique InChIKey-14 in {name}", len(sets[name]))

    # ============== Q1: confirmed split provenance ==============
    section("Q1 - 'confirmed' split provenance")

    # Load the confirmed regression split (modeled compounds in confirmed split)
    confirmed_split_ids = set()
    for subset in ("train", "val", "test_locked"):
        sp = SPLITS_ROOT / "regression" / "confirmed" / f"{subset}.csv"
        df = safe_read(sp)
        confirmed_split_ids |= id_set(df)
    show("unique InChIKey-14 in confirmed regression split (train+val+test_locked)",
         len(confirmed_split_ids))

    # Compare to candidate sources
    in_t1_confirmed  = len(confirmed_split_ids & sets["t1_confirmed"])
    in_t1_noncov     = len(confirmed_split_ids & sets["t1_non_covalent"])
    in_t1_aggregated = len(confirmed_split_ids & sets["t1_aggregated"])
    in_intersection  = len(confirmed_split_ids
                           & sets["t1_confirmed"]
                           & sets["t1_non_covalent"])
    show("confirmed-split compounds also in pad_t1_confirmed",     in_t1_confirmed)
    show("confirmed-split compounds also in pad_t1_non_covalent",  in_t1_noncov)
    show("confirmed-split compounds also in pad_t1_aggregated",    in_t1_aggregated)
    show("confirmed-split compounds in BOTH confirmed AND non_covalent",
         in_intersection)

    # Is the confirmed split EXACTLY pad_t1_confirmed, or the intersection?
    # If confirmed_split_ids == sets['t1_confirmed'], it's direct.
    # If confirmed_split_ids == sets['t1_confirmed'] & sets['t1_non_covalent'],
    # it's the intersection (i.e., covalent-free).
    is_pure_confirmed = (confirmed_split_ids == sets["t1_confirmed"])
    is_intersection   = (confirmed_split_ids
                         == (sets["t1_confirmed"] & sets["t1_non_covalent"]))
    is_subset_confirmed = confirmed_split_ids.issubset(sets["t1_confirmed"])
    is_subset_noncov    = confirmed_split_ids.issubset(sets["t1_non_covalent"])
    show("is confirmed-split == pad_t1_confirmed exactly?",     is_pure_confirmed)
    show("is confirmed-split == intersection(confirmed,noncov)?", is_intersection)
    show("is confirmed-split ⊆ pad_t1_confirmed?",   is_subset_confirmed)
    show("is confirmed-split ⊆ pad_t1_non_covalent?", is_subset_noncov)

    # Does the confirmed split include any covalent compounds?
    if t1_covalent is not None:
        cov_ids = id_set(t1_covalent)
        cov_in_confirmed_split = confirmed_split_ids & cov_ids
        show("covalent compounds in confirmed split", len(cov_in_confirmed_split))
        if cov_in_confirmed_split:
            print(f"    examples: {list(cov_in_confirmed_split)[:5]}", flush=True)

    report["Q1_confirmed_provenance"] = {
        "confirmed_split_n_unique_ids": len(confirmed_split_ids),
        "overlap_with_t1_confirmed":     in_t1_confirmed,
        "overlap_with_t1_non_covalent":  in_t1_noncov,
        "overlap_with_t1_aggregated":    in_t1_aggregated,
        "overlap_with_intersection_confirmed_AND_noncov": in_intersection,
        "is_exactly_t1_confirmed":       bool(is_pure_confirmed),
        "is_exactly_intersection":       bool(is_intersection),
        "is_subset_of_t1_confirmed":     bool(is_subset_confirmed),
        "is_subset_of_t1_non_covalent":  bool(is_subset_noncov),
        "covalent_compounds_in_confirmed_split":
            int(len(confirmed_split_ids & id_set(t1_covalent)))
            if t1_covalent is not None else None,
    }

    # ============== Q2: replicate coverage ==============
    section("Q2 - pad_replicates_full coverage of modeled compounds")
    if replicates is not None:
        rep_ids = id_set(replicates)
        show("unique InChIKey-14 in replicates_full", len(rep_ids))
        # Coverage of regression modeling set (2,618 non_covalent)
        in_noncov = len(rep_ids & sets["t1_non_covalent"])
        in_class  = len(rep_ids & sets["classification"])
        in_both   = len(rep_ids & sets["t1_non_covalent"] & sets["classification"])
        show("replicates compounds also in pad_t1_non_covalent",  in_noncov)
        show("replicates compounds also in pad_classification",    in_class)
        show("replicates compounds in both",                        in_both)
        coverage_noncov = in_noncov / max(1, len(sets["t1_non_covalent"]))
        coverage_class  = in_class  / max(1, len(sets["classification"]))
        show("fraction of non_covalent set covered by replicates",
             f"{coverage_noncov:.3f}")
        show("fraction of classification set covered by replicates",
             f"{coverage_class:.3f}")

        # How many replicates per compound? Distribution
        if "inchikey_14" in replicates.columns:
            rep_per = replicates.groupby("inchikey_14").size()
            show("replicates per compound: median", int(rep_per.median()))
            show("replicates per compound: mean",   round(rep_per.mean(), 2))
            show("replicates per compound: max",    int(rep_per.max()))
            show("compounds with >= 2 replicates",  int((rep_per >= 2).sum()))
            show("compounds with >= 3 replicates",  int((rep_per >= 3).sum()))
            show("compounds with >= 5 replicates",  int((rep_per >= 5).sum()))

            # Replicates with pIC50 column?
            if "pIC50" in replicates.columns:
                # Per-compound pIC50 std among compounds with >= 2 measurements
                df = replicates[["inchikey_14", "pIC50"]].dropna()
                pic_std = (df.groupby("inchikey_14")["pIC50"]
                           .agg(["count", "std", "mean"]))
                multi = pic_std[pic_std["count"] >= 2]
                show("compounds with >= 2 pIC50 measurements", len(multi))
                if len(multi):
                    show("median pIC50 std across multi-measurement compounds",
                         round(multi["std"].median(), 3))
                    show("mean pIC50 std across multi-measurement compounds",
                         round(multi["std"].mean(), 3))
                    show("p95 pIC50 std", round(multi["std"].quantile(0.95), 3))

        report["Q2_replicate_coverage"] = {
            "replicates_n_rows": int(len(replicates)),
            "replicates_n_unique_ids": int(len(rep_ids)),
            "covers_n_of_t1_non_covalent": int(in_noncov),
            "covers_n_of_classification":  int(in_class),
            "covers_both_sets":            int(in_both),
            "coverage_fraction_non_covalent": float(coverage_noncov),
            "coverage_fraction_classification": float(coverage_class),
        }
    else:
        report["Q2_replicate_coverage"] = {"error": "pad_replicates_full.csv not loaded"}

    # ============== Regression × Classification overlap (stacking foundation) ==============
    section("Regression set × Classification set overlap")
    reg_set = sets["t1_non_covalent"]
    cls_set = sets["classification"]
    overlap = reg_set & cls_set
    show("regression set (non_covalent)", len(reg_set))
    show("classification set",            len(cls_set))
    show("intersection",                  len(overlap))
    show("reg-only",                      len(reg_set - cls_set))
    show("cls-only",                      len(cls_set - reg_set))
    show("union",                         len(reg_set | cls_set))
    show("fraction of reg set covered by cls",
         f"{len(overlap) / max(1, len(reg_set)):.3f}")
    show("fraction of cls set covered by reg",
         f"{len(overlap) / max(1, len(cls_set)):.3f}")
    report["task_overlap"] = {
        "regression_n":   len(reg_set),
        "classification_n": len(cls_set),
        "intersection_n": len(overlap),
        "regression_only_n": len(reg_set - cls_set),
        "classification_only_n": len(cls_set - reg_set),
        "union_n": len(reg_set | cls_set),
    }

    # ============== Activity cliffs scope check ==============
    section("Activity cliffs file: scope inside Paper 1 modeling sets")
    if cliffs is not None:
        # cliffs has inchikey_1 and inchikey_2 (27-char). Truncate to 14.
        cliff_ids = set()
        for col in cliffs.columns:
            if "inchikey" in col.lower():
                cliff_ids |= set(cliffs[col].dropna().astype(str).str[:14])
        show("unique InChIKey-14 in activity_cliffs", len(cliff_ids))
        show("cliff compounds in t1_non_covalent",
             len(cliff_ids & sets["t1_non_covalent"]))
        show("cliff compounds in classification",
             len(cliff_ids & sets["classification"]))
        show("cliff compounds OUTSIDE t1_non_covalent",
             len(cliff_ids - sets["t1_non_covalent"]))
        # Severity distribution if present
        if "delta_pIC50" in cliffs.columns:
            show("delta_pIC50 mean",   round(cliffs["delta_pIC50"].mean(), 3))
            show("delta_pIC50 median", round(cliffs["delta_pIC50"].median(), 3))
            show("delta_pIC50 max",    round(cliffs["delta_pIC50"].max(), 3))
        if "tanimoto_similarity" in cliffs.columns:
            show("tanimoto mean", round(cliffs["tanimoto_similarity"].mean(), 3))
            show("tanimoto min",  round(cliffs["tanimoto_similarity"].min(), 3))
        report["cliffs_scope"] = {
            "n_pairs": int(len(cliffs)),
            "n_unique_compounds": int(len(cliff_ids)),
            "in_t1_non_covalent": int(len(cliff_ids & sets["t1_non_covalent"])),
            "in_classification": int(len(cliff_ids & sets["classification"])),
            "outside_t1_non_covalent": int(len(cliff_ids - sets["t1_non_covalent"])),
        }

    # ============== Source contribution to modeled sets ==============
    section("Source contribution to modeled sets")
    for set_name, df in [("t1_non_covalent", t1_noncov),
                          ("classification", classif)]:
        if df is None:
            continue
        source_cols = [c for c in df.columns
                       if c in ("source_list", "source_db", "source_count")]
        show(f"{set_name} candidate source columns", source_cols)
        if "source_list" in df.columns:
            # source_list is comma-separated string of sources
            flat = []
            for s in df["source_list"].dropna().astype(str):
                flat.extend([x.strip() for x in s.split(",") if x.strip()])
            counter = Counter(flat)
            show(f"{set_name} source token counts (compound-record level)",
                 dict(counter))
        if "source_count" in df.columns:
            show(f"{set_name} mean source_count per compound",
                 round(df["source_count"].mean(), 2))
            show(f"{set_name} compounds with source_count >= 2",
                 int((df["source_count"] >= 2).sum()))

    # ============== Covalent exclusion check ==============
    section("Covalent inhibitor exclusion check")
    if t1_noncov is not None and "is_covalent" in t1_noncov.columns:
        cov_in_noncov = int(t1_noncov["is_covalent"].astype(bool).sum())
        show("compounds with is_covalent=True inside pad_t1_non_covalent",
             cov_in_noncov)
        if cov_in_noncov > 0:
            show("WARNING", "pad_t1_non_covalent contains is_covalent=True rows")
    if t1_aggregated is not None and "is_covalent" in t1_aggregated.columns:
        cov_in_agg = int(t1_aggregated["is_covalent"].astype(bool).sum())
        show("compounds with is_covalent=True inside pad_t1_aggregated",
             cov_in_agg)
        show("expected diff (aggregated - non_covalent)",
             sizes["t1_aggregated"] - sizes["t1_non_covalent"]
             if sizes["t1_aggregated"] and sizes["t1_non_covalent"] else None)
    if t1_confirmed is not None and "is_covalent" in t1_confirmed.columns:
        cov_in_conf = int(t1_confirmed["is_covalent"].astype(bool).sum())
        show("compounds with is_covalent=True inside pad_t1_confirmed",
             cov_in_conf)
        report["confirmed_includes_covalents"] = bool(cov_in_conf > 0)

    # ============== Raw data scan ==============
    section("Raw data: file counts per source")
    raw_counts = {}
    for source_dir in sorted(RAW_ROOT.iterdir()) if RAW_ROOT.exists() else []:
        if not source_dir.is_dir():
            continue
        files = sorted(source_dir.glob("*"))
        info = {}
        for f in files:
            if f.is_file() and f.suffix in (".tsv", ".csv"):
                try:
                    # Just count rows quickly
                    with open(f, "rb") as fh:
                        n_lines = sum(1 for _ in fh)
                    info[f.name] = max(0, n_lines - 1)  # minus header
                except Exception as e:
                    info[f.name] = f"ERR: {e}"
        raw_counts[source_dir.name] = info
        show(source_dir.name, info)
    report["raw_counts"] = raw_counts

    # ============== Splits summary cross-check ==============
    section("Splits summary cross-check vs CSVs")
    summary_path = SPLITS_ROOT / "splits_summary.csv"
    if summary_path.exists():
        ssum = pd.read_csv(summary_path)
        show("splits_summary.csv shape", ssum.shape)
        show("columns", list(ssum.columns))
        for _, row in ssum.iterrows():
            print(f"    {dict(row)}", flush=True)
        report["splits_summary"] = ssum.to_dict(orient="records")

    # ============== Write report ==============
    json_path = OUT_ROOT / "data_intro_report.json"
    json_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote structured report to {json_path}", flush=True)

    # Pretty markdown answer to the two questions
    md = []
    md.append("# PAD4_BENCH Data Introspection — Paper 1 Scope Questions\n")
    md.append("## Q1: Confirmed split provenance\n")
    q1 = report.get("Q1_confirmed_provenance", {})
    md.append(f"- Confirmed regression split has **{q1.get('confirmed_split_n_unique_ids')}** unique compounds.")
    md.append(f"- Overlap with `pad_t1_confirmed`: {q1.get('overlap_with_t1_confirmed')}")
    md.append(f"- Overlap with `pad_t1_non_covalent`: {q1.get('overlap_with_t1_non_covalent')}")
    md.append(f"- Overlap with intersection: {q1.get('overlap_with_intersection_confirmed_AND_noncov')}")
    md.append(f"- Is split exactly == `pad_t1_confirmed`? **{q1.get('is_exactly_t1_confirmed')}**")
    md.append(f"- Is split exactly == intersection? **{q1.get('is_exactly_intersection')}**")
    md.append(f"- Is split ⊆ `pad_t1_non_covalent`? **{q1.get('is_subset_of_t1_non_covalent')}**")
    md.append(f"- Covalent compounds inside confirmed split: **{q1.get('covalent_compounds_in_confirmed_split')}**\n")
    md.append("### Conclusion (interpret manually):\n")
    md.append("If is_subset_of_t1_non_covalent == True AND covalent_compounds_in_confirmed_split == 0,")
    md.append("then the confirmed split is covalent-free and Paper 1's 'we exclude covalents' framing holds.\n")

    md.append("## Q2: Replicate coverage\n")
    q2 = report.get("Q2_replicate_coverage", {})
    md.append(f"- Replicates file: {q2.get('replicates_n_rows')} rows, {q2.get('replicates_n_unique_ids')} unique compounds.")
    md.append(f"- Covers {q2.get('covers_n_of_t1_non_covalent')} of {report['task_overlap'].get('regression_n')} regression compounds ({q2.get('coverage_fraction_non_covalent'):.1%}).")
    md.append(f"- Covers {q2.get('covers_n_of_classification')} of {report['task_overlap'].get('classification_n')} classification compounds ({q2.get('coverage_fraction_classification'):.1%}).\n")
    md.append("### Conclusion (interpret manually):\n")
    md.append("If coverage is >50% of modeled compounds, a noise-floor figure (D10) is valid.")
    md.append("If <50%, the noise floor figure should be labeled 'subset of compounds with replicate data'.\n")

    md_path = OUT_ROOT / "data_intro_report.md"
    md_path.write_text("\n".join(md))
    print(f"Wrote markdown summary to {md_path}", flush=True)
    print(f"\nDONE.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
