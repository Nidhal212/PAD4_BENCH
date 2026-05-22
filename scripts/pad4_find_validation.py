#!/usr/bin/env python3
"""
PAD4-Bench: deep search for the validation set (AID 1805620).

Goal: determine where, if anywhere, the validation records ended up.
Tries many column-name conventions and inspects every CSV in PAD_RESULTS
for traces of AID 1805620.

Run: python pad4_find_validation.py
Outputs:
  ./pad4_validation_search.md          (full diagnostic report)
  ./pad4_validation_search_SUMMARY.md  (short answers to the 5 gating
                                        questions from the v2 handoff)
"""

import argparse
import json
from pathlib import Path
from io import StringIO
import re

import pandas as pd

VALIDATION_AID = "1805620"


# ---------------------------------------------------------------------------
# Findings accumulator
# ---------------------------------------------------------------------------
# Each search_* function appends to this dict so main() can write a short
# summary at the end without re-parsing the long markdown.

def _new_findings():
    return {
        "replicates":      {"hit_columns": [], "tier_breakdown": {}, "total_hits": 0,
                            "file_present": False, "n_rows": None},
        "t1_aggregated":   {"hit_columns": [], "total_hits": 0,
                            "file_present": False, "n_rows": None},
        "classification":  {"is_validation_counts": {}, "hit_columns": [],
                            "total_hits": 0, "file_present": False, "n_rows": None},
        "pipeline_source": {"aid_mentions": [], "validation_aid_const": [],
                            "is_validation_writes": [], "any_match": False},
        "raw_input":       {"input_dir": None, "input_dir_exists": False,
                            "aid_file_exists": False, "aid_file_size": None,
                            "aid_file_rows": None},
        "tree_scan":       {"files_with_hits": []},  # list of (relpath, n_hits)
    }


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _safe_read(path, **kw):
    try:
        return pd.read_csv(path, low_memory=False, **kw)
    except Exception as e:
        return e


def _hit_summary(df, fname, hits, columns_searched, hit_columns):
    """Produce a brief summary of what we found in one file."""
    out = StringIO()
    out.write(f"### `{fname}` ({len(df):,} rows × {len(df.columns)} cols)\n\n")
    out.write(f"Columns searched: `{', '.join(columns_searched)}`\n\n")
    if hits == 0:
        out.write(f"**No rows match AID {VALIDATION_AID}.**\n\n")
        return out.getvalue()
    out.write(f"**{hits:,} rows match AID {VALIDATION_AID}** in column(s): `{', '.join(hit_columns)}`\n\n")
    return out.getvalue()


# ---------------------------------------------------------------------------
# Search functions (each returns (markdown, findings_update))
# ---------------------------------------------------------------------------

def search_replicates(base):
    """
    pad_replicates_full.csv is the key file: it should contain per-record
    source attribution before aggregation. If AID 1805620 records exist
    anywhere, they should be here.
    """
    out = StringIO()
    update = {"hit_columns": [], "tier_breakdown": {}, "total_hits": 0,
              "file_present": False, "n_rows": None}

    p = base / "pad_replicates_full.csv"
    if not p.exists():
        out.write("`pad_replicates_full.csv` not found.\n")
        return out.getvalue(), update

    df = _safe_read(p)
    if isinstance(df, Exception):
        out.write(f"Error reading: {df}\n")
        return out.getvalue(), update

    update["file_present"] = True
    update["n_rows"] = len(df)
    out.write(f"Shape: {df.shape}\n\n")
    out.write(f"All columns: `{', '.join(df.columns)}`\n\n")

    aid_like_cols = [c for c in df.columns
                     if any(k in c.lower() for k in
                            ["aid", "source", "assay", "origin", "ref",
                             "doc", "publication", "citation"])]
    out.write(f"AID-like columns: `{', '.join(aid_like_cols)}`\n\n")

    for col in aid_like_cols:
        ser = df[col].astype(str)
        mask = ser.str.contains(VALIDATION_AID, na=False, regex=False)
        n = int(mask.sum())
        out.write(f"  `{col}`: {n:,} rows match `{VALIDATION_AID}`\n")
        if n > 0:
            update["total_hits"] += n
            update["hit_columns"].append({"column": col, "n": n})
            out.write(f"\n  Top value counts in `{col}` for matched rows:\n```\n")
            out.write(df.loc[mask, col].value_counts().head(10).to_string())
            out.write("\n```\n\n")
            out.write("  Tier assignment of matched rows:\n```\n")
            for tcol in ["tier", "tier_subgroup", "fidelity_tier", "subset"]:
                if tcol in df.columns:
                    vc = df.loc[mask, tcol].value_counts()
                    out.write(f"  {tcol}: {vc.to_string()}\n")
                    update["tier_breakdown"][tcol] = vc.to_dict()
            out.write("\n```\n\n")

    for col in aid_like_cols:
        unique_vals = df[col].astype(str).unique()
        out.write(f"  All unique values in `{col}` (first 50): "
                  f"`{', '.join(sorted(map(str, unique_vals[:50])))}`\n\n")
    return out.getvalue(), update


def search_t1_aggregated(base):
    out = StringIO()
    update = {"hit_columns": [], "total_hits": 0,
              "file_present": False, "n_rows": None}

    p = base / "pad_t1_ic50_aggregated.csv"
    if not p.exists():
        out.write("`pad_t1_ic50_aggregated.csv` not found.\n")
        return out.getvalue(), update

    df = _safe_read(p)
    if isinstance(df, Exception):
        out.write(f"Error reading: {df}\n")
        return out.getvalue(), update

    update["file_present"] = True
    update["n_rows"] = len(df)
    out.write(f"Shape: {df.shape}\n\n")
    out.write(f"All columns: `{', '.join(df.columns)}`\n\n")

    hit_counts = {}
    for col in df.columns:
        try:
            ser = df[col].astype(str)
            n = int(ser.str.contains(VALIDATION_AID, na=False, regex=False).sum())
            if n > 0:
                hit_counts[col] = n
        except Exception:
            pass

    if hit_counts:
        out.write(f"**Columns containing AID {VALIDATION_AID}:**\n\n```\n")
        for col, n in sorted(hit_counts.items(), key=lambda x: -x[1]):
            out.write(f"  {col}: {n} rows\n")
            update["total_hits"] += n
            update["hit_columns"].append({"column": col, "n": n})
        out.write("```\n\n")
    else:
        out.write(f"**No column contains the substring `{VALIDATION_AID}`.**\n\n")

    for col in df.columns:
        if any(k in col.lower() for k in ["source", "aid", "assay", "tier",
                                           "validation", "subset", "origin"]):
            try:
                vc = df[col].value_counts(dropna=False).head(20)
                out.write(f"`{col}` value counts (top 20):\n```\n{vc.to_string()}\n```\n\n")
            except Exception:
                pass
    return out.getvalue(), update


def search_classification(base):
    out = StringIO()
    update = {"is_validation_counts": {}, "hit_columns": [],
              "total_hits": 0, "file_present": False, "n_rows": None}

    p = base / "pad_classification_v17.csv"
    if not p.exists():
        out.write("`pad_classification_v17.csv` not found.\n")
        return out.getvalue(), update

    df = _safe_read(p)
    if isinstance(df, Exception):
        out.write(f"Error reading: {df}\n")
        return out.getvalue(), update

    update["file_present"] = True
    update["n_rows"] = len(df)
    out.write(f"Shape: {df.shape}\n\n")
    out.write(f"All columns: `{', '.join(df.columns)}`\n\n")

    val_cols = [c for c in df.columns
                if any(k in c.lower() for k in ["valid", "is_val", "split",
                                                 "tier", "subset", "origin"])]
    for col in val_cols:
        try:
            vc = df[col].value_counts(dropna=False)
            out.write(f"**`{col}` value counts:**\n```\n{vc.to_string()}\n```\n\n")
            if "valid" in col.lower():
                update["is_validation_counts"][col] = {
                    str(k): int(v) for k, v in vc.to_dict().items()}
        except Exception:
            pass

    hit_counts = {}
    for col in df.columns:
        try:
            n = int(df[col].astype(str).str.contains(
                VALIDATION_AID, na=False, regex=False).sum())
            if n > 0:
                hit_counts[col] = n
        except Exception:
            pass
    if hit_counts:
        out.write(f"**Columns containing AID {VALIDATION_AID}:**\n```\n")
        for col, n in sorted(hit_counts.items(), key=lambda x: -x[1]):
            out.write(f"  {col}: {n}\n")
            update["total_hits"] += n
            update["hit_columns"].append({"column": col, "n": n})
        out.write("```\n\n")
    else:
        out.write(f"**No column contains the substring `{VALIDATION_AID}`.**\n\n")
    return out.getvalue(), update


def grep_pipeline_for_validation(base):
    out = StringIO()
    update = {"aid_mentions": [], "validation_aid_const": [],
              "is_validation_writes": [], "any_match": False}

    candidates = list(base.glob("*.py"))
    pipeline_candidates = [p for p in candidates
                           if "curation" in p.name.lower()
                           or "pipeline" in p.name.lower()
                           or "v17" in p.name.lower()]
    out.write(f"Python files found in {base}: {len(candidates)}\n")
    out.write(f"Pipeline candidates: {[str(p.name) for p in pipeline_candidates]}\n\n")

    any_match = False
    for path in pipeline_candidates + candidates[:20]:
        try:
            txt = path.read_text(errors="replace")
        except Exception as e:
            out.write(f"  `{path.name}`: cannot read ({e})\n")
            continue
        for m in re.finditer(r".*1805620.*", txt):
            line = m.group(0).strip()[:200]
            out.write(f"`{path.name}`: `{line}`\n")
            update["aid_mentions"].append({"file": path.name, "line": line})
            any_match = True
        for m in re.finditer(
                r"^\s*(VALIDATION_AID[S]?|HELDOUT_AID|VAL_AID|EXTERNAL_AID)\s*=.*$",
                txt, re.MULTILINE):
            line = m.group(0).strip()[:200]
            out.write(f"`{path.name}`: `{line}`\n")
            update["validation_aid_const"].append({"file": path.name, "line": line})
            any_match = True
        for m in re.finditer(
                r"^\s*.*['\"]is_validation['\"].*$",
                txt, re.MULTILINE):
            line = m.group(0).strip()[:200]
            out.write(f"`{path.name}`: `{line}`\n")
            update["is_validation_writes"].append({"file": path.name, "line": line})
            any_match = True

    update["any_match"] = any_match
    if not any_match:
        out.write("No matches found in any .py file.\n")
    return out.getvalue(), update


def check_input_dir(base):
    out = StringIO()
    update = {"input_dir": None, "input_dir_exists": False,
              "aid_file_exists": False, "aid_file_size": None,
              "aid_file_rows": None}

    config_path = base / "pipeline_config.json"
    if not config_path.exists():
        return "pipeline_config.json missing\n", update
    config = json.loads(config_path.read_text())
    input_dir = Path(config.get("input_dir", ""))
    update["input_dir"] = str(input_dir)
    out.write(f"Pipeline `input_dir`: `{input_dir}`\n\n")

    if not input_dir.exists():
        out.write("Input directory does not currently exist on this machine.\n")
        return out.getvalue(), update

    update["input_dir_exists"] = True
    aid_file = input_dir / f"AID_{VALIDATION_AID}_datatable_all.csv"
    if aid_file.exists():
        update["aid_file_exists"] = True
        update["aid_file_size"] = aid_file.stat().st_size
        out.write(f"`{aid_file.name}` exists ({aid_file.stat().st_size:,} bytes).\n\n")
        try:
            df = pd.read_csv(aid_file, low_memory=False)
            update["aid_file_rows"] = len(df)
            out.write(f"Shape: {df.shape}\n")
            out.write(f"Columns (first 20): `{', '.join(df.columns[:20])}`\n\n")
            out.write("Sample (first 3 rows, selected columns):\n```\n")
            interesting = [c for c in df.columns
                          if any(k in c.lower() for k in
                                  ["smiles", "ic50", "activity", "outcome",
                                   "value", "qualifier", "type", "unit", "cid"])][:8]
            if interesting:
                out.write(df[interesting].head(3).to_string())
            out.write("\n```\n")
        except Exception as e:
            out.write(f"Could not parse: {e}\n")
    else:
        out.write(f"`{aid_file.name}` does NOT exist in input dir.\n")
    return out.getvalue(), update


def scan_all_csvs(base):
    out = StringIO()
    update = {"files_with_hits": []}

    csv_files = sorted(base.rglob("*.csv"))
    out.write(f"Total CSVs scanned: {len(csv_files)}\n\n")
    out.write("Files containing AID 1805620 anywhere:\n\n")
    out.write("File | Total hits across all string columns\n")
    out.write("--- | ---\n")
    found_any = False
    for path in csv_files:
        try:
            sz_mb = path.stat().st_size / (1024 ** 2)
        except Exception:
            continue
        if sz_mb > 100:
            out.write(f"`{path.relative_to(base)}` | SKIPPED ({sz_mb:.0f} MB)\n")
            continue
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        total = 0
        for col in df.columns:
            try:
                total += int(df[col].astype(str).str.contains(
                    VALIDATION_AID, na=False, regex=False).sum())
            except Exception:
                pass
        if total > 0:
            found_any = True
            relpath = str(path.relative_to(base))
            out.write(f"`{relpath}` | **{total}** hits\n")
            update["files_with_hits"].append({"path": relpath, "n": total})
    if not found_any:
        out.write("(none found in any non-giant CSV)\n")
    return out.getvalue(), update


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------

def _verdict(findings):
    """
    Translate the findings into one of the three outcomes named in §6 of
    the v2 handoff:
      A. Records found in some tier
      B. No records anywhere
      C. Records found but routed to T2/T3 (or other unexpected location)
    """
    rep = findings["replicates"]
    t1 = findings["t1_aggregated"]
    cls = findings["classification"]
    tree = findings["tree_scan"]

    any_dataset_hit = (rep["total_hits"] > 0
                       or t1["total_hits"] > 0
                       or cls["total_hits"] > 0
                       or len(tree["files_with_hits"]) > 0)

    # Also count "is_validation == True" rows in the classification file as a hit
    # for the validation flag, even if the AID substring isn't present.
    is_val_true = 0
    for col, counts in cls["is_validation_counts"].items():
        for k, v in counts.items():
            if k.lower() in ("true", "1", "yes"):
                is_val_true += v
    has_is_validation_true = is_val_true > 0

    if not any_dataset_hit and not has_is_validation_true:
        return ("B", "No records anywhere",
                "AID 1805620 does not appear in any searched dataset CSV "
                "and the `is_validation` column has no truthy rows.")

    # If hits exist, figure out where
    landing = []
    if rep["total_hits"] > 0:
        tier_str = ""
        if rep["tier_breakdown"]:
            tier_str = "; tiers: " + ", ".join(
                f"{tcol}={d}" for tcol, d in rep["tier_breakdown"].items())
        landing.append(f"replicates file ({rep['total_hits']} rows{tier_str})")
    if t1["total_hits"] > 0:
        landing.append(f"T1 aggregated ({t1['total_hits']} rows)")
    if cls["total_hits"] > 0:
        landing.append(f"classification ({cls['total_hits']} rows)")
    if has_is_validation_true:
        landing.append(f"`is_validation==True` rows ({is_val_true})")
    if tree["files_with_hits"]:
        extras = [f"{f['path']}({f['n']})" for f in tree["files_with_hits"]]
        landing.append("tree scan: " + ", ".join(extras))

    # Heuristic: if classification file has is_validation==True AND AID hits,
    # the validation set was constructed correctly → outcome A.
    if has_is_validation_true and t1["total_hits"] > 0:
        return ("A", "Records found, validation flag populated",
                "Records exist in the dataset and the `is_validation` column "
                "is populated. Validation set construction appears intact.")

    # If hits exist but there is no is_validation==True, the records were
    # ingested but ended up in some tier (likely T2/T3), not flagged → outcome C.
    return ("C", "Records found, but routed to a tier (no validation flag)",
            "AID 1805620 records appear in the dataset, but the "
            "`is_validation` column is empty. Records were ingested and "
            "tier-assigned rather than held out. Landing locations: "
            + "; ".join(landing) + ".")


def write_summary(findings, out_path):
    code, headline, detail = _verdict(findings)

    s = StringIO()
    s.write("# PAD4-Bench validation set search — SHORT SUMMARY\n\n")
    s.write(f"Generated by `pad4_find_validation.py`. ")
    s.write("See `pad4_validation_search.md` for full diagnostic output.\n\n")
    s.write("---\n\n")

    s.write(f"## Verdict: outcome **{code}** — {headline}\n\n")
    s.write(f"{detail}\n\n")
    s.write("Mapping to v2 handoff §6:\n\n")
    s.write("- **A** = Records found in some tier (validation flag populated). "
            "§1/§4/§6.3 framing stands; §2 introduces the validation set normally.\n")
    s.write("- **B** = No records anywhere. Drop \"external held-out validation set\" "
            "language from §1; do not mention in §2; drop §6.3 validation subsection.\n")
    s.write("- **C** = Records found, routed elsewhere (no validation flag). "
            "§2 explains honestly what happened; §1/§4/§6.3 reframed.\n\n")

    s.write("---\n\n")
    s.write("## Five gating answers\n\n")

    rep = findings["replicates"]
    s.write("**1. Does AID 1805620 appear in `pad_replicates_full.csv` "
            "(per-record, pre-aggregation)?**\n\n")
    if not rep["file_present"]:
        s.write("- File not found.\n")
    elif rep["total_hits"] == 0:
        s.write(f"- No. File has {rep['n_rows']:,} rows; "
                f"no AID-like column contains the substring.\n")
    else:
        cols = ", ".join(f"`{c['column']}` ({c['n']})" for c in rep["hit_columns"])
        s.write(f"- Yes. {rep['total_hits']} rows match in: {cols}.\n")
        if rep["tier_breakdown"]:
            s.write("- Tier assignment of matched rows:\n")
            for tcol, d in rep["tier_breakdown"].items():
                s.write(f"  - `{tcol}`: {d}\n")
    s.write("\n")

    t1 = findings["t1_aggregated"]
    s.write("**2. Does AID 1805620 appear in `pad_t1_ic50_aggregated.csv` (T1)?**\n\n")
    if not t1["file_present"]:
        s.write("- File not found.\n")
    elif t1["total_hits"] == 0:
        s.write(f"- No. File has {t1['n_rows']:,} rows; "
                f"no column contains the substring.\n")
    else:
        cols = ", ".join(f"`{c['column']}` ({c['n']})" for c in t1["hit_columns"])
        s.write(f"- Yes. {t1['total_hits']} rows match in: {cols}.\n")
    s.write("\n")

    cls = findings["classification"]
    s.write("**3. Classification file: `is_validation` state and AID hits?**\n\n")
    if not cls["file_present"]:
        s.write("- File not found.\n")
    else:
        s.write(f"- Rows: {cls['n_rows']:,}.\n")
        if cls["is_validation_counts"]:
            for col, counts in cls["is_validation_counts"].items():
                s.write(f"- `{col}` value counts: {counts}\n")
        else:
            s.write("- No `is_validation`-style column found.\n")
        if cls["total_hits"] == 0:
            s.write("- No column contains the substring `1805620`.\n")
        else:
            cols = ", ".join(f"`{c['column']}` ({c['n']})" for c in cls["hit_columns"])
            s.write(f"- Substring `1805620` found in: {cols}.\n")
    s.write("\n")

    src = findings["pipeline_source"]
    s.write("**4. Does the v17 source code reference VALIDATION_AID or 1805620?**\n\n")
    if not src["any_match"]:
        s.write("- No matches in any `.py` file scanned. "
                "(Either the v17 file is not in the search path, "
                "or no validation-set logic was implemented.)\n")
    else:
        if src["validation_aid_const"]:
            s.write("- Constant definitions:\n")
            for m in src["validation_aid_const"]:
                s.write(f"  - `{m['file']}`: `{m['line']}`\n")
        if src["aid_mentions"]:
            s.write(f"- Mentions of `1805620` ({len(src['aid_mentions'])}):\n")
            for m in src["aid_mentions"][:10]:
                s.write(f"  - `{m['file']}`: `{m['line']}`\n")
            if len(src["aid_mentions"]) > 10:
                s.write(f"  - ... and {len(src['aid_mentions']) - 10} more "
                        "(see full report).\n")
        if src["is_validation_writes"]:
            s.write(f"- `is_validation` references "
                    f"({len(src['is_validation_writes'])}):\n")
            for m in src["is_validation_writes"][:10]:
                s.write(f"  - `{m['file']}`: `{m['line']}`\n")
            if len(src["is_validation_writes"]) > 10:
                s.write(f"  - ... and {len(src['is_validation_writes']) - 10} more.\n")
    s.write("\n")

    raw = findings["raw_input"]
    s.write("**5. Is the raw AID 1805620 input file still on disk?**\n\n")
    s.write(f"- Pipeline `input_dir`: `{raw['input_dir']}`\n")
    if not raw["input_dir_exists"]:
        s.write("- Input directory does not exist on this machine.\n")
    elif not raw["aid_file_exists"]:
        s.write("- Directory exists, but `AID_1805620_datatable_all.csv` is not in it.\n")
    else:
        s.write(f"- Yes. Size: {raw['aid_file_size']:,} bytes; "
                f"rows: {raw['aid_file_rows']}.\n")
    s.write("\n")

    s.write("---\n\n")
    s.write("## Tree scan\n\n")
    tree = findings["tree_scan"]
    if not tree["files_with_hits"]:
        s.write("No CSV in the tree contains the substring `1805620` "
                "(excluding files >100 MB which were skipped).\n")
    else:
        s.write("CSVs containing `1805620`:\n\n")
        for f in tree["files_with_hits"]:
            s.write(f"- `{f['path']}`: {f['n']} hits\n")
    s.write("\n")

    Path(out_path).write_text(s.getvalue())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/home/nidhal/PAD_RESULTS")
    ap.add_argument("--out",  default="./pad4_validation_search.md")
    ap.add_argument("--summary-out", default="./pad4_validation_search_SUMMARY.md")
    args = ap.parse_args()
    base = Path(args.base).expanduser().resolve()

    parts = [f"# PAD4-Bench: validation set deep search\n\n"
             f"Searching for AID **{VALIDATION_AID}** across `{base}`.\n"
             f"Generated by `pad4_find_validation.py`.\n\n"
             f"See `{Path(args.summary_out).name}` for the short version.\n\n---\n"]

    findings = _new_findings()

    sections = [
        ("1. Replicates file (per-record, pre-aggregation)", search_replicates,    "replicates"),
        ("2. T1 aggregated file (post-aggregation T1)",      search_t1_aggregated, "t1_aggregated"),
        ("3. Classification file",                            search_classification,"classification"),
        ("4. Grep .py files for VALIDATION_AID logic",        grep_pipeline_for_validation, "pipeline_source"),
        ("5. Raw input directory (does the AID file exist?)", check_input_dir,      "raw_input"),
        ("6. Last-resort scan of every CSV in tree",          scan_all_csvs,        "tree_scan"),
    ]

    for title, fn, key in sections:
        print(f"  {title}")
        try:
            body, update = fn(base)
            findings[key] = update
        except Exception as e:
            import traceback
            body = f"ERROR: {e}\n```\n{traceback.format_exc()}\n```\n"
        parts.append(f"\n## {title}\n\n{body}\n")

    out_path = Path(args.out)
    out_path.write_text("".join(parts))
    print(f"\nFull report:    {out_path.resolve()}")

    summary_path = Path(args.summary_out)
    write_summary(findings, summary_path)
    print(f"Short summary:  {summary_path.resolve()}")


if __name__ == "__main__":
    main()