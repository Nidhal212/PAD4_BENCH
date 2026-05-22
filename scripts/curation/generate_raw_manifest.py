#!/usr/bin/env python3
"""
generate_raw_manifest.py
========================

Walks ~/PAD4_BENCH/data/raw/ and produces the canonical MANIFEST.md
plus hashes.txt.

VERSION 2: AID-to-role lookup is now hardcoded in this script (see
PUBCHEM_AID_ROLES below). This makes the manifest fully reproducible:
re-running the script regenerates an identical manifest, and any role
correction is a single dict edit in the script rather than a manual
edit of the output Markdown.

Run from anywhere; targets ~/PAD4_BENCH/data/raw/ unless --raw-dir given.

Outputs:
    <raw-dir>/MANIFEST.md
    <raw-dir>/hashes.txt
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# =============================================================================
# Canonical role assignments
# =============================================================================
# Roles for the 23 PubChem AIDs. Determined by inspecting the columns of
# each file (Standard Type, Standard Relation, IC50 vs. percent inhibition)
# and the row count. See the manuscript Table 1 for the full source manifest.
PUBCHEM_AID_ROLES: dict[int, str] = {
    # T3 — high-throughput screens (percent inhibition, large)
    485272:  "T3 input (HTS qHTS, primary)",
    463073:  "T3 input (HTS confirmation)",
    488796:  "T3 input (HTS counter-screen)",

    # T1 — confirmatory IC50 (Standard Type=IC50, Standard Relation == "=")
    1919095: "T1 input (confirmatory IC50)",
    1920200: "T1 input (confirmatory IC50)",
    1963715: "T1 input (confirmatory IC50)",
    1330527: "T1 input (confirmatory IC50, ChEMBL-mirrored)",
    1813806: "T1 input (confirmatory IC50, ChEMBL-mirrored)",
    1875531: "T1 input (confirmatory IC50, ChEMBL-mirrored)",
    2134413: "T1 input (confirmatory IC50, ChEMBL-mirrored)",

    # T1 — averaged IC50 with separate SD column (Format F in §3.1)
    492970:  "T1 input (averaged IC50 format)",

    # T1 — PAD isoform panel; needs demultiplexing by target
    588487:  "T1 input (PAD isoform panel; demuxed by target)",
    588560:  "T1 input (PAD isoform panel; demuxed by target)",

    # T1 + T2 mixed — ChEMBL-mirrored with `<` or `>` qualifiers in some rows
    1806182: "T1 + T2 input (ChEMBL-mirrored confirmatory)",
    1806183: "T1 + T2 input (ChEMBL-mirrored confirmatory)",
    1806764: "T1 + T2 input (ChEMBL-mirrored confirmatory)",
    1806765: "T1 + T2 input (ChEMBL-mirrored confirmatory)",
    1804546: "T1 + T2 input (ChEMBL-mirrored confirmatory)",
    1920046: "T1 + T2 input (ChEMBL-mirrored confirmatory)",
    2202442: "T1 + T2 input (ChEMBL-mirrored confirmatory)",
    2071731: "T1 + T2 input (ChEMBL-mirrored confirmatory; 1 censored)",

    # Ki only
    1804627: "Ki input",

    # Held-out external validation
    1805620: "**VALIDATION (held-out)**",
}


# Roles for non-PubChem files. Filename-based since BindingDB and ChEMBL
# files are organized by isoform name, not by AID.
def _role_for_isoform_file(name: str) -> str:
    if any(iso in name for iso in ["PAD1", "PAD2", "PAD3", "PAD6"]):
        return "Excluded (cross-isoform reference)"
    if "PAD4" in name:
        return "T1 + Ki input"
    return "?"


# Number of header rows in PubChem datatable_all CSV exports.
# Row 1 is the column header; rows 2-3 are PubChem metadata
# (Standard Type=STRING, Standard Relation=qualifier description, etc.).
# Subtracting 3 from total file lines (1 header + 3 metadata = 4) gives
# the count of actual data rows.
PUBCHEM_HEADER_ROWS = 3


# =============================================================================
# File walking and metadata extraction
# =============================================================================
def assign_role(path: Path) -> str:
    name = path.name
    parent = path.parent.name

    # PubChem AIDs: extract AID from filename and look up
    if name.startswith("AID_"):
        aid_str = name.replace("AID_", "").replace("_datatable_all.csv", "")
        try:
            aid = int(aid_str)
        except ValueError:
            log.warning(f"  could not parse AID from filename: {name}")
            return "?"
        if aid in PUBCHEM_AID_ROLES:
            return PUBCHEM_AID_ROLES[aid]
        log.warning(f"  AID {aid} not in PUBCHEM_AID_ROLES dict; "
                    f"add it to the script.")
        return f"? (AID {aid} not in role dict)"

    # BindingDB and ChEMBL: filename heuristic by isoform
    if parent in ("bindingdb", "chembl"):
        return _role_for_isoform_file(name)

    return "?"


def hash_file(path: Path, block_size: int = 65536) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            h.update(chunk)
    return h.hexdigest()


def line_count(path: Path) -> int:
    """Exact line count for files <50MB; estimated for larger."""
    size = path.stat().st_size
    if size > 50 * 1024 * 1024:
        with open(path, "rb") as f:
            sample = f.read(1024 * 1024)
        if not sample:
            return 0
        sample_lines = sample.count(b"\n")
        return int(sample_lines * (size / len(sample)))
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def data_row_count(path: Path, total_lines: int) -> int:
    """Approximate count of actual data rows (excluding headers).

    For PubChem AID files: subtract column header (1) + PubChem metadata
    rows (3) = 4 lines.
    For BindingDB/ChEMBL: subtract column header only.
    """
    if path.name.startswith("AID_"):
        return max(0, total_lines - 1 - PUBCHEM_HEADER_ROWS)
    return max(0, total_lines - 1)


def first_columns(path: Path, n: int = 3) -> str:
    """First n column names from a CSV/TSV header. Uses csv parser to
    handle quoted fields correctly."""
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            header = f.readline().strip()
        reader = csv.reader([header], delimiter=sep)
        cols = next(reader)[:n]
        return ", ".join(c.strip() for c in cols)
    except Exception as e:
        return f"(read error: {e})"


def fmt_size(b: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} TB"


def walk_raw(raw_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name in ["MANIFEST.md", "hashes.txt", "README.md"]:
            continue
        if any(part.startswith(".") for part in path.parts):
            continue
        log.info(f"  hashing {path.relative_to(raw_dir)}")
        total_lines = line_count(path)
        rows.append({
            "path":       str(path.relative_to(raw_dir)),
            "filename":   path.name,
            "size":       path.stat().st_size,
            "size_h":     fmt_size(path.stat().st_size),
            "lines":      total_lines,
            "data_rows":  data_row_count(path, total_lines),
            "first_cols": first_columns(path),
            "sha256":     hash_file(path),
            "role":       assign_role(path),
        })
    return rows


# =============================================================================
# Manifest writing
# =============================================================================
SCRIPT_VERSION = "2.0"


def categorize_role(role: str) -> str:
    """Bucket a fine-grained role into a coarse category for the tally."""
    if "VALIDATION" in role:
        return "Validation (held-out)"
    if "Excluded" in role:
        return "Excluded (cross-isoform)"
    if "T3 input" in role:
        return "T3 input"
    if "T1 + T2" in role:
        return "T1 + T2 mixed input"
    if "T1 + Ki" in role:
        return "T1 + Ki input"
    if "T1 input" in role:
        return "T1 input"
    if "Ki input" in role:
        return "Ki input"
    return "Other / unclassified"


def write_manifest(rows: list[dict], output_path: Path) -> None:
    md = []
    md.append("# Raw data manifest")
    md.append("")
    md.append(f"Source data files for PAD4-Bench curation. Generated by "
              f"`generate_raw_manifest.py` v{SCRIPT_VERSION}. The role "
              f"column reflects the canonical role assignments defined in "
              f"`PUBCHEM_AID_ROLES` in the generator script; do not edit "
              f"the role values here directly — edit the script and "
              f"regenerate.")
    md.append("")
    md.append(f"**Total files:** {len(rows)}")
    md.append(f"**Total size:** "
              f"{fmt_size(sum(r['size'] for r in rows))}")
    md.append(f"**Total data rows (excl. headers):** "
              f"{sum(r['data_rows'] for r in rows):,}")
    md.append("")
    md.append("## File listing")
    md.append("")
    md.append("| File | Size | Data rows | First columns | SHA-256 (first 16) | Role |")
    md.append("|------|------|----------:|---------------|--------------------|------|")
    for r in rows:
        sha_short = r["sha256"][:16]
        first_cols_short = r["first_cols"][:60] + ("…" if len(r["first_cols"]) > 60 else "")
        md.append(f"| `{r['path']}` | {r['size_h']} | {r['data_rows']:,} | "
                  f"`{first_cols_short}` | `{sha_short}` | {r['role']} |")
    md.append("")

    # Tally by role category
    role_tally = Counter(categorize_role(r["role"]) for r in rows)

    md.append("## Role tally")
    md.append("")
    md.append("| Role | n files |")
    md.append("|------|--------:|")
    for role, n in sorted(role_tally.items(), key=lambda kv: -kv[1]):
        md.append(f"| {role} | {n} |")
    md.append(f"| **Total** | **{len(rows)}** |")
    md.append("")

    md.append("## Provenance")
    md.append("")
    md.append("- **PubChem AIDs:** downloaded via the PubChem bioassay "
              "datatable export interface. See "
              "`docs/data_provenance/pubchem_bioassay_audit.json` for the "
              "audit log of the download session.")
    md.append("- **PubChem header rows:** PubChem datatable CSV exports "
              "include 3 metadata rows after the column header (Standard "
              "Type definition, qualifier description, taxonomy info). The "
              "ingestion stage of the curation pipeline must skip these. "
              "The 'Data rows' column above reports row counts after this "
              "subtraction.")
    md.append("- **ChEMBL:** queried target IDs for human PAD1-PAD6 isoforms. "
              "**[VERIFY: paste the exact ChEMBL target IDs queried — "
              "CHEMBL6111 for PAD4, plus PAD1/2/3/6 IDs.]**")
    md.append("- **BindingDB:** PAD-isoform extracts. "
              "**[VERIFY: extraction date.]**")
    md.append("")
    md.append("## Notes")
    md.append("")
    md.append("- **Cross-isoform exclusions:** the 8 cross-isoform files "
              "(`*_PAD1.tsv`, `*_PAD2.tsv`, `*_PAD3.tsv`, `*_PAD6.tsv` from "
              "both BindingDB and ChEMBL) are retained in the raw data for "
              "completeness but are excluded from PAD4 curation. They are "
              "available for future cross-isoform selectivity modeling.")
    md.append("- **Validation set:** AID 1805620 records are routed to "
              "`data/processed/pad4_validation.csv` and are explicitly "
              "excluded from all five split protocols' train and test "
              "partitions. This is enforced by an InChIKey-14 disjointness "
              "assertion in the curation pipeline.")
    md.append("- **PubChem mirror handling:** ChEMBL records are mirrored "
              "into PubChem AID-format records. The curation pipeline drops "
              "ChEMBL Source ID = 37 records at ingestion to prevent "
              "double-counting; see `pad4bench/curation/constants.py` "
              "(once Phase 2 is complete).")
    md.append("")
    md.append("## Verifying integrity")
    md.append("")
    md.append("To verify that the raw data files have not been modified "
              "since the manifest was generated:")
    md.append("")
    md.append("```bash")
    md.append("cd data/raw/")
    md.append("sha256sum -c hashes.txt")
    md.append("```")
    md.append("")

    output_path.write_text("\n".join(md))
    log.info(f"  wrote: {output_path}")


def write_hashes(rows: list[dict], output_path: Path) -> None:
    """sha256sum-format hash file. Compatible with `sha256sum -c`."""
    lines = []
    for r in rows:
        lines.append(f"{r['sha256']}  {r['path']}")
    output_path.write_text("\n".join(lines) + "\n")
    log.info(f"  wrote: {output_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", default="~/PAD4_BENCH/data/raw")
    p.add_argument("--manifest", default="MANIFEST.md")
    p.add_argument("--hashes", default="hashes.txt")
    args = p.parse_args()

    raw_dir = Path(args.raw_dir).expanduser().resolve()
    if not raw_dir.exists():
        log.error(f"Not found: {raw_dir}")
        sys.exit(1)

    log.info(f"Walking {raw_dir} (script v{SCRIPT_VERSION})")
    rows = walk_raw(raw_dir)
    if not rows:
        log.error("No files found.")
        sys.exit(1)

    log.info(f"\n[1/2] Writing manifest")
    write_manifest(rows, raw_dir / args.manifest)

    log.info(f"\n[2/2] Writing hashes file")
    write_hashes(rows, raw_dir / args.hashes)

    n_unknown = sum(1 for r in rows if r['role'].startswith('?'))
    log.info("\n" + "=" * 60)
    log.info(f"DONE.")
    log.info(f"  manifest: {raw_dir / args.manifest}")
    log.info(f"  hashes:   {raw_dir / args.hashes}")
    log.info(f"  files:    {len(rows)}")
    if n_unknown:
        log.warning(f"  unresolved roles: {n_unknown} (search '?' in MANIFEST.md)")
    else:
        log.info(f"  all roles assigned")
    log.info("=" * 60)


if __name__ == "__main__":
    main()