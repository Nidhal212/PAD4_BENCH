#!/usr/bin/env python3
"""
make_submission_dir.py
----------------------
Build a clean submission snapshot of PAD4_BENCH into a sibling directory
called `paper_pad4bench_2026`.

DESIGN PRINCIPLES (read before running):
  * COPY-ONLY. This script never moves or deletes anything in the source.
    If it crashes mid-run, your original PAD4_BENCH/ is completely untouched.
  * IDEMPOTENT-ISH. If the target dir already exists it ABORTS rather than
    overwrite — delete the target yourself if you want a fresh run.
  * VERBOSE. Prints what it copies and skips; writes MANIFEST.txt at the end.

USAGE:
  Place this file at /home/nidhal/PAD4_BENCH/ and run from there:
      python make_submission_dir.py
  Or do a dry run first (recommended — copies nothing, just reports):
      python make_submission_dir.py --dry-run

The target is created NEXT TO the source, not inside it, so the snapshot
never tries to copy itself.
"""

from __future__ import annotations
import argparse
import fnmatch
import shutil
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# CONFIG — edit here if you disagree with any decision.
# --------------------------------------------------------------------------

TARGET_NAME = "paper_pad4bench_2026"

# Top-level items to COPY into the snapshot. Anything not listed is skipped.
# This is an allowlist on purpose: safer than a denylist for a release dir.
INCLUDE_TOPLEVEL = [
    "data",                 # all tiers incl. T2/T3/Ki (released, not modeled)
    "features_v18",         # canonical features
    "models_v1",            # all 190 model cells + results
    "paper",                # figures + tables asset store
    "paper_intro",          # 4 intro/audit reports
    "scripts",              # curation, featurization, split scripts
    "docs",                 # data provenance docs
    "tests",                # whatever test coverage exists
    "results",              # diagnostics + per-task READMEs
    # root-level python scripts:
    "covalent_inspection.py",
    "external_validation_inventory.py",
    "overnight_followup.py",
    "paper1_reviewer_proof.py",
    "paper_ad_stats.py",
    "paper_calibration_ad.py",
    "paper_data_figures.py",
    "paper_data_introspection.py",
    "paper_results_figures.py",
    "paper_tables.py",
    "reviewer_audit.py",
    "sweep_classification.py",
    "sweep_regression.py",
    # root-level config / metadata:
    "environment.yml",
    "requirements.txt",
    "README.md",
    "LICENSE",
    "run_featurize_v18.sh",
]

# Top-level items deliberately EXCLUDED. Listed only for the run log so you
# can see the decisions being made. Edit INCLUDE_TOPLEVEL to change them.
EXCLUDE_TOPLEVEL_NOTED = {
    "features_v17":    "superseded by features_v18 (provenance only)",
    "manuscript":      "stale scaffold + 4 outdated figures; paper/ is canonical",
    "smoke_test_xgb_regression.py": "smoke test, not a release artifact",
    "run_fix_a.sh":    "ad-hoc fix script, not a release artifact",
    ".git":            "version-control internals",
}

# Glob patterns pruned from every copied subtree (logs, pids, caches, etc.).
PRUNE_PATTERNS = [
    "*.log", "*.pid", "*.pyc", "*.pyo", "*.tmp", "*.swp",
    "__pycache__", ".ipynb_checkpoints", ".DS_Store", ".pytest_cache",
    "smoke_test*.py",
]


def is_pruned(path: Path) -> bool:
    """True if this file/dir name matches a prune pattern."""
    return any(fnmatch.fnmatch(path.name, pat) for pat in PRUNE_PATTERNS)


def copy_tree(src: Path, dst: Path, dry_run: bool, stats: dict) -> None:
    """Recursively copy src -> dst, skipping pruned names."""
    if is_pruned(src):
        stats["pruned"] += 1
        print(f"  prune   {src}")
        return
    if src.is_dir():
        if not dry_run:
            dst.mkdir(parents=True, exist_ok=True)
        for child in sorted(src.iterdir()):
            copy_tree(child, dst / child.name, dry_run, stats)
    else:
        stats["files"] += 1
        stats["bytes"] += src.stat().st_size
        if not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def main() -> int:
    ap = argparse.ArgumentParser(description="Build PAD4_BENCH submission snapshot.")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be copied; copy nothing")
    ap.add_argument("--source", default=".",
                    help="source PAD4_BENCH directory (default: current dir)")
    args = ap.parse_args()

    source = Path(args.source).resolve()
    target = source.parent / TARGET_NAME

    # --- safety checks -----------------------------------------------------
    if not source.is_dir():
        print(f"ERROR: source {source} is not a directory.", file=sys.stderr)
        return 1
    # confirm we are really in PAD4_BENCH by spot-checking a known item
    if not (source / "models_v1").is_dir() and not (source / "data").is_dir():
        print(f"ERROR: {source} does not look like PAD4_BENCH "
              f"(no models_v1/ or data/). Aborting so we don't snapshot the "
              f"wrong directory.", file=sys.stderr)
        return 1
    if target.exists():
        print(f"ERROR: target {target} already exists. Delete it yourself "
              f"and re-run, so nothing is silently overwritten.", file=sys.stderr)
        return 1

    print(f"Source : {source}")
    print(f"Target : {target}")
    print(f"Mode   : {'DRY RUN (no files written)' if args.dry_run else 'COPY'}")
    print("-" * 70)

    # report exclusions
    print("Deliberately excluded (edit INCLUDE_TOPLEVEL to change):")
    for name, reason in EXCLUDE_TOPLEVEL_NOTED.items():
        present = "present" if (source / name).exists() else "absent"
        print(f"  - {name:<32} [{present}]  {reason}")
    print("-" * 70)

    if not args.dry_run:
        target.mkdir(parents=True)

    stats = {"files": 0, "bytes": 0, "pruned": 0, "missing": 0}

    for name in INCLUDE_TOPLEVEL:
        src_item = source / name
        if not src_item.exists():
            stats["missing"] += 1
            print(f"  MISSING {name}  (listed in INCLUDE_TOPLEVEL but not on disk)")
            continue
        print(f"  copy    {name}")
        copy_tree(src_item, target / name, args.dry_run, stats)

    print("-" * 70)
    print(f"Files copied : {stats['files']}")
    print(f"Total size   : {human(stats['bytes'])}")
    print(f"Pruned items : {stats['pruned']}  (logs/pids/caches)")
    print(f"Missing items: {stats['missing']}  (check INCLUDE_TOPLEVEL if > 0)")

    # --- write manifest ----------------------------------------------------
    if not args.dry_run:
        manifest = target / "MANIFEST.txt"
        with manifest.open("w") as fh:
            fh.write("PAD4_BENCH submission snapshot\n")
            fh.write(f"Built from : {source}\n")
            fh.write(f"Files      : {stats['files']}\n")
            fh.write(f"Total size : {human(stats['bytes'])}\n")
            fh.write(f"Pruned     : {stats['pruned']} logs/pids/caches\n\n")
            fh.write("Top-level contents:\n")
            for child in sorted(target.iterdir()):
                kind = "dir " if child.is_dir() else "file"
                fh.write(f"  [{kind}] {child.name}\n")
        print(f"\nManifest written: {manifest}")
        print("\nNEXT STEPS (do yourself, verify before each):")
        print("  1. Inspect the snapshot:  ls -R paper_pad4bench_2026 | less")
        print("  2. Spot-check sizes match the original.")
        print("  3. Only then zip it:  zip -r paper_pad4bench_2026.zip "
              "paper_pad4bench_2026")
    else:
        print("\nDry run complete. Re-run without --dry-run to copy for real.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
