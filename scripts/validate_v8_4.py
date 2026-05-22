#!/usr/bin/env python3
"""
PAD4 v8.4 — Determinism + Multi-Seed Robustness Test
=====================================================
Verifies two things reviewers care about:

  (A) DETERMINISM: same seed → byte-identical splits across runs.
      Establishes that any reported numbers are reproducible.

  (B) ROBUSTNESS: split health is stable across seeds 42, 43, 44.
      A benchmark that swings wildly with seed is not a benchmark.
      We watch test-set size, scaffold count, KS stat, and (for
      cliff_aware) cliff_test_coverage_pct.

Usage:
  cd /home/nidhal/PAD4_BENCH
  python scripts/validate_v8_4.py

Run time: ~30 seconds on a 2.8K-compound dataset.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO        = Path("/home/nidhal/PAD4_BENCH")
SPLITTER    = REPO / "scripts" / "pad_split_v8_4.py"
SPLITS_BASE = REPO / "data" / "splits"
TMP_BASE    = REPO / "data" / "splits_validation"

SEEDS_TO_TEST = [42, 43, 44]
SPLITS = ["scaffold", "random", "similarity", "confirmed", "lead_opt", "cliff_aware"]


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(65536), b""):
            h.update(blk)
    return h.hexdigest()


def hash_split_dir(split_dir: Path) -> dict:
    """MD5 every CSV in a split dir, return {filename: md5}."""
    out = {}
    if not split_dir.exists():
        return out
    for csv in sorted(split_dir.glob("*.csv")):
        out[csv.name] = md5(csv)
    return out


def run_splitter(seed: int, output_dir: Path) -> int:
    """Run the splitter with given seed and output dir. Returns exit code."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(SPLITTER),
        "--mode", "both",
        "--seed", str(seed),
        "--output_dir", str(output_dir),
        "--force",
        "--cliff_pairs_file", str(REPO / "data/processed/pad_activity_cliffs.csv"),
    ]
    print(f"    Running seed={seed} -> {output_dir.name}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    FAILED (exit {result.returncode})")
        print(result.stderr[-2000:])
    return result.returncode


def read_diagnostics(output_dir: Path) -> dict:
    """Parse split_diagnostics.json for a given run."""
    p = output_dir / "split_diagnostics.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


# ─────────────────────────────────────────────────────────────────────
# (A) DETERMINISM CHECK
# ─────────────────────────────────────────────────────────────────────
def test_determinism():
    print("\n" + "=" * 60)
    print("(A) DETERMINISM: same seed twice should give identical CSVs")
    print("=" * 60)

    dir1 = TMP_BASE / "det_run1"
    dir2 = TMP_BASE / "det_run2"
    for d in (dir1, dir2):
        if d.exists():
            shutil.rmtree(d)

    rc1 = run_splitter(seed=42, output_dir=dir1)
    rc2 = run_splitter(seed=42, output_dir=dir2)
    if rc1 != 0 or rc2 != 0:
        print("  FAIL: splitter errored")
        return False

    all_match = True
    for split in SPLITS:
        h1 = hash_split_dir(dir1 / "regression" / split)
        h2 = hash_split_dir(dir2 / "regression" / split)
        if not h1 or not h2:
            print(f"    {split:<14} SKIP (no CSVs)")
            continue
        if h1 == h2:
            print(f"    {split:<14} OK (3 files md5-identical)")
        else:
            diff_files = [k for k in h1 if h1.get(k) != h2.get(k)]
            print(f"    {split:<14} FAIL: differs in {diff_files}")
            all_match = False

    # also check classification side
    for split in SPLITS:
        h1 = hash_split_dir(dir1 / "classification" / split)
        h2 = hash_split_dir(dir2 / "classification" / split)
        if not h1 or not h2:
            continue
        if h1 != h2:
            diff_files = [k for k in h1 if h1.get(k) != h2.get(k)]
            print(f"    classification/{split:<14} FAIL: {diff_files}")
            all_match = False

    print()
    print(f"  RESULT: {'PASS — fully deterministic' if all_match else 'FAIL — non-determinism present'}")
    return all_match


# ─────────────────────────────────────────────────────────────────────
# (B) MULTI-SEED ROBUSTNESS
# ─────────────────────────────────────────────────────────────────────
def test_robustness():
    print("\n" + "=" * 60)
    print(f"(B) ROBUSTNESS: split health across seeds {SEEDS_TO_TEST}")
    print("=" * 60)

    diag_per_seed = {}
    for seed in SEEDS_TO_TEST:
        out = TMP_BASE / f"seed_{seed}"
        if out.exists():
            shutil.rmtree(out)
        rc = run_splitter(seed=seed, output_dir=out)
        if rc != 0:
            print(f"  FAIL: seed {seed} errored")
            return False
        diag_per_seed[seed] = read_diagnostics(out)

    # Build a comparison table
    print(f"\n  Per-split metric stability across {len(SEEDS_TO_TEST)} seeds:")
    print()
    print(f"    {'split':<14} {'metric':<25} " +
          " ".join(f"seed={s:<6}" for s in SEEDS_TO_TEST) +
          "  spread")
    print("    " + "-" * 75)

    metrics_to_watch = [
        ("n_test",                   "test_size"),
        ("n_scaffolds_test",         "test_scaffolds"),
        ("ks_stat",                  "ks_stat"),
        ("activity_drift_pp",        "activity_drift"),
        ("health_score",             "health_score"),
    ]
    cliff_metrics = [
        ("cliff_test_count",         "cliff_count"),
        ("cliff_test_coverage_pct",  "cliff_coverage%"),
    ]

    issues = []
    for split in SPLITS:
        vals_by_metric = {}
        watch = metrics_to_watch + (cliff_metrics if split == "cliff_aware" else [])
        for key, label in watch:
            row = []
            for seed in SEEDS_TO_TEST:
                d = diag_per_seed[seed].get(split, {})
                v = d.get(key)
                row.append(v)
            vals_by_metric[label] = row

        for label, row in vals_by_metric.items():
            cells = []
            nums  = []
            for v in row:
                if v is None:
                    cells.append("    -   ")
                elif isinstance(v, float):
                    cells.append(f"{v:>9.3f}")
                    nums.append(v)
                else:
                    cells.append(f"{v:>9}")
                    if isinstance(v, (int, float)):
                        nums.append(float(v))
            spread = f"{max(nums)-min(nums):.2f}" if len(nums) >= 2 else "  -  "
            print(f"    {split:<14} {label:<25} " + " ".join(cells) + f"   {spread:>6}")

            # Flag big swings
            if label == "test_size" and len(nums) >= 2:
                if (max(nums) - min(nums)) / max(nums) > 0.10:
                    issues.append(f"{split}: test size varies >10% across seeds")
            if label == "ks_stat" and len(nums) >= 2:
                if max(nums) - min(nums) > 0.10:
                    issues.append(f"{split}: KS stat varies > 0.10 across seeds")
            if label == "health_score" and len(nums) >= 2:
                if max(nums) - min(nums) > 15:
                    issues.append(f"{split}: health score varies > 15 across seeds")
            if label == "cliff_coverage%" and len(nums) >= 2:
                if max(nums) - min(nums) > 10:
                    issues.append(f"cliff_aware: coverage varies > 10pp across seeds")
        print()

    if issues:
        print("  CONCERNS:")
        for i in issues:
            print(f"    !! {i}")
        print()
        print("  Some metrics are seed-sensitive. Consider reporting mean ± std")
        print("  across seeds in your paper, not single-seed numbers.")
        return False

    print("  RESULT: PASS — all metrics stable across seeds")
    print("  Single-seed (42) numbers are representative.")
    return True


# ─────────────────────────────────────────────────────────────────────
# CLEANUP
# ─────────────────────────────────────────────────────────────────────
def cleanup():
    print("\n" + "=" * 60)
    if TMP_BASE.exists():
        print(f"  Removing temporary validation runs: {TMP_BASE}")
        shutil.rmtree(TMP_BASE)
    print(f"  Production splits at {SPLITS_BASE} are untouched.")


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not SPLITTER.exists():
        print(f"ERROR: splitter not found at {SPLITTER}")
        sys.exit(1)

    det_ok = test_determinism()
    rob_ok = test_robustness()
    cleanup()

    print("\n" + "=" * 60)
    print(f"  Determinism: {'PASS' if det_ok else 'FAIL'}")
    print(f"  Robustness:  {'PASS' if rob_ok else 'WARN (see above)'}")
    print("=" * 60)
    sys.exit(0 if (det_ok and rob_ok) else 1)