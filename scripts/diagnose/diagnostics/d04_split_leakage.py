#!/usr/bin/env python3
"""
d04 — Cross-split leakage and difficulty diagnostic.

Question this answers: given that the v6 splitter produces five split
protocols (scaffold, random, similarity, confirmed, lead_opt) with
strict structural guarantees (scaffold-disjointness, InChIKey-14
disjointness, activity-imbalance audits), are the resulting test sets
actually distributed across a difficulty gradient — and does the
dominant-scaffold concern (190-compound family) survive scrutiny?

What this script does NOT do:
  - Re-validate scaffold disjointness. v6 already raises on leakage.
  - Re-validate InChIKey-14 disjointness. Same.
  - Re-validate activity drift. Same.
  Those audits live in pad_split_v6.py and run automatically when
  splits are produced. d04 reads splits_v6/ as a snapshot and reports
  metrics v6 doesn't compute.

What this script DOES:
  1. Cross-split nearest-neighbor Tanimoto distribution. For each
     test set, compute every test compound's best Tanimoto to its
     own training set. Reports mean / median / 10th / 90th percentile,
     and fraction of test compounds with NN < 0.40 (genuinely OOD).

  2. Activity cliff leakage. From pad_activity_cliffs.csv, count
     cliff pairs that straddle the train/test boundary in each split
     (one compound in train, the other in test). A model can
     memorize the activity relationship from such pairs, inflating
     the apparent difficulty of "predicting" the test compound.

  3. Dominant-scaffold coverage. For each split, where do the top-5
     largest scaffolds end up (train / val / test)? This directly
     addresses the reviewer's "190-compound family memorization"
     concern.

  4. Per-split scaffold-disjointness summary. Reports, doesn't enforce
     (v6 already enforces). Documents the design decision per split.

Outputs:
  manuscript/figures/d04_split_leakage.png — three-panel figure:
    (a) NN-Tanimoto distribution per split (boxplot)
    (b) Activity-cliff straddling counts per split (bar)
    (c) Top-5 scaffold coverage per split (stacked bar)
  results/diagnostics/d04_split_leakage.json — full numbers

This script does NOT modify the splits. It reads splits_v6/ as
input and produces a report. To regenerate splits, run pad_split_v6.py.
"""
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    OKABE_ITO, default_input_dir, find_repo_root, log_table,
    save_figure, save_summary, setup_logging, setup_matplotlib,
)

NAME = "d04_split_leakage"

# Order matters for the figures: easy → hard, so the difficulty gradient reads left-to-right.
SPLIT_ORDER = ["random", "lead_opt", "scaffold", "confirmed", "similarity"]
SPLIT_COLOR = {
    "random":     OKABE_ITO[2],   # sky blue (in-distribution)
    "lead_opt":   OKABE_ITO[1],   # orange (within-scaffold)
    "scaffold":   OKABE_ITO[5],   # blue (chemotype OOD)
    "confirmed":  OKABE_ITO[3],   # green (high-confidence subset)
    "similarity": OKABE_ITO[6],   # vermilion (Butina OOD)
}

# Required columns in each split CSV
REQUIRED_COLS = ("inchikey", "canonical_smiles")


# ── RDKit (required for fingerprints) ─────────────────────────────────
try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    from rdkit.DataStructs import ConvertToNumpyArray
    RDLogger.logger().setLevel(RDLogger.ERROR)
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


def _featurize(smiles: List[str], n_bits: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
    """ECFP4 with chirality. Returns (fps_matrix, valid_mask)."""
    if not HAS_RDKIT:
        raise RuntimeError("RDKit required for d04")
    arr = np.zeros((len(smiles), n_bits), dtype=np.uint8)
    valid = np.zeros(len(smiles), dtype=bool)
    for i, smi in enumerate(smiles):
        if not isinstance(smi, str):
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=2, nBits=n_bits, useChirality=True)
        ConvertToNumpyArray(fp, arr[i])
        valid[i] = True
    return arr, valid


def _tanimoto_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise Tanimoto between row sets (matches v6's implementation)."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    dot = a @ b.T
    sum_a = a.sum(1, keepdims=True)
    sum_b = b.sum(1, keepdims=True).T
    denom = sum_a + sum_b - dot
    denom = np.where(denom == 0, 1e-9, denom)
    return (dot / denom).astype(np.float32)


def _nn_tanimoto(query_fps: np.ndarray, ref_fps: np.ndarray,
                 batch: int = 512) -> np.ndarray:
    """For each query, return its best Tanimoto to any reference."""
    n = len(query_fps)
    if n == 0 or len(ref_fps) == 0:
        return np.array([])
    out = np.zeros(n, dtype=np.float32)
    for s in range(0, n, batch):
        e = min(s + batch, n)
        sim = _tanimoto_matrix(query_fps[s:e], ref_fps)
        out[s:e] = sim.max(axis=1)
    return out


# ── Loaders ───────────────────────────────────────────────────────────
def _load_split(split_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load train/val/test CSVs from a split directory.
    Prefers test_locked.csv over test.csv if both exist."""
    out = {}
    for sub in ("train", "val", "test"):
        if sub == "test":
            path = split_dir / "test_locked.csv"
            if not path.exists():
                path = split_dir / "test.csv"
        else:
            path = split_dir / f"{sub}.csv"
        if not path.exists():
            logging.warning(f"  {split_dir.name}/{sub}: {path.name} missing")
            continue
        out[sub] = pd.read_csv(path)
    return out


def _load_cliffs(input_dir: Path) -> pd.DataFrame:
    """Load activity cliff pairs from the curation outputs.

    The v17 pipeline writes pad_activity_cliffs.csv with columns
    inchikey_1 / inchikey_2 (full InChIKeys, not the truncated 14-char form).
    Earlier formats may have used different names, so we tolerate
    several conventions and normalize to inchikey_1 / inchikey_2 here.
    """
    path = input_dir / "pad_activity_cliffs.csv"
    if not path.exists():
        logging.warning(f"  {path} missing — cliff analysis will be skipped")
        return pd.DataFrame()
    df = pd.read_csv(path)
    cols = set(df.columns)
    # Already in canonical form?
    if {"inchikey_1", "inchikey_2"}.issubset(cols):
        return df
    # Try tolerated alternatives
    for a, b in [("inchikey_a", "inchikey_b"),
                 ("ik_1", "ik_2"),
                 ("ik_a", "ik_b"),
                 ("compound_a", "compound_b")]:
        if {a, b}.issubset(cols):
            df = df.rename(columns={a: "inchikey_1", b: "inchikey_2"})
            return df
    logging.error(
        f"  pad_activity_cliffs.csv schema not recognized. Found columns: "
        f"{sorted(cols)}. Expected pair columns like inchikey_1/inchikey_2."
    )
    return pd.DataFrame()


# ── Main analyses ─────────────────────────────────────────────────────
def _nn_tanimoto_per_split(splits: Dict[str, Dict[str, pd.DataFrame]],
                            smiles_col: str) -> Dict[str, dict]:
    """For each split, compute NN-Tanimoto from test → train."""
    results = {}
    for split_name, subsets in splits.items():
        if "train" not in subsets or "test" not in subsets:
            continue
        train_df = subsets["train"]
        test_df  = subsets["test"]
        if smiles_col not in train_df.columns or smiles_col not in test_df.columns:
            logging.warning(f"  {split_name}: no {smiles_col} column")
            continue

        logging.info(f"  {split_name}: featurizing {len(test_df):,} test "
                     f"+ {len(train_df):,} train")
        train_fps, tr_valid = _featurize(train_df[smiles_col].tolist())
        test_fps,  te_valid = _featurize(test_df[smiles_col].tolist())
        train_fps = train_fps[tr_valid]
        test_fps  = test_fps[te_valid]
        if len(train_fps) == 0 or len(test_fps) == 0:
            continue
        nn = _nn_tanimoto(test_fps, train_fps)
        results[split_name] = {
            "n_test_valid": int(len(nn)),
            "mean":   round(float(nn.mean()),   4),
            "median": round(float(np.median(nn)), 4),
            "p10":    round(float(np.percentile(nn, 10)), 4),
            "p25":    round(float(np.percentile(nn, 25)), 4),
            "p75":    round(float(np.percentile(nn, 75)), 4),
            "p90":    round(float(np.percentile(nn, 90)), 4),
            "max":    round(float(nn.max()), 4),
            "frac_below_0.40": round(float((nn < 0.40).mean()), 4),
            "frac_below_0.30": round(float((nn < 0.30).mean()), 4),
            "frac_above_0.70": round(float((nn > 0.70).mean()), 4),
            "_nn_array": nn,   # popped later before JSON write
        }
    return results


def _cliff_leakage_per_split(splits: Dict[str, Dict[str, pd.DataFrame]],
                              cliffs: pd.DataFrame,
                              id_col: str) -> Dict[str, dict]:
    """Count activity-cliff pairs that straddle the train/test boundary."""
    if len(cliffs) == 0:
        return {}
    results = {}
    for split_name, subsets in splits.items():
        train_df = subsets.get("train", pd.DataFrame())
        test_df  = subsets.get("test",  pd.DataFrame())
        if id_col not in train_df.columns or id_col not in test_df.columns:
            continue
        tr_keys = set(train_df[id_col].astype(str))
        te_keys = set(test_df[id_col].astype(str))
        a = cliffs["inchikey_1"].astype(str)
        b = cliffs["inchikey_2"].astype(str)
        a_tr, a_te = a.isin(tr_keys), a.isin(te_keys)
        b_tr, b_te = b.isin(tr_keys), b.isin(te_keys)
        both_train  = (a_tr & b_tr).sum()
        both_test   = (a_te & b_te).sum()
        straddle    = ((a_tr & b_te) | (a_te & b_tr)).sum()
        either_only = a_tr | a_te | b_tr | b_te
        n_relevant  = int(either_only.sum())
        results[split_name] = {
            "n_total_cliffs":    int(len(cliffs)),
            "n_relevant_cliffs": n_relevant,
            "both_in_train":     int(both_train),
            "both_in_test":      int(both_test),
            "straddle_train_test": int(straddle),
            "straddle_pct_of_relevant": round(
                float(straddle / max(n_relevant, 1)), 4),
        }
    return results


def _dominant_scaffold_coverage(splits: Dict[str, Dict[str, pd.DataFrame]],
                                 scaffold_col: str,
                                 t1_df: pd.DataFrame,
                                 top_n: int = 5) -> Dict[str, dict]:
    """For each split, where do the top-N largest T1 scaffolds end up?"""
    if scaffold_col not in t1_df.columns:
        return {}
    counts = t1_df[scaffold_col].value_counts()
    top_scaffolds = counts.head(top_n).index.tolist()

    results = {}
    for split_name, subsets in splits.items():
        if scaffold_col not in subsets.get("train", pd.DataFrame()).columns:
            continue
        per_scaffold = []
        for rank, scaf in enumerate(top_scaffolds, 1):
            n_train = int((subsets.get("train", pd.DataFrame()).get(
                scaffold_col, pd.Series(dtype=str)) == scaf).sum())
            n_val   = int((subsets.get("val", pd.DataFrame()).get(
                scaffold_col, pd.Series(dtype=str)) == scaf).sum())
            n_test  = int((subsets.get("test", pd.DataFrame()).get(
                scaffold_col, pd.Series(dtype=str)) == scaf).sum())
            per_scaffold.append({
                "rank":    rank,
                "size_t1": int(counts.iloc[rank - 1]),
                "n_train": n_train,
                "n_val":   n_val,
                "n_test":  n_test,
                "test_fraction": round(
                    n_test / max(n_train + n_val + n_test, 1), 4),
            })
        results[split_name] = per_scaffold
    return results


# ── Figure ────────────────────────────────────────────────────────────
def _make_figure(plt, nn_stats: dict, cliff_stats: dict,
                 scaffold_cov: dict, top_n: int) -> "Figure":
    """Three-panel figure: NN distribution, cliff straddling, scaffold coverage."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # (a) NN-Tanimoto boxplot per split
    ax = axes[0]
    splits_present = [s for s in SPLIT_ORDER if s in nn_stats]
    box_data = [nn_stats[s]["_nn_array"] for s in splits_present]
    box_labels = [
        f"{s}\n(n={nn_stats[s]['n_test_valid']:,})" for s in splits_present
    ]
    box_colors = [SPLIT_COLOR[s] for s in splits_present]
    if box_data:
        try:
            bp = ax.boxplot(box_data, tick_labels=box_labels, widths=0.55,
                            patch_artist=True, showfliers=False)
        except TypeError:
            bp = ax.boxplot(box_data, labels=box_labels, widths=0.55,
                            patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.axhline(0.40, color="black", linewidth=0.7, linestyle="--",
                   label="Tanimoto = 0.40 (OOD)")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("test compound NN-Tanimoto to train")
        ax.set_title("(a) test-set difficulty gradient\n"
                     "(lower = harder)")
        ax.legend(loc="lower left", fontsize=8)
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

    # (b) Activity-cliff straddling counts per split
    ax = axes[1]
    if cliff_stats:
        splits_c = [s for s in SPLIT_ORDER if s in cliff_stats]
        straddle_n = [cliff_stats[s]["straddle_train_test"] for s in splits_c]
        relevant_n = [cliff_stats[s]["n_relevant_cliffs"]    for s in splits_c]
        x = np.arange(len(splits_c))
        ax.bar(x, relevant_n, color="lightgray", edgecolor="white",
               linewidth=0.5, label="relevant cliff pairs")
        ax.bar(x, straddle_n, color=[SPLIT_COLOR[s] for s in splits_c],
               edgecolor="white", linewidth=0.5,
               label="straddling train/test")
        ax.set_xticks(x)
        ax.set_xticklabels(splits_c, rotation=15, ha="right")
        ax.set_ylabel("number of cliff pairs")
        ax.set_title("(b) activity cliff leakage\n"
                     "(straddling pairs share information across train/test)")
        ax.legend(loc="upper right", fontsize=8)
        # Annotate with straddle %
        for xi, sn, rn in zip(x, straddle_n, relevant_n):
            if rn > 0:
                pct = sn / rn * 100
                ax.text(xi, sn + max(relevant_n) * 0.02, f"{pct:.0f}%",
                        ha="center", va="bottom", fontsize=8)
    else:
        ax.text(0.5, 0.5, "pad_activity_cliffs.csv\nnot found",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("(b) activity cliff leakage")

    # (c) Top-N scaffold coverage per split (stacked bar)
    ax = axes[2]
    if scaffold_cov:
        splits_s = [s for s in SPLIT_ORDER if s in scaffold_cov]
        # For each split: total compounds in top-N scaffolds, split by subset
        train_totals = []
        val_totals   = []
        test_totals  = []
        for split in splits_s:
            tr = sum(d["n_train"] for d in scaffold_cov[split])
            va = sum(d["n_val"]   for d in scaffold_cov[split])
            te = sum(d["n_test"]  for d in scaffold_cov[split])
            train_totals.append(tr)
            val_totals.append(va)
            test_totals.append(te)
        x = np.arange(len(splits_s))
        ax.bar(x, train_totals,
               color=OKABE_ITO[5], edgecolor="white", linewidth=0.5,
               label="train")
        ax.bar(x, val_totals, bottom=train_totals,
               color=OKABE_ITO[1], edgecolor="white", linewidth=0.5,
               label="val")
        ax.bar(x, test_totals, bottom=np.array(train_totals) + np.array(val_totals),
               color=OKABE_ITO[6], edgecolor="white", linewidth=0.5,
               label="test")
        ax.set_xticks(x)
        ax.set_xticklabels(splits_s, rotation=15, ha="right")
        ax.set_ylabel(f"compounds in top-{top_n} T1 scaffolds")
        ax.set_title(f"(c) dominant-scaffold coverage\n"
                     f"(top-{top_n} largest T1 scaffolds)")
        ax.legend(loc="upper right", fontsize=8)
    else:
        ax.text(0.5, 0.5, "scaffold info unavailable",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"(c) top-{top_n} scaffold coverage")

    fig.tight_layout()
    return fig


# ── Main ──────────────────────────────────────────────────────────────
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits-dir", type=Path, default=None,
                        help="Splits directory (default: <repo>/splits_v6/)")
    parser.add_argument("--input-dir", type=Path, default=None,
                        help="Curated data dir (default: <repo>/data/processed/)")
    parser.add_argument("--t1-file", type=Path, default=None,
                        help="T1 file for scaffold rankings "
                             "(default: <input>/pad_t1_ic50_aggregated.csv)")
    parser.add_argument("--scaffold-col", default="stereo_stripped_scaffold",
                        help="Scaffold column name. Default "
                             "'stereo_stripped_scaffold' (the canonical SMILES) "
                             "matches across files. Avoid 'scaffold_id' here — "
                             "it's a file-local integer that won't be "
                             "consistent between splits CSVs and T1 reference.")
    parser.add_argument("--id-col", default="inchikey",
                        help="Compound ID column for cliff matching. "
                             "Default 'inchikey' matches the full InChIKeys "
                             "stored in pad_activity_cliffs.csv. Use "
                             "'inchikey_14' only if you've verified that's "
                             "what the cliffs file contains.")
    parser.add_argument("--smiles-col", default="canonical_smiles",
                        help="SMILES column for fingerprinting")
    parser.add_argument("--top-n", type=int, default=5,
                        help="Number of largest scaffolds to track")
    args = parser.parse_args(argv)

    setup_logging()
    plt = setup_matplotlib()

    if not HAS_RDKIT:
        logging.error("RDKit required for d04 — install via conda or pip")
        return 1

    # ── Resolve inputs ────────────────────────────────────────────────
    repo = find_repo_root()
    splits_dir = args.splits_dir or (repo / "splits_v6")
    input_dir  = args.input_dir  or default_input_dir()
    t1_file    = args.t1_file    or (input_dir / "pad_t1_ic50_aggregated.csv")

    if not splits_dir.exists():
        logging.error(f"Splits directory not found: {splits_dir}")
        logging.error("Run pad_split_v6.py first to produce splits.")
        return 1
    if not t1_file.exists():
        logging.error(f"T1 file not found: {t1_file}")
        return 1

    logging.info(f"Splits dir : {splits_dir}")
    logging.info(f"Input dir  : {input_dir}")
    logging.info(f"T1 file    : {t1_file}")

    # ── Load all available splits ────────────────────────────────────
    splits: Dict[str, Dict[str, pd.DataFrame]] = {}
    for split_name in SPLIT_ORDER:
        sd = splits_dir / split_name
        if sd.exists():
            data = _load_split(sd)
            if data:
                splits[split_name] = data
                sizes = " / ".join(f"{k}={len(v):,}" for k, v in data.items())
                logging.info(f"  loaded {split_name}: {sizes}")
    if not splits:
        logging.error("No splits found. Did you run pad_split_v6.py?")
        return 1

    t1_df = pd.read_csv(t1_file)
    logging.info(f"  loaded T1: {len(t1_df):,} compounds")

    # Check that the scaffold column exists in both T1 and at least one split
    if args.scaffold_col not in t1_df.columns:
        logging.error(
            f"Column '{args.scaffold_col}' not in T1 file ({t1_file}). "
            f"Available: {sorted(t1_df.columns)[:20]}..."
        )
        return 1
    sample_split = next(iter(splits.values()))
    sample_train = sample_split.get("train", pd.DataFrame())
    if args.scaffold_col not in sample_train.columns:
        logging.warning(
            f"Column '{args.scaffold_col}' not in splits CSVs. Available in "
            f"first split: {sorted(sample_train.columns)[:20]}... "
            f"Analysis 3 (dominant-scaffold coverage) will be skipped."
        )

    cliffs = _load_cliffs(input_dir)
    if len(cliffs):
        logging.info(f"  loaded {len(cliffs):,} activity cliff pairs")

    # ── Analysis 1: NN-Tanimoto distribution per split ───────────────
    logging.info("\n" + "=" * 60)
    logging.info("Analysis 1: NN-Tanimoto from test → train per split")
    logging.info("=" * 60)
    nn_stats = _nn_tanimoto_per_split(splits, args.smiles_col)
    rows = []
    for s in SPLIT_ORDER:
        if s not in nn_stats:
            continue
        d = nn_stats[s]
        rows.append((
            s,
            f"{d['n_test_valid']:,}",
            f"{d['mean']:.3f}",
            f"{d['median']:.3f}",
            f"{d['p10']:.3f}",
            f"{d['p90']:.3f}",
            f"{d['frac_below_0.40']*100:.1f}%",
        ))
    log_table(rows, ["split", "n_test", "mean", "median", "p10", "p90",
                     "% NN<0.40"])

    # ── Analysis 2: Activity cliff straddling ────────────────────────
    if len(cliffs):
        logging.info("\n" + "=" * 60)
        logging.info("Analysis 2: activity cliff leakage")
        logging.info("=" * 60)
        cliff_stats = _cliff_leakage_per_split(splits, cliffs, args.id_col)
        rows = []
        for s in SPLIT_ORDER:
            if s not in cliff_stats:
                continue
            d = cliff_stats[s]
            rows.append((
                s,
                f"{d['n_relevant_cliffs']:,}",
                f"{d['both_in_train']:,}",
                f"{d['both_in_test']:,}",
                f"{d['straddle_train_test']:,}",
                f"{d['straddle_pct_of_relevant']*100:.1f}%",
            ))
        log_table(rows, ["split", "relevant", "both_train", "both_test",
                         "straddle", "straddle %"])
    else:
        cliff_stats = {}

    # ── Analysis 3: Dominant scaffold coverage ───────────────────────
    logging.info("\n" + "=" * 60)
    logging.info(f"Analysis 3: top-{args.top_n} scaffold coverage per split")
    logging.info("=" * 60)
    scaffold_cov = _dominant_scaffold_coverage(
        splits, args.scaffold_col, t1_df, top_n=args.top_n)
    if scaffold_cov:
        rows = []
        for s in SPLIT_ORDER:
            if s not in scaffold_cov:
                continue
            for d in scaffold_cov[s]:
                rows.append((
                    s,
                    d["rank"],
                    f"{d['size_t1']:,}",
                    d["n_train"], d["n_val"], d["n_test"],
                    f"{d['test_fraction']*100:.0f}%",
                ))
        log_table(rows, ["split", "rank", "T1 size", "n_train", "n_val",
                         "n_test", "test %"])

    # ── Figure ────────────────────────────────────────────────────────
    fig = _make_figure(plt, nn_stats, cliff_stats, scaffold_cov, args.top_n)
    save_figure(fig, NAME)
    plt.close(fig)

    # ── Summary JSON ──────────────────────────────────────────────────
    # Drop the _nn_array fields before serializing
    nn_stats_clean = {
        k: {kk: vv for kk, vv in v.items() if kk != "_nn_array"}
        for k, v in nn_stats.items()
    }
    summary = {
        "n_splits_analyzed": len(splits),
        "splits_present": list(splits.keys()),
        "scaffold_col": args.scaffold_col,
        "id_col": args.id_col,
        "nn_tanimoto_per_split":  nn_stats_clean,
        "cliff_leakage_per_split": cliff_stats,
        "top_n_scaffold_coverage": scaffold_cov,
        "interpretation": _interpret(nn_stats_clean, cliff_stats),
    }
    save_summary(summary, NAME)
    return 0


def _interpret(nn_stats: dict, cliff_stats: dict) -> str:
    """Plain-language interpretation for the JSON summary."""
    if not nn_stats:
        return "No splits analyzed."
    parts = []

    # Difficulty gradient
    means = {s: nn_stats[s]["mean"] for s in nn_stats}
    if means:
        easy   = max(means, key=means.get)
        hard   = min(means, key=means.get)
        spread = means[easy] - means[hard]
        parts.append(
            f"Difficulty gradient: '{easy}' is the easiest split "
            f"(mean test→train Tanimoto = {means[easy]:.2f}); "
            f"'{hard}' is the hardest (mean = {means[hard]:.2f}). "
            f"Spread: {spread:.2f}."
        )

    # OOD fraction in hardest split
    if "similarity" in nn_stats:
        ood_frac = nn_stats["similarity"].get("frac_below_0.40", 0)
        parts.append(
            f"In the similarity split, {ood_frac*100:.0f}% of test "
            f"compounds have NN-Tanimoto < 0.40 (genuinely OOD)."
        )

    # Cliff leakage warning
    if cliff_stats:
        max_straddle = max(
            (d["straddle_pct_of_relevant"] for d in cliff_stats.values()),
            default=0,
        )
        if max_straddle > 0.30:
            worst = max(cliff_stats,
                        key=lambda s: cliff_stats[s]["straddle_pct_of_relevant"])
            parts.append(
                f"WARNING: {worst} has {max_straddle*100:.0f}% of relevant "
                f"activity cliff pairs straddling train/test. Models may "
                f"appear to handle activity cliffs well by memorizing the "
                f"in-train half of these pairs. Consider stratifying "
                f"benchmarks by 'cliff-straddling' vs 'cliff-internal' "
                f"test compounds."
            )

    return " ".join(parts)


if __name__ == "__main__":
    sys.exit(main())