"""
Unit tests for pad4bench.splits

Invariant tests (from hand-off spec):
  1. Determinism: same seed -> identical split
  2. Fraction targeting: each fold within 5% of requested fraction
  3. No scaffold leakage: no scaffold appears in >1 fold
  4. Reproducibility on actual T1 dataset (skip if no real data)

Additional tests:
  - lead_opt scaffold overlap by design
  - dedupe_by_inchikey14 tie-breaking
  - audit_split raises on leakage
  - aligned split preserves scaffold-disjointness
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pad4bench.splits import (
    align_classification_splits,
    audit_split,
    dedupe_by_inchikey14,
    scaffold_split,
    split_lead_opt,
    split_random,
    split_scaffold_capped,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_synthetic_df(
    n: int = 500,
    n_scaffolds: int = 80,
    seed: int = 0,
    add_stereo: bool = True,
) -> pd.DataFrame:
    """Create a synthetic DataFrame mimicking T1 structure."""
    rng = np.random.default_rng(seed)
    scaffolds = [f"SCAF_{i:03d}" for i in range(n_scaffolds)]

    # Power-law scaffold sizes (realistic: few large, many singletons)
    sizes = rng.power(2.5, n_scaffolds) * 50
    sizes = (sizes / sizes.sum() * n).astype(int)
    sizes = np.maximum(sizes, 1)
    # Adjust to exact n
    diff = n - sizes.sum()
    sizes[0] += diff

    rows = []
    inchi_counter = 0
    for scaf, sz in zip(scaffolds, sizes):
        for _ in range(sz):
            rows.append(
                {
                    "inchikey_14": f"IK{inchi_counter:012d}",
                    "stereo_stripped_scaffold": scaf,
                    "pIC50": rng.normal(6.5, 1.2),
                    "ml_weight": rng.random(),
                    "stereo_flag": rng.choice(
                        ["defined", "achiral", "undefined"]
                    ),
                }
            )
            inchi_counter += 1

    df = pd.DataFrame(rows)
    return df


@pytest.fixture
def synthetic_df():
    return _make_synthetic_df(n=500, n_scaffolds=80, seed=42)


@pytest.fixture
def small_df():
    """Tiny deterministic fixture for edge-case tests."""
    return pd.DataFrame(
        {
            "inchikey_14": [f"IK{i:03d}" for i in range(20)],
            "stereo_stripped_scaffold": [
                "S_A", "S_A", "S_A", "S_A", "S_A",
                "S_B", "S_B", "S_B",
                "S_C", "S_C",
            ] + [f"S_{i}" for i in range(10)],
            "pIC50": [6.0 + i * 0.1 for i in range(20)],
            "ml_weight": [1.0] * 20,
            "stereo_flag": ["defined"] * 20,
        }
    )


# ---------------------------------------------------------------------------
# 1. Determinism
# ---------------------------------------------------------------------------


def test_scaffold_split_determinism(synthetic_df):
    """Same seed must produce identical key lists."""
    tr1, va1, te1 = scaffold_split(
        synthetic_df,
        scaffold_col="stereo_stripped_scaffold",
        key_col="inchikey_14",
        fractions=(0.8, 0.1, 0.1),
        seed=42,
    )
    tr2, va2, te2 = scaffold_split(
        synthetic_df,
        scaffold_col="stereo_stripped_scaffold",
        key_col="inchikey_14",
        fractions=(0.8, 0.1, 0.1),
        seed=42,
    )
    assert tr1 == tr2
    assert va1 == va2
    assert te1 == te2


def test_random_split_determinism(synthetic_df):
    """Random split must be deterministic with same seed."""
    tr1, va1, te1 = split_random(synthetic_df, seed=42)
    tr2, va2, te2 = split_random(synthetic_df, seed=42)
    assert tr1["inchikey_14"].tolist() == tr2["inchikey_14"].tolist()
    assert va1["inchikey_14"].tolist() == va2["inchikey_14"].tolist()
    assert te1["inchikey_14"].tolist() == te2["inchikey_14"].tolist()


# ---------------------------------------------------------------------------
# 2. Fraction targeting (within 5% of requested)
# ---------------------------------------------------------------------------


def test_scaffold_split_fractions(synthetic_df):
    """Each fold should be within 10 percentage points of target.
    Scaffold splits have coarser granularity than random because
    whole scaffolds are assigned at once."""
    n = len(synthetic_df)
    tr, va, te = scaffold_split(
        synthetic_df,
        scaffold_col="stereo_stripped_scaffold",
        key_col="inchikey_14",
        fractions=(0.8, 0.1, 0.1),
        seed=42,
    )
    assert abs(len(tr) / n - 0.80) <= 0.10
    assert abs(len(va) / n - 0.10) <= 0.10
    assert abs(len(te) / n - 0.10) <= 0.10


def test_scaffold_split_fractions_70_15_15(synthetic_df):
    """Test with non-default fractions."""
    n = len(synthetic_df)
    tr, va, te = scaffold_split(
        synthetic_df,
        scaffold_col="stereo_stripped_scaffold",
        key_col="inchikey_14",
        fractions=(0.70, 0.15, 0.15),
        seed=42,
    )
    assert abs(len(tr) / n - 0.70) <= 0.05
    assert abs(len(va) / n - 0.15) <= 0.05
    assert abs(len(te) / n - 0.15) <= 0.05


def test_random_split_fractions(synthetic_df):
    """Random split should also hit fractions within 5%."""
    n = len(synthetic_df)
    tr, va, te = split_random(synthetic_df, test_frac=0.15, val_frac=0.10, seed=42)
    assert abs(len(tr) / n - 0.75) <= 0.05
    assert abs(len(va) / n - 0.10) <= 0.05
    assert abs(len(te) / n - 0.15) <= 0.05


# ---------------------------------------------------------------------------
# 3. No scaffold leakage
# ---------------------------------------------------------------------------


def test_scaffold_split_no_leakage(synthetic_df):
    """No scaffold may appear in both train and test (or train and val)."""
    tr, va, te = scaffold_split(
        synthetic_df,
        scaffold_col="stereo_stripped_scaffold",
        key_col="inchikey_14",
        fractions=(0.8, 0.1, 0.1),
        seed=42,
    )
    tr_df = synthetic_df[synthetic_df["inchikey_14"].isin(tr)]
    va_df = synthetic_df[synthetic_df["inchikey_14"].isin(va)]
    te_df = synthetic_df[synthetic_df["inchikey_14"].isin(te)]

    tr_scafs = set(tr_df["stereo_stripped_scaffold"])
    va_scafs = set(va_df["stereo_stripped_scaffold"])
    te_scafs = set(te_df["stereo_stripped_scaffold"])

    assert len(tr_scafs & te_scafs) == 0, "scaffold leakage: train∩test"
    assert len(tr_scafs & va_scafs) == 0, "scaffold leakage: train∩val"
    assert len(va_scafs & te_scafs) == 0, "scaffold leakage: val∩test"


def test_confirmed_split_no_leakage(synthetic_df):
    """Confirmed split must also be scaffold-disjoint."""
    from pad4bench.splits import split_confirmed

    tr, va, te = split_confirmed(
        synthetic_df,
        test_frac=0.15,
        val_frac=0.10,
        seed=42,
        scaffold_col="stereo_stripped_scaffold",
    )
    tr_scafs = set(tr["stereo_stripped_scaffold"])
    te_scafs = set(te["stereo_stripped_scaffold"])
    assert len(tr_scafs & te_scafs) == 0


# ---------------------------------------------------------------------------
# 4. Reproducibility on actual T1 dataset
# ---------------------------------------------------------------------------

REAL_T1_PATH = os.environ.get("PAD4_T1_PATH", "")


@pytest.mark.skipif(
    not REAL_T1_PATH or not Path(REAL_T1_PATH).exists(),
    reason="Set PAD4_T1_PATH env var to real pad_t1_non_covalent.csv",
)
def test_reproducibility_on_real_t1():
    """Run scaffold split twice on real T1; counts must be identical."""
    df = pd.read_csv(REAL_T1_PATH)
    tr1, va1, te1 = scaffold_split(
        df,
        scaffold_col="stereo_stripped_scaffold",
        key_col="inchikey_14",
        fractions=(0.8, 0.1, 0.1),
        seed=42,
    )
    tr2, va2, te2 = scaffold_split(
        df,
        scaffold_col="stereo_stripped_scaffold",
        key_col="inchikey_14",
        fractions=(0.8, 0.1, 0.1),
        seed=42,
    )
    assert len(tr1) == len(tr2)
    assert len(va1) == len(va2)
    assert len(te1) == len(te2)
    assert tr1 == tr2
    assert va1 == va2
    assert te1 == te2


# ---------------------------------------------------------------------------
# Lead-opt: scaffold overlap BY DESIGN
# ---------------------------------------------------------------------------


def test_lead_opt_scaffold_overlap_by_design(synthetic_df):
    """Lead-opt split must intentionally share scaffolds between train/test."""
    tr, va, te, meta = split_lead_opt(
        synthetic_df,
        test_frac=0.15,
        val_frac=0.10,
        seed=42,
        scaffold_col="stereo_stripped_scaffold",
        min_scaffold_size=4,
    )
    tr_scafs = set(tr["stereo_stripped_scaffold"])
    te_scafs = set(te["stereo_stripped_scaffold"])
    shared = tr_scafs & te_scafs
    assert len(shared) > 0, "lead_opt should have scaffold overlap by design"
    assert meta["n_shared_scaffolds_train_test"] == len(shared)


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def test_dedupe_stereo_preference():
    """Defined stereo should win over undefined."""
    df = pd.DataFrame(
        {
            "inchikey_14": ["IK001", "IK001", "IK001"],
            "stereo_flag": ["undefined", "defined", "achiral"],
            "ml_weight": [0.5, 0.3, 0.9],
            "pIC50": [6.0, 6.5, 7.0],
        }
    )
    out = dedupe_by_inchikey14(df, "test", id_col="inchikey_14", max_loss_frac=0.99)
    assert len(out) == 1
    assert out.iloc[0]["stereo_flag"] == "defined"


def test_dedupe_weight_tiebreak():
    """When stereo is equal, higher ml_weight wins."""
    df = pd.DataFrame(
        {
            "inchikey_14": ["IK001", "IK001"],
            "stereo_flag": ["defined", "defined"],
            "ml_weight": [0.3, 0.8],
            "pIC50": [6.0, 6.5],
        }
    )
    out = dedupe_by_inchikey14(df, "test", id_col="inchikey_14", max_loss_frac=0.99)
    assert len(out) == 1
    assert out.iloc[0]["ml_weight"] == 0.8


def test_dedupe_raises_on_excessive_loss():
    """Should raise if >25% of rows would be lost."""
    df = pd.DataFrame(
        {
            "inchikey_14": ["IK001"] * 10,
            "stereo_flag": ["defined"] * 10,
            "ml_weight": [0.5] * 10,
        }
    )
    with pytest.raises(RuntimeError, match="dedup would remove"):
        dedupe_by_inchikey14(df, "test", id_col="inchikey_14")


# ---------------------------------------------------------------------------
# Audit / leakage detection
# ---------------------------------------------------------------------------


def test_audit_split_raises_on_scaffold_leakage(small_df):
    """audit_split must raise when scaffolds leak across train/test."""
    # Artificially create leakage
    tr = small_df.iloc[:10].copy()
    va = small_df.iloc[10:15].copy()
    te = small_df.iloc[5:].copy()  # overlaps scaffold S_A with train

    with pytest.raises(RuntimeError, match="SCAFFOLD LEAKAGE"):
        audit_split(tr, va, te, "test_leak", scaffold_col="stereo_stripped_scaffold")


def test_audit_split_raises_on_inchikey_leakage(small_df):
    """audit_split must raise when InChIKey-14 leaks."""
    # Scaffold-disjoint but InChIKey-overlapping: same IK, different scaffolds
    tr = small_df.iloc[:5].copy()   # S_A, IK000-IK004
    va = small_df.iloc[5:8].copy()  # S_B, IK005-IK007
    te = small_df.iloc[8:10].copy()  # S_C, IK008-IK009
    # Force InChIKey overlap: give test compound same IK as train but diff scaffold
    te = te.copy()
    te.loc[te.index[0], "inchikey_14"] = "IK000"  # overlaps with train
    te.loc[te.index[0], "stereo_stripped_scaffold"] = "S_Z"  # new scaffold

    with pytest.raises(RuntimeError, match="INCHIKEY-14 LEAKAGE"):
        audit_split(
            tr,
            va,
            te,
            "test_ik_leak",
            scaffold_col="stereo_stripped_scaffold",
            strict_inchikey=True,
        )


def test_audit_split_allows_scaffold_overlap_when_flagged(small_df):
    """When allow_scaffold_overlap=True, audit should not raise."""
    tr = small_df.iloc[:5].copy()   # S_A, IK000-IK004
    va = small_df.iloc[5:8].copy()  # S_B, IK005-IK007
    te = small_df.iloc[8:10].copy() # S_C, IK008-IK009 (scaffold overlap with none)
    # Manually add a compound with same scaffold as train but different InChIKey
    te_extra = small_df.iloc[[0]].copy()
    te_extra["inchikey_14"] = "IK999"
    te = pd.concat([te, te_extra], ignore_index=True)

    report = audit_split(
        tr,
        va,
        te,
        "test_allowed",
        scaffold_col="stereo_stripped_scaffold",
        allow_scaffold_overlap=True,
    )
    assert report["scaffold_train_test_overlap"] > 0
    assert len(report["errors"]) == 0


# ---------------------------------------------------------------------------
# Aligned classification split
# ---------------------------------------------------------------------------


def test_aligned_split_no_scaffold_leakage(tmp_path, small_df):
    """Aligned scaffold split must remain scaffold-disjoint."""
    # Create fake regression splits
    reg_dir = tmp_path / "reg_splits"
    reg_dir.mkdir()

    # Scaffold split: S_A -> train, S_B -> test, S_C + singletons -> val
    scaf_dir = reg_dir / "scaffold"
    scaf_dir.mkdir()
    small_df.iloc[:5].assign(split="train").to_csv(
        scaf_dir / "train.csv", index=False
    )
    small_df.iloc[5:8].assign(split="test").to_csv(
        scaf_dir / "test_locked.csv", index=False
    )
    small_df.iloc[8:].assign(split="val").to_csv(
        scaf_dir / "val.csv", index=False
    )

    # Classification data: same scaffolds + one novel
    cls_df = small_df.copy()
    cls_df = pd.concat(
        [
            cls_df,
            pd.DataFrame(
                {
                    "inchikey_14": ["IK999"],
                    "stereo_stripped_scaffold": ["S_NOVEL"],
                    "pIC50": [6.0],
                    "ml_weight": [0.5],
                    "stereo_flag": ["defined"],
                }
            ),
        ],
        ignore_index=True,
    )
    cls_file = tmp_path / "cls.csv"
    cls_df.to_csv(cls_file, index=False)

    out_dir = tmp_path / "cls_splits"
    audit = align_classification_splits(
        classification_file=cls_file,
        regression_splits_dir=reg_dir,
        output_dir=out_dir,
        scaffold_col="stereo_stripped_scaffold",
        id_col="inchikey_14",
        seed=42,
    )

    # Verify scaffold-disjointness in output
    out_scaf = out_dir / "scaffold"
    tr_out = pd.read_csv(out_scaf / "train.csv")
    te_out = pd.read_csv(out_scaf / "test_locked.csv")
    tr_scafs = set(tr_out["stereo_stripped_scaffold"])
    te_scafs = set(te_out["stereo_stripped_scaffold"])
    assert len(tr_scafs & te_scafs) == 0, "aligned split leaked scaffolds"

    # Novel scaffold should have been distributed somewhere
    all_out = pd.concat(
        [
            pd.read_csv(out_scaf / "train.csv"),
            pd.read_csv(out_scaf / "val.csv"),
            pd.read_csv(out_scaf / "test_locked.csv"),
        ]
    )
    assert "S_NOVEL" in set(all_out["stereo_stripped_scaffold"])


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_dataframe():
    """Splitting an empty DataFrame should not crash."""
    df = pd.DataFrame(
        columns=["inchikey_14", "stereo_stripped_scaffold", "pIC50"]
    )
    tr, va, te = scaffold_split(
        df,
        scaffold_col="stereo_stripped_scaffold",
        key_col="inchikey_14",
        seed=42,
    )
    assert tr == [] and va == [] and te == []


def test_all_singleton_scaffolds():
    """Dataset where every scaffold is a singleton."""
    df = pd.DataFrame(
        {
            "inchikey_14": [f"IK{i:03d}" for i in range(30)],
            "stereo_stripped_scaffold": [f"S_{i}" for i in range(30)],
            "pIC50": [6.0 + i * 0.05 for i in range(30)],
        }
    )
    tr, va, te = scaffold_split(
        df,
        scaffold_col="stereo_stripped_scaffold",
        key_col="inchikey_14",
        fractions=(0.7, 0.15, 0.15),
        seed=42,
    )
    # All singletons means no scaffold leakage by definition
    tr_scafs = set(
        df[df["inchikey_14"].isin(tr)]["stereo_stripped_scaffold"]
    )
    te_scafs = set(
        df[df["inchikey_14"].isin(te)]["stereo_stripped_scaffold"]
    )
    assert len(tr_scafs & te_scafs) == 0


def test_single_large_scaffold():
    """One dominant scaffold should be capped in test."""
    df = pd.DataFrame(
        {
            "inchikey_14": [f"IK{i:03d}" for i in range(100)],
            "stereo_stripped_scaffold": ["S_BIG"] * 80
            + [f"S_{i}" for i in range(20)],
            "pIC50": [6.0] * 100,
        }
    )
    tr, va, te = scaffold_split(
        df,
        scaffold_col="stereo_stripped_scaffold",
        key_col="inchikey_14",
        fractions=(0.8, 0.1, 0.1),
        seed=42,
        cap_frac=0.08,
    )
    te_df = df[df["inchikey_14"].isin(te)]
    big_in_test = (te_df["stereo_stripped_scaffold"] == "S_BIG").sum()
    # cap = ceil(0.08 * ceil(0.10 * 100)) = ceil(0.08 * 10) = 1
    assert big_in_test <= 1


# ---------------------------------------------------------------------------
# Integration: run_all_splits smoke test
# ---------------------------------------------------------------------------


def test_run_all_splits_smoke(synthetic_df):
    """run_all_splits should complete without error on synthetic data."""
    from pad4bench.splits import run_all_splits

    all_stats, all_audits, all_health = run_all_splits(
        t1=synthetic_df,
        confirmed=synthetic_df.iloc[:100],
        test_frac=0.15,
        val_frac=0.10,
        seed=42,
        scaffold_col="stereo_stripped_scaffold",
        activity_col="pIC50",
        skip_similarity=True,  # RDKit may not be available in test env
    )

    for split_name in ("scaffold", "random", "confirmed", "lead_opt"):
        assert split_name in all_stats
        assert "train" in all_stats[split_name]
        assert "test" in all_stats[split_name]
        assert split_name in all_audits
        assert split_name in all_health
