#!/usr/bin/env python3
"""
PAD4_BENCH Paper 1 - data figures.

Produces 14 figures (9 main, 5 supplementary) characterizing the data used in
Paper 1, with explicit scope: only T1 non-covalent (n=2,618 regression) and
classification (n=2,758) are visualized. T2/T3/Ki are not visualized to avoid
implying they were used.

All figures saved at ≥600 DPI as PNG and as vector PDF.
Paper-grade typography: serif body, sans-serif labels, consistent palette.

Output layout:
  paper/figures/data/main/    D1-D9.{png,pdf}
  paper/figures/data/supp/    D10-D14.{png,pdf}

Usage:
    cd /home/nidhal/PAD4_BENCH
    python paper_data_figures.py
"""

import json
import sys
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyBboxPatch
from scipy.stats import ks_2samp

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED   = PROJECT_ROOT / "data" / "processed"
SPLITS_ROOT      = PROJECT_ROOT / "data" / "splits"
FEATURES_ROOT    = PROJECT_ROOT / "features_v18"
MODELS_ROOT      = PROJECT_ROOT / "models_v1"
LEAKAGE_PATH     = MODELS_ROOT / "leakage_verification.json"

OUT_MAIN = PROJECT_ROOT / "paper" / "figures" / "data" / "main"
OUT_SUPP = PROJECT_ROOT / "paper" / "figures" / "data" / "supp"

STRATEGIES = ["random", "scaffold", "confirmed", "lead_opt", "similarity", "cliff_aware"]
STRAT_PRETTY = {
    "random": "Random",
    "scaffold": "Scaffold",
    "confirmed": "Confirmed",
    "lead_opt": "Lead-Opt",
    "similarity": "Similarity",
    "cliff_aware": "Cliff-Aware",
}

# Paper-grade palette: high-contrast, colorblind-safe, print-friendly.
PALETTE = {
    "primary":   "#1f4e79",
    "secondary": "#c5504b",
    "accent":    "#e8b73a",
    "neutral":   "#5a5a5a",
    "light":     "#d4d4d4",
    "good":      "#2d7a4f",
    "warn":      "#c5504b",
}
SPLIT_COLOR = {
    "random":      "#1f4e79",
    "scaffold":    "#2d7a4f",
    "confirmed":   "#5a5a5a",
    "lead_opt":    "#7b3294",
    "similarity":  "#c5504b",
    "cliff_aware": "#e8731a",
}
TASK_COLOR = {"regression": "#1f4e79", "classification": "#c5504b"}

DPI_RASTER = 600

# -----------------------------------------------------------------------------
# Matplotlib paper style
# -----------------------------------------------------------------------------
def set_paper_style() -> None:
    mpl.rcParams.update({
        # Type
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 11,
        # Axes
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#222222",
        "axes.labelcolor": "#222222",
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        # Ticks
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        # Lines / patches
        "lines.linewidth": 1.4,
        "patch.linewidth": 0.6,
        # Grids
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#d9d9d9",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.6,
        # Figure
        "figure.dpi": 110,
        "savefig.dpi": DPI_RASTER,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_fig(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{name}.png"
    pdf = out_dir / f"{name}.pdf"
    fig.savefig(png, dpi=DPI_RASTER)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"    wrote {png.relative_to(PROJECT_ROOT)} and .pdf", flush=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# -----------------------------------------------------------------------------
# Loaders (cached)
# -----------------------------------------------------------------------------
_cache: dict = {}

def load_csv(name: str, **kw) -> pd.DataFrame | None:
    key = f"csv::{name}::{sorted(kw.items())}"
    if key in _cache:
        return _cache[key]
    path = DATA_PROCESSED / name
    if not path.exists():
        return None
    df = pd.read_csv(path, low_memory=False, **kw)
    _cache[key] = df
    return df


def load_split(task: str, strategy: str, subset: str) -> pd.DataFrame | None:
    key = f"split::{task}/{strategy}/{subset}"
    if key in _cache:
        return _cache[key]
    p = SPLITS_ROOT / task / strategy / f"{subset}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, low_memory=False)
    _cache[key] = df
    return df


def load_leakage() -> dict:
    if "leakage" in _cache:
        return _cache["leakage"]
    if not LEAKAGE_PATH.exists():
        raise FileNotFoundError(f"{LEAKAGE_PATH} missing — run paper1_reviewer_proof.py")
    _cache["leakage"] = json.loads(LEAKAGE_PATH.read_text())
    return _cache["leakage"]


# =============================================================================
# D1. Provenance funnel
# =============================================================================
def fig_D1_provenance() -> None:
    log("D1: provenance funnel")
    # Raw counts (from raw/ directory, verified earlier)
    chembl = {"PAD1": 168, "PAD2": 181, "PAD3": 121, "PAD4": 4925, "PAD6": 6}
    bdb    = {"PAD1": 46,  "PAD2": 76,  "PAD3": 22,  "PAD4": 3043, "PAD6": 6}
    pubchem_total = 331_947  # 23 AIDs combined

    # Counts we trust from the audit
    t1_aggregated_unique = 2654
    reg_set     = 2618
    classif_set = 2758
    covalent_excluded = 36
    confirmed_set = 1623

    fig, ax = plt.subplots(figsize=(8.5, 5.6))

    # Build a vertical funnel with stages on the right, counts on the left.
    stages = [
        ("Raw assay records\n(ChEMBL + BindingDB + PubChem)",
         chembl["PAD4"] + bdb["PAD4"] + pubchem_total,
         f"ChEMBL PAD4: {chembl['PAD4']:,}\nBindingDB PAD4: {bdb['PAD4']:,}\nPubChem (23 AIDs): {pubchem_total:,}",
         PALETTE["neutral"]),
        ("Filter to PAD4 isoform\n+ unit standardization\n+ structure curation",
         t1_aggregated_unique,
         f"unique compounds with\nPAD4 IC50 measurements",
         PALETTE["primary"]),
        ("Exclude irreversible covalent\ninhibitors",
         reg_set,
         f"{covalent_excluded} irreversible covalent\ncompounds excluded",
         PALETTE["primary"]),
        ("Paper 1 regression set\n(pad_t1_non_covalent)",
         reg_set,
         f"used for 6 split strategies",
         PALETTE["good"]),
    ]

    # Geometry
    n_stages = len(stages)
    y_positions = np.linspace(0.92, 0.08, n_stages)
    x_center = 0.55
    max_w = 0.55
    min_w = 0.30
    max_count = max(s[1] for s in stages)

    for i, (label, n, note, color) in enumerate(stages):
        # Box width scaled by row count, but with floor so the bottom is readable
        w = min_w + (max_w - min_w) * (n / max_count)
        h = 0.13
        box = FancyBboxPatch((x_center - w / 2, y_positions[i] - h / 2), w, h,
                             boxstyle="round,pad=0.01,rounding_size=0.012",
                             facecolor=to_rgba(color, 0.18),
                             edgecolor=color, linewidth=1.4,
                             transform=ax.transAxes)
        ax.add_patch(box)
        ax.text(x_center, y_positions[i] + 0.012, label,
                ha="center", va="center", fontsize=9, weight="bold",
                color="#1a1a1a", transform=ax.transAxes)
        ax.text(x_center, y_positions[i] - 0.028, f"n = {n:,}",
                ha="center", va="center", fontsize=10, weight="bold",
                color=color, transform=ax.transAxes)
        # Side annotation
        ax.text(0.02, y_positions[i], note, ha="left", va="center",
                fontsize=7.5, color=PALETTE["neutral"], style="italic",
                transform=ax.transAxes)
        # Arrow to next stage
        if i < n_stages - 1:
            ax.annotate("", xy=(x_center, y_positions[i + 1] + 0.065),
                        xytext=(x_center, y_positions[i] - 0.065),
                        xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="-|>",
                                        color=PALETTE["neutral"],
                                        lw=1.2,
                                        mutation_scale=14))

    # Right-side parallel branch for classification + confirmed
    ax.text(0.93, y_positions[0], "Auxiliary curated subsets:",
            ha="right", va="center", fontsize=8, weight="bold",
            color=PALETTE["neutral"], transform=ax.transAxes)
    ax.text(0.93, y_positions[1],
            f"Classification set\n(pad_classification_v17)\nn = {classif_set:,}",
            ha="right", va="center", fontsize=8,
            color=PALETTE["secondary"], transform=ax.transAxes)
    ax.text(0.93, y_positions[2],
            f"Confirmed-quality subset\n(pad_t1_confirmed)\nn = {confirmed_set:,}",
            ha="right", va="center", fontsize=8,
            color=PALETTE["accent"], transform=ax.transAxes)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.suptitle("Figure D1.  Paper 1 data provenance funnel",
                 fontsize=11, weight="bold", y=0.99)
    fig.text(0.5, 0.01,
             "Counts above the green box show the regression modeling set. "
             "T2 censored, T3 HTS, and Ki tiers are not used in Paper 1.",
             ha="center", fontsize=7.5, style="italic", color=PALETTE["neutral"])
    save_fig(fig, OUT_MAIN, "D1_provenance_funnel")


# =============================================================================
# D2. pIC50 distribution of modeled set
# =============================================================================
def fig_D2_pic50_distribution() -> None:
    log("D2: pIC50 distribution")
    noncov = load_csv("pad_t1_non_covalent.csv")
    classif = load_csv("pad_classification_v17.csv")
    if noncov is None or classif is None:
        return

    # Use the unique-compound aggregated pIC50 per InChIKey (mean across rows)
    def aggregate(df: pd.DataFrame) -> np.ndarray:
        return (df.dropna(subset=["pIC50"])
                  .groupby("inchikey_14")["pIC50"].mean().values)

    pic_reg = aggregate(noncov)
    pic_cls = aggregate(classif)

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4), sharey=True)
    bins = np.linspace(2, 9, 36)

    ax = axes[0]
    ax.hist(pic_reg, bins=bins, color=PALETTE["primary"],
            alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.axvline(np.median(pic_reg), color=PALETTE["secondary"],
               linestyle="--", linewidth=1.0, label=f"median = {np.median(pic_reg):.2f}")
    ax.set_title(f"Regression set: T1 non-covalent  (n = {len(pic_reg):,})")
    ax.set_xlabel("pIC50")
    ax.set_ylabel("compounds")
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1]
    ax.hist(pic_cls, bins=bins, color=PALETTE["secondary"],
            alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.axvline(6.0, color=PALETTE["good"], linestyle="-", linewidth=1.2,
               label="active/inactive threshold (pIC50 = 6)")
    ax.axvline(np.median(pic_cls), color=PALETTE["neutral"],
               linestyle="--", linewidth=1.0, label=f"median = {np.median(pic_cls):.2f}")
    ax.set_title(f"Classification set  (n = {len(pic_cls):,})")
    ax.set_xlabel("pIC50")
    ax.legend(frameon=False, loc="upper left")

    fig.suptitle("Figure D2.  pIC50 distributions of Paper 1 modeling sets",
                 fontsize=10.5, y=1.02)
    save_fig(fig, OUT_MAIN, "D2_pic50_distribution")


# =============================================================================
# D3. Activity class balance
# =============================================================================
def fig_D3_class_balance() -> None:
    log("D3: class balance")
    classif = load_csv("pad_classification_v17.csv")
    if classif is None:
        return
    cl = classif.dropna(subset=["activity_class"])
    cl = cl.drop_duplicates(subset=["inchikey_14"])

    counts = cl["activity_class"].value_counts().sort_index()
    n_neg = int(counts.get(0, 0))
    n_pos = int(counts.get(1, 0))
    total = n_pos + n_neg

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))

    # Left: bar
    ax = axes[0]
    bars = ax.bar(["Inactive\n(pIC50 < 6)", "Active\n(pIC50 ≥ 6)"],
                  [n_neg, n_pos],
                  color=[PALETTE["neutral"], PALETTE["good"]],
                  edgecolor="white", linewidth=1.2)
    for b, v in zip(bars, [n_neg, n_pos]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 30,
                f"{v:,}\n({v/total:.1%})", ha="center", va="bottom",
                fontsize=9, weight="bold")
    ax.set_ylabel("compounds")
    ax.set_title(f"Class balance  (n = {total:,})")
    ax.set_ylim(0, max(n_neg, n_pos) * 1.18)

    # Right: pIC50 violin by class
    ax = axes[1]
    pos_vals = cl[cl["activity_class"] == 1]["pIC50"].values
    neg_vals = cl[cl["activity_class"] == 0]["pIC50"].values
    parts = ax.violinplot([neg_vals, pos_vals], positions=[0, 1],
                          showmedians=True, widths=0.7)
    for pc, c in zip(parts["bodies"], [PALETTE["neutral"], PALETTE["good"]]):
        pc.set_facecolor(c)
        pc.set_alpha(0.6)
        pc.set_edgecolor(c)
    for k in ("cmins", "cmaxes", "cbars", "cmedians"):
        if k in parts:
            parts[k].set_color(PALETTE["secondary"])
            parts[k].set_linewidth(1.0)
    ax.axhline(6.0, color=PALETTE["secondary"], linestyle="--", linewidth=0.8,
               alpha=0.7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Inactive", "Active"])
    ax.set_ylabel("pIC50")
    ax.set_title("pIC50 vs activity class")
    ax.grid(True, axis="y", alpha=0.5)

    fig.suptitle("Figure D3.  Classification labels and pIC50 binarization",
                 fontsize=10.5, y=1.02)
    save_fig(fig, OUT_MAIN, "D3_class_balance")


# =============================================================================
# D4. Molecular property panel (Lipinski + extras)
# =============================================================================
def fig_D4_property_panel() -> None:
    log("D4: molecular property panel")
    noncov = load_csv("pad_t1_non_covalent.csv")
    if noncov is None:
        return
    # One row per compound
    df = noncov.drop_duplicates(subset=["inchikey_14"])

    panels = [
        ("mw", "Molecular weight (Da)", None, (50, 1000)),
        ("logP", "ALogP", None, (-3, 9)),
        ("tpsa", "TPSA (Å²)", None, (0, 250)),
        ("hba", "H-bond acceptors", None, (0, 16)),
        ("hbd", "H-bond donors", None, (0, 10)),
        ("rot_bonds", "Rotatable bonds", None, (0, 22)),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(8.6, 4.8))
    for ax, (col, xlabel, _, xlim) in zip(axes.flat, panels):
        if col not in df.columns:
            ax.set_visible(False)
            continue
        vals = df[col].dropna().values
        ax.hist(vals, bins=30, color=PALETTE["primary"],
                alpha=0.85, edgecolor="white", linewidth=0.4)
        ax.axvline(np.median(vals), color=PALETTE["secondary"],
                   linestyle="--", linewidth=0.9,
                   label=f"median = {np.median(vals):.1f}")
        ax.set_xlim(xlim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")
        ax.legend(frameon=False, fontsize=7.5, loc="upper right")

    # Lipinski compliance note
    if all(c in df.columns for c in ("mw", "logP", "hba", "hbd")):
        lip_ok = ((df["mw"] <= 500) & (df["logP"] <= 5) &
                  (df["hba"] <= 10) & (df["hbd"] <= 5)).sum()
        frac = lip_ok / len(df)
        fig.text(0.5, -0.02,
                 f"Lipinski Ro5 compliance: {lip_ok:,} of {len(df):,} ({frac:.1%}).",
                 ha="center", fontsize=8, style="italic", color=PALETTE["neutral"])

    fig.suptitle("Figure D4.  Molecular properties of the regression modeling set "
                 f"(n = {len(df):,})", fontsize=10.5, y=1.02)
    fig.tight_layout()
    save_fig(fig, OUT_MAIN, "D4_property_panel")


# =============================================================================
# D5. Scaffold diversity
# =============================================================================
def fig_D5_scaffold_diversity() -> None:
    log("D5: scaffold diversity")
    noncov = load_csv("pad_t1_non_covalent.csv")
    if noncov is None or "stereo_stripped_scaffold" not in noncov.columns:
        return
    df = noncov.drop_duplicates(subset=["inchikey_14"])
    df = df.dropna(subset=["stereo_stripped_scaffold"])
    sc_counts = df["stereo_stripped_scaffold"].value_counts()
    n_unique = sc_counts.shape[0]
    n_singletons = int((sc_counts == 1).sum())

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.0),
                              gridspec_kw={"width_ratios": [1.2, 1]})

    # Left: top-20 scaffolds bar
    ax = axes[0]
    top = sc_counts.head(20)
    y_pos = np.arange(len(top))
    ax.barh(y_pos, top.values, color=PALETTE["primary"], edgecolor="white",
            linewidth=0.6)
    for i, v in enumerate(top.values):
        ax.text(v + 5, i, str(v), va="center", fontsize=7.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"#{i+1}" for i in range(len(top))], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("compounds")
    ax.set_title(f"Top-20 scaffolds by frequency")
    ax.grid(True, axis="x", alpha=0.5)

    # Right: scaffold-size distribution
    ax = axes[1]
    ax.hist(sc_counts.values, bins=np.logspace(0, np.log10(sc_counts.max() + 1), 30),
            color=PALETTE["accent"], edgecolor="white", linewidth=0.4)
    ax.set_xscale("log")
    ax.set_xlabel("compounds per scaffold")
    ax.set_ylabel("scaffolds")
    ax.set_title("Scaffold-size distribution")
    ax.grid(True, alpha=0.5)
    ax.text(0.55, 0.92,
            f"unique scaffolds: {n_unique:,}\n"
            f"singletons: {n_singletons:,} ({n_singletons/n_unique:.1%})\n"
            f"compounds/scaffold ratio: {len(df)/n_unique:.2f}",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(facecolor="white", edgecolor=PALETTE["neutral"],
                      boxstyle="round,pad=0.4", linewidth=0.6))

    fig.suptitle("Figure D5.  Scaffold diversity (stereo-stripped Bemis-Murcko)",
                 fontsize=10.5, y=1.02)
    fig.tight_layout()
    save_fig(fig, OUT_MAIN, "D5_scaffold_diversity")


# =============================================================================
# D6. Label quality (ml_weight + label_uncertainty)
# =============================================================================
def fig_D6_label_quality() -> None:
    log("D6: label quality")
    noncov = load_csv("pad_t1_non_covalent.csv")
    if noncov is None:
        return
    df = noncov.drop_duplicates(subset=["inchikey_14"])

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4))

    # ml_weight
    if "ml_weight" in df.columns:
        ax = axes[0]
        vals = df["ml_weight"].dropna().values
        ax.hist(vals, bins=30, color=PALETTE["primary"],
                alpha=0.85, edgecolor="white", linewidth=0.4)
        ax.axvline(np.median(vals), color=PALETTE["secondary"],
                   linestyle="--", linewidth=0.9,
                   label=f"median = {np.median(vals):.2f}")
        ax.set_xlabel("ml_weight")
        ax.set_ylabel("compounds")
        ax.set_title("Sample weight distribution")
        ax.legend(frameon=False, fontsize=7.5)

    # label_uncertainty
    if "label_uncertainty_score_v2" in df.columns:
        ax = axes[1]
        vals = df["label_uncertainty_score_v2"].dropna().values
        ax.hist(vals, bins=30, color=PALETTE["accent"],
                alpha=0.85, edgecolor="white", linewidth=0.4)
        ax.axvline(np.median(vals), color=PALETTE["secondary"],
                   linestyle="--", linewidth=0.9,
                   label=f"median = {np.median(vals):.3f}")
        ax.set_xlabel("label uncertainty score")
        ax.set_ylabel("compounds")
        ax.set_title("Label uncertainty distribution")
        ax.legend(frameon=False, fontsize=7.5)

    fig.suptitle("Figure D6.  Label quality of the modeling set",
                 fontsize=10.5, y=1.02)
    fig.tight_layout()
    save_fig(fig, OUT_MAIN, "D6_label_quality")


# =============================================================================
# D7. Split sizes and class balance per split
# =============================================================================
def fig_D7_split_overview() -> None:
    log("D7: split overview")
    ss = pd.read_csv(SPLITS_ROOT / "splits_summary.csv")
    # Reorder
    ss = ss.set_index("method").reindex(STRATEGIES).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.4))

    # Left: stacked sizes train/val/test
    ax = axes[0]
    x = np.arange(len(STRATEGIES))
    width = 0.66
    p_train = ax.bar(x, ss["n_train"], width, label="train",
                     color=PALETTE["primary"], edgecolor="white", linewidth=0.6)
    p_val = ax.bar(x, ss["n_val"], width, bottom=ss["n_train"],
                   label="val", color=PALETTE["accent"],
                   edgecolor="white", linewidth=0.6)
    p_test = ax.bar(x, ss["n_test"], width,
                    bottom=ss["n_train"] + ss["n_val"], label="test",
                    color=PALETTE["secondary"], edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([STRAT_PRETTY[s] for s in STRATEGIES],
                       rotation=30, ha="right")
    ax.set_ylabel("compounds")
    ax.set_title("Regression-task split sizes")
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper right")
    for i, (tr, v, t) in enumerate(zip(ss["n_train"], ss["n_val"], ss["n_test"])):
        total = tr + v + t
        ax.text(i, total + 25, f"{total:,}", ha="center", va="bottom",
                fontsize=7.5, weight="bold")

    # Right: test pct_active per split
    ax = axes[1]
    colors = [SPLIT_COLOR[s] for s in STRATEGIES]
    bars = ax.bar(x, ss["test_pct_active"], width, color=colors,
                  edgecolor="white", linewidth=0.6)
    for b, v in zip(bars, ss["test_pct_active"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=7.5, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([STRAT_PRETTY[s] for s in STRATEGIES],
                       rotation=30, ha="right")
    ax.set_ylabel("% active in test")
    ax.set_title("Test-set positive-class rate")
    ax.set_ylim(0, 100)

    fig.suptitle("Figure D7.  Split sizes and class balance across strategies",
                 fontsize=10.5, y=1.02)
    fig.tight_layout()
    save_fig(fig, OUT_MAIN, "D7_split_overview")


# =============================================================================
# D8. Train/test pIC50 per split
# =============================================================================
def fig_D8_pic50_per_split() -> None:
    log("D8: train/test pIC50 per split")
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.4),
                              sharex=True, sharey=True)
    bins = np.linspace(2, 9, 30)
    for ax, strategy in zip(axes.flat, STRATEGIES):
        tr = load_split("regression", strategy, "train")
        te = load_split("regression", strategy, "test_locked")
        if tr is None or te is None or "pIC50" not in tr.columns:
            ax.set_visible(False)
            continue
        tr_pic = tr["pIC50"].dropna().values
        te_pic = te["pIC50"].dropna().values
        ax.hist(tr_pic, bins=bins, alpha=0.55, color=PALETTE["primary"],
                density=True, label=f"train (n={len(tr_pic):,})",
                edgecolor="white", linewidth=0.3)
        ax.hist(te_pic, bins=bins, alpha=0.65, color=PALETTE["secondary"],
                density=True, label=f"test (n={len(te_pic):,})",
                edgecolor="white", linewidth=0.3)
        ks_stat, ks_p = ks_2samp(tr_pic, te_pic)
        ax.set_title(STRAT_PRETTY[strategy], fontsize=9.5)
        # KS annotation
        sci_p = f"{ks_p:.2e}" if ks_p < 1e-3 else f"{ks_p:.3f}"
        ax.text(0.04, 0.96, f"KS p = {sci_p}", transform=ax.transAxes,
                fontsize=7.5, va="top",
                bbox=dict(facecolor="white", edgecolor="none",
                          alpha=0.7, pad=1.5))
        ax.legend(frameon=False, fontsize=7, loc="upper left",
                  bbox_to_anchor=(0.04, 0.85))
        ax.grid(True, alpha=0.4)
    for ax in axes[-1]:
        ax.set_xlabel("pIC50")
    for ax in axes[:, 0]:
        ax.set_ylabel("density")
    fig.suptitle("Figure D8.  Train-test pIC50 distributions per split  "
                 "(regression task; KS p-value annotated)",
                 fontsize=10.5, y=1.00)
    fig.tight_layout()
    save_fig(fig, OUT_MAIN, "D8_pic50_per_split")


# =============================================================================
# D9. Tanimoto distributions per split (from leakage_verification.json)
# =============================================================================
def fig_D9_tanimoto_per_split() -> None:
    log("D9: train→test Tanimoto per split")
    leak = load_leakage()
    # We have summary statistics per cell, not the per-test distribution.
    # Use the regression cells; show summary statistics as a bar+errorbar plot
    # with max/p95/p99/mean values.
    rows = []
    for cell in leak.get("cells", []):
        if cell.get("task") != "regression":
            continue
        strat = cell.get("strategy")
        tan = cell.get("ecfp4_tanimoto_train_to_test", {})
        rows.append({
            "strategy": strat,
            "max": tan.get("max"),
            "mean": tan.get("mean_of_max_per_test"),
            "p95": tan.get("p95_of_max_per_test"),
            "p99": tan.get("p99_of_max_per_test"),
            "frac_sim_above_05": tan.get("frac_test_with_train_sim_above_0_5"),
            "frac_sim_above_07": tan.get("frac_test_with_train_sim_above_0_7"),
        })
    df = pd.DataFrame(rows).set_index("strategy").reindex(STRATEGIES).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))

    # Left: max-vs-mean-vs-p95 per split as grouped bars
    ax = axes[0]
    x = np.arange(len(STRATEGIES))
    width = 0.27
    ax.bar(x - width, df["mean"], width, label="mean of max",
           color=PALETTE["primary"], edgecolor="white", linewidth=0.4)
    ax.bar(x, df["p95"], width, label="95th pct",
           color=PALETTE["accent"], edgecolor="white", linewidth=0.4)
    ax.bar(x + width, df["max"], width, label="max",
           color=PALETTE["secondary"], edgecolor="white", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([STRAT_PRETTY[s] for s in STRATEGIES],
                       rotation=30, ha="right")
    ax.set_ylabel("Tanimoto similarity (ECFP4)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-test-compound max-Tanimoto train→test")
    ax.legend(frameon=False, fontsize=7.5)
    # Highlight similarity split
    sim_idx = STRATEGIES.index("similarity")
    ax.axvspan(sim_idx - 0.45, sim_idx + 0.45, alpha=0.08,
               color=PALETTE["secondary"], zorder=0)
    ax.text(sim_idx, 1.02, "stress-test", ha="center", va="bottom",
            fontsize=7, style="italic", color=PALETTE["secondary"])

    # Right: fraction of test compounds with high train similarity
    ax = axes[1]
    width = 0.38
    ax.bar(x - width / 2, df["frac_sim_above_05"], width,
           label="≥ 0.5", color=PALETTE["primary"],
           edgecolor="white", linewidth=0.4)
    ax.bar(x + width / 2, df["frac_sim_above_07"], width,
           label="≥ 0.7", color=PALETTE["secondary"],
           edgecolor="white", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([STRAT_PRETTY[s] for s in STRATEGIES],
                       rotation=30, ha="right")
    ax.set_ylabel("fraction of test compounds")
    ax.set_ylim(0, 1.05)
    ax.set_title("Fraction of test with high train Tanimoto")
    ax.legend(frameon=False, fontsize=7.5, title="max sim ≥",
              title_fontsize=7.5)

    fig.suptitle("Figure D9.  Chemistry-space separation between train and test",
                 fontsize=10.5, y=1.02)
    fig.tight_layout()
    save_fig(fig, OUT_MAIN, "D9_tanimoto_per_split")


# =============================================================================
# D10. Replicate noise floor
# =============================================================================
def fig_D10_replicate_noise() -> None:
    log("D10: replicate noise floor")
    reps = load_csv("pad_replicates_full.csv")
    noncov = load_csv("pad_t1_non_covalent.csv")
    if reps is None or noncov is None:
        return
    modeled_ids = set(noncov["inchikey_14"].astype(str))
    rep = reps.dropna(subset=["inchikey_14", "pIC50"])
    rep = rep[rep["inchikey_14"].astype(str).isin(modeled_ids)]
    per = rep.groupby("inchikey_14")["pIC50"].agg(["count", "std"])
    multi = per[per["count"] >= 2].dropna(subset=["std"])
    stds = multi["std"].values

    p50 = float(np.median(stds))
    p95 = float(np.quantile(stds, 0.95))
    p99 = float(np.quantile(stds, 0.99))

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6))

    ax = axes[0]
    # Use linear axis with clip at p99 for visibility
    bins = np.linspace(0, max(p99 * 1.5, 0.6), 50)
    ax.hist(stds, bins=bins, color=PALETTE["primary"],
            alpha=0.85, edgecolor="white", linewidth=0.4)
    for q, label, color in [(p50, "median", PALETTE["neutral"]),
                             (p95, "p95",    PALETTE["secondary"]),
                             (p99, "p99",    PALETTE["good"])]:
        ax.axvline(q, color=color, linestyle="--", linewidth=1.0,
                   label=f"{label} = {q:.3f}")
    ax.set_xlabel("per-compound pIC50 std across replicates")
    ax.set_ylabel("compounds")
    ax.set_title(f"Noise floor (n = {len(multi):,} compounds with ≥ 2 measurements)")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    # Replicate-count distribution
    counts = per["count"].values
    bins = np.arange(1, counts.max() + 2)
    ax.hist(counts, bins=bins, color=PALETTE["accent"],
            alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.set_xlabel("measurements per compound")
    ax.set_ylabel("compounds")
    ax.set_title("Replicate-count distribution")
    ax.set_yscale("log")
    ax.set_xlim(0.5, max(20, counts.max() + 1))

    fig.suptitle(f"Figure D10.  Experimental noise floor in the modeling set "
                 f"({len(stds):,} multi-measurement compounds; 100% coverage)",
                 fontsize=10.5, y=1.02)
    fig.tight_layout()
    save_fig(fig, OUT_SUPP, "D10_replicate_noise")


# =============================================================================
# D11. Assay-source bias
# =============================================================================
def fig_D11_assay_bias() -> None:
    log("D11: assay-source bias")
    bias = load_csv("pad_assay_bias_report.csv")
    if bias is None or "assay_shift" not in bias.columns:
        return
    df = bias.copy().sort_values("assay_shift").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    y = np.arange(len(df))
    colors = [PALETTE["primary"] if v >= 0 else PALETTE["secondary"]
              for v in df["assay_shift"]]
    err = df["assay_shift_std"] if "assay_shift_std" in df.columns else None
    ax.barh(y, df["assay_shift"], xerr=err, color=colors,
            edgecolor="white", linewidth=0.4, capsize=2,
            error_kw=dict(elinewidth=0.7, ecolor=PALETTE["neutral"]))
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(df["source_id"].astype(str).values, fontsize=7)
    ax.set_xlabel("assay shift (Δ pIC50)")
    ax.set_title("Per-source assay-shift estimates")
    if "n_anchors" in df.columns:
        for i, n in enumerate(df["n_anchors"]):
            ax.text(df["assay_shift"].iloc[i],
                    i, f"  n={int(n)}", va="center", fontsize=6.5,
                    color=PALETTE["neutral"])
    max_abs = float(df["assay_shift"].abs().max())
    fig.suptitle(f"Figure D11.  Inter-source assay-bias correction  "
                 f"(max |shift| = {max_abs:.2f} pIC50 across {len(df)} sources)",
                 fontsize=10.5, y=1.00)
    fig.tight_layout()
    save_fig(fig, OUT_SUPP, "D11_assay_bias")


# =============================================================================
# D12. Cliff characterization
# =============================================================================
def fig_D12_cliff() -> None:
    log("D12: cliff characterization")
    cl = load_csv("pad_activity_cliffs.csv")
    if cl is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))

    ax = axes[0]
    delta = cl["delta_pIC50"].dropna().values
    ax.hist(delta, bins=30, color=PALETTE["secondary"],
            alpha=0.85, edgecolor="white", linewidth=0.4)
    for q, lab, c in [(np.median(delta), "median", PALETTE["neutral"]),
                       (np.max(delta), "max", PALETTE["good"])]:
        ax.axvline(q, linestyle="--", color=c, linewidth=0.9,
                   label=f"{lab} = {q:.2f}")
    ax.set_xlabel("Δ pIC50 between cliff pair")
    ax.set_ylabel("pairs")
    ax.set_title("Cliff magnitudes")
    ax.legend(frameon=False, fontsize=7.5)

    ax = axes[1]
    if "tanimoto_similarity" in cl.columns:
        ax.scatter(cl["tanimoto_similarity"], cl["delta_pIC50"],
                   s=14, alpha=0.55, color=PALETTE["primary"],
                   edgecolor="white", linewidth=0.3)
        ax.set_xlabel("Tanimoto similarity (ECFP4)")
        ax.set_ylabel("Δ pIC50")
        ax.set_title("Cliff pairs in (similarity, Δactivity) space")
        ax.set_xlim(0.78, 1.02)
    fig.suptitle(f"Figure D12.  Activity-cliff characterization  "
                 f"(n = {len(cl):,} pairs)",
                 fontsize=10.5, y=1.02)
    fig.tight_layout()
    save_fig(fig, OUT_SUPP, "D12_cliff")


# =============================================================================
# D13. Covalent disclosure table (rendered as image)
# =============================================================================
def fig_D13_covalent_disclosure() -> None:
    log("D13: covalent disclosure table")
    rows = [
        ["Set", "n total", "Irrev. covalent", "Rev. covalent", "Treatment"],
        ["pad_t1_covalent (reference)",     "36",   "36", "0",  "EXCLUDED from all splits"],
        ["pad_t1_non_covalent (regression)", "2,618", "0",  "9",  "USED — reversible covalents retained"],
        ["pad_classification_v17 (classification)", "2,758", "36", "9",
                                                                  "USED — both classes retained"],
        ["pad_t1_confirmed (Confirmed split)", "1,623", "20", "9",
                                                                  "USED — covalents retained for quality stratification"],
    ]

    fig, ax = plt.subplots(figsize=(9.0, 2.5))
    ax.axis("off")
    n_rows, n_cols = len(rows), len(rows[0])
    col_widths = [0.28, 0.08, 0.13, 0.12, 0.39]
    row_h = 1 / n_rows
    for r, row in enumerate(rows):
        x = 0
        for c, cell in enumerate(row):
            w = col_widths[c]
            face = "#f5f5f5" if r == 0 else ("#ffffff" if r % 2 == 1 else "#fafafa")
            edge = PALETTE["neutral"]
            ax.add_patch(plt.Rectangle((x, 1 - (r + 1) * row_h), w, row_h,
                                        facecolor=face, edgecolor=edge,
                                        linewidth=0.7, transform=ax.transAxes))
            weight = "bold" if r == 0 else "normal"
            color = "#1a1a1a" if r == 0 else "#222222"
            size = 8.5 if r == 0 else 8
            ax.text(x + w / 2, 1 - (r + 0.5) * row_h, cell,
                    ha="center", va="center", fontsize=size,
                    weight=weight, color=color, transform=ax.transAxes)
            x += w
    fig.suptitle("Figure D13.  Covalent-inhibitor accounting across data tiers",
                 fontsize=10.5, y=1.02)
    save_fig(fig, OUT_SUPP, "D13_covalent_disclosure")


# =============================================================================
# D14. Reliability diagrams (paper-quality re-render)
# =============================================================================
def fig_D14_reliability() -> None:
    log("D14: reliability diagrams (paper-quality)")
    # Pull from each classification XGB fingerprints metrics.json
    fig, axes = plt.subplots(2, 3, figsize=(9.0, 5.7),
                              sharex=True, sharey=True)
    for ax, strategy in zip(axes.flat, STRATEGIES):
        mpath = (MODELS_ROOT / "classification" / strategy /
                  "fingerprints" / "xgboost" / "metrics.json")
        if not mpath.exists():
            ax.set_visible(False)
            continue
        m = json.loads(mpath.read_text())
        cal = m.get("test_calibration", {})
        bins = cal.get("bins", [])
        ece = cal.get("ece", float("nan"))
        ax.plot([0, 1], [0, 1], color=PALETTE["neutral"], linestyle="--",
                linewidth=0.8, alpha=0.7, label="perfect")
        xs, ys, ws = [], [], []
        for b in bins:
            if b.get("avg_proba") is not None and b.get("obs_pos_rate") is not None:
                xs.append(b["avg_proba"])
                ys.append(b["obs_pos_rate"])
                ws.append(b["n_eff"])
        if xs:
            ws_arr = np.asarray(ws, dtype=np.float64)
            sizes = 40 + 240 * (ws_arr / max(1.0, ws_arr.max()))
            ax.plot(xs, ys, color=SPLIT_COLOR[strategy], linewidth=1.0,
                    alpha=0.65, zorder=2)
            ax.scatter(xs, ys, s=sizes, color=SPLIT_COLOR[strategy],
                       edgecolor="white", linewidth=0.7, zorder=3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{STRAT_PRETTY[strategy]}  (ECE = {ece:.3f})",
                     fontsize=9.5)
        ax.grid(True, alpha=0.4)
    for ax in axes[-1]:
        ax.set_xlabel("predicted probability")
    for ax in axes[:, 0]:
        ax.set_ylabel("observed positive rate")
    fig.suptitle("Figure D14.  Reliability diagrams across splits  "
                 "(XGBoost on fingerprints variant; point size ∝ bin weight)",
                 fontsize=10.5, y=1.00)
    fig.tight_layout()
    save_fig(fig, OUT_SUPP, "D14_reliability")


# =============================================================================
# Main
# =============================================================================
def main() -> int:
    set_paper_style()
    OUT_MAIN.mkdir(parents=True, exist_ok=True)
    OUT_SUPP.mkdir(parents=True, exist_ok=True)
    log(f"output (main): {OUT_MAIN}")
    log(f"output (supp): {OUT_SUPP}")

    fig_D1_provenance()
    fig_D2_pic50_distribution()
    fig_D3_class_balance()
    fig_D4_property_panel()
    fig_D5_scaffold_diversity()
    fig_D6_label_quality()
    fig_D7_split_overview()
    fig_D8_pic50_per_split()
    fig_D9_tanimoto_per_split()

    fig_D10_replicate_noise()
    fig_D11_assay_bias()
    fig_D12_cliff()
    fig_D13_covalent_disclosure()
    fig_D14_reliability()

    log("DONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
