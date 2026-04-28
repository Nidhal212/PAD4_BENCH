# PAD4-Bench

A fidelity-tiered IC50 dataset and benchmark for PAD4 inhibitor QSAR.

## Overview

PAD4-Bench provides:

- A curated dataset of PAD4 inhibitor measurements from PubChem, ChEMBL,
  and BindingDB, organized into fidelity tiers (T1 high-confidence IC50,
  T2 censored, T3 HTS, Ki).
- Five complementary train/test split protocols spanning scaffold-,
  random-, confirmed-, lead-optimization-, and similarity-stratified
  generalization regimes.
- A held-out external validation set from PubChem AID 1805620.
- Baseline benchmark across 4 model families × 5 feature variants × 5
  splits, for both regression and classification tasks.
- Two methodological findings on cross-validation bias under within-
  scaffold splitting and on single-permutation y-scramble unreliability.

## Repository structure

```
PAD4_BENCH/
├── data/                        # raw and processed datasets
├── pad4bench/                   # python package (curation, splits, benchmark)
├── scripts/                     # entry-point CLI scripts
├── results/                     # compute outputs (regression, classification, y-scramble)
├── manuscript/                  # paper sections, figures, tables
├── tests/                       # unit and integration tests
└── docs/                        # additional documentation
```

See [docs/](docs/) for module-level documentation.

## Quick start

```bash
# 1. Set up environment
conda env create -f environment.yml
conda activate pad4bench

# 2. Place raw source files in data/raw/
#    See data/raw/README.md for required files

# 3. Run curation
python scripts/run_curation.py

# 4. Generate splits
python scripts/run_splits.py

# 5. Run benchmark
python scripts/run_benchmark.py

# 6. Generate figures and tables
python scripts/run_figures.py
python scripts/run_tables.py
```

## Reproducibility

This release was produced with:

- Python 3.10.19
- Conda environment `pad4bench` (specification in `environment.yml`)
- See `requirements.txt` for pinned package versions

Random seeds are fixed throughout the pipeline. See `pad4bench/utils/seeds.py`
for the canonical seed registry.

## Citation

If you use PAD4-Bench in your work, please cite:

> [REF — to be filled in upon publication]

## License

Code is released under [LICENSE]. Data is released under [DATA_LICENSE].
See LICENSE files for details.

## Contact

[REF — corresponding author]
