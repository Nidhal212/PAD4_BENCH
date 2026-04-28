# pad4bench/figures — figure generators

```
figures/
├── __init__.py
├── _style.py        # Shared matplotlib style + color palettes
├── fig01_curation_flow.py
├── fig02_source_tier_crosstab.py
├── fig03_dataset_characterization.py
├── fig04_split_protocols.py
├── fig05_regression_heatmap.py
├── fig06_classification_heatmap.py
├── fig07_cv_test_gap.py
├── fig08_yscramble_distribution.py
└── figS1_stratified.py
```

Each module exposes a `render(results_dir, output_dir)` function. The
master runner is `scripts/run_figures.py`.
