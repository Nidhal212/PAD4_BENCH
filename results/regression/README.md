# Regression task results

Output of `scripts/run_benchmark.py --task regression`.

```
regression/
├── summary_all.csv         # 100 cells: split × variant × model
├── stratified_all.csv      # Per-stratum metrics
├── per_cell/               # One subdir per (split, variant, model) cell
│   └── <cell>/
│       ├── predictions.csv
│       ├── metrics.json
│       └── feature_importance.json   (if applicable)
└── manifest.json           # Hashes of input data + this run's config
```
