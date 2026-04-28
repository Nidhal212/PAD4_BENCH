# y-Scramble distribution results

Output of `scripts/run_benchmark.py --y-scramble`.

```
y_scramble/
├── y_scramble_distribution.csv     # Per-cell mean, std, p05, p95
├── per_cell/                       # Per-permutation metrics
│   └── <cell>/
│       └── permutations.csv
└── manifest.json
```

Distribution is computed from ≥10 permutations per cell. Single-
permutation results are NOT a supported reporting mode.
