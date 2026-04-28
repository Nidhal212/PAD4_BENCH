# pad4bench/benchmark — training and evaluation

```
benchmark/
├── __init__.py
├── features.py        # 5 feature variants: full, fingerprints, physchem, mordred, fragments
├── models.py          # 4 model families per task (regression/classification)
├── train.py           # Single-cell training
├── evaluate.py        # Held-out test metrics + stratified breakdown
├── cross_validate.py  # 5-fold CV (and CV-test gap)
├── y_scramble.py      # Distribution-based y-scramble
└── grid.py            # Run the full split × variant × model grid
```

Note: `y_scramble.py` is distribution-based by default (≥10 permutations).
Single-permutation y-scrambles are NOT supported as a primary mode; we
learned the hard way (§5.5).
