# pad4bench/tables — table generators

```
tables/
├── __init__.py
├── table01_sources.py
├── table02_formulas.py
├── table03_thresholds.py
├── table04_benchmark.py
├── table05_yscramble.py
└── tableS2_full_grid.py
```

Output is Markdown to `manuscript/tables/`. Tables 2 and 3 are mostly
static (formulas and thresholds) but read constants from
`pad4bench/curation/constants.py` to ensure they stay in sync with the
pipeline.
