# Five split protocols

Each split has its own subdirectory with `train.csv` and `test.csv`.

```
splits/
├── scaffold/
├── random/
├── confirmed/
├── lead_opt/
└── similarity/
```

Each train/test pair is constructed from `data/processed/pad4_t1_full.csv`
(plus T2 records with valid `activity_label` for classification splits).
The held-out validation set (`pad4_validation.csv`) is excluded from all
splits' training and test partitions; this is enforced by an InChIKey-14
disjointness check in `pad4bench/splits/validate.py`.
