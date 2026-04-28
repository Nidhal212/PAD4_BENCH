# Raw source data

Place source files here, organized by source database. Files are NOT
included in the git repository (they are too large and have their own
licenses); they must be obtained from the original sources.

```
raw/
├── pubchem/
│   ├── AID_463073_datatable_all.csv     # qHTS PAD4
│   ├── AID_485272_datatable_all.csv     # qHTS confirmation
│   ├── ...
│   └── AID_1805620_datatable_all.csv    # External validation set
├── chembl/
│   └── activities_CHEMBL6111.tsv        # PAD4 (CHEMBL6111)
└── bindingdb/
    └── BindingDB_PAD4.tsv
```

## Required files (33 sources total)

See `manuscript/tables/table1_sources.md` for the full source manifest.

## Validation set

`AID_1805620_datatable_all.csv` is the **held-out external validation set**.
It is treated specially by the pipeline: records from this AID are routed
to `data/processed/pad4_validation.csv` and are explicitly excluded from
all five split protocols' training and test partitions. See
`pad4bench/curation/validation.py` for the implementation.

If this file is missing, `scripts/run_curation.py` will fail loudly rather
than silently producing a dataset without external validation.
