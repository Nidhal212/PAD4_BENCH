# pad4bench/curation — 13-stage curation pipeline

Implements the curation methodology described in §3 of the manuscript.

Module structure (planned):

```
curation/
├── __init__.py
├── stage01_ingestion.py         # Per-source ingestion adapters
├── stage02_standardization.py   # Structure standardization
├── stage03_units.py             # Unit normalization
├── stage04_tiering.py           # Tier assignment
├── stage05_replicate_weight.py  # Per-record replicate weighting
├── stage06_aggregation.py       # Per-compound aggregation (weighted median)
├── stage07_label_uncertainty.py # LU_v2 score
├── stage08_subtier.py           # T1_high/T1_confirmed/T1_standard
├── stage09_hts_denoising.py     # T3 denoising
├── stage10_cliffs.py            # Activity cliff detection
├── stage11_ad.py                # Applicability domain scoring
├── stage12_metadata.py          # Final metadata columns
├── stage13_qc.py                # 25 reproducible QC assertions
├── validation.py                # AID 1805620 routing (held-out)
└── constants.py                 # All thresholds and weights (Tables 2, 3)
```

The validation routing is in its own module to make the held-out logic
easy to audit.
