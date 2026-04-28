# Final processed datasets

Output of the curation pipeline. These are the files referenced in the
paper.

```
processed/
├── pad4_t1_high.csv             # T1_high sub-tier
├── pad4_t1_confirmed.csv        # T1_confirmed sub-tier
├── pad4_t1_standard.csv         # T1_standard sub-tier
├── pad4_t1_full.csv             # All T1 (concatenation of three sub-tiers)
├── pad4_t2_censored.csv         # T2 censored measurements
├── pad4_t3_hts_denoised.csv     # T3 high-throughput screening
├── pad4_ki.csv                  # Ki measurements
├── pad4_validation.csv          # Held-out external validation (AID 1805620)
├── dataset_summary.json         # Compound counts and provenance
├── pipeline_attrition.json      # Per-stage record counts (for Figure 1)
└── pipeline_config.json         # Pipeline configuration for this build
```
