# pad4bench/utils — shared helpers

```
utils/
├── __init__.py
├── paths.py        # Project-relative path resolution
├── seeds.py        # Canonical seed registry
├── logging.py      # Structured logging
├── io.py           # CSV/JSON readers and writers
└── chem.py         # RDKit helpers (InChIKey, scaffold, fingerprint)
```

The `seeds.py` registry is the single source of truth for all RNG seeds
in the pipeline. Anything that uses an RNG must import its seed from here
rather than hardcoding.
