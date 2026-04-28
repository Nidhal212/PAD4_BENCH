# pad4bench/splits — 5 split protocols

Each split protocol is implemented as a generator function.

```
splits/
├── __init__.py
├── scaffold.py
├── random.py
├── confirmed.py
├── lead_opt.py
├── similarity.py
└── validate.py    # InChIKey disjointness check + validation hold-out
```

The `validate.py` module enforces:

1. No InChIKey-14 appears in both train and test of the same split.
2. No InChIKey-14 from `pad4_validation.csv` appears in train or test of
   any split.

Failures are loud (exception raised). This is enforced as part of the
split generation, not as a post-hoc check.
