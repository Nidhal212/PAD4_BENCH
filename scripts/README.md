# Entry-point scripts

CLI scripts that orchestrate the package modules. These are the user-
facing scripts, expected to be run in this order:

1. `run_curation.py`   — Run the 13-stage pipeline (data/raw → data/processed)
2. `run_splits.py`     — Generate 5 split protocols (data/processed → data/splits)
3. `run_benchmark.py`  — Run the 200-cell grid + y-scramble distribution
4. `run_figures.py`    — Generate all figures
5. `run_tables.py`     — Generate all tables
6. `run_smoke_tests.py` — Verify outputs match expected hashes

Each script supports `--help`, `--dry-run`, and `--from-stage N`.
