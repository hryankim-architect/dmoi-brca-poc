# `scripts/`

Operational helpers, not the pipeline itself.

- `run_lab.sh`, one-liner to invoke `make run` on a lab node
  with the substrate env vars set to lab defaults.

## Pipeline & analysis scripts

The numbered build/eval scripts (`build_cohort*.py`,
`build_metabric_cohort*.py`, `eval_dmoi*.py`, `eval_metabric*.py`,
`calibrate_transfer.py`) are self-documenting via their module docstrings, and
each writes a dated audit artifact under `audit/`. Headline-result reproduce
blocks live in the top-level `README.md` and the per-version `audit/dmoi_v0.*.md`
files — this list is deliberately not exhaustive.

Most recent additions:

- `calibrate_transfer.py` — v0.13 cross-cohort calibration transfer
  (→ `audit/dmoi_calibration_transfer_v0.13.md`).
- `build_cohort_v4.py` / `build_metabric_cohort_v4.py` — v0.14 HER2-vs-Luminal
  TCGA + METABRIC cohorts.
- `eval_dmoi_v0.14.py` / `eval_dmoi_v0.14_cv.py` — v0.14 HER2 axis, single-split
  (TCGA + METABRIC) and 5-fold CV (→ `audit/dmoi_v0.14.md`, `audit/dmoi_v0.14_cv.md`).
