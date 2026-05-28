# DMOI POC v0.2

Adds two external generalization tests to the v0.1 single-cohort POC, plus
a clean analysis of how calibration and class-prior shift behave when the
model moves to a new cohort.

This is a capability portrait, not a research result. Reproduces end-to-end
in about 5 minutes on an M-series Mac (after a one-time ~690 MB METABRIC
download).

## What's new since v0.1

| Capability | v0.1 | v0.2 |
|---|---|---|
| 5-fold CV on TCGA (full cohort) | AUROC 0.961 | — (rescoped) |
| 5-fold CV on TCGA train-only split | — | AUROC 0.954 (smaller train) |
| Nested calibration split | ✓ | ✓ |
| Held-out TCGA test (Path C, n=84) | — | **AUROC 0.968** |
| External validation on independent cohort | — | **METABRIC AUROC 0.909 (n=1,175)** |
| Calibration transfer analysis | — | T_TCGA=0.634 vs T_METABRIC=0.934 |
| LumB sensitivity decomposition | — | Prior shift (~50%) + meth silencing (~50%) |

## v0.2 headline numbers

| Metric | Value |
|---|---|
| 5-fold CV AUROC (TCGA train split, n=333) | 0.954 ± 0.017 |
| **Held-out TCGA test AUROC (n=84)** | **0.968** |
| **METABRIC external AUROC (n=1,175, RNA-only)** | **0.909** |
| ECE after T-scaling on held-out TCGA test | 0.079 (T=0.634) |
| ECE on METABRIC eval slice (cohort-specific T) | 0.074 (T_METABRIC=0.934) |

## The four-act story

1. **Baseline saturated the easy signal.** LogReg on concat(RNA, methylation)
   already lands at 0.963 AUROC on TCGA cohort_v2.

2. **Hypothesis-conditioned attention did NOT lift AUROC.** Architectural
   value is in the secondary disagreement signal, not the headline metric.

3. **Calibration was the v0.1 win.** Temperature scaling on a held-out
   15% cal split cuts ECE roughly in half on TCGA (T < 1 — model is
   under-confident).

4. **External generalization is the v0.2 win.** TCGA holdout AUROC 0.968.
   METABRIC AUROC 0.909. Calibration parameters are cohort-specific
   (TCGA's T over-sharpens METABRIC's meth-silenced predictions; a
   METABRIC-fit T near 1.0 is the right answer). The LumB sensitivity
   asymmetry on METABRIC decomposes cleanly into class-prior shift
   (~half, correctable via Bayes adjustment) and methylation silencing
   (~half, unrecoverable without a multi-modal external cohort that
   simply doesn't exist publicly).

## Honest caveats

- **Methylation branch is silenced for METABRIC.** METABRIC has no HM450
  data. No public BRCA cohort outside TCGA has paired RNA-seq + HM450 —
  see `docs/v0.2-design-external-validation.md` for the recon. The METABRIC
  result validates the RNA encoder's cross-cohort generalization, NOT the
  dual-modality story.
- **TCGA test split is small (n=84).** It catches gross overfit but isn't a
  large external cohort. METABRIC at n=1,175 is the heavier evidence.
- **Calibration didn't transfer.** A real, interpretable finding — but it
  means deploying DMOI to a new cohort requires fitting a cohort-specific T
  on a small labeled subset.

## What's in the repo (v0.2 additions vs v0.1)

- `src/dmoi_brca/external.py` — gene-symbol alignment, quantile
  normalization, meth-silenced inference helpers.
- `src/dmoi_brca/cohort.py` — adds `train_test_split_cohort` (stratified
  80/20 with fixed seed 2024).
- `src/dmoi_brca/train.py` — adds `pick_best_epoch` flag for held-out
  test scoring (prevents val-AUC peeking when val == test).
- `scripts/fetch_metabric.py` + `scripts/build_metabric_cohort.py` —
  METABRIC data ingestion via cBioPortal datahub.
- `scripts/eval_external.py` — full external-validation driver including
  the cohort-specific calibration comparison and LumB sensitivity
  decomposition.

## Reproduce

```bash
uv sync
python scripts/build_cohort_v2.py          # TCGA cohort + train/test split
python scripts/run_baseline_v2.py          # LogReg/RF baselines
python scripts/eval_dmoi.py                # TCGA CV + held-out test
python scripts/fetch_metabric.py           # one-time ~690 MB download
python scripts/build_metabric_cohort.py
python scripts/eval_external.py            # METABRIC external + cal + sens
```

## Test status

All unit tests pass. ruff clean. CJK gate clean (English-only on all public
artifacts).
