# DMOI v0.13 — Cross-Cohort Calibration Transfer (TCGA → METABRIC)

Generated: 2026-06-02T17:34:58Z

## Framing

Temperature scaling and the affine/odds transforms below are **monotonic**, so AUROC is invariant across all conditions (AUROC = 0.9115 on the eval slice). This is a **calibration-quality** result (ECE + Brier), not an accuracy claim.

- TCGA train priors: π_LumB = 0.306; METABRIC: π_LumB = 0.404
- METABRIC cal pool: n=176 (stratified 15%, seed=2024); eval slice: n=999
- Gene overlap (METABRIC vs TCGA train): shared=16890

## Conditions (fixed METABRIC eval slice)

| Condition | T | ECE | Brier |
|---|---|---|---|
| A_uncalibrated | — | 0.0745 | 0.1294 |
| B_TCGA_T | 0.634 | 0.1051 | 0.1349 |
| C_METABRIC_oracle_T | 0.934 | 0.0738 | 0.1292 |
| D2_labelfree_align_TCGA_T | 0.634 | 0.0974 | 0.1335 |
| D3_prior_odds | — | 0.1004 | 0.1221 |

Legend: A uncalibrated · B naive TCGA cal-split T (v0.2 failure case) · C METABRIC oracle T (labelled cal pool) · D2 label-free logit alignment + TCGA T · D3 class-prior odds correction.

## D1 — METABRIC-mini learning curve (mean ECE over seeds)

| labelled n | mean ECE |
|---|---|
| 30 | 0.0852 |
| 50 | 0.0855 |
| 100 | 0.0821 |

Full per-(n, seed) detail: `dmoi_calibration_transfer_v0.13_learning_curve.tsv`. Reliability bins: `dmoi_calibration_transfer_v0.13_reliability.tsv`.

## Verdict

- The model is **already calibrated on METABRIC out of the box**: raw ECE 0.0745 vs labelled oracle 0.0738 (headroom only +0.0007). There is almost no calibration to recover.
- Naive TCGA-T worsens calibration (ECE 0.0745 -> 0.1051) — reproduces and sharpens the v0.2 finding: TCGA's temperature should NOT be imported.
- **No transfer method beats doing nothing.** The best attempt (D1_mini_n100, ECE 0.0821) is still worse than the uncalibrated baseline (0.0745). Recommended cross-cohort policy: apply no temperature; the raw probabilities are the best-calibrated available without a fully labelled target cohort.
- Brier nuance: class-prior odds correction (D3) gives the best Brier (0.1221 vs raw 0.1294) by matching METABRIC's higher LumB base rate, even though its binned ECE is worse — a probability-accuracy vs bin-calibration trade-off worth noting.
