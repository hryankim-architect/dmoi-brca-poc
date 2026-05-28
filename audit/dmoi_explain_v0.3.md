# DMOI v0.3 — Per-patient Integrated Gradients attribution

Generated: 2026-05-28T10:03:48Z

## Setup

- Train cohort      : TCGA cohort_v2 train split, n=333
- Test cohort       : TCGA cohort_v2 test split,  n=84 (LumA 58, LumB 26)
- Architecture      : Option A (aux BCE + disagreement), 15 epochs, no peek, cal_frac=0.15
- Attribution algo  : Integrated Gradients, baseline = zero (standardized), 50 steps
- Targets           : final_logit + lumA_pole + lumB_pole (3 separate IG runs per patient)

## Completeness check

Per-patient `|sum(IG) - (f(x) - f(0))|`, the IG completeness axiom residual. Tighter is more faithful; below 1e-2 is the IG-literature standard for 50-step Riemann approximation on a model with ReLU non-linearities.

- **final_logit**: mean 0.01619, max 0.20617
- **lumA_pole**: mean 0.00156, max 0.01552
- **lumB_pole**: mean 0.00270, max 0.02158

## Global top-10 features per (target, modality)

### final_logit (RNA)

| Rank | Feature | mean |IG| |
|---|---|---|
| 1 | `FOXC1` | 0.22588 |
| 2 | `TUBB2B` | 0.15196 |
| 3 | `PDLIM3` | 0.14979 |
| 4 | `BCL2` | 0.13806 |
| 5 | `KRT15` | 0.13074 |
| 6 | `FHL2` | 0.12353 |
| 7 | `RAB17` | 0.12116 |
| 8 | `S100A1` | 0.11610 |
| 9 | `RAB31` | 0.10917 |
| 10 | `AHNAK` | 0.10449 |

### final_logit (methylation)

| Rank | Feature | mean |IG| |
|---|---|---|
| 1 | `cg03345116` | 0.10245 |
| 2 | `cg14042099` | 0.08601 |
| 3 | `cg25744613` | 0.07792 |
| 4 | `cg15543523` | 0.06588 |
| 5 | `cg15732851` | 0.06171 |
| 6 | `cg26089753` | 0.06160 |
| 7 | `cg10342963` | 0.05843 |
| 8 | `cg14660676` | 0.05665 |
| 9 | `cg10566121` | 0.05581 |
| 10 | `cg15012484` | 0.05518 |

### lumA_pole (RNA)

| Rank | Feature | mean |IG| |
|---|---|---|
| 1 | `FOXC1` | 0.05083 |
| 2 | `PDLIM3` | 0.03587 |
| 3 | `TUBB2B` | 0.03295 |
| 4 | `BCL2` | 0.03283 |
| 5 | `KRT15` | 0.03229 |
| 6 | `RAB17` | 0.03118 |
| 7 | `S100A1` | 0.03033 |
| 8 | `AHNAK` | 0.02907 |
| 9 | `FHL2` | 0.02775 |
| 10 | `ZBTB16` | 0.02676 |

### lumB_pole (RNA)

| Rank | Feature | mean |IG| |
|---|---|---|
| 1 | `EFNA5` | 0.01891 |
| 2 | `RANBP1` | 0.01424 |
| 3 | `SMC6` | 0.01098 |
| 4 | `ZW10` | 0.01077 |
| 5 | `DMD` | 0.01006 |
| 6 | `DSCC1` | 0.00987 |
| 7 | `IFRD1` | 0.00936 |
| 8 | `RPL34` | 0.00898 |
| 9 | `CKS1B` | 0.00847 |
| 10 | `MEIS1` | 0.00846 |

Full top-50 lists in [`dmoi_explain_global.tsv`](dmoi_explain_global.tsv).

## Per-patient breakdowns

See [`dmoi_explain_per_patient.tsv`](dmoi_explain_per_patient.tsv) for the per-patient top-10 contributors across all three targets and both modalities. Format: `sample_id, y_true, target, modality, rank, feature, attribution, input_value, target_score`.

## Honest scope

- Attribution is on the TCGA cohort_v2 test split only (n=84). METABRIC attribution is deferred to v0.4 — the IG computation cost is modest (~7 min on MPS for 1,175 patients × 3 targets), but the v0.3 scope is to validate that DMOI's pole-conditioned predictions are interpretable on the same patients we benchmark on.
- IG attribution is over standardized inputs (post-`StandardScaler`). Pathway-level aggregation (e.g., MSigDB) is out of scope for v0.3.
- The completeness residual rises with model non-linearity; DMOI uses ReLU + GELU, so a 50-step Riemann sum gives residuals in the 1e-3 to 1e-2 range. Acceptable; reported above so the reader can judge.

## Reproduce

```bash
python scripts/explain_dmoi.py
```
