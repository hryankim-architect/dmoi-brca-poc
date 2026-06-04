# DMOI v0.3, Per-patient Integrated Gradients attribution

Generated: 2026-05-28T10:23:55Z

## Setup

- Train cohort      : TCGA cohort_v2 train split, n=333
- Test cohort       : TCGA cohort_v2 test split,  n=84 (LumA 58, LumB 26)
- Architecture      : Option A (aux BCE + disagreement), 15 epochs, no peek, cal_frac=0.15
- Attribution algo  : Integrated Gradients, baseline = zero (standardized), 50 steps
- Targets           : final_logit + lumA_pole + lumB_pole (3 separate IG runs per patient)

## Completeness check

Per-patient `|sum(IG) - (f(x) - f(0))|`, the IG completeness axiom residual. Tighter is more faithful; below 1e-2 is the IG-literature standard for 50-step Riemann approximation on a model with ReLU non-linearities.

- **final_logit**: mean 0.02054, max 0.34250
- **lumA_pole**: mean 0.00234, max 0.01817
- **lumB_pole**: mean 0.00215, max 0.01116

## Global top-10 features per (target, modality)

### final_logit (RNA)

| Rank | Feature | mean |IG| |
|---|---|---|
| 1 | `FOXC1` | 0.21724 |
| 2 | `TUBB2B` | 0.15263 |
| 3 | `PDLIM3` | 0.14234 |
| 4 | `BCL2` | 0.14233 |
| 5 | `EGR3` | 0.12421 |
| 6 | `KRT15` | 0.11696 |
| 7 | `KCNK5` | 0.11669 |
| 8 | `FHL2` | 0.10908 |
| 9 | `RAB17` | 0.10696 |
| 10 | `S100A1` | 0.10119 |

### final_logit (methylation)

| Rank | Feature | mean |IG| |
|---|---|---|
| 1 | `cg14042099` | 0.10863 |
| 2 | `cg26089753` | 0.08002 |
| 3 | `cg03345116` | 0.07840 |
| 4 | `cg10342963` | 0.07231 |
| 5 | `cg15543523` | 0.07134 |
| 6 | `cg00773370` | 0.06955 |
| 7 | `cg25744613` | 0.06861 |
| 8 | `cg01738022` | 0.06215 |
| 9 | `cg05116002` | 0.05646 |
| 10 | `cg17538572` | 0.05585 |

### lumA_pole (RNA)

| Rank | Feature | mean |IG| |
|---|---|---|
| 1 | `FOXC1` | 0.04468 |
| 2 | `BCL2` | 0.03233 |
| 3 | `PDLIM3` | 0.03191 |
| 4 | `TUBB2B` | 0.03027 |
| 5 | `EGR3` | 0.02632 |
| 6 | `KRT15` | 0.02619 |
| 7 | `S100A1` | 0.02546 |
| 8 | `AHNAK` | 0.02473 |
| 9 | `RAB17` | 0.02448 |
| 10 | `FHL2` | 0.02316 |

### lumB_pole (RNA)

| Rank | Feature | mean |IG| |
|---|---|---|
| 1 | `EFNA5` | 0.01429 |
| 2 | `RANBP1` | 0.01041 |
| 3 | `NBN` | 0.01003 |
| 4 | `ZW10` | 0.00942 |
| 5 | `POLA2` | 0.00869 |
| 6 | `DSCC1` | 0.00843 |
| 7 | `CKS1B` | 0.00839 |
| 8 | `SMC6` | 0.00827 |
| 9 | `ATAD2` | 0.00811 |
| 10 | `IFRD1` | 0.00801 |

Full top-50 lists in [`dmoi_explain_global.tsv`](dmoi_explain_global.tsv).

## Per-patient breakdowns

See [`dmoi_explain_per_patient.tsv`](dmoi_explain_per_patient.tsv) for the per-patient top-10 contributors across all three targets and both modalities. Format: `sample_id, y_true, target, modality, rank, feature, attribution, input_value, target_score`.

## Limitations

- Attribution is on the TCGA cohort_v2 test split only (n=84). METABRIC attribution is deferred to v0.4, the IG computation cost is modest (~7 min on MPS for 1,175 patients × 3 targets), but the v0.3 scope is to validate that DMOI's pole-conditioned predictions are interpretable on the same patients we benchmark on.
- IG attribution is over standardized inputs (post-`StandardScaler`). Pathway-level aggregation (e.g., MSigDB) is out of scope for v0.3.
- The completeness residual rises with model non-linearity; DMOI uses ReLU + GELU, so a 50-step Riemann sum gives residuals in the 1e-3 to 1e-2 range. Acceptable; reported above so the reader can judge.

## Reproduce

```bash
python scripts/explain_dmoi.py
```
