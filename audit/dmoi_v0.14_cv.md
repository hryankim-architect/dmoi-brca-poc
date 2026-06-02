# DMOI v0.14 CV -- 5-fold stability check for HER2-vs-Luminal

Generated: 2026-06-02T18:08:15Z

## Setup

- Architecture: v0.6 base (same as v0.14 single-split), n_pathways=0.
- Cohort: TCGA cohort_v4 (HER2 + Luminal dual-modality, n=436; HER2=58, Luminal=378).
- Split: 5-fold StratifiedKFold (random_state=42), pick_best_epoch=True.
- POLE_HER2 = PI3K_AKT_MTOR + MTORC1 + G2M_CHECKPOINT; POLE_LUMINAL_ER = ER_EARLY + ER_LATE.
- v0.14 single-split references: TCGA test AUROC 0.891 / bacc 0.849 (n_test=88, HER2=12); METABRIC external AUROC 0.893.

## Aggregate AUROC + bacc (5-fold)

| Metric | mean | std |
|---|---|---|
| AUROC | **0.8921** | 0.0558 |
| bacc  | **0.7934** | 0.0564 |

## Per-fold table

| Fold | AUROC | bacc | Luminal IG top-5 ∩ priors | HER2 IG top-5 ∩ priors | best epoch |
|---|---|---|---|---|---|
| 1 | 0.9002 | 0.8487 | 2 / 2 | 3 / 3 | 15 |
| 2 | 0.8816 | 0.7398 | 2 / 2 | 3 / 3 | 13 |
| 3 | 0.9833 | 0.8571 | 2 / 2 | 3 / 3 | 15 |
| 4 | 0.8444 | 0.7433 | 2 / 2 | 3 / 3 | 13 |
| 5 | 0.8511 | 0.7783 | 2 / 2 | 3 / 3 | 9 |

## Cross-fold pathway frequency

### Luminal pole -- frequency in per-fold top-5 (out of 5)

| Pathway | Frequency |
|---|---|
| `HALLMARK_ESTROGEN_RESPONSE_LATE` | 5 / 5 |
| `HALLMARK_ESTROGEN_RESPONSE_EARLY` | 5 / 5 |

### HER2 pole -- frequency in per-fold top-5 (out of 5)

| Pathway | Frequency |
|---|---|
| `HALLMARK_G2M_CHECKPOINT` | 5 / 5 |
| `HALLMARK_PI3K_AKT_MTOR_SIGNALING` | 5 / 5 |
| `HALLMARK_MTORC1_SIGNALING` | 5 / 5 |

## Cross-fold top-3 stability (pairwise mean Jaccard)

- Luminal pole top-3 mean pairwise Jaccard : **0.6000**
- HER2    pole top-3 mean pairwise Jaccard : **1.0000**

## Honest scope

- Same architecture and priors as v0.14 single-split. Only the train/val split changes across folds.
- HER2 is the small class (~12 per val fold); the AUROC band is wider than a larger-cohort axis would give -- that width is the honest deliverable, and METABRIC (n=224 HER2) carries the cross-cohort weight.
- No METABRIC scoring here; eval_dmoi_v0.14.py covers cross-cohort. v0.14 CV is purely a TCGA stability check.

## Reproduce

```bash
python scripts/build_cohort_v4.py     # if cohort_v4.tsv not built
python scripts/eval_dmoi_v0.14_cv.py
```
