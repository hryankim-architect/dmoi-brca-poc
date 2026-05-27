# DMOI POC Cohort v2 Summary (Day-5A — Week-2 re-scope)

Generated: 2026-05-27T20:59:17Z

## Rationale

Day-4 baseline saturated at AUROC=1.0 on H+ luminal vs H- basal (cohort v1).
Re-scoped Week-2 target to **within-luminal LumA vs LumB** — both poles ER+,
discriminating axis is proliferation rate (LumB high Ki67/cell cycle).
Literature baseline AUC ~0.70-0.85 on single-omic, much harder.

## Inputs

- Clinical matrix: `BRCA_clinicalMatrix.tsv` (1247 rows)
- RNA-seq samples (HiSeqV2): 1218
- HM450 methylation samples: 888

## Cohort v2 splits

| Pole | Definition | n |
|---|---|---|
| LumA | PAM50 = LumA (low proliferation, ER+) | 437 |
| LumB | PAM50 = LumB (high proliferation, ER+) | 198 |
| **Total** | | **635** |

## Modality coverage

- Both RNA + methylation: 417 (DMOI dual-modality v2 training set)
- RNA only: 218
- Methylation only: 0

## Reproduce

```bash
python scripts/build_cohort_v2.py
```
