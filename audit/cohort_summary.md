# DMOI POC Cohort Summary (Day-3)

Generated: 2026-05-27T20:20:59Z

## Inputs

- Clinical matrix: `BRCA_clinicalMatrix.tsv` (1247 rows)
- RNA-seq samples (HiSeqV2): 1218
- HM450 methylation samples: 888

## Cohort splits

| Pole | Definition | n |
|---|---|---|
| H+ (luminal) | PAM50 in {LumA, LumB} AND ER positive | 547 |
| H- (basal/TN) | PAM50 = Basal AND ER/PR/HER2 all negative | 103 |
| **Total** | | **650** |

## Modality coverage

- Both RNA + methylation: 395 (DMOI dual-modality training set)
- RNA only: 255
- Methylation only: 0

## Reproduce

```bash
make data           # or bash scripts/download_tcga_brca.sh
python scripts/build_cohort.py
```

## Notes

- PAM50 source: `PAM50Call_RNAseq` (primary, ~956/1247 coverage) with fallback to `PAM50_mRNA_nature2012` (long-form labels normalized).
- Other PAM50 subtypes (Her2-enriched, Normal-like) are excluded from the POC — DMOI POC contrasts the H+ vs H- poles only.
- `cohort.tsv` lives under `data/` and is gitignored (sample IDs are TCGA barcodes, derived from open-tier data but kept out of git per the scaffold's data-handling convention).
