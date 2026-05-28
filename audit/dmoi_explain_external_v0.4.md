# DMOI v0.4 — METABRIC per-patient Integrated Gradients attribution

Generated: 2026-05-28T10:31:44Z

## Setup

- Train cohort  : TCGA cohort_v2 train split, n=333
- External      : METABRIC LumA/LumB with mRNA, n=1175 (LumA=700, LumB=475)
- Architecture  : Option A (aux BCE + disagreement), 15 epochs, no peek, cal_frac=0.15
- External AUROC: 0.9095
- Methylation   : silenced (METABRIC has no HM450)
- IG steps      : 50

## Completeness check

- **final_logit**: mean 0.01547, max 0.22759
- **lumA_pole**: mean 0.00147, max 0.01629
- **lumB_pole**: mean 0.00226, max 0.02042

## Global top-10 RNA features per target (METABRIC)

### final_logit

| Rank | Feature | mean |IG| |
|---|---|---|
| 1 | `TUBB2B` | 0.17719 |
| 2 | `FOXC1` | 0.17117 |
| 3 | `PDLIM3` | 0.15813 |
| 4 | `BCL2` | 0.15478 |
| 5 | `KRT15` | 0.13376 |
| 6 | `EGR3` | 0.13234 |
| 7 | `RAB17` | 0.11814 |
| 8 | `DUSP4` | 0.11649 |
| 9 | `KCNK15` | 0.11295 |
| 10 | `ZBTB16` | 0.10698 |

### lumA_pole

| Rank | Feature | mean |IG| |
|---|---|---|
| 1 | `FOXC1` | 0.03805 |
| 2 | `TUBB2B` | 0.03559 |
| 3 | `BCL2` | 0.03355 |
| 4 | `PDLIM3` | 0.03312 |
| 5 | `EGR3` | 0.03072 |
| 6 | `ZBTB16` | 0.03051 |
| 7 | `KRT15` | 0.02883 |
| 8 | `RAB17` | 0.02578 |
| 9 | `DLC1` | 0.02516 |
| 10 | `AHNAK` | 0.02448 |

### lumB_pole

| Rank | Feature | mean |IG| |
|---|---|---|
| 1 | `POLA2` | 0.01147 |
| 2 | `EFNA5` | 0.01089 |
| 3 | `CKS1B` | 0.01059 |
| 4 | `RANBP1` | 0.01014 |
| 5 | `ZW10` | 0.00934 |
| 6 | `DBF4` | 0.00932 |
| 7 | `NBN` | 0.00929 |
| 8 | `NDC80` | 0.00900 |
| 9 | `IFRD1` | 0.00868 |
| 10 | `DSCC1` | 0.00843 |

## Cross-cohort comparison vs TCGA test (v0.3)

The interpretability headline: do the same genes dominate when the trained model attributes on a completely different cohort? If yes, the biology the model learned generalizes; if no, the v0.3 finding was cohort-specific.

### final_logit

- Jaccard(top-10) = **0.538** · Jaccard(top-50) = **0.724**
- Shared top-10 genes: `BCL2`, `EGR3`, `FOXC1`, `KRT15`, `PDLIM3`, `RAB17`, `TUBB2B`

| Rank | TCGA test top-10 | METABRIC top-10 |
|---|---|---|
| 1 | `FOXC1` | `TUBB2B` |
| 2 | `TUBB2B` | `FOXC1` |
| 3 | `PDLIM3` | `PDLIM3` |
| 4 | `BCL2` | `BCL2` |
| 5 | `EGR3` | `KRT15` |
| 6 | `KRT15` | `EGR3` |
| 7 | `KCNK5` | `RAB17` |
| 8 | `FHL2` | `DUSP4` |
| 9 | `RAB17` | `KCNK15` |
| 10 | `S100A1` | `ZBTB16` |

### lumA_pole

- Jaccard(top-10) = **0.667** · Jaccard(top-50) = **0.786**
- Shared top-10 genes: `AHNAK`, `BCL2`, `EGR3`, `FOXC1`, `KRT15`, `PDLIM3`, `RAB17`, `TUBB2B`

| Rank | TCGA test top-10 | METABRIC top-10 |
|---|---|---|
| 1 | `FOXC1` | `FOXC1` |
| 2 | `BCL2` | `TUBB2B` |
| 3 | `PDLIM3` | `BCL2` |
| 4 | `TUBB2B` | `PDLIM3` |
| 5 | `EGR3` | `EGR3` |
| 6 | `KRT15` | `ZBTB16` |
| 7 | `S100A1` | `KRT15` |
| 8 | `AHNAK` | `RAB17` |
| 9 | `RAB17` | `DLC1` |
| 10 | `FHL2` | `AHNAK` |

### lumB_pole

- Jaccard(top-10) = **0.667** · Jaccard(top-50) = **0.538**
- Shared top-10 genes: `CKS1B`, `DSCC1`, `EFNA5`, `IFRD1`, `NBN`, `POLA2`, `RANBP1`, `ZW10`

| Rank | TCGA test top-10 | METABRIC top-10 |
|---|---|---|
| 1 | `EFNA5` | `POLA2` |
| 2 | `RANBP1` | `EFNA5` |
| 3 | `NBN` | `CKS1B` |
| 4 | `ZW10` | `RANBP1` |
| 5 | `POLA2` | `ZW10` |
| 6 | `DSCC1` | `DBF4` |
| 7 | `CKS1B` | `NBN` |
| 8 | `SMC6` | `NDC80` |
| 9 | `ATAD2` | `IFRD1` |
| 10 | `IFRD1` | `DSCC1` |

## Honest caveats

- **Methylation silenced.** METABRIC has no HM450 data, so the methylation branch receives a fixed zero (raw-domain) tensor, which after the train-fitted StandardScaler becomes a fixed `-mean/std` per probe. The lumA/lumB attribution focuses on the RNA branch only; the methylation attribution column is reported for completeness but is uninformative (all patients see the same meth input).
- **Cross-cohort gene set differs slightly.** 16890 of TCGA's 20530 RNA genes are shared with METABRIC; the remainder are mean-imputed to zero in METABRIC. A gene that is informative on TCGA but absent from METABRIC cannot appear in METABRIC attributions.
- **Quantile normalization is applied per gene.** METABRIC's per-gene empirical distribution is mapped to TCGA train's per-gene CDF before standardization, so each gene's attribution is computed on inputs that match the training distribution.

## Reproduce

```bash
python scripts/explain_dmoi.py        # TCGA test attribution (v0.3)
python scripts/explain_metabric.py    # METABRIC external attribution (v0.4)
```
