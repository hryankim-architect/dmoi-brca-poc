# DMOI v0.6, Full 50-set Hallmark IG aggregation

Generated: 2026-05-28T11:06:31Z

## Setup

- Train cohort     : TCGA cohort_v2 train split, n=333
- TCGA test cohort : n=84 (AUROC 0.9682)
- METABRIC cohort  : n=1175 (RNA-only, meth silenced)
- Pathway catalog  : 50 MSigDB Hallmark v2024.1.Hs sets loaded from `data/msigdb/h.all.v2024.1.Hs.symbols.gmt`
- Aggregation      : per-pathway `mean |IG|`, `sum_signed`, `signed_mean` over per-patient × per-gene attributions

## v0.5 finding survives the 50-set widening?

All v0.5 top pathways are still in the v0.6 (50-set) top-3 on both cohorts. The 5-set rollup wasn't an artifact of which sets were loaded, the same pathways win out of 50.

### lumA_pole

- **TCGA test** top-3 (of 50) = `HALLMARK_ESTROGEN_RESPONSE_EARLY`, `HALLMARK_ESTROGEN_RESPONSE_LATE`, `HALLMARK_IL2_STAT5_SIGNALING`. v0.5 finding (`HALLMARK_ESTROGEN_RESPONSE_EARLY`, `HALLMARK_ESTROGEN_RESPONSE_LATE`): all present.
- **METABRIC** top-3 (of 50) = `HALLMARK_ESTROGEN_RESPONSE_EARLY`, `HALLMARK_ESTROGEN_RESPONSE_LATE`, `HALLMARK_IL2_STAT5_SIGNALING`. v0.5 finding (`HALLMARK_ESTROGEN_RESPONSE_EARLY`, `HALLMARK_ESTROGEN_RESPONSE_LATE`): all present.

### lumB_pole

- **TCGA test** top-3 (of 50) = `HALLMARK_E2F_TARGETS`, `HALLMARK_G2M_CHECKPOINT`, `HALLMARK_MYC_TARGETS_V1`. v0.5 finding (`HALLMARK_E2F_TARGETS`, `HALLMARK_G2M_CHECKPOINT`, `HALLMARK_MYC_TARGETS_V1`): all present.
- **METABRIC** top-3 (of 50) = `HALLMARK_E2F_TARGETS`, `HALLMARK_G2M_CHECKPOINT`, `HALLMARK_MYC_TARGETS_V1`. v0.5 finding (`HALLMARK_E2F_TARGETS`, `HALLMARK_G2M_CHECKPOINT`, `HALLMARK_MYC_TARGETS_V1`): all present.

### final_logit

- **TCGA test** top-3 (of 50) = `HALLMARK_ESTROGEN_RESPONSE_EARLY`, `HALLMARK_ESTROGEN_RESPONSE_LATE`, `HALLMARK_G2M_CHECKPOINT`. v0.5 finding (`HALLMARK_ESTROGEN_RESPONSE_EARLY`, `HALLMARK_ESTROGEN_RESPONSE_LATE`): all present.
- **METABRIC** top-3 (of 50) = `HALLMARK_ESTROGEN_RESPONSE_EARLY`, `HALLMARK_ESTROGEN_RESPONSE_LATE`, `HALLMARK_G2M_CHECKPOINT`. v0.5 finding (`HALLMARK_ESTROGEN_RESPONSE_EARLY`, `HALLMARK_ESTROGEN_RESPONSE_LATE`): all present.

## Cross-cohort top-3 (of 50)

### lumA_pole

- TCGA test top-3 (of 50): `HALLMARK_ESTROGEN_RESPONSE_EARLY`, `HALLMARK_ESTROGEN_RESPONSE_LATE`, `HALLMARK_IL2_STAT5_SIGNALING`
- METABRIC top-3 (of 50) : `HALLMARK_ESTROGEN_RESPONSE_EARLY`, `HALLMARK_ESTROGEN_RESPONSE_LATE`, `HALLMARK_IL2_STAT5_SIGNALING`
- Shared : `HALLMARK_ESTROGEN_RESPONSE_EARLY`, `HALLMARK_ESTROGEN_RESPONSE_LATE`, `HALLMARK_IL2_STAT5_SIGNALING`

### lumB_pole

- TCGA test top-3 (of 50): `HALLMARK_MYC_TARGETS_V1`, `HALLMARK_E2F_TARGETS`, `HALLMARK_G2M_CHECKPOINT`
- METABRIC top-3 (of 50) : `HALLMARK_E2F_TARGETS`, `HALLMARK_G2M_CHECKPOINT`, `HALLMARK_MYC_TARGETS_V1`
- Shared : `HALLMARK_E2F_TARGETS`, `HALLMARK_G2M_CHECKPOINT`, `HALLMARK_MYC_TARGETS_V1`

### final_logit

- TCGA test top-3 (of 50): `HALLMARK_ESTROGEN_RESPONSE_EARLY`, `HALLMARK_ESTROGEN_RESPONSE_LATE`, `HALLMARK_G2M_CHECKPOINT`
- METABRIC top-3 (of 50) : `HALLMARK_ESTROGEN_RESPONSE_EARLY`, `HALLMARK_ESTROGEN_RESPONSE_LATE`, `HALLMARK_G2M_CHECKPOINT`
- Shared : `HALLMARK_ESTROGEN_RESPONSE_EARLY`, `HALLMARK_ESTROGEN_RESPONSE_LATE`, `HALLMARK_G2M_CHECKPOINT`

## Top-10 pathways per target × cohort

### lumA_pole, top 10

| Rank | TCGA test pathway | mean \|IG\| | METABRIC pathway | mean \|IG\| |
|---|---|---|---|---|
| 1 | `HALLMARK_ESTROGEN_RESPONSE_EARLY` | 0.00629 | `HALLMARK_ESTROGEN_RESPONSE_EARLY` | 0.00681 |
| 2 | `HALLMARK_ESTROGEN_RESPONSE_LATE` | 0.00424 | `HALLMARK_ESTROGEN_RESPONSE_LATE` | 0.00455 |
| 3 | `HALLMARK_IL2_STAT5_SIGNALING` | 0.00096 | `HALLMARK_IL2_STAT5_SIGNALING` | 0.00101 |
| 4 | `HALLMARK_TNFA_SIGNALING_VIA_NFKB` | 0.00086 | `HALLMARK_TNFA_SIGNALING_VIA_NFKB` | 0.00099 |
| 5 | `HALLMARK_ANDROGEN_RESPONSE` | 0.00068 | `HALLMARK_ANDROGEN_RESPONSE` | 0.00076 |
| 6 | `HALLMARK_HYPOXIA` | 0.00058 | `HALLMARK_HYPOXIA` | 0.00064 |
| 7 | `HALLMARK_UNFOLDED_PROTEIN_RESPONSE` | 0.00056 | `HALLMARK_UNFOLDED_PROTEIN_RESPONSE` | 0.00063 |
| 8 | `HALLMARK_UV_RESPONSE_UP` | 0.00055 | `HALLMARK_UV_RESPONSE_UP` | 0.00059 |
| 9 | `HALLMARK_UV_RESPONSE_DN` | 0.00055 | `HALLMARK_UV_RESPONSE_DN` | 0.00059 |
| 10 | `HALLMARK_P53_PATHWAY` | 0.00049 | `HALLMARK_P53_PATHWAY` | 0.00055 |

### lumB_pole, top 10

| Rank | TCGA test pathway | mean \|IG\| | METABRIC pathway | mean \|IG\| |
|---|---|---|---|---|
| 1 | `HALLMARK_MYC_TARGETS_V1` | 0.00328 | `HALLMARK_E2F_TARGETS` | 0.00355 |
| 2 | `HALLMARK_E2F_TARGETS` | 0.00325 | `HALLMARK_G2M_CHECKPOINT` | 0.00347 |
| 3 | `HALLMARK_G2M_CHECKPOINT` | 0.00321 | `HALLMARK_MYC_TARGETS_V1` | 0.00340 |
| 4 | `HALLMARK_MYC_TARGETS_V2` | 0.00126 | `HALLMARK_MYC_TARGETS_V2` | 0.00134 |
| 5 | `HALLMARK_MITOTIC_SPINDLE` | 0.00072 | `HALLMARK_MITOTIC_SPINDLE` | 0.00081 |
| 6 | `HALLMARK_MTORC1_SIGNALING` | 0.00056 | `HALLMARK_MTORC1_SIGNALING` | 0.00058 |
| 7 | `HALLMARK_DNA_REPAIR` | 0.00051 | `HALLMARK_DNA_REPAIR` | 0.00056 |
| 8 | `HALLMARK_UNFOLDED_PROTEIN_RESPONSE` | 0.00049 | `HALLMARK_UNFOLDED_PROTEIN_RESPONSE` | 0.00050 |
| 9 | `HALLMARK_APICAL_SURFACE` | 0.00038 | `HALLMARK_NOTCH_SIGNALING` | 0.00043 |
| 10 | `HALLMARK_NOTCH_SIGNALING` | 0.00036 | `HALLMARK_SPERMATOGENESIS` | 0.00036 |

### final_logit, top 10

| Rank | TCGA test pathway | mean \|IG\| | METABRIC pathway | mean \|IG\| |
|---|---|---|---|---|
| 1 | `HALLMARK_ESTROGEN_RESPONSE_EARLY` | 0.02900 | `HALLMARK_ESTROGEN_RESPONSE_EARLY` | 0.03138 |
| 2 | `HALLMARK_ESTROGEN_RESPONSE_LATE` | 0.01962 | `HALLMARK_ESTROGEN_RESPONSE_LATE` | 0.02118 |
| 3 | `HALLMARK_G2M_CHECKPOINT` | 0.01223 | `HALLMARK_G2M_CHECKPOINT` | 0.01284 |
| 4 | `HALLMARK_MYC_TARGETS_V1` | 0.01220 | `HALLMARK_E2F_TARGETS` | 0.01266 |
| 5 | `HALLMARK_E2F_TARGETS` | 0.01201 | `HALLMARK_MYC_TARGETS_V1` | 0.01223 |
| 6 | `HALLMARK_MYC_TARGETS_V2` | 0.00530 | `HALLMARK_MYC_TARGETS_V2` | 0.00558 |
| 7 | `HALLMARK_TNFA_SIGNALING_VIA_NFKB` | 0.00438 | `HALLMARK_TNFA_SIGNALING_VIA_NFKB` | 0.00499 |
| 8 | `HALLMARK_IL2_STAT5_SIGNALING` | 0.00435 | `HALLMARK_IL2_STAT5_SIGNALING` | 0.00496 |
| 9 | `HALLMARK_UNFOLDED_PROTEIN_RESPONSE` | 0.00423 | `HALLMARK_UNFOLDED_PROTEIN_RESPONSE` | 0.00450 |
| 10 | `HALLMARK_ANDROGEN_RESPONSE` | 0.00373 | `HALLMARK_ANDROGEN_RESPONSE` | 0.00395 |

## Full 50-row tables

Full per-pathway tables (one CSV per (target, cohort) combination, ranked by `mean |IG|`):

- `audit/dmoi_pathway_v0.6_lumA_pole__TCGA_test.csv`
- `audit/dmoi_pathway_v0.6_lumA_pole__METABRIC.csv`
- `audit/dmoi_pathway_v0.6_lumB_pole__TCGA_test.csv`
- `audit/dmoi_pathway_v0.6_lumB_pole__METABRIC.csv`
- `audit/dmoi_pathway_v0.6_final_logit__TCGA_test.csv`
- `audit/dmoi_pathway_v0.6_final_logit__METABRIC.csv`

## Reading

- `mean |IG|`, how loudly the pathway speaks (magnitude).
- `signed_mean`, direction (positive = pushes toward LumB; negative = pushes toward LumA for the final logit; for the pole scores, positive = pushes toward 'this is the pole's class').
- A pathway with high `mean |IG|` but `signed_mean ≈ 0` means the pathway has both pro- and anti- genes that roughly cancel.

## Limitations

- 50 Hallmark sets loaded, the entire Hallmark v2024.1.Hs catalog. v0.6 closes the v0.5 caveat ('did you only load the 5 sets that work?'). The C2 curated catalog (~5,000 sets) remains out of scope.
- Aggregation is over the RNA modality only. METABRIC's methylation branch is silenced; even on TCGA the meth features are HM450 probes, not gene symbols, so a Hallmark rollup of meth IG would need a probe -> gene crosswalk.
- The pathway scores are interpretation artifacts, not training signals. The model still attends to genes, not to pathways. Pathway-level *attention* (vs aggregation) is the natural v0.7+ candidate.

## Reproduce

```bash
python scripts/aggregate_pathway_ig_full.py
```
