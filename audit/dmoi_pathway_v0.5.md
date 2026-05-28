# DMOI v0.5 — Pathway-level IG aggregation (MSigDB Hallmark)

Generated: 2026-05-28T10:46:44Z

## Setup

- Train cohort     : TCGA cohort_v2 train split, n=333
- TCGA test cohort : n=84 (AUROC 0.9682)
- METABRIC cohort  : n=1175 (RNA-only, meth silenced)
- Pathway sets     : 5 MSigDB Hallmark sets from `priors.py` (ESTROGEN_RESPONSE_EARLY/LATE, E2F_TARGETS, G2M_CHECKPOINT, MYC_TARGETS_V1)
- Aggregation      : per-pathway `mean |IG|`, `sum_signed`, `signed_mean` over per-patient × per-gene attributions

## Cross-cohort pathway agreement (top-3 per target)

### lumA_pole

- TCGA test top-3 pathways: `HALLMARK_ESTROGEN_RESPONSE_EARLY`, `HALLMARK_ESTROGEN_RESPONSE_LATE`, `HALLMARK_G2M_CHECKPOINT`
- METABRIC top-3 pathways : `HALLMARK_ESTROGEN_RESPONSE_EARLY`, `HALLMARK_ESTROGEN_RESPONSE_LATE`, `HALLMARK_G2M_CHECKPOINT`
- Shared : `HALLMARK_ESTROGEN_RESPONSE_EARLY`, `HALLMARK_ESTROGEN_RESPONSE_LATE`, `HALLMARK_G2M_CHECKPOINT`

### lumB_pole

- TCGA test top-3 pathways: `HALLMARK_G2M_CHECKPOINT`, `HALLMARK_MYC_TARGETS_V1`, `HALLMARK_E2F_TARGETS`
- METABRIC top-3 pathways : `HALLMARK_G2M_CHECKPOINT`, `HALLMARK_E2F_TARGETS`, `HALLMARK_MYC_TARGETS_V1`
- Shared : `HALLMARK_E2F_TARGETS`, `HALLMARK_G2M_CHECKPOINT`, `HALLMARK_MYC_TARGETS_V1`

### final_logit

- TCGA test top-3 pathways: `HALLMARK_ESTROGEN_RESPONSE_EARLY`, `HALLMARK_ESTROGEN_RESPONSE_LATE`, `HALLMARK_G2M_CHECKPOINT`
- METABRIC top-3 pathways : `HALLMARK_ESTROGEN_RESPONSE_EARLY`, `HALLMARK_ESTROGEN_RESPONSE_LATE`, `HALLMARK_G2M_CHECKPOINT`
- Shared : `HALLMARK_ESTROGEN_RESPONSE_EARLY`, `HALLMARK_ESTROGEN_RESPONSE_LATE`, `HALLMARK_G2M_CHECKPOINT`

## Detailed scores per pathway × cohort

### lumA_pole

| Pathway | TCGA genes in inputs | TCGA test mean \|IG\| | METABRIC mean \|IG\| |
|---|---|---|---|
| `HALLMARK_ESTROGEN_RESPONSE_EARLY` | 107 | 0.00991 (signed_mean -0.00090) | 0.01076 (signed_mean -0.00087) |
| `HALLMARK_ESTROGEN_RESPONSE_LATE` | 115 | 0.00946 (signed_mean -0.00066) | 0.01030 (signed_mean -0.00071) |
| `HALLMARK_E2F_TARGETS` | 189 | 0.00003 (signed_mean +0.00000) | 0.00004 (signed_mean +0.00000) |
| `HALLMARK_G2M_CHECKPOINT` | 185 | 0.00013 (signed_mean -0.00004) | 0.00016 (signed_mean -0.00003) |
| `HALLMARK_MYC_TARGETS_V1` | 188 | 0.00003 (signed_mean +0.00000) | 0.00004 (signed_mean +0.00000) |

### lumB_pole

| Pathway | TCGA genes in inputs | TCGA test mean \|IG\| | METABRIC mean \|IG\| |
|---|---|---|---|
| `HALLMARK_ESTROGEN_RESPONSE_EARLY` | 107 | 0.00008 (signed_mean -0.00000) | 0.00008 (signed_mean -0.00002) |
| `HALLMARK_ESTROGEN_RESPONSE_LATE` | 115 | 0.00007 (signed_mean -0.00000) | 0.00008 (signed_mean -0.00002) |
| `HALLMARK_E2F_TARGETS` | 189 | 0.00327 (signed_mean +0.00061) | 0.00360 (signed_mean +0.00050) |
| `HALLMARK_G2M_CHECKPOINT` | 185 | 0.00334 (signed_mean +0.00040) | 0.00362 (signed_mean +0.00042) |
| `HALLMARK_MYC_TARGETS_V1` | 188 | 0.00329 (signed_mean -0.00018) | 0.00341 (signed_mean -0.00015) |

### final_logit

| Pathway | TCGA genes in inputs | TCGA test mean \|IG\| | METABRIC mean \|IG\| |
|---|---|---|---|
| `HALLMARK_ESTROGEN_RESPONSE_EARLY` | 107 | 0.04570 (signed_mean +0.00404) | 0.04968 (signed_mean +0.00408) |
| `HALLMARK_ESTROGEN_RESPONSE_LATE` | 115 | 0.04295 (signed_mean +0.00364) | 0.04717 (signed_mean +0.00332) |
| `HALLMARK_E2F_TARGETS` | 189 | 0.01206 (signed_mean +0.00240) | 0.01276 (signed_mean +0.00227) |
| `HALLMARK_G2M_CHECKPOINT` | 185 | 0.01272 (signed_mean +0.00190) | 0.01341 (signed_mean +0.00211) |
| `HALLMARK_MYC_TARGETS_V1` | 188 | 0.01221 (signed_mean -0.00085) | 0.01226 (signed_mean -0.00074) |

## Reading

- `mean |IG|` — how loudly the pathway speaks (magnitude).
- `signed_mean` — direction (positive = pushes toward LumB; negative = pushes toward LumA for the final logit; for the pole scores, positive = pushes toward 'this is the pole's class').
- A pathway with high `mean |IG|` but `signed_mean ≈ 0` means the pathway has both pro- and anti- genes that roughly cancel — the pathway is important but ambiguous in direction.

## Honest scope

- Only 5 Hallmark sets are loaded (the ones already in `priors.py` for the pole masks). The full 50-set MSigDB Hallmark catalog is out of scope for v0.5 — keeping the dependency surface tight. Future work: add a `gmt`-file loader and roll up the full Hallmark catalog (or even C2 curated pathways).
- Aggregation is over the RNA modality only. METABRIC's methylation branch is silenced; even on TCGA the methylation pathway aggregation isn't meaningful because the meth features are HM450 probes, not gene symbols.

## Reproduce

```bash
python scripts/aggregate_pathway_ig.py
```
