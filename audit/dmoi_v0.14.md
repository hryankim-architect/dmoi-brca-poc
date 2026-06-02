# DMOI v0.14 -- HER2-vs-Luminal cross-task + cross-cohort generalization

Generated: 2026-06-02T18:02:26Z

## Setup

- Architecture: v0.6 base (no model changes; n_pathways=0). Only cohort and pole-defining Hallmark sets differ.
- Pole pair: HER2 (clinical HER2+) vs Luminal (LumA+LumB).
- POLE_HER2 = PI3K_AKT_MTOR_SIGNALING + MTORC1_SIGNALING + G2M_CHECKPOINT.
- POLE_LUMINAL_ER = ESTROGEN_RESPONSE_EARLY + ESTROGEN_RESPONSE_LATE.
- TCGA train: cohort_v4 train, n=348 (HER2=46, Luminal=302).
- TCGA test:  n=88 (HER2=12, Luminal=76).
- METABRIC external: n=1399 (Luminal=1175, HER2=224). RNA-only + meth silenced + QN to TCGA train RNA.

## Headline AUROC

| Cohort | AUROC | bacc |
|---|---|---|
| TCGA held-out test | **0.8914** | 0.8487 |
| METABRIC external  | **0.8927** | 0.7411 |

## Per-pole IG top-10 pathways (METABRIC)

### Luminal pole

| Rank | Pathway | mean \|IG\| | signed_mean |
|---|---|---|---|
| 1 | `HALLMARK_ESTROGEN_RESPONSE_EARLY` | 0.00184 | -0.00010 |
| 2 | `HALLMARK_ESTROGEN_RESPONSE_LATE` | 0.00184 | -0.00014 |
| 3 | `HALLMARK_MYC_TARGETS_V2` | 0.00021 | -0.00000 |
| 4 | `HALLMARK_ANDROGEN_RESPONSE` | 0.00017 | +0.00003 |
| 5 | `HALLMARK_UV_RESPONSE_UP` | 0.00017 | +0.00001 |
| 6 | `HALLMARK_TNFA_SIGNALING_VIA_NFKB` | 0.00016 | +0.00000 |
| 7 | `HALLMARK_TGF_BETA_SIGNALING` | 0.00016 | -0.00001 |
| 8 | `HALLMARK_APOPTOSIS` | 0.00015 | -0.00002 |
| 9 | `HALLMARK_IL2_STAT5_SIGNALING` | 0.00015 | -0.00002 |
| 10 | `HALLMARK_GLYCOLYSIS` | 0.00015 | -0.00003 |

### HER2 pole

| Rank | Pathway | mean \|IG\| | signed_mean |
|---|---|---|---|
| 1 | `HALLMARK_MTORC1_SIGNALING` | 0.00114 | +0.00008 |
| 2 | `HALLMARK_PI3K_AKT_MTOR_SIGNALING` | 0.00105 | +0.00000 |
| 3 | `HALLMARK_G2M_CHECKPOINT` | 0.00104 | -0.00002 |
| 4 | `HALLMARK_E2F_TARGETS` | 0.00040 | +0.00000 |
| 5 | `HALLMARK_CHOLESTEROL_HOMEOSTASIS` | 0.00034 | +0.00005 |
| 6 | `HALLMARK_MYC_TARGETS_V1` | 0.00025 | +0.00001 |
| 7 | `HALLMARK_MYC_TARGETS_V2` | 0.00025 | +0.00001 |
| 8 | `HALLMARK_MITOTIC_SPINDLE` | 0.00022 | -0.00002 |
| 9 | `HALLMARK_HYPOXIA` | 0.00021 | +0.00002 |
| 10 | `HALLMARK_REACTIVE_OXYGEN_SPECIES_PATHWAY` | 0.00020 | +0.00002 |

### final_logit

| Rank | Pathway | mean \|IG\| | signed_mean |
|---|---|---|---|
| 1 | `HALLMARK_ESTROGEN_RESPONSE_EARLY` | 0.01355 | +0.00096 |
| 2 | `HALLMARK_ESTROGEN_RESPONSE_LATE` | 0.01319 | +0.00104 |
| 3 | `HALLMARK_MTORC1_SIGNALING` | 0.00637 | +0.00049 |
| 4 | `HALLMARK_G2M_CHECKPOINT` | 0.00566 | +0.00008 |
| 5 | `HALLMARK_PI3K_AKT_MTOR_SIGNALING` | 0.00539 | +0.00003 |
| 6 | `HALLMARK_MYC_TARGETS_V2` | 0.00314 | +0.00003 |
| 7 | `HALLMARK_CHOLESTEROL_HOMEOSTASIS` | 0.00292 | +0.00056 |
| 8 | `HALLMARK_E2F_TARGETS` | 0.00231 | +0.00019 |
| 9 | `HALLMARK_GLYCOLYSIS` | 0.00205 | +0.00042 |
| 10 | `HALLMARK_ANDROGEN_RESPONSE` | 0.00198 | -0.00014 |

## Cross-pole biology sanity check (METABRIC)

Expected Luminal-pole top-5 to include {ER_EARLY, ER_LATE}.
Expected HER2-pole top-5 to include {PI3K_AKT_MTOR, MTORC1, G2M_CHECKPOINT}.

- Luminal pole top-5 ∩ expected = 2 / 2 : `HALLMARK_ESTROGEN_RESPONSE_EARLY`, `HALLMARK_ESTROGEN_RESPONSE_LATE`
- HER2 pole top-5 ∩ expected = 3 / 3 : `HALLMARK_G2M_CHECKPOINT`, `HALLMARK_MTORC1_SIGNALING`, `HALLMARK_PI3K_AKT_MTOR_SIGNALING`

## Honest scope

- HER2+ is the small TCGA class (train HER2=46, test HER2=12). The single-split TCGA AUROC is noisy; the METABRIC external (HER2 n=224) carries the statistical weight. A 5-fold CV variant is the natural follow-up before quoting a TCGA headline.
- Definitional difference across cohorts: TCGA HER2 = clinical HER2+ (HER2_Final_Status); METABRIC HER2 = PAM50 'Her2' (CLAUDIN_SUBTYPE). Recorded as a cross-cohort caveat.
- METABRIC has no HM450 methylation -> meth branch silenced + per-gene QN to TCGA train RNA (v0.2/v0.4/v0.6/v0.10 protocol).
- Same architecture as v0.6/v0.9 (no model changes); only cohort + pole priors change. This is a reusability demonstration, not a powered result.

## Reproduce

```bash
python scripts/build_cohort_v4.py            # TCGA cohort_v4
python scripts/build_metabric_cohort_v4.py   # METABRIC cohort_v4
python scripts/eval_dmoi_v0.14.py
```
