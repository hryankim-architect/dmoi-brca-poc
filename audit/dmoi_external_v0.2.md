# DMOI v0.2 External Validation — METABRIC (RNA-only, Path A')

Generated: 2026-05-28T03:40:34Z

## Setup

- Train cohort  : TCGA-BRCA cohort_v2 train split — 333 patients (LumA=231, LumB=102)
- External      : METABRIC (Curtis 2012 + Pereira 2016) — 1175 patients (LumA=700, LumB=475)
- Architecture  : Option A (aux BCE on sub-classifiers, disagreement IN), trained once on full TCGA train.
- n_epochs      : 15 (CV mean best epoch from Step A; no early stopping, no test-AUC-driven epoch selection).
- Calibration   : T fit on a stratified 15% cal split of TCGA train; applied to METABRIC logits.

## Cross-cohort alignment

- TCGA HiSeqV2 genes (training-time order) : 20530
- METABRIC unique Hugo symbols              : 20384
- Shared                                     : 16890
- TCGA-only (mean-imputed to 0)              : 3640
- METABRIC-only (dropped)                    : 3494

Per-gene quantile normalization maps each METABRIC gene's empirical distribution to the TCGA train gene's distribution before the TCGA-fitted StandardScaler is applied.

## Headline external metrics

- **External AUROC** : **0.9095** (sanity recompute: 0.9095)
- **External BalAcc**: 0.7877
- **External ECE before T-scaling** : 0.0729
- **External ECE after T-scaling**  : 0.0992  (T=0.634)

| | pred LumA | pred LumB |
|---|---|---|
| true LumA | 668 | 32 |
| true LumB | 180 | 295 |

External accuracy: 0.8196  ·  LumB sensitivity: 0.6211  ·  LumB specificity: 0.9543

## Calibration transfer investigation

First v0.2 run applied T fit on TCGA cohort_v2 cal-split (T=0.634) to METABRIC, and it made ECE WORSE — the meth-silenced METABRIC predictions were already reasonably calibrated, and the TCGA T over-sharpened them. To test that, we carved a 15% stratified cal slice out of METABRIC, fit T on it, and applied it to the remaining 85% eval slice. All three ECEs below are computed on the SAME eval slice for an apples-to-apples comparison:

- METABRIC cal slice (15%, stratified): 176 patients (LumA=105, LumB=71)
- METABRIC eval slice (85%): 999 patients

| Calibration | T | ECE on eval slice |
|---|---|---|
| Uncalibrated | 1.000 | 0.0745 |
| T from TCGA cal-split (naive transfer) | 0.634 | 0.1051 |
| T from METABRIC cal-split (cohort-specific) | 0.934 | 0.0738 |

Takeaway: **calibration parameters don't blindly transfer across cohorts/modalities.** The TCGA T was fit on a model that had both RNA + methylation; on METABRIC the methylation branch is silenced, so the logit distribution is different and the TCGA T over-sharpens. A METABRIC-specific T does the right thing (closer to 1.0 if naive ECE was already low, sharpens or softens as the cohort actually needs).

## LumB sensitivity investigation

At the default 0.5 threshold the meth-silenced model has LumB sensitivity 0.621 / specificity 0.954 on METABRIC — it calls LumA too often. Two corrections, both reported on the same 85% METABRIC eval slice (n=999):

1. **Bayes class-prior adjustment** (principled, no tuning):
   - TCGA train LumB prior = 0.306, METABRIC LumB prior = 0.404.
   - Adjust each METABRIC probability via
     `adj = p · (π_test/π_train) / (p · (π_test/π_train) + (1-p) · ((1-π_test)/(1-π_train)))`.
2. **Threshold tuned on METABRIC cal slice** (pragmatic):
   - Sweep thresholds in [0.30, 0.60] on the 15% METABRIC cal slice, pick the one that maximizes BalAcc on cal.
   - Best threshold: 0.425 (cal BalAcc 0.8188).

Both then evaluated on the 85% eval slice:

| Strategy | LumB sens | LumB spec | BalAcc | F1 LumB |
|---|---|---|---|---|
| Default @0.5 | 0.619 | 0.956 | 0.788 | 0.735 |
| Bayes prior-adjusted @0.5 | 0.691 | 0.933 | 0.812 | 0.772 |
| Tuned threshold (0.425) | 0.656 | 0.943 | 0.799 | 0.754 |

Interpretation: the sensitivity asymmetry has two plausible drivers — (1) class-prior shift (METABRIC has ~40% LumB vs TCGA train's ~31%), and (2) modality silencing (the meth branch normally contributes signal toward the harder LumB calls).

## Honest caveats

- **Methylation branch is silenced.** METABRIC has no HM450 data (it's an Illumina HT-12 v3 expression-only cohort). The methylation pole encoder receives a zero-tensor at inference, so this test does NOT validate the dual-modality story. It validates only that the RNA pole encoder + classifier head generalizes across cohorts.
- **Platform difference.** TCGA uses HiSeq RNA-seq (FPKM log2 scale); METABRIC uses Illumina HT-12 v3 expression microarray. Quantile normalization is applied per gene, which is the standard correction for cross-platform validation.
- **Mean-imputed train-only genes.** Genes present in TCGA but not METABRIC are filled with 0 (the post-StandardScaler train mean). This is a permissive choice — the model sees those features as neutral rather than missing.
- **No multi-modal external validation available on public data.** No public BRCA cohort outside TCGA has paired RNA-seq + HM450 methylation; see `docs/v0.2-design-external-validation.md` for the recon trail.

## Reproduce

```bash
python scripts/fetch_metabric.py        # one-time ~690 MB download
python scripts/build_metabric_cohort.py
python scripts/eval_external.py
```
