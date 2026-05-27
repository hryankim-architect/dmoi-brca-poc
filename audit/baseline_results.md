# DMOI POC Baseline Results (Day-4)

Generated: 2026-05-27T20:39:52Z

## Cohort

- Dual-modality patients: **395**
- H+ luminal: 338 (85.6%)
- H- basal/TN: 57 (14.4%)

## Features

- RNA-seq (HiSeqV2): 20530 genes (all retained)
- Methylation (HM450): 10000 probes (top-variance filter from 485,577)
- Concatenated (early fusion): 30530 features

## Models

- `logreg`: L2 LogisticRegression, class_weight='balanced', max_iter=2000
- `rf`: RandomForest, 300 trees, class_weight='balanced'
- StandardScaler upstream of both
- StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

## Results (mean ± std across 5 folds)

| Feature set | Model | AUROC | Balanced accuracy |
|---|---|---|---|
| concat | logreg | 1.0000 ± 0.0000 | 0.9985 ± 0.0033 |
| concat | rf | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| meth | logreg | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| meth | rf | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| rna | logreg | 1.0000 ± 0.0000 | 0.9985 ± 0.0033 |
| rna | rf | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |

## Saturation finding (honest)

Every (feature_set, model) combination hits AUROC >= 0.99 in 5-fold CV.
This is **not** a successful baseline — it means the H+ (luminal) vs
H- (basal/TN) task is **too easy** for a DMOI POC discrimination target:

- The PAM50 labels (LumA/LumB/Basal) used to define the poles are themselves
  derived from RNA-seq, so RNA-based classification is partially circular.
- The two poles are biologically very distinct cell-of-origin states.
  Single-modality discrimination has saturated decades of literature.
- Without baseline headroom, the Week-2 DMOI hypothesis-conditioned encoder
  cannot demonstrate value on this task.

**Honest next step**: re-scope the Week-2 discrimination target to a harder
task on the same cohort. Candidates:

- Within-luminal LumA vs LumB (PAM50 mRNA_nature2012 sub-call).
- 5-year overall survival / progression-free survival prediction.
- Response to neoadjuvant chemotherapy on the basal subset.
- Methylation-only prediction of an *RNA-derived* signature score where
  the cross-modal task is non-trivial.

The Day-4 deliverable is therefore **a negative finding + scope decision**
rather than a comparison number to beat. The substrate, cohort, and
feature pipeline are validated; the Week-2 target needs adjustment.

## Reproduce

```bash
python scripts/build_cohort.py     # if cohort.tsv missing
python scripts/run_baseline.py
```
