# Task generalization — binary LumA vs LumB (RNA-only)

n = 628 luminal samples (LumA 434, LumB 194). Same
label-free RNA selectors as the 5-class comparison; LogisticRegression, stratified
5-fold. AUROC is DMOI's headline metric for this pole task.

| selector (label-free, RNA) | AUROC | LR wF1 | CHI ↑ |
|---|---|---|---|
| DMOI-prior(5-set) | 0.948 | 0.880 | 52.0 |
| DMOI-prior(50-set) | 0.840 | 0.782 | 26.3 |
| top-variance | 0.825 | 0.767 | 19.7 |

## Reading

- If the biological prior keeps its edge over top-variance here, the v0.15 finding
  generalizes from the 5-class task to DMOI's native binary pole task — it is not a
  5-class artifact.
- Note: these are *label-free selectors + a plain LR*, not DMOI's full supervised,
  prior-conditioned model (which reports AUROC ~0.97 on LumA-vs-LumB). This isolates the
  feature-selection contribution, consistent with the rest of this comparison.
