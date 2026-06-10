# Biological prior vs unsupervised selection — one page across three omics

*Cross-repo synthesis · TCGA-BRCA · all results label-free, locally reproducible.*
*Canonical home: this file. Companion repo: [`multiomics-cnv-conditioned-poc`](https://github.com/hryankim-architect/multiomics-cnv-conditioned-poc) (CNV).*

## The question

Does restricting the feature space with **biological knowledge** beat generic
statistical / deep-learning multi-omics integration for breast-cancer subtyping? The
landscape pointed at Omran et al. 2025 (J Transl Med,
[doi:10.1186/s12967-025-06662-5](https://doi.org/10.1186/s12967-025-06662-5)), which
benchmarked MOFA+ (statistical) and MoGCN (deep learning) as **unsupervised feature
selectors** for PAM50. (An earlier candidate, moBRCA-net, was dropped — its dataset is
not public.)

## The method (removes the supervised-vs-unsupervised confound)

A model's *prior* is a feature selector. So is MOFA+/MoGCN. Compare them on equal
footing: each picks features **without seeing labels**, then the *same* downstream
LogisticRegression / linear SVC (stratified 5-fold) scores them. The prior's biological
feature restriction is label-free (knowledge, not labels), so this is apples-to-apples —
the downstream classifier is the only supervised step. Baseline = top-variance.

## Results — the prior wins, where the modality's biology is informative

| modality (repo) | task | prior selector | prior | top-variance | metric |
|---|---|---|---|---|---|
| **RNA** (dmoi) | 5-class PAM50 | Hallmark 5-set (RNA+meth, 100/omics) | **0.876** | 0.813 | weighted-F1 |
| **RNA** (dmoi) | LumA-vs-LumB | Hallmark 5-set (RNA) | **0.948** | 0.825 | AUROC |
| **Methylation** (dmoi) | 5-class PAM50 | HM450 cis-mapped to Hallmark | *(in RNA+meth 0.876)* | — | weighted-F1 |
| **CNV** (mocnv) | HER2-vs-rest | amplicon (ERBB2/MYC/CCND1, 20 genes) | **0.830** | 0.767 (k=100) | AUROC |
| **CNV** (mocnv) | LumA-vs-LumB | amplicon | **0.728** | 0.703 (k=100) | AUROC |
| *reference* | 5-class PAM50 | *MOFA+ (3-omics, diff. cohort)* | *0.75* | — | weighted-F1 |

## Four findings that hold across modalities

1. **Prior > statistical selection at matched budget** — RNA 0.876 vs 0.813; CNV 0.830
   vs 0.812 (k=20) / 0.767 (k=100), with 5× fewer features.
2. **Specificity, not breadth** — RNA: the 5 curated proliferation/ER Hallmark sets beat
   the full 50-set catalog (0.876 vs 0.819). CNV: a 20-gene amplicon prior outperforms
   100 high-variance loci. Narrow, well-chosen priors win.
3. **The edge is biological, not variance re-discovery** — RNA prior genes barely overlap
   top-variance (Jaccard 0.036) and are more clinically coherent (73% vs 61% associated
   with stage/node/age, independent of subtype — mirroring the paper's MOFA+>MoGCN
   0.59>0.47). CNV attribution lands on the expected ERBB2 amplicon.
4. **The win is largest where the modality's biology defines the axis** — RNA helps the
   proliferation/ER axes broadly; CNV helps *sharply* on the ERBB2-amplicon HER2 axis and
   modestly off it (via the CCND1/MYC proliferation amplicon). The prior helps where the
   biology is informative, not uniformly.

## Honest scope

- The MOFA+ 0.75 is a **literature reference** (3-omics incl. shotgun microbiome,
  non-identical cohort) — **not** a controlled head-to-head. The controlled comparison
  throughout is prior vs top-variance on identical data, same downstream model.
- A **microbiome** third omic was deferred (absent from the standard cBioPortal BRCA
  study; prior-free, so it doesn't exercise a gene-centric prior).
- These measure the **feature-selection contribution** only — label-free selectors + a
  plain classifier, not the full supervised pole-conditioned DMOI/mocnv models.
- CNV reuses PAM50 labels from this (dmoi) repo (the two repos sit side by side).

## Reproduce (public TCGA-BRCA, ~seconds–40s each)

```bash
# RNA + methylation (dmoi-brca-poc)
make compare                                  # prior vs baselines, 5-class
python scripts/ablate_hallmark_sets.py        # per-Hallmark-set ablation
python scripts/clinical_association.py         # OncoDB-style clinical coherence
python scripts/compare_binary_task.py          # LumA-vs-LumB generalization
python scripts/plot_comparison_summary.py      # synthesis figure

# CNV (multiomics-cnv-conditioned-poc)
make compare-cnv                               # amplicon prior vs baselines, per axis
```

**Bottom line:** across RNA, methylation, and CNV, a compact label-free biological prior
beats generic statistical feature selection — and does so most where that modality's
biology defines the axis. Knowledge-conditioning is not a flattering benchmark artifact;
it is a real, modality-appropriate, and honestly-bounded effect.
