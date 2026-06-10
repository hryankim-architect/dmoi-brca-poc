# DMOI biological prior vs unsupervised feature selection (PAM50, TCGA-BRCA)

n = 620 samples with RNA + HM450 methylation + a PAM50 call
({'Basal': 87, 'Her2': 31, 'LumA': 288, 'LumB': 127, 'Normal': 87}). 5-class weighted-F1, stratified 5-fold; **every selector is
label-free** (priors use knowledge, top-variance uses statistics — neither sees `y`),
so the downstream classifier is the only supervised step. This puts DMOI's prior on
the same footing as the unsupervised integrators in Omran et al. 2025.

| selector (label-free) | n_feat | LR wF1 | SVC wF1 | CHI ↑ | DBI ↓ |
|---|---|---|---|---|---|
| DMOI-prior RNA+meth (100+100) | 200 | 0.875 | 0.856 | 66.7 | 2.80 |
| DMOI-prior RNA+meth (full) | 12640 | 0.839 | 0.854 | 12.2 | 5.82 |
| DMOI-prior RNA-only | 605 | 0.868 | 0.876 | 56.6 | 3.06 |
| top-variance RNA+meth (100+100) | 200 | 0.813 | 0.824 | 40.2 | 3.54 |

## Reading

- **At a matched, paper-comparable budget (100 features/omics), DMOI's biological
  prior beats top-variance selection** on weighted-F1 and gives markedly better-
  separated subtype clusters (higher Calinski-Harabasz, lower Davies-Bouldin). This is
  the apples-to-apples result: prior vs statistical selection, same downstream model.
- More features is not better: the full prior set (thousands of methylation probes)
  underperforms the 100+100 budget — echoing the feature-selection motivation of the
  source paper.

## Literature reference (NOT a controlled head-to-head)

Omran et al. 2025 (J Transl Med, doi:10.1186/s12967-025-06662-5) report MOFA+ as the
best unsupervised integrator at **weighted-F1 0.75**
(MoGCN lower) on 960 TCGA samples using **three** omics (transcriptome, epigenome,
**shotgun microbiome**), 100 features/omics. Our numbers are **not** directly
comparable to theirs: (a) we use **two** omics (RNA + methylation, no microbiome —
DMOI's gene-centric prior does not map to microbial taxa); (b) our sample set
(n=620, TCGA-BRCA PAM50) is reconstructed independently and is not their
exact cohort; (c) their feature selection is per-fold. Their 0.75 is included only as
a sanity-scale reference for what unsupervised multi-omics selection achieves on this
task; the controlled comparison here is DMOI-prior vs top-variance on identical data.
