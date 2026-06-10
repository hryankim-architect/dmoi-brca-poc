# DMOI v0.17, external comparison: biological prior vs unsupervised integration

**Date:** 2026-06-10
**Tag:** v0.17
**Type:** external method comparison (new analysis module + scripts; `src/dmoi_brca/` core model unchanged)

## TL;DR

How does DMOI's biological prior compare to general multi-omics integrators (MOFA+,
MoGCN)? Omran et al. 2025 (J Transl Med,
[doi:10.1186/s12967-025-06662-5](https://doi.org/10.1186/s12967-025-06662-5)) benchmarked
those two as **unsupervised feature selectors** for TCGA-BRCA PAM50 subtyping. DMOI's
Hallmark + HM450-cis feature restriction is **also label-free**, so it drops into the
exact same protocol — which removes the supervised-vs-unsupervised confound: every
selector picks features without labels, and a shared downstream LR/SVC is the only
supervised step. On public TCGA-BRCA, the biological prior beats statistical selection
at a matched budget, the advantage is biological (not variance re-discovery) and
clinical, it prefers a *narrow* curated prior over the full catalog, and it generalizes
to the binary luminal pole task.

## Headline result

5-class PAM50, n=620 (RNA + HM450 methylation), 100 features/omics, label-free
selection, LR 5-fold weighted-F1:

| selector (label-free) | LR weighted-F1 | Calinski-Harabasz ↑ |
|---|---|---|
| **DMOI-prior (5 curated Hallmark sets)** | **0.876** | 65.9 |
| DMOI-prior (full 50-set catalog) | 0.819 | 55.5 |
| top-variance | 0.813 | 40.2 |
| *MOFA+ (literature reference, 3-omics, diff. cohort)* | *0.75* | — |

## Four consistent findings

1. **Prior > statistical selection** at a matched 100-feature/omics budget (0.876 vs
   0.813), with better-separated subtype clusters (CHI 65.9 vs 40.2, DBI 2.82 vs 3.54).
2. **(a) Specificity, not breadth** — the 5 curated proliferation/ER sets (605 genes)
   beat the full 50-set catalog (4,191 genes): 0.876 vs 0.819. Widening the prior dilutes
   the luminal-axis signal back toward the variance baseline.
3. **(b) The edge is biological, not variance re-discovery** — selected RNA genes barely
   overlap top-variance (Jaccard 0.036), and **(1)** 73% are clinically associated with
   stage/node/age — variables independent of the PAM50 target — vs 61% for top-variance
   and 50% for the 50-set prior. Same direction as the paper's MOFA+ > MoGCN (0.59 > 0.47,
   OncoDB).
4. **(2) It generalizes** — on the binary LumA-vs-LumB pole the gap *widens*: AUROC 0.948
   (5-set prior) vs 0.825 (top-variance), 0.840 (50-set). **(d)** Per-set ablation: every
   single curated set alone beats top-variance(100), and `G2M_CHECKPOINT` is the strongest
   single set and the most load-bearing in leave-one-out.

## What changed

- `src/dmoi_brca/compare_integration.py` — new analysis module: label-free selectors
  (`prior_rna_indices`, `prior_meth_indices` via HM450 cis-mapping, `topvar_indices`,
  `topvar_within`), `eval_selector` (downstream LR + linear SVC, stratified 5-fold
  weighted-F1, Calinski-Harabasz / Davies-Bouldin), `jaccard_index`, `bh_fdr`.
- `scripts/compare_mofa_mogcn.py` (→ `audit/dmoi_vs_mofa_mogcn.md`) — 5-set vs 50-set
  prior vs top-variance + RNA Jaccard; `make compare` target.
- `scripts/ablate_hallmark_sets.py` (→ `audit/dmoi_prior_ablation.md`) — per-set + LOO.
- `scripts/clinical_association.py` (→ `audit/dmoi_clinical_association.md`) — OncoDB-style
  non-circular clinical coherence (Kruskal-Wallis / Spearman, BH-FDR).
- `scripts/compare_binary_task.py` (→ `audit/dmoi_binary_lumA_lumB.md`) — LumA-vs-LumB AUROC.
- `scripts/plot_comparison_summary.py` (→ `audit/dmoi_comparison_summary.{png,md}`) — figure + note.
- `tests/test_compare_integration.py` — 8 unit tests (selection, jaccard, BH-FDR, eval).
- `README.md` + `ROADMAP.md` — external-comparison section / v0.15–v0.17 entries.

No changes to the DMOI model in `src/dmoi_brca/` beyond the new analysis module; the
prior, pole structure, and trained pipeline are unchanged.

## Reproduce

```bash
make compare                                   # 5-class prior vs baselines (~40s)
PYTHONPATH=src python scripts/ablate_hallmark_sets.py
PYTHONPATH=src python scripts/clinical_association.py
PYTHONPATH=src python scripts/compare_binary_task.py
PYTHONPATH=src python scripts/plot_comparison_summary.py
make test && make lint
```

All inputs are public TCGA-BRCA (`data/tcga_brca/`, fetched via `make data`); no private
data and no microbiome download required.

## Limitations (honest scope)

- **Literature reference, not a controlled head-to-head.** The MOFA+ 0.75 figure is from
  a different study using **three** omics (transcriptome, epigenome, **shotgun
  microbiome**) on a non-identical 960-sample cohort with per-fold feature selection. It
  is shown only as a sanity-scale reference. The *controlled* comparison throughout is
  DMOI-prior vs top-variance on identical data with the same downstream model.
- **Two omics, not three.** DMOI uses RNA + methylation. A microbiome third omic was
  **deferred**: it is absent from the standard cBioPortal `brca_tcga_pan_can_atlas_2018`
  study, would require the multi-GB Poore et al. 2020 all-TCGA microbial dataset, and is
  *prior-free* — it does not exercise DMOI's gene-centric prior.
- **Feature-selection contribution only.** These runs are label-free selectors + a plain
  LR/SVC, not DMOI's full supervised, prior-conditioned model (which reports AUROC ~0.97
  on LumA-vs-LumB). That is by design — it isolates the prior's value as a selector for a
  fair comparison to MOFA+/MoGCN.
- **Sample set.** n=620 (RNA∩meth∩PAM50) reconstructed independently from TCGA-BRCA; not
  the source paper's exact cohort.
