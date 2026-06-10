# Roadmap — `dmoi-brca-poc`

DMOI: **hypothesis-conditioned multi-omics** (RNA-seq + DNA-methylation) for breast-cancer
subtyping, with honest calibration, per-patient attribution, and explicit generalization
seals. This roadmap is the **shipped arc** (v0.1 → v0.14) — retrospective and dated; every
tag carries a GitHub Release. A capability portrait, not a benchmark claim (see the README
+ `docs/` for the honest scope).

---

## Shipped (v0.1 → v0.14)

**Foundations — conditioning, calibration, attribution**
- [x] **v0.1** — hypothesis-conditioned multi-omics + honest calibration
- [x] **v0.2** — external generalization + calibration transfer
- [x] **v0.3** — per-patient Integrated Gradients attribution
- [x] **v0.4** — cross-cohort interpretability + train hygiene

**Pathway-level interpretability**
- [x] **v0.5** — pathway-level IG aggregation
- [x] **v0.6** — full Hallmark catalog rollup (50 gene sets)

**Architecture experiment — kept as honest negatives**
- [x] **v0.7 (Phase A)** — learnable pathway-pole attention: an honest negative
- [x] **v0.7.1 (Phase B)** — collapse fixed, but the wrong basin learned
- [x] **v0.8** — Variant C closes the v0.7+v0.8 architecture experiment

**Generalization seals**
- [x] **v0.9** — cross-task generalization confirmed (Luminal vs Basal)
- [x] **v0.10** — cross-cohort + cross-task (doubly generalized)
- [x] **v0.11** — 5-fold CV stability seal
- [x] **v0.12-A** — cross-cohort split-invariance seal

**Calibration transfer + a new axis**
- [x] **v0.13** — cross-cohort calibration transfer (Brier + reliability)
- [x] **v0.14** — HER2-vs-Luminal third task axis

**External method comparison (literature-grounded)**
- [x] **v0.15** — prior-as-feature-selector vs unsupervised baselines on 5-class PAM50.
  `compare_integration.py` exposes DMOI's *label-free* Hallmark + HM450-cis restriction
  as a feature selector, benchmarked against top-variance through the same downstream
  LR/SVC 5-fold weighted-F1 + Calinski-Harabasz / Davies-Bouldin
  (`scripts/compare_mofa_mogcn.py` → `audit/dmoi_vs_mofa_mogcn.md`). Resolves the
  supervised-vs-unsupervised confound by comparing label-free selectors only. Result
  (n=620, RNA+meth): at a matched 100-feature/omics budget the prior beats top-variance
  (LR wF1 0.876 vs 0.813; CHI 65.9 vs 40.2). MOFA+/MoGCN (Omran et al. 2025, F1 0.75,
  3-omics incl. microbiome) cited as a literature reference, not a controlled head-to-head
  (different omics + non-identical cohort — see the report's caveats).
  - **(a) prior breadth:** the 5 curated proliferation/ER sets (LR wF1 0.876) *beat* the
    full 50-set Hallmark catalog (0.819) — widening the prior dilutes the luminal-axis
    signal back toward the variance baseline. Specificity, not just "use a prior", is
    what helps.
  - **(b) interpretability (Jaccard):** the 5-set prior's selected RNA genes barely
    overlap the top-variance pick (Jaccard 0.036) — the F1/clustering edge is biology-
    driven, not a re-derivation of high-variance genes.
  - **(c) microbiome 3rd omic — deferred (documented blocker):** not present in the
    standard cBioPortal `brca_tcga_pan_can_atlas_2018` study; would require the
    multi-GB Poore et al. 2020 all-TCGA microbial dataset, and it is a *prior-free*
    input that does not exercise DMOI's gene-centric prior. Out of scope for this POC.

The CNV third-modality extension is a separate clean-room repo:
[`multiomics-cnv-conditioned-poc`](https://github.com/hryankim-architect/multiomics-cnv-conditioned-poc).

---

## Why this shape

The arc front-loads the **capability** (v0.1–v0.6: hypothesis conditioning + honest
calibration + pathway attribution), then **stress-tests** it (v0.7–v0.8: a learnable-attention
architecture experiment whose negatives are reported, not hidden), then **seals
generalization** (v0.9–v0.12: cross-task, cross-cohort, CV-stability, split-invariance) and
**calibration transfer** (v0.13–v0.14). Honest negatives (v0.7 / v0.7.1) stay in the record.

---

## Honest scope

A proof-of-concept on public BRCA cohorts (TCGA, METABRIC) at small N; the generalization
"seals" (including near-ceiling AUROC on well-separated task axes) are POC results on these
cohorts, not benchmark claims. Tests run from a fresh clone (`pytest`, `pythonpath=src`).
Part of the portfolio's multi-omics flagship; the substrate + honest-measurement discipline
are the reusable contribution.
