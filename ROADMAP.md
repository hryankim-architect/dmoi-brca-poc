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
