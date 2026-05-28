# `dmoi-brca-poc`

> **Capability portrait, not a research result.** Public data is intentionally
> subsetted to keep the demo small and reproducible on a single workstation.

**What this shows.** A hypothesis-conditioned multi-omics architecture
(Dialectical Multi-Omics Integration, DMOI) on TCGA-BRCA. Two "pole" perspectives
(LumA-like and LumB-like) each see only their pole-relevant RNA + DNA-methylation
features via Hallmark-gene priors and HM450 cis-mapping. Their predictions are
fused with a disagreement signal exposed to the classifier head. Temperature
scaling on a held-out calibration split delivers honest, well-calibrated
probabilities.

**Reproducibility.** `python scripts/eval_dmoi.py` reproduces the full
3-way ablation + calibration report in about 2 minutes on an M-series Mac.

**Substrate.** Emits hash-chained NDJSON audit entries, tracks MLflow runs,
exposes a canary smoke test for `lab_semantic_check.py`.

**Production framing.** A version of this method ran at full cohort scale on
proprietary multi-omics data during my time in industry. The lab version here
proves the *method* and the *engineering*, not the result. See
[`docs/what-is-out-of-scope.md`](docs/what-is-out-of-scope.md).

---

## v0.1 headline result

| Metric | DMOI v0.1 (Option A, nested-cal) | Baseline LogReg (concat) |
|---|---|---|
| AUROC | 0.9606 ± 0.017 | 0.9626 ± 0.015 |
| F1 LumA | 0.927 ± 0.023 | – |
| F1 LumB (minority) | 0.839 ± 0.054 | – |
| ECE uncalibrated | 0.138 ± 0.055 | – |
| **ECE after T-scaling (honest, nested cal-split)** | **0.077** | – |
| Disagreement AUC for misclass | 0.755 (4/5 folds significant) | – |

**The honest takeaway, in three acts:**

1. **Baseline saturated the easy signal.** Plain LogReg on
   concat(RNA, methylation) already lands at 0.963 AUROC on the 417-patient
   LumA vs LumB cohort. There isn't much headroom for an architectural
   upgrade to beat that — within-luminal class structure is already strongly
   expressed in the bulk transcriptome.

2. **Hypothesis-conditioned attention did NOT lift AUROC.** DMOI's primary
   metric ties baseline within noise (Δ ≈ −0.002). The dual-perspective
   architecture's value is **not** in the headline number — the LogReg
   ceiling reflects the structure of the data, not a model limitation we
   can engineer around.

3. **The real win is the secondary signal + calibration.** Option A's
   auxiliary BCE supervision on the sub-classifiers produces a disagreement
   score that **does** track misclassification (AUC 0.755, 4/5 folds with
   p < 0.05). Temperature scaling on a held-out 15% calibration split cuts
   ECE roughly in half (0.138 → 0.077) — honest, no double-dipping between
   fit and evaluation.

This is what a capability portrait looks like when the headline metric
doesn't move: you ship the parts that did move, name the parts that didn't,
and walk the reader through both.

---

## Architecture (one paragraph)

Two pole-specific input masks (LumA, LumB) are derived from MSigDB Hallmark
gene sets — `ESTROGEN_RESPONSE_EARLY` + `ESTROGEN_RESPONSE_LATE` for LumA;
`E2F_TARGETS` + `G2M_CHECKPOINT` + `MYC_TARGETS_V1` for LumB — and the HM450
probe-to-gene cis-mapping from UCSC Xena. Each pole branch sees a hypothesis-attended view of RNA + methylation
through an MLP encoder, ending in a sub-classifier that's supervised with an
auxiliary BCE loss (`aux_weight=0.3`). The two pole representations are fused
and concatenated with a scalar disagreement = `|s_LumA − (1 − s_LumB)|` before
the final classifier head. Temperature scaling is fit by LBFGS on a held-out
15% calibration split per fold.

See [`docs/architecture.md`](docs/architecture.md) for the diagram + module map.

---

## Headline tables (5-fold CV, n = 417)

### 3-way ablation

| Variant | AUROC | BalAcc |
|---|---|---|
| **Option A** (aux BCE + disagreement IN, **ships in v0.1**) | 0.9606 ± 0.017 | 0.888 ± 0.045 |
| Option B (no aux + disagreement IN) | 0.9679 ± 0.012 | 0.895 ± 0.014 |
| Ablation (no aux + no disagreement) | 0.9690 ± 0.013 | 0.912 ± 0.029 |
| Δ A − B | −0.007 | −0.007 |
| Δ A − Ablation | −0.008 | −0.024 |

Option A pays ~0.007–0.008 AUROC for the auxiliary supervision that surfaces
the disagreement signal. Option B and the Ablation are tied within noise.
The architecture neither helps nor hurts the primary metric.

### Calibration (Option A)

| T fit on | Mean T | Mean ECE on val | Interpretation |
|---|---|---|---|
| val (optimistic, upper bound) | 0.523 ± 0.267 | 0.0685 | T tuned to the same fold ECE is measured on |
| **held-out cal split (honest)** | **0.591 ± 0.127** | **0.0774** | T never saw the val data |

T < 1 means DMOI is **under-confident**: the pole-conditioned architecture
plus class-balanced BCE compress logits toward zero. Calibration *sharpens*
them. The nested mean T is much more stable (std 0.127 vs 0.267) than the
optimistic fit — Fold 5's optimistic T = 0.065 (an outlier) becomes
T = 0.468 when fit on a held-out cal split.

### Pooled OOF confusion matrix

|       | pred LumA | pred LumB |
|-------|-----------|-----------|
| true LumA | 265 | 24 |
| true LumB | 18 | 110 |

Pooled accuracy 0.899 · LumB sensitivity 0.859 · LumB specificity 0.917.

Per-fold detail in [`audit/dmoi_eval_per_fold.tsv`](audit/dmoi_eval_per_fold.tsv);
full report in [`audit/dmoi_eval_v0.md`](audit/dmoi_eval_v0.md).

---

## Reproduce

```bash
# 1. Install pinned deps.
uv sync

# 2. (One-time) Fetch the cohort: TCGA-BRCA RNA-seq + HM450 from UCSC Xena.
python scripts/build_cohort_v2.py     # produces data/tcga_brca/cohort_v2.tsv

# 3. Baseline (LogReg + RF on concat / rna / meth).
python scripts/run_baseline_v2.py     # writes audit/baseline_v2_*

# 4. DMOI 3-way ablation + calibration.
python scripts/eval_dmoi.py           # ~2 min on Apple Silicon (MPS)
                                      # writes audit/dmoi_eval_v0.md + per_fold.tsv
```

Pinned to Python 3.11+, `numpy 2.2`, `scikit-learn 1.7`, `torch 2.x`
(MPS-supported on Apple Silicon).

---

## Layout

```
src/dmoi_brca/
├── features.py             # cohort + RNA + streaming top-K methylation loader
├── priors.py               # Hallmark gene-set priors per pole
├── hypothesis_attention.py # cis-mapping + PoleMaskSet + make_pole_masks
├── encoder.py              # pole-conditioned MLP encoders
├── fusion.py               # dual-perspective fuser + disagreement scalar
├── dmoi_model.py           # end-to-end DMOIModel
├── train.py                # train_one_fold + run_dmoi_cv (StratifiedKFold)
├── eval.py                 # per-class metrics, ECE, disagreement-vs-misclass
├── calibration.py          # temperature scaling (LBFGS on log_T)
├── baseline.py             # sklearn baselines (LogReg, RF)
├── audit.py                # NDJSON hash-chained ledger
├── tracking.py             # MLflow run wrapper
└── canary.py               # smoke-test interface for lab_semantic_check.py

scripts/
├── build_cohort_v2.py      # cohort selection (LumA + LumB dual-modality)
├── run_baseline_v2.py      # baseline driver
├── train_dmoi.py           # Day-3 single-config DMOI driver
├── eval_dmoi.py            # Day-4+ 3-way ablation + calibration driver
└── check_english_only.py   # CJK gate enforced pre-push
```

---

## What's out of scope for v0.1

See [`docs/what-is-out-of-scope.md`](docs/what-is-out-of-scope.md) for the
full list. Key items deliberately deferred:

- **External validation** (e.g. METABRIC). v0.1 is a single-cohort POC.
- **Other pole hypotheses** (ER−/HER2+, basal vs claudin-low).
- **Full Hallmark gene-set incorporation**. Four sets used; the rest are
  in `priors.py` as documented constants but not yet routed to attention.
- **Nested CV for hyperparameter tuning**. `calibration_frac=0.15` is a
  scalar choice carried over from the Guo et al. recommendation, not
  swept.
- **Per-patient explainability**. The model exposes `pole_scores` +
  `disagreement` per patient, but a SHAP / IG layer is v0.2+.

---

## License

MIT. See [`LICENSE`](LICENSE).
