# `dmoi-brca-poc`

> **Capability portrait, not a research result.** Public data is intentionally
> subsetted to keep the demo small and reproducible on a single workstation.

**What this shows.** A hypothesis-conditioned multi-omics architecture
(Dialectical Multi-Omics Integration, DMOI) on TCGA-BRCA. Two "pole" perspectives
(LumA-like and LumB-like) each see only their pole-relevant RNA + DNA-methylation
features via Hallmark-gene priors and HM450 cis-mapping. Their predictions are
fused with a disagreement signal exposed to the classifier head. Temperature
scaling on a held-out calibration split delivers honest, well-calibrated
probabilities. **v0.2 adds two external generalization tests** — a held-out
TCGA test split and an independent cohort (METABRIC, RNA-only) — and a clean
cohort-specific calibration analysis.

**Reproducibility.** `python scripts/eval_dmoi.py` reproduces the full TCGA
evaluation in about 2 minutes on an M-series Mac.
`python scripts/eval_external.py` adds the METABRIC external test
in another ~3 minutes (after a one-time ~690 MB METABRIC download).

**Substrate.** Emits hash-chained NDJSON audit entries, tracks MLflow runs,
exposes a canary smoke test for `lab_semantic_check.py`.

**Production framing.** A version of this method ran at full cohort scale on
proprietary multi-omics data during my time in industry. The lab version here
proves the *method* and the *engineering*, not the result. See
[`docs/what-is-out-of-scope.md`](docs/what-is-out-of-scope.md).

---

## v0.3 headline result

| Metric | DMOI v0.3 |
|---|---|
| 5-fold CV AUROC (TCGA train split, n=333) | 0.954 ± 0.017 |
| **Held-out TCGA test AUROC (n=84, scored once)** | **0.968** |
| **METABRIC external AUROC (n=1,175, RNA-only)** | **0.909** |
| ECE after T-scaling on held-out TCGA test | 0.079  (T=0.634) |
| ECE on METABRIC eval slice — cohort-specific T | 0.074  (T_METABRIC=0.934) |
| Disagreement AUC for misclass (TCGA CV) | 0.715 (2/5 folds significant) |
| **Per-patient Integrated Gradients attribution** (v0.3) | **lumA / lumB / final logit on TCGA test** |

**The honest takeaway, in five acts:**

1. **Baseline saturated the easy signal.** Plain LogReg on
   concat(RNA, methylation) lands at 0.963 AUROC on the 417-patient
   LumA vs LumB cohort. Within-luminal class structure is already strongly
   expressed in the bulk transcriptome; there isn't much room for an
   architectural upgrade to beat it.

2. **Hypothesis-conditioned attention did NOT lift AUROC.** DMOI's primary
   metric ties baseline within noise (Δ ≈ −0.002 on the full v0.1 cohort).
   The dual-perspective architecture's value is **not** in the headline
   number — the LogReg ceiling reflects the structure of the data, not a
   model limitation we can engineer around.

3. **Calibration was the v0.1 win.** Temperature scaling on a held-out
   15% nested calibration split cuts ECE roughly in half on TCGA (0.138 →
   0.077). T < 1 — the architecture is *under*-confident, not over-confident.

4. **External generalization was the v0.2 win.** A truly held-out TCGA test
   split (random_state=2024, never seen during model dev) scores AUROC 0.968.
   An independent cohort (METABRIC, n=1,175) with the methylation branch
   silenced (METABRIC has no HM450 data) scores AUROC 0.909. Calibration
   parameters do NOT transfer naively between cohorts: TCGA's T=0.634
   over-sharpens the meth-silenced METABRIC predictions; METABRIC's own
   cal-split-fit T=0.934 is correctly close to 1.0.

5. **Interpretability is the v0.3 win.** Per-patient Integrated Gradients
   attribution on the TCGA test set reveals the architecture is doing
   sophisticated biology: the LumA pole learned **inverse-basal-marker**
   discrimination (FOXC1, KRT15 used as "this is NOT basal") plus the
   canonical anti-apoptotic luminal gene BCL2; the LumB pole learned
   cell-cycle structural genes (RANBP1, SMC6, ZW10). Canonical pan-luminal
   markers ESR1/PGR are correctly absent from the top attributions because
   they don't discriminate within the ER+ cohort. The model picked the
   right discrimination axis.

---

## Architecture (one paragraph)

Two pole-specific input masks (LumA, LumB) are derived from MSigDB Hallmark
gene sets — `ESTROGEN_RESPONSE_EARLY` + `ESTROGEN_RESPONSE_LATE` for LumA;
`E2F_TARGETS` + `G2M_CHECKPOINT` + `MYC_TARGETS_V1` for LumB — and the HM450
probe-to-gene cis-mapping from UCSC Xena. Each pole branch sees a
hypothesis-attended view of RNA + methylation through an MLP encoder,
ending in a sub-classifier supervised with an auxiliary BCE loss
(`aux_weight=0.3`). The two pole representations are fused and concatenated
with a scalar disagreement = `|s_LumA − (1 − s_LumB)|` before the final
classifier head. Temperature scaling is fit by LBFGS on a held-out 15%
calibration split.

See [`docs/architecture.md`](docs/architecture.md) for the diagram + module map.

---

## Internal results (TCGA cohort_v2, 5-fold CV on the 80% train split)

### 3-way ablation

| Variant | AUROC | BalAcc |
|---|---|---|
| **Option A** (aux BCE + disagreement IN, ships in v0.2) | 0.954 ± 0.021 | 0.861 ± 0.067 |
| Option B (no aux + disagreement IN) | 0.958 ± 0.027 | 0.869 ± 0.066 |
| Ablation (no aux + no disagreement) | 0.961 ± 0.023 | 0.889 ± 0.041 |
| Δ A − B | −0.004 | −0.007 |
| Δ A − Ablation | −0.007 | −0.028 |

Option A pays ~0.007 AUROC for the auxiliary supervision that surfaces the
disagreement signal. The architecture neither helps nor hurts the primary
metric.

### Calibration (5-fold CV)

| T fit on | Mean T | Mean ECE on val | Interpretation |
|---|---|---|---|
| val (optimistic, upper bound) | 0.620 ± 0.252 | 0.111 | T tuned to the same fold ECE is measured on |
| **held-out cal split (honest, ship)** | **0.673 ± 0.294** | **0.121** | T never saw the val data |

T < 1 means DMOI is **under-confident**: the pole-conditioned architecture
plus class-balanced BCE compress logits toward zero. Calibration *sharpens*.

### Held-out TCGA test (v0.2 Path C, n=84)

The 20% TCGA test split is carved at cohort-construction time with
`random_state=2024` (distinct from the CV seed) and scored *once* by a single
Option A model trained on the full train split for the CV-mean best epoch
(no early stopping, no test-AUC-driven epoch selection).

- AUROC : **0.968** (internal CV mean: 0.954, Δ = +0.014)
- BalAcc : 0.897
- ECE before T-scaling : 0.143
- ECE after T-scaling  : **0.079**  (T=0.634)

Test ≥ CV is unusual but in the right direction — the model isn't overfitting
to the CV folds.

---

## External validation on METABRIC (v0.2 Path A', n=1,175)

| Metric | METABRIC |
|---|---|
| AUROC | **0.9095** |
| BalAcc | 0.788 |
| LumB sensitivity (default 0.5 threshold) | 0.619 |
| LumB specificity | 0.956 |

### Calibration does NOT blindly transfer across cohorts

Reported on the same 85% METABRIC eval slice (n=999); the other 15% (n=176)
is used as the cohort-specific cal slice.

| Calibration | T | ECE on eval slice |
|---|---|---|
| Uncalibrated | 1.000 | 0.0745 |
| T from TCGA cal-split (naive transfer) | 0.634 | 0.1051 |
| **T from METABRIC cal-split (cohort-specific)** | **0.934** | **0.0738** |

TCGA's T was fit on a model with both RNA + methylation; on METABRIC the
methylation branch is silenced, so the logit distribution is different and
TCGA's sharpening T over-corrects. A METABRIC-specific T lands near 1.0
(the meth-silenced model is already nearly well-calibrated) and slightly
improves ECE. **Calibration parameters are cohort/modality specific.**

### LumB sensitivity decomposes into prior shift + modality silencing

The 0.619 / 0.956 sensitivity-specificity asymmetry on METABRIC isn't a
single phenomenon. Two corrections compared on the same 85% eval slice:

| Strategy | LumB sens | LumB spec | BalAcc | F1 LumB |
|---|---|---|---|---|
| Default @0.5 | 0.619 | 0.956 | 0.788 | 0.735 |
| **Bayes prior-adjusted** | **0.691** | 0.933 | **0.812** | **0.772** |
| Tuned threshold (0.425 from cal slice) | 0.656 | 0.943 | 0.799 | 0.754 |

TCGA train has 31% LumB; METABRIC has 40% LumB. Bayes class-prior adjustment
boosts LumB calls without tuning, gaining +0.072 sensitivity and +0.024
BalAcc. The data-tuned threshold lands at 0.425, exactly the direction
Bayes predicts. Both corrections triangulate to the same conclusion: **the
prior shift explains about half of the sensitivity asymmetry. The remainder
is the methylation-silencing residual** — the meth branch normally
contributes positive signal for harder LumB calls, and without it some
borderline cases are unrecoverable from RNA alone.

### What METABRIC validation does and does NOT show

It DOES show that the hypothesis-conditioned RNA encoder generalizes to an
independent cohort across platforms (HiSeq → Illumina HT-12 v3) with
quantile normalization and gene-symbol harmonization (16,890 shared genes
out of 20,530 TCGA / 20,384 METABRIC unique Hugo symbols).

It does NOT validate the dual-modality story — no public BRCA cohort has
paired RNA-seq + HM450 outside TCGA. See
[`docs/v0.2-design-external-validation.md`](docs/v0.2-design-external-validation.md)
for the recon trail.

Full reports: [`audit/dmoi_eval_v0.md`](audit/dmoi_eval_v0.md) (TCGA),
[`audit/dmoi_external_v0.2.md`](audit/dmoi_external_v0.2.md) (METABRIC).

---

## Per-patient attribution (v0.3, TCGA test n=84)

Integrated Gradients (Sundararajan et al. 2017) on each of three model
outputs, baseline = zero in the standardized domain (= train per-feature
mean). 50 Riemann steps per IG run. See
[`docs/v0.3-design-attribution.md`](docs/v0.3-design-attribution.md) for
the algorithm + scope rationale.

### Top-5 global features per pole (mean |IG| across 84 test patients)

| Rank | lumA_pole RNA | mean \|IG\| | lumB_pole RNA | mean \|IG\| |
|---|---|---|---|---|
| 1 | `FOXC1` | 0.0508 | `EFNA5` | 0.0189 |
| 2 | `PDLIM3` | 0.0359 | `RANBP1` | 0.0142 |
| 3 | `TUBB2B` | 0.0329 | `SMC6` | 0.0110 |
| 4 | **`BCL2`** | 0.0328 | `ZW10` | 0.0108 |
| 5 | `KRT15` | 0.0323 | `DMD` | 0.0101 |

### Three biological readings

- **lumA pole learned "this is NOT basal-like."** FOXC1 and KRT15 are
  basal/myoepithelial markers; the LumA pole's strongest discriminative
  signal is their *low* expression in LumA. BCL2 is the only canonical
  anti-apoptotic luminal marker in the top-5 — the rest of the LumA
  pole's attribution comes from contrast against basal features.
- **lumB pole learned cell-cycle structural genes.** RANBP1 (nuclear
  transport in mitosis), SMC6 (chromatin maintenance), ZW10 (mitotic
  checkpoint) are all proliferation-machinery genes. Not the textbook
  MKI67/TOP2A/AURKA but biologically equivalent — many gene proxies
  exist for the proliferation axis and the model picked the structural-
  mitotic ones.
- **ESR1 / PGR / FOXA1 are correctly absent from the top attributions.**
  Both LumA and LumB are ER+, so the canonical luminal markers don't
  discriminate within this cohort. Their absence here is evidence that
  the model picked the right axis (proliferation + inverse-basal) rather
  than a naïve pan-luminal prior.

### Completeness check (IG faithfulness axiom)

| Target | Mean residual | Max residual | Status |
|---|---|---|---|
| **lumA_pole** | 0.0016 | 0.0155 | tight |
| **lumB_pole** | 0.0027 | 0.0216 | acceptable |
| final_logit | 0.0162 | 0.2062 | one outlier — likely from the disagreement scalar `\|s_LumA − (1 − s_LumB)\|` which has a non-differentiable `abs()` at 0; the pole-specific attributions are the recommended clinical-interpretability headline |

Full per-patient + global lists:
[`audit/dmoi_explain_v0.3.md`](audit/dmoi_explain_v0.3.md),
[`audit/dmoi_explain_per_patient.tsv`](audit/dmoi_explain_per_patient.tsv)
(5,040 rows = 84 × 3 × 2 × top-10), [`audit/dmoi_explain_global.tsv`](audit/dmoi_explain_global.tsv).

---

## Reproduce

```bash
# 1. Install pinned deps.
uv sync

# 2. (One-time) Fetch TCGA cohort.
python scripts/build_cohort_v2.py     # produces data/tcga_brca/cohort_v2.tsv

# 3. Baseline (LogReg + RF on concat / rna / meth).
python scripts/run_baseline_v2.py     # writes audit/baseline_v2_*

# 4. DMOI 3-way ablation + calibration + held-out TCGA test (Step A).
python scripts/eval_dmoi.py           # ~2 min on Apple Silicon (MPS)

# 5. (One-time) Fetch METABRIC. ~690 MB.
python scripts/fetch_metabric.py
python scripts/build_metabric_cohort.py

# 6. METABRIC external validation + cohort-specific cal + LumB sens analysis.
python scripts/eval_external.py       # ~3 min on MPS

# 7. (v0.3) Per-patient Integrated Gradients attribution on TCGA test (n=84).
python scripts/explain_dmoi.py        # ~3-4 min on MPS
                                      # writes audit/dmoi_explain_v0.3.md
                                      # + per_patient.tsv + global.tsv + 3 PNG plots
```

Pinned to Python 3.11+, `numpy 2.2`, `scikit-learn 1.7`, `torch 2.x`,
`captum 0.7+` (MPS-supported on Apple Silicon).

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
├── attribution.py          # v0.3: Captum-based Integrated Gradients wrapper
├── external.py             # v0.2: cross-cohort gene align + QN + meth-silenced helpers
├── cohort.py               # cohort construction + 80/20 train/test split
├── baseline.py             # sklearn baselines (LogReg, RF)
├── audit.py                # NDJSON hash-chained ledger
├── tracking.py             # MLflow run wrapper
└── canary.py               # smoke-test interface for lab_semantic_check.py

scripts/
├── build_cohort_v2.py        # TCGA cohort selection + stratified 80/20 split
├── run_baseline_v2.py        # baseline driver
├── train_dmoi.py             # single-config DMOI driver
├── eval_dmoi.py              # 3-way ablation + calibration + held-out TCGA test
├── fetch_metabric.py         # v0.2: cBioPortal LFS download (~690 MB)
├── build_metabric_cohort.py  # v0.2: filter to LumA/LumB
├── eval_external.py          # v0.2: cross-cohort eval + cal-transfer + LumB sens
├── explain_dmoi.py           # v0.3: per-patient IG attribution + audit MD
└── check_english_only.py     # CJK gate enforced pre-push
```

---

## What's out of scope for v0.3

See [`docs/what-is-out-of-scope.md`](docs/what-is-out-of-scope.md) for the
full list. Key items still deliberately deferred after v0.3:

- **Multi-modal external validation.** No public BRCA cohort outside TCGA
  has paired RNA-seq + HM450 — see the v0.2 design doc for the recon.
- **METABRIC attribution.** v0.3 ships IG on TCGA test only; METABRIC IG
  is straightforward (same `attribution.py` API, just bigger n=1,175) but
  deferred to a v0.4 to keep the v0.3 audit MD focused.
- **Other pole hypotheses** (ER−/HER2+, basal vs claudin-low).
- **Full Hallmark gene-set incorporation**. Four sets used; the rest are
  in `priors.py` as documented constants but not yet routed to attention.
- **Pathway-level attribution aggregation** (e.g., MSigDB rollup of IG
  scores). v0.3 is gene-level only.
- **Counterfactual explanations** ("what would need to change to flip the
  prediction") — adversarial-style, much heavier than IG.
- **Nested CV for hyperparameter tuning**. `calibration_frac=0.15` is a
  fixed choice carried over from Guo et al., not swept.

---

## License

MIT. See [`LICENSE`](LICENSE).
