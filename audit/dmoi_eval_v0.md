# DMOI Full Evaluation + Ablation (Day-4)

Generated: 2026-05-28T03:22:07Z

## Cohort v2

- Dual-modality patients: **333** (LumA 231 / LumB 102)

## Headline metrics (Full DMOI, 5-fold CV)

- **AUROC** : 0.9540 ± 0.0213
- **BalAcc** : 0.8614 ± 0.0670
- **F1 LumA** : 0.9105 ± 0.0337
- **F1 LumB** : 0.8006 ± 0.0817  ← minority class
- **ECE** : 0.1567 ± 0.0274  (lower = better calibrated)
- **Disagreement AUC for misclass** : 0.7146 ± 0.1177  (0.5 = no signal, 1.0 = perfect)

## 3-way ablation: Option A vs Option B vs no-disagreement

Three architectural variants. All share the same encoder + attention
+ fuser; only the loss and ClassifierHead input vary:

- **Option A** (v0.2 candidate): aux BCE supervision on sub-classifiers
  with weight 0.3; disagreement scalar included as classifier-head input.
- **Option B** (v0.1 baseline): NO aux supervision; disagreement IN.
- **Ablation**: NO aux + disagreement OUT.

| Variant | AUROC | BalAcc |
|---|---|---|
| Option A (aux + disagreement IN) | 0.9540 ± 0.0213 | 0.8614 ± 0.0670 |
| Option B (no aux + disagreement IN) | 0.9577 ± 0.0273 | 0.8687 ± 0.0655 |
| Ablation (no aux + disagreement OUT) | 0.9607 ± 0.0226 | 0.8889 ± 0.0415 |
| **Δ A − B** | **-0.0037** | **-0.0073** |
| **Δ A − Ablation** | **-0.0067** | **-0.0275** |

Interpretation:
- **Δ A − B**: does the auxiliary supervision on sub-classifiers add value?
  + If > +0.005: aux supervision sharpens the disagreement signal, Option A wins.
  + If ≈ 0: aux didn't help meaningfully; v0.1 (Option B) was already near-optimal.
  + If < 0: aux supervision over-constrained the sub-classifiers; revisit weight.
- **Δ A − Ablation**: is the dual-perspective architecture (with supervised sub-clfs)
  better than dropping the disagreement / sub-clf branch entirely?

## Disagreement-vs-misclassification analysis

- Mean disagreement AUC for predicting misclass: 0.7146
- Per-fold AUCs: [0.7131147540983607, 0.6610169491525424, 0.6580645161290323, 0.6237816764132553, 0.9172714078374455]
- Point-biserial correlation r per fold: ['+0.188', '+0.278', '+0.196', '+0.205', '+0.616']
- Point-biserial p per fold: ['0.1221', '0.0198', '0.1066', '0.0938', '0.0000']

**2/5 folds** show statistically informative disagreement (mean dis on misclass > mean dis on correct AND p < 0.05). **Partial support** for DMOI's Option-B thesis. Disagreement is informative on some folds but not consistently. v0.2 should consider Option A (auxiliary BCE on sub-classifier scores) as an alternative.

## Temperature scaling calibration (Option A)

Single-parameter post-hoc calibration via Guo et al. 2017:
`calibrated_proba = sigmoid(logits / T)`. T fit by LBFGS on the
BCE NLL of each fold's val logits.

**Caveat (v0.1):** T is fit on the same val fold we measure ECE on,
which is **optimistic**, it's an upper bound on what post-hoc
calibration can buy with this architecture on this cohort.
v0.2+ should fit T on a nested calibration split carved out of
the train fold.

- Mean T : **0.620 ± 0.252**  (T > 1 = overconfident; T = 1 = already calibrated)
- Per-fold T : ['0.419', '0.639', '0.393', '0.626', '1.023']
- Per-fold ECE (uncalibrated) : ['0.1969', '0.1323', '0.1723', '0.1358', '0.1463']
- Per-fold ECE (T-calibrated) : ['0.1146', '0.0922', '0.1067', '0.0948', '0.1469']
- **Mean ECE before → after** : **0.1567 → 0.1110**
- **Δ ECE (improvement)** : **+0.0457**

## Temperature scaling calibration, nested split (Option A)

Each fold also held out 15% of its train data (stratified on y) as a
calibration split that the model never trained on. T is fit on those
cal logits and applied to val. This is the held-out number, no
double-dipping between fit and evaluation.

- Cal split fraction : **15%** of each train fold (stratified)
- Mean T (nested) : **0.673 ± 0.294**
- Per-fold T (nested) : ['0.515', '0.605', '0.551', '0.502', '1.194']
- Per-fold ECE on val (T fit on cal split) : ['0.1488', '0.0957', '0.1155', '0.0968', '0.1458']
- **Mean ECE before → after (nested)** : **0.1567 → 0.1205**
- **Δ ECE (improvement)** : **+0.0362**

Comparison:

| T fit on | Mean T | Mean ECE on val | Notes |
|---|---|---|---|
| val (optimistic) | 0.620 | 0.1110 | upper bound, T tuned to the same fold |
| held-out cal split (held-out) | 0.673 | 0.1205 | what generalizes |

## Held-out TCGA test (v0.2 Path C, 80/20 split)

A 84-patient stratified test split was carved at cohort construction time with random_state=2024 (distinct from the CV seed). It is scored **once** by a single Option A model trained on the full train split (333 patients) for 15 epochs (CV mean best epoch; no early stopping; `pick_best_epoch=False` so val/test AUC does not select an epoch).

- **Test AUROC** : **0.9682** (internal CV mean: 0.9540)
- **Test BalAcc**: 0.8972
- **Test ECE before T-scaling** : 0.1431
- **Test ECE after T-scaling**  : 0.0793  (T=0.634 fit on a 15% cal split of train)

| | pred LumA | pred LumB |
|---|---|---|
| true LumA | 55 | 3 |
| true LumB | 4 | 22 |

Test accuracy: 0.9167  ·  LumB sensitivity: 0.8462  ·  LumB specificity: 0.9483

## Pooled OOF confusion matrix (all 5 folds concatenated)

|       | pred LumA | pred LumB |
|-------|-----------|-----------|
| true LumA | 208 | 23 |
| true LumB | 18 | 84 |

Pooled accuracy: 0.8769  ·  LumB sensitivity: 0.8235  ·  LumB specificity: 0.9004

## Reproduce

```bash
python scripts/eval_dmoi.py
```
