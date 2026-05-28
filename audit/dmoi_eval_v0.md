# DMOI Full Evaluation + Ablation (Day-4)

Generated: 2026-05-28T02:55:21Z

## Cohort v2

- Dual-modality patients: **417** (LumA 289 / LumB 128)

## Headline metrics (Full DMOI, 5-fold CV)

- **AUROC** : 0.9606 ± 0.0173
- **BalAcc** : 0.8884 ± 0.0449
- **F1 LumA** : 0.9266 ± 0.0231
- **F1 LumB** : 0.8391 ± 0.0541  ← minority class
- **ECE** : 0.1375 ± 0.0552  (lower = better calibrated)
- **Disagreement AUC for misclass** : 0.7545 ± 0.0978  (0.5 = no signal, 1.0 = perfect)

## 3-way ablation: Option A vs Option B vs no-disagreement

Three architectural variants. All share the same encoder + attention
+ fuser; only the loss and ClassifierHead input vary:

- **Option A** (v0.2 candidate): aux BCE supervision on sub-classifiers
  with weight 0.3; disagreement scalar included as classifier-head input.
- **Option B** (v0.1 baseline): NO aux supervision; disagreement IN.
- **Ablation**: NO aux + disagreement OUT.

| Variant | AUROC | BalAcc |
|---|---|---|
| Option A (aux + disagreement IN) | 0.9606 ± 0.0173 | 0.8884 ± 0.0449 |
| Option B (no aux + disagreement IN) | 0.9679 ± 0.0117 | 0.8949 ± 0.0136 |
| Ablation (no aux + disagreement OUT) | 0.9690 ± 0.0127 | 0.9123 ± 0.0285 |
| **Δ A − B** | **-0.0072** | **-0.0065** |
| **Δ A − Ablation** | **-0.0083** | **-0.0239** |

Interpretation:
- **Δ A − B**: does the auxiliary supervision on sub-classifiers add value?
  + If > +0.005: aux supervision sharpens the disagreement signal — Option A wins.
  + If ≈ 0: aux didn't help meaningfully; v0.1 (Option B) was already near-optimal.
  + If < 0: aux supervision over-constrained the sub-classifiers; revisit weight.
- **Δ A − Ablation**: is the dual-perspective architecture (with supervised sub-clfs)
  better than dropping the disagreement / sub-clf branch entirely?

## Disagreement-vs-misclassification analysis

- Mean disagreement AUC for predicting misclass: 0.7545
- Per-fold AUCs: [0.7891891891891891, 0.7384806973848069, 0.610759493670886, 0.8816666666666666, 0.7522522522522522]
- Point-biserial correlation r per fold: ['+0.313', '+0.295', '+0.119', '+0.423', '+0.268']
- Point-biserial p per fold: ['0.0029', '0.0052', '0.2797', '0.0000', '0.0123']

**4/5 folds** show statistically informative disagreement (mean dis on misclass > mean dis on correct AND p < 0.05). DMOI's Option-B thesis (disagreement is INFORMATIVE rather than a regularization target) is **empirically supported** on cohort_v2: high-disagreement cases are disproportionately the misclassified ones, which are biologically the LumA/LumB borderline tumors where the two pole perspectives genuinely disagree.

## Temperature scaling calibration (Option A)

Single-parameter post-hoc calibration via Guo et al. 2017:
`calibrated_proba = sigmoid(logits / T)`. T fit by LBFGS on the
BCE NLL of each fold's val logits.

**Caveat (v0.1):** T is fit on the same val fold we measure ECE on,
which is **optimistic** — it's an upper bound on what post-hoc
calibration can buy with this architecture on this cohort.
v0.2+ should fit T on a nested calibration split carved out of
the train fold.

- Mean T : **0.523 ± 0.267**  (T > 1 = overconfident; T = 1 = already calibrated)
- Per-fold T : ['0.642', '0.721', '0.519', '0.669', '0.065']
- Per-fold ECE (uncalibrated) : ['0.1189', '0.0952', '0.1328', '0.1077', '0.2330']
- Per-fold ECE (T-calibrated) : ['0.0609', '0.0505', '0.0517', '0.0680', '0.1113']
- **Mean ECE before → after** : **0.1375 → 0.0685**
- **Δ ECE (improvement)** : **+0.0690**

## Temperature scaling calibration — nested split (Option A, honest)

Each fold also held out 15% of its train data (stratified on y) as a
calibration split that the model never trained on. T is fit on those
cal logits and applied to val. This is the **honest** number — no
double-dipping between fit and evaluation.

- Cal split fraction : **15%** of each train fold (stratified)
- Mean T (nested) : **0.591 ± 0.127**
- Per-fold T (nested) : ['0.443', '0.721', '0.674', '0.650', '0.468']
- Per-fold ECE on val (T fit on cal split) : ['0.0821', '0.0504', '0.0744', '0.0680', '0.1120']
- **Mean ECE before → after (nested)** : **0.1375 → 0.0774**
- **Δ ECE (honest improvement)** : **+0.0601**

Comparison:

| T fit on | Mean T | Mean ECE on val | Notes |
|---|---|---|---|
| val (optimistic) | 0.523 | 0.0685 | upper bound — T tuned to the same fold |
| held-out cal split (honest) | 0.591 | 0.0774 | what generalizes |

## Pooled OOF confusion matrix (all 5 folds concatenated)

|       | pred LumA | pred LumB |
|-------|-----------|-----------|
| true LumA | 265 | 24 |
| true LumB | 18 | 110 |

Pooled accuracy: 0.8993  ·  LumB sensitivity: 0.8594  ·  LumB specificity: 0.9170

## Reproduce

```bash
python scripts/eval_dmoi.py
```
