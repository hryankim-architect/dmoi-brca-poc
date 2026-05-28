# DMOI Full Evaluation + Ablation (Day-4)

Generated: 2026-05-28T02:47:35Z

## Cohort v2

- Dual-modality patients: **417** (LumA 289 / LumB 128)

## Headline metrics (Full DMOI, 5-fold CV)

- **AUROC** : 0.9680 ± 0.0148
- **BalAcc** : 0.9024 ± 0.0123
- **F1 LumA** : 0.9284 ± 0.0212
- **F1 LumB** : 0.8533 ± 0.0247  ← minority class
- **ECE** : 0.1496 ± 0.0338  (lower = better calibrated)
- **Disagreement AUC for misclass** : 0.6826 ± 0.1412  (0.5 = no signal, 1.0 = perfect)

## 3-way ablation: Option A vs Option B vs no-disagreement

Three architectural variants. All share the same encoder + attention
+ fuser; only the loss and ClassifierHead input vary:

- **Option A** (v0.2 candidate): aux BCE supervision on sub-classifiers
  with weight 0.3; disagreement scalar included as classifier-head input.
- **Option B** (v0.1 baseline): NO aux supervision; disagreement IN.
- **Ablation**: NO aux + disagreement OUT.

| Variant | AUROC | BalAcc |
|---|---|---|
| Option A (aux + disagreement IN) | 0.9680 ± 0.0148 | 0.9024 ± 0.0123 |
| Option B (no aux + disagreement IN) | 0.9679 ± 0.0117 | 0.8949 ± 0.0136 |
| Ablation (no aux + disagreement OUT) | 0.9690 ± 0.0127 | 0.9123 ± 0.0285 |
| **Δ A − B** | **+0.0002** | **+0.0075** |
| **Δ A − Ablation** | **-0.0009** | **-0.0099** |

Interpretation:
- **Δ A − B**: does the auxiliary supervision on sub-classifiers add value?
  + If > +0.005: aux supervision sharpens the disagreement signal — Option A wins.
  + If ≈ 0: aux didn't help meaningfully; v0.1 (Option B) was already near-optimal.
  + If < 0: aux supervision over-constrained the sub-classifiers; revisit weight.
- **Δ A − Ablation**: is the dual-perspective architecture (with supervised sub-clfs)
  better than dropping the disagreement / sub-clf branch entirely?

## Disagreement-vs-misclassification analysis

- Mean disagreement AUC for predicting misclass: 0.6826
- Per-fold AUCs: [0.6725925925925926, 0.7884972170686456, 0.5112781954887218, 0.854978354978355, 0.5858585858585859]
- Point-biserial correlation r per fold: ['+0.236', '+0.323', '+0.081', '+0.349', '+0.094']
- Point-biserial p per fold: ['0.0279', '0.0020', '0.4643', '0.0008', '0.3951']

**3/5 folds** show statistically informative disagreement (mean dis on misclass > mean dis on correct AND p < 0.05). DMOI's Option-B thesis (disagreement is INFORMATIVE rather than a regularization target) is **empirically supported** on cohort_v2: high-disagreement cases are disproportionately the misclassified ones, which are biologically the LumA/LumB borderline tumors where the two pole perspectives genuinely disagree.

## Temperature scaling calibration (Option A)

Single-parameter post-hoc calibration via Guo et al. 2017:
`calibrated_proba = sigmoid(logits / T)`. T fit by LBFGS on the
BCE NLL of each fold's val logits.

**Caveat (v0.1):** T is fit on the same val fold we measure ECE on,
which is **optimistic** — it's an upper bound on what post-hoc
calibration can buy with this architecture on this cohort.
v0.2+ should fit T on a nested calibration split carved out of
the train fold.

- Mean T : **0.555 ± 0.107**  (T > 1 = overconfident; T = 1 = already calibrated)
- Per-fold T : ['0.578', '0.638', '0.549', '0.635', '0.375']
- Per-fold ECE (uncalibrated) : ['0.1487', '0.1305', '0.1513', '0.1140', '0.2038']
- Per-fold ECE (T-calibrated) : ['0.0748', '0.0688', '0.0626', '0.0471', '0.1087']
- **Mean ECE before → after** : **0.1496 → 0.0724**
- **Δ ECE (improvement)** : **+0.0772**

## Pooled OOF confusion matrix (all 5 folds concatenated)

|       | pred LumA | pred LumB |
|-------|-----------|-----------|
| true LumA | 262 | 27 |
| true LumB | 13 | 115 |

Pooled accuracy: 0.9041  ·  LumB sensitivity: 0.8984  ·  LumB specificity: 0.9066

## Reproduce

```bash
python scripts/eval_dmoi.py
```
