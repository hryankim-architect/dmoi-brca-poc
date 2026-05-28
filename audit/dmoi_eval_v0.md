# DMOI Full Evaluation + Ablation (Day-4)

Generated: 2026-05-28T02:28:55Z

## Cohort v2

- Dual-modality patients: **417** (LumA 289 / LumB 128)

## Headline metrics (Full DMOI, 5-fold CV)

- **AUROC** : 0.9679 ± 0.0117
- **BalAcc** : 0.8949 ± 0.0136
- **F1 LumA** : 0.9229 ± 0.0217
- **F1 LumB** : 0.8425 ± 0.0227  ← minority class
- **ECE** : 0.1327 ± 0.0361  (lower = better calibrated)
- **Disagreement AUC for misclass** : 0.4133 ± 0.1116  (0.5 = no signal, 1.0 = perfect)

## Ablation: disagreement IN vs OUT of classifier head

Both runs use the same encoder + attention + fuser; only the final
ClassifierHead differs. The flag `use_disagreement` toggles whether
the scalar disagreement value is concatenated alongside [z_LumA, z_LumB].

| Variant | AUROC | BalAcc |
|---|---|---|
| Full DMOI (disagreement IN) | 0.9679 ± 0.0117 | 0.8949 ± 0.0136 |
| Ablation (disagreement OUT) | 0.9690 ± 0.0127 | 0.9123 ± 0.0285 |
| **Δ (full − ablation)** | **-0.0011** | **-0.0174** |

Interpretation:
- If Δ AUROC > +0.005 and the disagreement-vs-misclass AUC is > 0.6, the disagreement feature is empirically useful and DMOI's Option-B thesis is supported.
- If Δ AUROC ≈ 0 (within 1 std), the disagreement feature is **redundant** with what the pole-fused latents already encode.

## Disagreement-vs-misclassification analysis

- Mean disagreement AUC for predicting misclass: 0.4133
- Per-fold AUCs: [0.6036184210526316, 0.35855263157894735, 0.3157894736842105, 0.4083333333333333, 0.3802816901408451]
- Point-biserial correlation r per fold: ['+0.110', '-0.147', '-0.178', '-0.094', '-0.164']
- Point-biserial p per fold: ['0.3153', '0.1772', '0.1035', '0.3967', '0.1344']

**0/5 folds** show statistically informative disagreement (mean dis on misclass > mean dis on correct AND p < 0.05). **Thesis NOT supported** on this run: disagreement is not statistically elevated on misclassified cases. The signal is noise on cohort_v2. v0.2 should either drop disagreement or move to Option A.

## Pooled OOF confusion matrix (all 5 folds concatenated)

|       | pred LumA | pred LumB |
|-------|-----------|-----------|
| true LumA | 260 | 29 |
| true LumB | 14 | 114 |

Pooled accuracy: 0.8969  ·  LumB sensitivity: 0.8906  ·  LumB specificity: 0.8997

## Reproduce

```bash
python scripts/eval_dmoi.py
```
