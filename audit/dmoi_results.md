# DMOI Training Results (Day-3 smoke run)

Generated: 2026-05-28T02:13:16Z

## Cohort

- Dual-modality v2 patients: **417**
- LumA: 289 (69.3%)
- LumB: 128 (30.7%)

## Pole masks (input gating)

- **LumA**: rna on 161/20530, meth on 66/10000
- **LumB**: rna on 447/20530, meth on 72/10000

## Training config

- Latent dim: 128
- RNA encoder: 20530 → 1024 → 256 → 128
- Meth encoder: 10000 → 512 → 128
- Fuser per pole: 256 → 128 → 64; sub-classifier 64 → 1
- Head: [z_LumA, z_LumB, disagreement] → 32 → 1
- AdamW lr=1e-4 weight_decay=1e-4; batch=64; up to 50 epochs;
  early stop on val AUROC, patience=10
- BCEWithLogitsLoss + pos_weight = n_LumA / n_LumB (class-balanced)
- StratifiedKFold(n_splits=5, shuffle=True, random_state=42) [same as baseline_v2]

## Aggregate (mean ± std across 5 folds)

- **AUROC**: 0.9679 ± 0.0117
- **Balanced accuracy**: 0.8949 ± 0.0136
- Best epoch (mean / max): 11.8 / 20
- Total runtime: 11.5 s

## Head-to-head vs baseline_v2

Baseline_v2 best concat configuration (LogReg, 5-fold same folds):
- baseline concat LogReg: AUROC **0.963 ± 0.015**, BalAcc 0.892 ± 0.037
- baseline meth   LogReg: AUROC **0.880 ± 0.030**, BalAcc 0.763 ± 0.060
- baseline rna    LogReg: AUROC **0.961 ± 0.015**, BalAcc 0.891 ± 0.020

DMOI vs concat LogReg: ΔAUROC = +0.0049
DMOI vs meth  LogReg: ΔAUROC = +0.0879

## Scope

Baseline AUROC values range 0.9496 - 0.9800 (mean=0.9679, n=5).
This is a non-trivial comparison anchor for the next-step model.

## Reproduce

```bash
python scripts/train_dmoi.py
```
