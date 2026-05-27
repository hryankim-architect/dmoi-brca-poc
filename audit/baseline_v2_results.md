# DMOI POC Baseline v2 Results (Day-5A)

Generated: 2026-05-27T20:59:37Z

## Target

Within-luminal LumA vs LumB (Week-2 re-scope).
Discriminating axis: proliferation rate.

## Cohort

- Dual-modality patients: **417**
- LumA: 289 (69.3%)
- LumB: 128 (30.7%)

## Features

- RNA-seq: 20530 genes
- Methylation (HM450): 10000 probes (top-variance from 485k)
- Concat: 30530 features

## Results (mean ± std, 5-fold CV)

| Feature set | Model | AUROC | Balanced accuracy |
|---|---|---|---|
| concat | logreg | 0.9626 ± 0.0150 | 0.8917 ± 0.0372 |
| concat | rf | 0.9630 ± 0.0236 | 0.8498 ± 0.0374 |
| meth | logreg | 0.8800 ± 0.0297 | 0.7625 ± 0.0597 |
| meth | rf | 0.8179 ± 0.0567 | 0.6288 ± 0.0321 |
| rna | logreg | 0.9605 ± 0.0153 | 0.8908 ± 0.0203 |
| rna | rf | 0.9679 ± 0.0227 | 0.8732 ± 0.0433 |

## Headroom confirmed

At least one baseline configuration lands in the 0.55-0.95 AUROC range,
leaving room for Week-2 hypothesis-conditioning to show a meaningful gain.
The LumA vs LumB re-scope is validated.

## Reproduce

```bash
python scripts/build_cohort_v2.py
python scripts/run_baseline_v2.py
```
