# METABRIC Cohort v3 Summary (DMOI v0.10 cross-cohort + cross-task eval)

Generated: 2026-05-28T18:00:47Z

## Source

- Study: Curtis et al. *Nature* 2012 + Pereira et al. *Nat Commun* 2016
- cBioPortal study ID: `brca_metabric`
- Subset used: patients with CLAUDIN_SUBTYPE in {LumA, LumB, Basal} AND mRNA microarray (Illumina HT-12 v3) available
- Label assignment: LumA + LumB -> Luminal; Basal -> Basal

## What v0.10 uses METABRIC for

Cross-cohort + cross-task external validation for the v0.9
framework. v0.9 trained the same v0.6 architecture with
Luminal/Basal pole priors on TCGA cohort_v3 and reached AUROC
1.000 with 8/8 expected priors in per-pole IG top-5. v0.10 asks
whether that finding holds when the trained model is scored on
METABRIC (different microarray platform, different patient
demographics) using the same RNA-only + meth-silenced + QN-to-TCGA
protocol established in v0.2 / v0.4.

## Cohort sizes

| Subset | LumA | LumB | Basal | Luminal (LumA+LumB) | Total |
|---|---|---|---|---|---|
| All PAM50/CLAUDIN_SUBTYPE-called | 700 | 475 | 209 | 1175 | 1384 |
| With mRNA microarray (cohort_v3.tsv) | — | — | 209 | 1175 | 1384 |

## Note on excluded subtypes

METABRIC distribution of CLAUDIN_SUBTYPE (all 1,980 called):
- LumA 700 / LumB 475 / Basal 209 (this cohort)
- claudin-low 218 / Her2 224 / Normal 148 / NC 6
  (excluded -- out of scope for the v0.9 Luminal-vs-Basal target)

## Reproduce

```bash
python scripts/fetch_metabric.py             # one-time, ~690 MB
python scripts/build_metabric_cohort_v3.py
```
