# METABRIC Cohort Summary (DMOI v0.2 external validation)

Generated: 2026-05-28T03:27:19Z

## Source

- Study: Curtis et al. *Nature* 2012 + Pereira et al. *Nat Commun* 2016
- cBioPortal study ID: `brca_metabric`
- Subset used: patients with PAM50 / CLAUDIN_SUBTYPE in {LumA, LumB} AND mRNA microarray (Illumina HT-12 v3) available

## What v0.2 uses METABRIC for

External validation for DMOI's RNA branch (Path A'). METABRIC has
no HM450 methylation, so the methylation pole encoder is silenced
(`meth = zeros`) at inference time. This tests whether the
hypothesis-conditioned RNA encoder generalizes across cohorts, but
does NOT validate the dual-modality story.

## Cohort sizes

| Subset | LumA | LumB | Total |
|---|---|---|---|
| All PAM50-called METABRIC patients | 700 | 475 | 1175 |
| With mRNA microarray (cohort.tsv) | 700 | 475 | 1175 |

## Note on excluded subtypes

METABRIC distribution of CLAUDIN_SUBTYPE (all 1,980 PAM50-called):
- LumA 700 / LumB 475 (this cohort)
- claudin-low 218 / Her2 224 / Basal 209 / Normal 148 / NC 6
  (excluded — out of scope for DMOI's LumA-vs-LumB target)

## Reproduce

```bash
python scripts/fetch_metabric.py        # ~690 MB one-time download
python scripts/build_metabric_cohort.py
```
