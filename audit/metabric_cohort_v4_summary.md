# METABRIC Cohort v4 Summary (DMOI v0.14 HER2 external validation)

Generated: 2026-06-02T18:01:55Z

- Source: Curtis 2012 + Pereira 2016 (`brca_metabric`)
- HER2 = CLAUDIN_SUBTYPE == 'Her2' (PAM50-style; differs slightly from TCGA clinical HER2+ — see eval caveat)
- Luminal = CLAUDIN_SUBTYPE in {LumA, LumB}
- RNA-only (methylation pole silenced at inference, v0.10 pattern)

| Subset | with mRNA |
|---|---|
| HER2 | 224 |
| Luminal | 1175 |
