# Clinical association of selected genes (interpretability, OncoDB-style)

n = 956 TCGA-BRCA samples. For each label-free RNA selector (100 genes), the
fraction whose expression is significantly associated (Kruskal-Wallis for categorical
stage / node, Spearman for age; BH-FDR q < 0.05) with clinical variables that
are **independent of the PAM50 target**. Higher = the selected feature set is more
clinically meaningful, not just predictive of subtype.

| selector (label-free, RNA) | any variable | stage | node | age |
|---|---|---|---|---|
| DMOI-prior(5-set) | 0.730 | 0.320 | 0.040 | 0.560 |
| DMOI-prior(50-set) | 0.500 | 0.000 | 0.010 | 0.500 |
| top-variance | 0.610 | 0.000 | 0.150 | 0.530 |

## Reading

- A higher *any-variable* fraction for the biological prior than for top-variance means
  the prior selects genes that carry clinical signal beyond subtype separation — the
  same kind of evidence Omran et al. 2025 reported via OncoDB (MOFA+ 59% vs MoGCN 47%).
- Stage/node/age are deliberately not the classification target, so this is a
  non-circular biological-coherence check (unlike Hallmark over-representation, which
  the prior would satisfy by construction).
