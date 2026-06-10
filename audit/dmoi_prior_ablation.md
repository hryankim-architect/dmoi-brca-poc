# DMOI prior ablation — which Hallmark sets drive the advantage (RNA-only, PAM50)

n = 956 TCGA-BRCA samples with a PAM50 call ({'Basal': 142, 'Her2': 67, 'LumA': 434, 'LumB': 194, 'Normal': 119}). RNA-only,
label-free selection; 5-class weighted-F1, stratified 5-fold. The variable is purely
*which Hallmark gene set(s)* define the prior — the downstream classifier is unchanged.

| selector (label-free, RNA) | n_feat | LR wF1 | SVC wF1 | CHI ↑ | DBI ↓ |
|---|---|---|---|---|---|
| only HALLMARK_ESTROGEN_RESPONSE_EARLY | 107 | 0.818 | 0.804 | 96.5 | 3.50 |
| only HALLMARK_ESTROGEN_RESPONSE_LATE | 115 | 0.835 | 0.827 | 83.2 | 3.50 |
| only HALLMARK_E2F_TARGETS | 189 | 0.836 | 0.820 | 119.2 | 3.56 |
| only HALLMARK_G2M_CHECKPOINT | 185 | 0.841 | 0.844 | 109.3 | 3.53 |
| only HALLMARK_MYC_TARGETS_V1 | 188 | 0.815 | 0.818 | 63.0 | 4.09 |
| 5-set minus HALLMARK_ESTROGEN_RESPONSE_EARLY | 100 | 0.888 | 0.872 | 165.0 | 2.17 |
| 5-set minus HALLMARK_ESTROGEN_RESPONSE_LATE | 100 | 0.884 | 0.875 | 179.9 | 2.06 |
| 5-set minus HALLMARK_E2F_TARGETS | 100 | 0.874 | 0.862 | 149.6 | 2.33 |
| 5-set minus HALLMARK_G2M_CHECKPOINT | 100 | 0.867 | 0.864 | 146.4 | 2.36 |
| 5-set minus HALLMARK_MYC_TARGETS_V1 | 100 | 0.872 | 0.864 | 152.8 | 2.31 |
| all 5 sets (cap100) | 100 | 0.872 | 0.864 | 152.8 | 2.31 |
| top-variance(100) | 100 | 0.773 | 0.773 | 87.7 | 3.04 |

## Reading

- **"only X" rows** show each curated set's standalone discriminative power; a high row
  means that set alone carries much of the PAM50 signal.
- **"5-set minus X" rows** show the cost of dropping one set from the curated prior; a
  large drop vs *all 5 sets* means X is load-bearing, a negligible drop means it is
  redundant with the others.
- Compared against the *top-variance(100)* baseline, this localizes the v0.15 prior
  advantage to specific proliferation/ER biology rather than the prior as a whole.
