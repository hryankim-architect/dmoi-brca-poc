# DMOI v0.7, Pathway-pole attention (Variant D)

Generated: 2026-05-28T11:50:10Z (Phase B run, raw expression + warmer init)

## Setup

- Architecture: v0.6 base + `PathwayPoleAttention(n_pathways=50)`.
  Per-pole softmax over the full v2024.1.Hs Hallmark catalog.
- Train cohort: TCGA cohort_v2 train split, n=333.
- TCGA test:    n=84 (held out, scored once).
- METABRIC:     n=1175 (RNA-only + meth silenced + QN).
- Epochs: 15, optimizer: AdamW(lr=1e-4, wd=1e-4), BCEWithLogitsLoss + aux=0.3, pick_best_epoch=False (no val peeking).

## Headline AUROC

| Cohort | v0.7 AUROC | v0.6 reference | Delta |
|---|---|---|---|
| TCGA held-out test | 0.9595 | 0.9682 | -0.0087 |
| METABRIC external  | 0.8976 | 0.9091 | -0.0115 |

**Verdict**: AUROC DROPPED.

AUROC was never the success criterion, the v0.6 baseline is at the LogReg ceiling. The v0.7 question is whether learned pathway-pole attention reproduces the v0.6 IG-derived ranking from scratch.

## Learned pathway-pole attention (top 10 per pole)

### LumA

| Rank | Pathway | softmax weight |
|---|---|---|
| 1 | `HALLMARK_WNT_BETA_CATENIN_SIGNALING` | 0.0689 |
| 2 | `HALLMARK_INTERFERON_ALPHA_RESPONSE` | 0.0591 |
| 3 | `HALLMARK_MTORC1_SIGNALING` | 0.0383 |
| 4 | `HALLMARK_ALLOGRAFT_REJECTION` | 0.0362 |
| 5 | `HALLMARK_CHOLESTEROL_HOMEOSTASIS` | 0.0300 |
| 6 | `HALLMARK_SPERMATOGENESIS` | 0.0285 |
| 7 | `HALLMARK_APOPTOSIS` | 0.0282 |
| 8 | `HALLMARK_UV_RESPONSE_DN` | 0.0261 |
| 9 | `HALLMARK_ANDROGEN_RESPONSE` | 0.0256 |
| 10 | `HALLMARK_INFLAMMATORY_RESPONSE` | 0.0247 |

### LumB

| Rank | Pathway | softmax weight |
|---|---|---|
| 1 | `HALLMARK_MTORC1_SIGNALING` | 0.0475 |
| 2 | `HALLMARK_KRAS_SIGNALING_UP` | 0.0461 |
| 3 | `HALLMARK_PEROXISOME` | 0.0400 |
| 4 | `HALLMARK_XENOBIOTIC_METABOLISM` | 0.0391 |
| 5 | `HALLMARK_P53_PATHWAY` | 0.0380 |
| 6 | `HALLMARK_PANCREAS_BETA_CELLS` | 0.0337 |
| 7 | `HALLMARK_HEME_METABOLISM` | 0.0336 |
| 8 | `HALLMARK_COAGULATION` | 0.0310 |
| 9 | `HALLMARK_ANGIOGENESIS` | 0.0289 |
| 10 | `HALLMARK_TNFA_SIGNALING_VIA_NFKB` | 0.0289 |

## v0.6 (post-hoc IG) vs v0.7 (learned attention), top-3 agreement

v0.6 top-3 per pole comes from `audit/dmoi_pathway_v0.6.md` (50-set Hallmark IG rollup, identical on TCGA test + METABRIC).

| Pole | v0.7 learned top-3 | v0.6 IG top-3 | Shared (n / 3) |
|---|---|---|---|
| **LumA** | `HALLMARK_INTERFERON_ALPHA_RESPONSE`, `HALLMARK_MTORC1_SIGNALING`, `HALLMARK_WNT_BETA_CATENIN_SIGNALING` | `HALLMARK_ESTROGEN_RESPONSE_EARLY`, `HALLMARK_ESTROGEN_RESPONSE_LATE`, `HALLMARK_IL2_STAT5_SIGNALING` | 0 / 3 |
| **LumB** | `HALLMARK_KRAS_SIGNALING_UP`, `HALLMARK_MTORC1_SIGNALING`, `HALLMARK_PEROXISOME` | `HALLMARK_E2F_TARGETS`, `HALLMARK_G2M_CHECKPOINT`, `HALLMARK_MYC_TARGETS_V1` | 0 / 3 |

## Reading

- `softmax weight`, each pole's attention weights sum to 1.0 across the 50 pathways. Weight = 0.02 means "uniform" (1/50). Anything above 0.05 is a meaningful preference; above 0.20 is strong concentration.
- Agreement count is informational, not a hypothesis test. With 50 pathways the chance of a random 3-set match is (50 choose 3 with k hits), not zero but small.

## Two-phase architecture experiment

v0.7 ran in two phases to isolate the failure mode cleanly.

### Phase A, standardized inputs + tight init (collapse)

| Cohort | AUROC | vs v0.6 |
|---|---|---|
| TCGA test  | 0.957 | -0.011 |
| METABRIC   | 0.913 | +0.004 |

- Top weights spanned 0.0203, 0.0205 across all 50 pathways on
  both poles, i.e. effectively uniform (1 / 50 = 0.0200).
- 0 of 3 v0.6 IG top pathways made the v0.7 Phase A top-3.
- **Mechanism**: pathway scores were standardized to zero mean before
  feeding `PathwayPoleAttention`. Softmax-uniform attention multiplied
  by a zero-centered input produced a near-zero per-pole feature,
  which the classifier head learned to ignore, which zeroed out the
  gradient flowing back to the attention. Combined with `wd=1e-4`
  pulling logits toward zero, the attention stayed uniform forever,
  a self-reinforcing equilibrium of uselessness.

### Phase B, raw expression + warmer init (collapse fixed; different basin learned)

| Cohort | AUROC | vs v0.6 | vs Phase A |
|---|---|---|---|
| TCGA test  | 0.960 | -0.009 | +0.003 |
| METABRIC   | 0.898 | -0.012 | -0.016 |

- Top weights now differentiate: LumA top weight 0.069 (3.4x uniform);
  LumB top weight 0.048 (2.4x uniform).
- 0 of 3 v0.6 IG top pathways still make the v0.7 Phase B top-3 on
  either pole. The model learned an entirely different ranking.
- **Mechanism**: with raw mean expression as input, uniform attention
  no longer produces zero output, the head gets a non-zero feature
  with variance across patients, gradient flows, attention learns.
  But what it learns is **magnitude-driven** rather than
  **direction-driven**. The scalar `pole_pathway_feat = sum_k w_k *
  mean_expression_k(patient)` rewards pathways whose member genes
  have high absolute expression baseline (WNT_BETA_CATENIN,
  INTERFERON_ALPHA, MTORC1, KRAS_UP, PEROXISOME, all pathways with
  high-magnitude expression profiles). The LumA-vs-LumB discriminative
  signal, however, is in the *direction* (ER program up for LumA,
  cell-cycle program up for LumB), not in absolute magnitude. So the
  Phase B attention learns to discriminate based on a signal that is
  not class-discriminative, and AUROC drops on both cohorts.

The fix doesn't undo the failure mode, it shifts it from
"collapse" to "wrong-basin". Both findings are publishable.

### Lesson candidates

- **Lvarphi candidate (softmax-attention collapse under standardized
  inputs)**, documented in Phase A. Softmax-mixed input
  architectures need the mixed feature output to be non-zero under
  uniform weights, or no gradient flows back to the attention.
  Currently 1 occurrence in DMOI v0.7 Phase A; scaffold-template
  promotion requires a 2nd independent occurrence.
- **(provisional, post-v0.7.1) scalar-pole softmax captures
  magnitude not direction**, documented in Phase B. A `(batch,
  n_poles) = pathway_scores @ attn_weights.T` projection has only
  the per-pole pathway *magnitude* as discriminative axis. Direction
  signals (which pole-relevant pathway is up vs down per patient)
  require either a per-pathway projection (Variant C) or auxiliary
  pathway-direction supervision.

## v0.8 plan, Variant C

Project `pole_pathway_feat` from scalar, (batch, n_poles), to
vector, (batch, n_poles, proj_dim). Architecture sketch:

```
pathway_scores  in  (batch, n_pathways)
attn_weights    in  (n_poles, n_pathways)  (softmax-normalized per pole)
for each pole P:
    gated_P[batch, k] = pathway_scores[batch, k] * attn_weights[P, k]
    pole_vec_P[batch, :] = Linear(n_pathways, proj_dim)(gated_P)
pole_pathway_feat[batch, P, :] = pole_vec_P
```

The head sees a `(batch, n_poles * proj_dim)` flattened vector.
Each pathway gets a learnable embedding (the row of the projection
matrix), so the head can read per-pathway direction signals weighted
by the pole's attention.

Backward-compatibility: `proj_dim=None` (default) keeps the v0.7.1
scalar path so existing scripts and checkpoints continue to work.

## Scope

- Single-fold final-model run (no CV). The v0.7 architecture diff is what's under test; held-out test scoring matches the v0.6 protocol.
- The pathway branch sees a per-pole scalar feature (weighted-sum of the 50 pathway-mean expressions). A richer projection (per-pole vector instead of scalar) is the v0.8 Variant C upgrade.
- Gene-level interpretation (v0.3 / v0.4 IG) is unaffected, the gene-level branch is unchanged from v0.6.

## Reproduce

```bash
python scripts/eval_dmoi_v0.7.py  # Phase B (current)
# To reproduce Phase A, revert the v0.7.1 patches to train.py +
# pathway_attention.py (StandardScaler back; init_std back to 0.01).
```
