# DMOI v0.7 (Phase A) — learnable pathway-pole attention: honest negative

## TL;DR

v0.7 attempted to replace v0.6's hand-picked pole masks with a
learnable softmax distribution over the full 50-set MSigDB Hallmark
catalog per pole (Variant D from the v0.7 design doc). The
architecture-level question: *can the model rediscover v0.6's
ER-for-LumA / cell-cycle-for-LumB alignment from scratch?*

**Answer (Phase A): No, the softmax attention collapsed to uniform.**
TCGA AUROC slipped 1.1pp; METABRIC AUROC moved up 0.4pp (roughly
neutral). 0 of 3 v0.6 top pathways made the v0.7 top-3 on either
pole. A self-reinforcing equilibrium-of-uselessness failure mode,
fully documented in [`audit/dmoi_v0.7.md`](audit/dmoi_v0.7.md).

**v0.6 remains the canonical architecture.** v0.7 ships as a recorded
honest-negative architecture experiment with a planned Phase B retry
(v0.7.1) that fixes the root cause.

## Nine-act narrative

1. **v0.0** — 5-fold CV on `cohort_v2` → AUROC 0.968.
2. **v0.1** — Temperature scaling on a nested calibration split.
3. **v0.2** — METABRIC RNA-only external validation; cohort-specific T.
4. **v0.3** — Per-patient IG on TCGA test; lumA=FOXC1/BCL2,
   lumB=RANBP1/NBN/ZW10/POLA2.
5. **v0.4 (cleanup + METABRIC IG)** — Cross-cohort Jaccard top-10 = 0.667.
6. **v0.5** — Pathway-level rollup over 5 priors-Hallmark sets:
   ~300× ER vs cell-cycle on LumA; ~45× cell-cycle vs ER on LumB.
7. **v0.6** — Full 50-set Hallmark rollup: all v0.5 top pathways
   stay in top-3 of 50 on both cohorts.
8. **v0.7 (this release, Phase A)** — Tried learnable softmax pathway-
   pole attention (Variant D minimal). Attention collapsed to uniform
   (0.0203–0.0205 across 50 sets vs uniform 0.0200). Diagnosed as a
   mechanical failure mode of softmax-over-zero-centered-inputs;
   documented + planned Phase B retry.

## Headline AUROC

| Cohort | v0.7 Phase A | v0.6 reference | Δ |
|---|---|---|---|
| TCGA held-out test | 0.957 | 0.968 | −0.011 |
| METABRIC external  | 0.913 | 0.909 | +0.004 |

Roughly neutral on average, consistent with the head needing to learn
to ignore two noisy extra input dimensions.

## Top-3 pathway agreement (v0.7 learned vs v0.6 IG-derived)

| Pole | v0.7 Phase A top-3 | v0.6 IG top-3 | Shared |
|---|---|---|---|
| LumA | `INTERFERON_ALPHA`, `MTORC1`, `WNT` | `ESTROGEN_RESPONSE_EARLY`, `ESTROGEN_RESPONSE_LATE`, `IL2_STAT5_SIGNALING` | **0 / 3** |
| LumB | `KRAS_UP`, `MTORC1`, `PEROXISOME` | `E2F_TARGETS`, `G2M_CHECKPOINT`, `MYC_TARGETS_V1` | **0 / 3** |

The "top-3" lists are picked among near-tied softmax weights (0.0203
–0.0205) and carry no signal.

## Why the collapse (mechanism)

1. v0.7 standardized per-patient pathway-expression scores (mean → 0,
   std → 1) before feeding `PathwayPoleAttention`.
2. Softmax-uniform attention × zero-centered inputs → per-pole feature
   averages ~0 across patients.
3. Classifier head learns to ignore the near-zero input.
4. No downstream signal → tiny gradient back to `attn_logits`. With
   `wd=1e-4` pulling logits toward zero, attention stays uniform.

This is a real architecture lesson candidate (provisional name **Lϟ**)
for the Polish-Phase5 canon: *softmax-mixed inputs need non-zero
output under uniform weights, or no gradient flows back to the
attention*. Will be promoted to scaffold-template if observed in a
second project (currently one occurrence; scaffold lesson promotion
requires two).

## What changed in code

- **New module**: `src/dmoi_brca/pathway_attention.py` (~155 LOC)
  - `compute_pathway_expression_scores(rna, feature_names, pathways)`
    per-patient mean expression per Hallmark pathway.
  - `PathwayPoleAttention(n_pathways, pole_order, init_std=0.01)` —
    learnable softmax distribution per pole, exposes `attn_weights`
    and `top_k_pathways(names, k)` helper.
- **`DMOIModel`** gets opt-in `n_pathways: int = 0` kwarg. n_pathways=0
  keeps v0.6 backward compatibility exactly; n_pathways>0 wires the
  pathway branch into the ClassifierHead.
- **`ClassifierHead`** gets opt-in `n_pole_pathway_feats: int = 0`
  extending its input dim. Same backward-compat pattern.
- **`train.train_one_fold`** gets `pathway_genes` +
  `rna_feature_names` kwargs. Pathway scores are pre-computed from
  standardized RNA (← the Phase A failure mode) and run through a
  separate StandardScaler before model input. FoldResult gains
  `pathway_scaler` + `pathway_names` for downstream introspection.
- **New tests**: `tests/test_pathway_attention.py` (22 tests, all
  pass) — compute_pathway_expression_scores correctness +
  PathwayPoleAttention forward/grad/softmax-normalization/top_k +
  DMOIModel n_pathways=0 vs n_pathways>0 behavior + opt-in errors.
- **New driver**: `scripts/eval_dmoi_v0.7.py` — trains the v0.7
  model with `pathway_genes=hallmark` on TCGA train, scores TCGA
  test + METABRIC, extracts learned attention, compares to v0.6 IG
  top-3, writes audit MD with auto-verdict.
- **New design doc**:
  `docs/v0.7-design-pathway-attention.md` (4 variants A/B/C/D + per-
  variant pros/cons + success-criteria table that explicitly named
  this Phase A outcome as a possible read).

No changes to v0.6 / v0.5 / v0.4 / v0.3 / v0.2 / v0.1 / v0.0 code.
v0.7 is purely additive.

## Honest scope

- Single fold, no CV. The architecture diff is what's under test.
- Pathway branch is a per-pole scalar. Variant C (richer projection)
  is a v0.8+ candidate if Phase B also collapses.
- Gene-level interpretation (v0.3 / v0.4 IG) is unaffected.
- The negative finding does *not* invalidate v0.5 / v0.6 — those are
  post-hoc rollups of a successful v0.4 model and remain canonical.

## Phase B plan (next release, v0.7.1)

- Drop the StandardScaler on pathway scores. Use raw per-patient mean
  expression so uniform-attention output is no longer zero-centered.
- Bump `PathwayPoleAttention.init_std` from 0.01 to 0.5. Asymmetric
  init gives gradient a direction on epoch 1 before weight decay
  symmetrizes things.

If Phase B also collapses: Variant C upgrade (project pole pathway
feature scalar → vector for richer head interface).

## Reproduce

```bash
python scripts/eval_dmoi_v0.7.py
# writes: audit/dmoi_v0.7.md
```

## Audit

[`audit/dmoi_v0.7.md`](audit/dmoi_v0.7.md) for full per-pole top-10
attention weights + collapse-mode diagnosis + Phase B plan.
