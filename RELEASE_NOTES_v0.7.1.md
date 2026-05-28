# DMOI v0.7.1 — Phase B added: collapse fixed, wrong basin learned

## TL;DR

v0.7.0 shipped Phase A as a documented softmax-attention collapse
(uniform weights, no learning). v0.7.1 lands the planned Phase B
fixes — drop the StandardScaler on pathway scores, bump
`PathwayPoleAttention.init_std` from 0.01 → 0.5 — and re-runs the
smoke. Result: **the collapse is fixed, but the model learns a
completely different basin than v0.6's IG-derived ranking, and AUROC
drops on both cohorts**.

Both phases are now recorded in
[`audit/dmoi_v0.7.md`](audit/dmoi_v0.7.md). v0.6 remains canonical.
The next attempt is **v0.8 Variant C** — scalar → vector pole
feature via a learnable per-pathway embedding.

## What changed in code from v0.7.0

Three small patches, all consistent with the Phase B plan published
in v0.7.0's audit:

- `src/dmoi_brca/train.py` — compute pathway scores from RAW RNA
  (not standardized); remove the per-pathway StandardScaler.
  `FoldResult.pathway_scaler` is now always `None` for the v0.7.1
  pathway-branch run.
- `src/dmoi_brca/pathway_attention.py` — bump `init_std` default
  from 0.01 to 0.5. Asymmetric init gives gradient a direction on
  epoch 1 before weight decay symmetrizes things.
- `scripts/eval_dmoi_v0.7.py` — `_score_cohort` no longer needs a
  `pathway_scaler` argument; pathway scores computed from raw RNA at
  inference time too. Audit MD timestamp now notes "Phase B run".

No model-architecture changes. v0.6 backward-compat preserved
(n_pathways=0 path unchanged).

## Phase A vs Phase B side-by-side

| Metric | Phase A | Phase B | v0.6 |
|---|---|---|---|
| TCGA test AUROC      | 0.957 | 0.960 | 0.968 |
| METABRIC AUROC       | 0.913 | 0.898 | 0.909 |
| LumA top weight      | 0.0205 | **0.0689** | (uniform 0.0200) |
| LumB top weight      | 0.0205 | **0.0475** | (uniform 0.0200) |
| LumA top-3 ∩ v0.6 IG | 0 / 3 | 0 / 3 | — |
| LumB top-3 ∩ v0.6 IG | 0 / 3 | 0 / 3 | — |

Phase B's LumA top weight (0.069) is **3.4× uniform** — clear
evidence the attention learned. The model just learned a
different signal than the IG-derived ranking from v0.6.

## What Phase B actually learned

### LumA pole top-5 (v0.7 Phase B, raw-input + warm-init)

| Rank | Pathway | softmax weight |
|---|---|---|
| 1 | `HALLMARK_WNT_BETA_CATENIN_SIGNALING` | 0.0689 |
| 2 | `HALLMARK_INTERFERON_ALPHA_RESPONSE`  | 0.0591 |
| 3 | `HALLMARK_MTORC1_SIGNALING`           | 0.0383 |
| 4 | `HALLMARK_ALLOGRAFT_REJECTION`        | 0.0362 |
| 5 | `HALLMARK_CHOLESTEROL_HOMEOSTASIS`    | 0.0300 |

### LumB pole top-5

| Rank | Pathway | softmax weight |
|---|---|---|
| 1 | `HALLMARK_MTORC1_SIGNALING`        | 0.0475 |
| 2 | `HALLMARK_KRAS_SIGNALING_UP`       | 0.0461 |
| 3 | `HALLMARK_PEROXISOME`              | 0.0400 |
| 4 | `HALLMARK_XENOBIOTIC_METABOLISM`   | 0.0391 |
| 5 | `HALLMARK_P53_PATHWAY`             | 0.0380 |

These are real, biologically coherent pathways (WNT crosstalks with
ESR1; MTORC1 is downstream of many growth signals), but they are
**not** the canonical LumA-vs-LumB discriminative pathways v0.6
identified post-hoc via IG.

## Why the wrong basin (mechanism)

The scalar `pole_pathway_feat = sum_k w_k × mean_expression_k(patient)`
gives the classifier head exactly one number per pole per patient.
That number can only carry **magnitude variance** information — "how
much of this pole's weighted-average pathway expression does this
patient have?". It cannot carry **direction** information — "which
specific genes within the pole's pathways are up vs down for this
patient compared to the cohort baseline?".

The LumA-vs-LumB classification problem lives in direction. Both
classes are ER+ luminal; ESR1 and PGR are *equally expressed* on
average across LumA and LumB. The discriminative signal is "ER program
relative balance vs cell-cycle program relative balance," not "high vs
low overall pathway expression magnitude."

So with a scalar interface, the attention finds the next-best signal
the head can use: pathways with **large patient-to-patient variance
in absolute magnitude** that happen to weakly correlate with the
class label. WNT, MTORC1, KRAS_UP have large between-patient variance
in absolute expression because they contain many highly-expressed
genes with high natural variance. The attention concentrates there.

But that variance isn't the discriminative variance, so AUROC drops.

### Lesson candidate (provisional name post-v0.7.1)

> A scalar-per-pole softmax-attention-over-pathway-scores
> architecture has a fundamental information bottleneck: the per-pole
> output is a single number, and a single number can only encode
> magnitude variance. Direction-signal discrimination requires either
> a per-pathway projection (Variant C, vector per pole), per-pathway
> direction supervision, or per-pole multi-head attention.

This is a second occurrence-candidate for a future Polish-Phase5
lesson — but distinct from the Phase A "softmax collapse under
standardized inputs" candidate (Lvarphi).

## v0.8 plan — Variant C

Project the pole pathway feature from scalar → vector per pole:

```
pathway_scores  in  (batch, n_pathways)
attn_weights    in  (n_poles, n_pathways)   (softmax-normalized per pole)
for each pole P:
    gated_P  = pathway_scores * attn_weights[P, :]      # (batch, n_pathways)
    pole_vec_P = Linear(n_pathways, proj_dim)(gated_P)  # (batch, proj_dim)
pole_pathway_feat = stack(pole_vec_P, P)               # (batch, n_poles, proj_dim)
```

The head sees a `(batch, n_poles * proj_dim)` flattened vector. Each
pathway gets a learnable embedding (the row of the projection matrix),
so the head can read per-pathway direction signals.

Backward-compatibility: `proj_dim=None` (default) keeps the v0.7.1
scalar path.

## Reproduce

```bash
python scripts/eval_dmoi_v0.7.py
# writes: audit/dmoi_v0.7.md (Phase B run -- raw expression + warmer init)
```

To reproduce Phase A: revert the patches to `train.py`
(StandardScaler back) and `pathway_attention.py` (`init_std=0.01`),
then re-run.

## Audit

See [`audit/dmoi_v0.7.md`](audit/dmoi_v0.7.md) for the full two-phase
write-up + mechanism diagnosis + v0.8 plan.
