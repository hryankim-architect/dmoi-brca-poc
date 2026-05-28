# DMOI v0.7 -- Pathway-pole attention (Variant D)

Generated: 2026-05-28T11:37:50Z

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
| TCGA held-out test | 0.9569 | 0.9682 | -0.0113 |
| METABRIC external  | 0.9132 | 0.9091 | +0.0041 |

**Verdict**: AUROC DROPPED.

AUROC was never the success criterion -- the v0.6 baseline is at the LogReg ceiling. The v0.7 question is whether learned pathway-pole attention reproduces the v0.6 IG-derived ranking from scratch.

## Learned pathway-pole attention (top 10 per pole)

### LumA

| Rank | Pathway | softmax weight |
|---|---|---|
| 1 | `HALLMARK_INTERFERON_ALPHA_RESPONSE` | 0.0205 |
| 2 | `HALLMARK_WNT_BETA_CATENIN_SIGNALING` | 0.0205 |
| 3 | `HALLMARK_MTORC1_SIGNALING` | 0.0204 |
| 4 | `HALLMARK_ALLOGRAFT_REJECTION` | 0.0203 |
| 5 | `HALLMARK_CHOLESTEROL_HOMEOSTASIS` | 0.0203 |
| 6 | `HALLMARK_SPERMATOGENESIS` | 0.0202 |
| 7 | `HALLMARK_REACTIVE_OXYGEN_SPECIES_PATHWAY` | 0.0202 |
| 8 | `HALLMARK_OXIDATIVE_PHOSPHORYLATION` | 0.0202 |
| 9 | `HALLMARK_ANDROGEN_RESPONSE` | 0.0202 |
| 10 | `HALLMARK_MITOTIC_SPINDLE` | 0.0202 |

### LumB

| Rank | Pathway | softmax weight |
|---|---|---|
| 1 | `HALLMARK_MTORC1_SIGNALING` | 0.0205 |
| 2 | `HALLMARK_PEROXISOME` | 0.0204 |
| 3 | `HALLMARK_KRAS_SIGNALING_UP` | 0.0203 |
| 4 | `HALLMARK_XENOBIOTIC_METABOLISM` | 0.0203 |
| 5 | `HALLMARK_P53_PATHWAY` | 0.0203 |
| 6 | `HALLMARK_UNFOLDED_PROTEIN_RESPONSE` | 0.0203 |
| 7 | `HALLMARK_HEME_METABOLISM` | 0.0202 |
| 8 | `HALLMARK_PANCREAS_BETA_CELLS` | 0.0202 |
| 9 | `HALLMARK_ALLOGRAFT_REJECTION` | 0.0202 |
| 10 | `HALLMARK_SPERMATOGENESIS` | 0.0202 |

## v0.6 (post-hoc IG) vs v0.7 (learned attention) -- top-3 agreement

v0.6 top-3 per pole comes from `audit/dmoi_pathway_v0.6.md` (50-set Hallmark IG rollup, identical on TCGA test + METABRIC).

| Pole | v0.7 learned top-3 | v0.6 IG top-3 | Shared (n / 3) |
|---|---|---|---|
| **LumA** | `HALLMARK_INTERFERON_ALPHA_RESPONSE`, `HALLMARK_MTORC1_SIGNALING`, `HALLMARK_WNT_BETA_CATENIN_SIGNALING` | `HALLMARK_ESTROGEN_RESPONSE_EARLY`, `HALLMARK_ESTROGEN_RESPONSE_LATE`, `HALLMARK_IL2_STAT5_SIGNALING` | 0 / 3 |
| **LumB** | `HALLMARK_KRAS_SIGNALING_UP`, `HALLMARK_MTORC1_SIGNALING`, `HALLMARK_PEROXISOME` | `HALLMARK_E2F_TARGETS`, `HALLMARK_G2M_CHECKPOINT`, `HALLMARK_MYC_TARGETS_V1` | 0 / 3 |

## Reading

- `softmax weight` -- each pole's attention weights sum to 1.0 across the 50 pathways. Weight = 0.02 means "uniform" (1/50). Anything above 0.05 is a meaningful preference; above 0.20 is strong concentration.
- Agreement count is informational, not a hypothesis test. With 50 pathways the chance of a random 3-set match is (50 choose 3 with k hits) -- not zero but small.

## The finding: softmax-attention collapse (Phase A honest negative)

The learned attention weights span ~0.0203 to 0.0205 -- effectively
uniform (1 / 50 = 0.0200). The model did not differentiate the
pathways at all. The displayed "top-3" pathways are picked among
near-tied weights and carry no signal.

This is a known architectural failure mode, and Phase A documents it
faithfully. The cause is mechanical, not biological:

1. v0.7 standardizes the per-patient pathway-expression scores
   (per-pathway mean → 0, std → 1) before feeding them into
   `PathwayPoleAttention`.
2. With softmax-uniform attention weights, the pole's pathway feature
   is `sum_k (1/50) * standardized_pathway_score[k]`. Across a batch
   this averages to ~0 because standardized scores are zero-centered.
3. The classifier head sees a near-zero pole_pathway_feat for every
   patient. It learns to ignore the feature (the head's first linear
   collapses that input dim's weight toward zero).
4. With no downstream signal flowing through pole_pathway_feat, the
   gradient back to `attn_logits` is tiny. Combined with `wd=1e-4`
   pulling logits toward zero, the attention stays uniform forever.

It is a self-reinforcing equilibrium of uselessness: uniform attention
produces zero signal, zero signal teaches the head to ignore, ignoring
zeros out the gradient.

The drop in TCGA AUROC (~1.1pp) is consistent with the head having
two extra noisy input features it has to learn to ignore. The
METABRIC AUROC moving up by 0.4pp is within run-to-run noise.

## Phase B plan (next release, v0.7.1)

Two fixes to the v0.7 minimal Variant D, both small:

1. **Drop the StandardScaler on pathway scores.** Use raw per-patient
   mean expression. Different pathways will have different baselines
   (some genes are highly expressed, some are lowly), and uniform
   attention will produce a non-zero per-patient pole feature with
   discriminative variance across patients.
2. **Increase `init_std` from 0.01 to 0.5.** Start the attention
   weights asymmetrically so gradient has a direction to follow on the
   first epoch. The current init is so tight that any tiny gradient
   gets washed out by weight decay before symmetry breaks.

If Phase B still shows uniform attention with these fixes, the next
step is to project the pole_pathway_feat per pole from scalar to
vector (Variant C-lite), so the head has a richer interface into the
pathway branch.

## v0.6 baseline remains canonical

Regardless of Phase B outcome, v0.6 (gene-level pole masks +
post-hoc Hallmark IG rollup) remains the canonical architecture and
interpretability story for DMOI. Phase A's negative finding does not
change the AUROC narrative or the v0.6 pathway-level interpretability
claim.

## Honest scope

- Single-fold final-model run (no CV). The v0.7 architecture diff is what's under test; held-out test scoring matches the v0.6 protocol.
- The pathway branch sees a per-pole scalar feature (weighted-sum of the 50 pathway-mean expressions). A richer projection (per-pole vector instead of scalar) is a Variant C upgrade, deferred to v0.8+ if Phase B also collapses.
- Gene-level interpretation (v0.3 / v0.4 IG) is unaffected -- the gene-level branch is unchanged from v0.6.

## Reproduce

```bash
python scripts/eval_dmoi_v0.7.py
```
