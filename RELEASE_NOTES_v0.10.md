# DMOI v0.10 — Cross-cohort + cross-task generalization (doubly generalized)

## TL;DR

v0.9 showed the v0.6 framework transfers to a different classification
axis on the same cohort (TCGA Luminal-vs-Basal, AUROC 1.000, 8/8
priors hit). v0.10 composes that with cross-cohort generalization:
**train the v0.9 model on TCGA cohort_v3, score METABRIC cohort_v3**
(Luminal-vs-Basal, n=1,384 -- Luminal 1,175 + Basal 209, Illumina
HT-12 v3 microarray + RNA-only + meth silenced + quantile-normalized
to TCGA train per the v0.2 / v0.4 / v0.6 protocol).

**Result: METABRIC AUROC = 0.965, bacc 0.842.** Per-pole IG top-3 on
METABRIC is **identical** to TCGA cohort_v3: Luminal pole loads
`ER_EARLY` + `ER_LATE` + `ANDROGEN_RESPONSE`; Basal pole loads
`MYC_TARGETS_V1` + `G2M_CHECKPOINT` + `EPITHELIAL_MESENCHYMAL_TRANSITION`.
**8/8 expected priors in METABRIC top-5 AND 3/3 + 3/3 top-3 stable
between TCGA and METABRIC.** The v0.6 framework is simultaneously
task-invariant and cohort-invariant within DMOI scope -- doubly
generalized.

The v0.4 LumA-vs-LumB METABRIC reference was 0.909; v0.10
Luminal-vs-Basal METABRIC is 0.965 (+5.6 pp lift) with perfect prior
recovery.

## What changed in code from v0.9

Backward-compatible additions only.

- **`scripts/build_metabric_cohort_v3.py`** -- new METABRIC cohort
  builder. Parallel of v0.2's `build_metabric_cohort.py` but extended
  to include Basal (CLAUDIN_SUBTYPE = "Basal") alongside Luminal
  (LumA + LumB → "Luminal"). Output: `data/metabric/cohort_v3.tsv`
  with 1,384 patients (Luminal 1,175 + Basal 209), all with mRNA.
- **`scripts/eval_metabric_v0.10.py`** -- new driver. Loads TCGA
  cohort_v3 + METABRIC cohort_v3 + full Hallmark gmt; builds
  Luminal/Basal pole masks using the v0.9 `make_pole_masks(...,
  hallmark_sets=...)` override; trains the same v0.9 architecture on
  TCGA train (no changes); scores TCGA test for reference + METABRIC
  for the new cross-cohort metric (RNA-only with meth silenced and
  quantile-normalized to TCGA train RNA per the v0.2 / v0.4 / v0.6
  protocol); runs IG attribution on METABRIC for Luminal_pole /
  Basal_pole / final_logit; rolls up to the 50-set Hallmark catalog;
  performs an automated cross-pole biology sanity check; writes
  `audit/dmoi_v0.10.md`.

No model, training-loop, or architecture changes. v0.9 / v0.6
backward compatibility preserved.

## Four-axis framework reusability table

The v0.6 → v0.10 sequence now reads as a complete falsifiable
architectural inquiry with **all four axes of reusability**
empirically validated:

| Axis | Evidence | Where |
|---|---|---|
| Calibration transfer | Cohort-specific T_TCGA = 0.634 vs T_METABRIC = 0.934 | v0.1 / v0.2 |
| Cross-cohort same-task | LumA-vs-LumB METABRIC AUROC 0.909, Jaccard 0.667 gene-level | v0.4 |
| Cross-task same-cohort | Luminal-vs-Basal TCGA AUROC 1.000, 8/8 priors in top-5 | v0.9 |
| **Cross-cohort + cross-task** | **Luminal-vs-Basal METABRIC AUROC 0.965, 8/8 priors, 3/3 + 3/3 top-3 stable** | **v0.10 (this release)** |

The v0.7 + v0.8 three-variant architecture experiment showed
separately that adding a trainable pathway-attention branch on top
of this framework is structurally redundant (gene-level commitment
captures all the discriminative direction signal; learnable
pathway-level attention can only find magnitude variance regardless
of interface dimensionality).

## Cross-cohort pole-biology stability (TCGA vs METABRIC, same trained model)

| Pole | TCGA test top-3 | METABRIC top-3 | Shared (n / 3) |
|---|---|---|---|
| **Luminal** | `ESTROGEN_RESPONSE_EARLY`, `ESTROGEN_RESPONSE_LATE`, `ANDROGEN_RESPONSE` | `ESTROGEN_RESPONSE_EARLY`, `ESTROGEN_RESPONSE_LATE`, `ANDROGEN_RESPONSE` | **3 / 3** |
| **Basal**   | `MYC_TARGETS_V1`, `G2M_CHECKPOINT`, `EPITHELIAL_MESENCHYMAL_TRANSITION` | `MYC_TARGETS_V1`, `G2M_CHECKPOINT`, `EPITHELIAL_MESENCHYMAL_TRANSITION` | **3 / 3** |

The per-pole biology recovered by the v0.9 model is cohort-invariant.
METABRIC microarray RNA on the HT-12 v3 platform, after
quantile-normalization to TCGA HiSeq, gives the same per-pole IG
ranking as the source TCGA RNA-seq cohort.

## What this means for the portfolio narrative

DMOI v0.0 → v0.10 reads as: **found, tested, generalized across
cohort, generalized across task, generalized across both.** The
framework -- gene-level hypothesis-conditioned attention + hand-picked
pole priors + post-hoc Hallmark IG rollup -- is empirically the right
architectural commitment for multi-omics binary subtype classification
within DMOI scope, validated across:

- Calibration regime (cohort-specific T)
- Cohort (TCGA HiSeq vs METABRIC HT-12 v3 microarray)
- Task (LumA-vs-LumB within-luminal vs Luminal-vs-Basal cross-lineage)
- Architecture variants (3-variant pathway-attention experiment all
  confirming gene-level commitment)

This is a rare four-axis validation pattern in a real-world POC.

## Honest scope

- Same architecture as v0.9 / v0.6 (no model changes); only the
  external scoring cohort changes.
- METABRIC microarray RNA is on a different platform (Illumina HT-12
  v3) than TCGA's HiSeqV2. Quantile normalization is applied
  column-by-column to match the TCGA train RNA distribution -- the
  same protocol used in v0.2 / v0.4 / v0.6 for METABRIC LumA-vs-LumB.
- METABRIC has no HM450 methylation, so the methylation branch is
  silenced at inference time. Cross-cohort generalization is validated
  on the RNA branch only (consistent with v0.4 protocol).
- Class imbalance is 5.6 : 1 (Luminal majority); Basal n=209 in
  METABRIC. AUROC variance is wider than v0.9's TCGA test, but the
  3/3 + 3/3 top-3 stability and 8/8 priors-hit rate are the more
  informative metrics.
- The v0.4 LumA-vs-LumB METABRIC reference (AUROC 0.909) is the
  closest comparable cross-cohort number; the v0.10 0.965 is a
  +5.6 pp lift, but the easier intrinsic separability of cross-lineage
  vs within-luminal makes that lift expected -- the 8/8 priors-hit +
  3/3 top-3 stability is the decisive evidence.

## Reproduce

```bash
python scripts/build_cohort_v3.py             # TCGA cohort_v3 (if not built)
python scripts/build_metabric_cohort_v3.py    # METABRIC cohort_v3
python scripts/eval_metabric_v0.10.py         # ~10 min on MPS
# writes audit/dmoi_v0.10.md
```

## Audit

[`audit/dmoi_v0.10.md`](audit/dmoi_v0.10.md) for the full result
tables, cross-cohort biology stability table, four-axis closure
analysis, and the closing statement on framework reusability.
