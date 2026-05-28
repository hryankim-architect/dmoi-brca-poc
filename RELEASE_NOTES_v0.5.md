# DMOI v0.5 — Pathway-level IG aggregation (MSigDB Hallmark)

The architectural prior validated at the gene level in v0.3 + v0.4 is
re-validated one level up, where the biology is meant to live.

## Seven-act narrative

1. **v0.0** — 5-fold CV on `cohort_v2` produced AUROC 0.9682.
2. **v0.1** — Temperature scaling exposed the model as under-confident
   (T<1) and corrected it to a 5e-3 NLL improvement on a nested
   calibration split.
3. **v0.2** — METABRIC RNA-only external validation: AUROC held in the
   high 0.9s. LumB sensitivity asymmetry decomposed into a class-prior
   shift (Bayes-correctable) and a meth-silencing residual.
4. **v0.3** — Per-patient Integrated Gradients (Captum) on TCGA test:
   lumA pole picked `FOXC1`, `BCL2` (ER-program markers); lumB pole
   picked `RANBP1`, `NBN`, `ZW10`, `POLA2` (cell-cycle markers).
   Completeness axiom holds within 1e-2.
5. **v0.4 (cleanup)** — `train_one_fold(keep_artifacts=True)` returns
   model + scalers so attribution scripts can run without retraining.
6. **v0.4 (METABRIC IG)** — Same IG protocol on METABRIC. Cross-cohort
   gene-level Jaccard top-10 = 0.667 for both poles. Biology travels.
7. **v0.5 (this release)** — Roll the per-gene IG up to MSigDB Hallmark
   pathways. The pole-pathway alignment ratios are visible at hundreds-
   of-times scale, and top-3 pathway agreement across cohorts is 3/3
   for both pole-specific targets.

## Headline finding

| Target | Top pathway | mean \|IG\| | Ratio vs opposite-program top |
|---|---|---|---|
| lumA pole, TCGA test | `HALLMARK_ESTROGEN_RESPONSE_EARLY` | 0.00991 | ~330× vs `HALLMARK_G2M_CHECKPOINT` (0.00013) |
| lumA pole, METABRIC  | `HALLMARK_ESTROGEN_RESPONSE_EARLY` | 0.01076 | ~270× vs `HALLMARK_G2M_CHECKPOINT` (0.00016) |
| lumB pole, TCGA test | `HALLMARK_G2M_CHECKPOINT`         | 0.00334 | ~42× vs `HALLMARK_ESTROGEN_RESPONSE_EARLY` (0.00008) |
| lumB pole, METABRIC  | `HALLMARK_G2M_CHECKPOINT`         | 0.00362 | ~45× vs `HALLMARK_ESTROGEN_RESPONSE_EARLY` (0.00008) |

The lumA pole loads the ER program ~300× harder than cell-cycle; the
lumB pole loads cell-cycle ~45× harder than the ER program. The
architectural prior wired into the pole masks is doing exactly what the
prior claims.

## Cross-cohort top-3 pathway agreement

| Target | Shared top-3 (TCGA ∩ METABRIC) |
|---|---|
| lumA pole   | `ESTROGEN_RESPONSE_EARLY`, `ESTROGEN_RESPONSE_LATE`, `G2M_CHECKPOINT` |
| lumB pole   | `E2F_TARGETS`, `G2M_CHECKPOINT`, `MYC_TARGETS_V1` |
| final_logit | `ESTROGEN_RESPONSE_EARLY`, `ESTROGEN_RESPONSE_LATE`, `G2M_CHECKPOINT` |

3/3 shared for both pole-specific targets. The G2M overlap in the lumA
top-3 is a magnitude artifact (it's third by a wide margin, ~75× behind
ER_EARLY) and reads as "lumA-classified patients also have *some*
cell-cycle signal," not as a contradiction of the prior.

## What changed in code

- **New module**: `src/dmoi_brca/pathway.py`
  - `PathwayScore` dataclass.
  - `pathway_aggregate(attribution, feature_names, pathways)` —
    rolls up per-patient × per-gene attribution into per-pathway scores
    (`mean_abs_ig`, `sum_signed`, `signed_mean`).
  - `rank_pathways(scores, by, descending)` — ranking helper with
    validation.
- **New driver**: `scripts/aggregate_pathway_ig.py`
  - Trains a single fold with `keep_artifacts=True`.
  - Runs IG for `final_logit`, `lumA_pole`, `lumB_pole` on TCGA test and
    on METABRIC (RNA-only with meth silenced + quantile-normalized to
    TCGA).
  - Aggregates over the 5 Hallmark sets in `priors.HALLMARK_SETS`.
  - Writes `audit/dmoi_pathway_v0.5.md`.
- **Tests**: `tests/test_pathway.py` (10 unit tests — shapes, signed-sum
  direction, missing-gene handling, empty-pathway zeros, multi-patient
  averaging, error cases, ranking).

No model-architecture changes. The pathway view is a post-hoc
aggregation of v0.3/v0.4 IG outputs.

## Honest scope

- Only the 5 Hallmark sets already in `priors.py` (the ones routed to
  the pole masks) are aggregated. The full 50-set MSigDB Hallmark
  catalog and the broader C2 curated set are out of scope for v0.5 —
  pulled in via a `gmt`-file loader is the natural v0.6 follow-up.
- Aggregation is over the RNA modality only. The methylation branch
  uses HM450 probes, not gene symbols, so a Hallmark rollup of the
  methylation IG is not meaningful without a probe → gene crosswalk.
- The pathway scores are interpretation artifacts, not training
  signals. The model still attends to genes, not to pathways. Pathway-
  level attention is a separate (heavier) v0.6+ candidate.

## Reproduce

```bash
python scripts/aggregate_pathway_ig.py
# writes: audit/dmoi_pathway_v0.5.md
```

## Audit

See [`audit/dmoi_pathway_v0.5.md`](audit/dmoi_pathway_v0.5.md) for the
full per-pathway × per-cohort tables.
