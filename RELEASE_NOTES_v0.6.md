# DMOI v0.6 — Full Hallmark catalog rollup (50 sets, MSigDB v2024.1.Hs)

v0.5 rolled per-gene IG into the 5 Hallmark sets that were already
routed to the pole masks. That left an obvious objection: *did those
5 sets win because they were the only ones loaded?* v0.6 closes the
caveat by loading the full 50-set Hallmark v2024.1.Hs catalog and
re-running the same rollup on TCGA test + METABRIC.

## Eight-act narrative

1. **v0.0** — 5-fold CV on `cohort_v2` → AUROC 0.9682.
2. **v0.1** — Temperature scaling on a nested calibration split;
   under-confident (T<1) correction.
3. **v0.2** — METABRIC RNA-only external validation; cohort-specific T.
4. **v0.3** — Per-patient IG (Captum) on TCGA test; lumA = FOXC1/BCL2,
   lumB = RANBP1/NBN/ZW10/POLA2, ESR1/PGR correctly absent.
5. **v0.4 (cleanup)** — `train_one_fold(keep_artifacts=True)` returns
   model + scalers.
6. **v0.4 (METABRIC IG)** — Cross-cohort gene-level Jaccard top-10 =
   0.667 on both poles.
7. **v0.5** — Pathway-level IG rollup over the 5 Hallmark sets in
   `priors.py`: lumA loads ER ~300× harder than cell-cycle, lumB
   loads cell-cycle ~45× harder than ER, 3/3 top-3 shared
   cross-cohort.
8. **v0.6 (this release)** — Same IG rollup, but over the full 50-set
   MSigDB Hallmark catalog (v2024.1.Hs, CC-BY 4.0). On both TCGA
   test and METABRIC, **every v0.5 top pathway stays in the v0.6
   top-3 out of 50**. The 5-set finding wasn't an artifact.

## Headline finding

| Target | TCGA test top-3 of 50 | METABRIC top-3 of 50 | v0.5 top-pathway(s) survive? |
|---|---|---|---|
| `lumA_pole`   | `ESTROGEN_RESPONSE_EARLY`, `ESTROGEN_RESPONSE_LATE`, `IL2_STAT5_SIGNALING` | identical | **Yes** — both ER sets |
| `lumB_pole`   | `MYC_TARGETS_V1`, `E2F_TARGETS`, `G2M_CHECKPOINT` | `E2F_TARGETS`, `G2M_CHECKPOINT`, `MYC_TARGETS_V1` | **Yes** — same 3 cell-cycle sets, rank order swaps within a near-tie |
| `final_logit` | `ESTROGEN_RESPONSE_EARLY`, `ESTROGEN_RESPONSE_LATE`, `G2M_CHECKPOINT` | identical | **Yes** — identical top-3 cross-cohort |

3/3 v0.5 top-pathway groups survive the 50-set widening on both cohorts.

## Honest secondary findings (only visible with the full catalog)

- **`IL2_STAT5_SIGNALING` joins lumA top-3** on both cohorts (TCGA
  test mean |IG| 0.00096; METABRIC 0.00101). STAT5 is a known
  co-regulator of `ESR1` transcriptional activity in luminal breast
  cancer, so this reads as a biologically coherent secondary signal
  rather than noise. It's still ~6.6× below `ESTROGEN_RESPONSE_EARLY`
  on both cohorts.
- **`MYC_TARGETS_V2` (rank 4) and `MITOTIC_SPINDLE` (rank 5)** join the
  lumB top-5 on both cohorts. Both are additional proliferation
  programs; the model is loading the entire cell-cycle / growth axis,
  not just one pathway.
- **No surprise non-proliferation, non-ER pathway** appears in either
  pole's top-5 — the architectural prior catches the biology that's
  actually there.

## What changed in code

- **Data**: `data/msigdb/h.all.v2024.1.Hs.symbols.gmt` (CC-BY 4.0, ~48
  KB) + `data/msigdb/README.md` with provenance, license attribution,
  and reproduce-curl command. Allowlisted in `data/.gitignore`.
- **New module**: `src/dmoi_brca/hallmark.py`
  - `load_hallmark_gmt(path)` — tiny single-pass gmt parser
    (`set_name<TAB>description_url<TAB>gene1<TAB>gene2<TAB>...`)
    with dedup, blank-line skip, validation.
  - `summarize_hallmark(sets)` — `{name: gene_count}` for audit
    tables.
  - **Zero new dependencies.** No `gseapy`, no Enrichr API.
- **New driver**: `scripts/aggregate_pathway_ig_full.py`
  - Loads the full 50-set catalog via `load_hallmark_gmt`.
  - Trains the same Option A model (`keep_artifacts=True`) on TCGA
    train split, scores TCGA test (AUROC 0.9682) and METABRIC, runs
    IG for `final_logit`, `lumA_pole`, `lumB_pole`.
  - Aggregates over all 50 Hallmark sets via
    `dmoi_brca.pathway.pathway_aggregate`.
  - Writes `audit/dmoi_pathway_v0.6.md` (top-10 tables × 3 targets +
    automated "v0.5-finding survives" assertion) plus six per-(target,
    cohort) CSVs with all 50 rows ranked by mean |IG|.
- **Tests**: `tests/test_hallmark.py` (12 unit tests — 50-set count,
  canonical set names present, no duplicate genes within a set,
  gene-count bounds 30-200, minimal-fixture parsing, missing-file +
  short-line + empty-file errors, blank-line + dedup behavior).

No model-architecture changes. The v0.6 view is a wider post-hoc
aggregation of the same IG outputs.

## Honest scope

- 50 Hallmark sets loaded — the entire Hallmark v2024.1.Hs catalog.
  The C2 curated catalog (~5,000 sets) and other MSigDB collections
  remain out of scope.
- Aggregation is RNA-only. METABRIC has no methylation; even on TCGA
  the meth features are HM450 probes, not gene symbols. A Hallmark
  rollup of methylation IG would need a probe → gene crosswalk.
- The pathway scores are interpretation artifacts, not training
  signals. The model still attends to genes. Pathway-level *attention*
  (feeding pathway embeddings into the model) is the v0.7+ candidate.

## Reproduce

```bash
python scripts/aggregate_pathway_ig_full.py
# writes: audit/dmoi_pathway_v0.6.md + 6 CSVs in audit/
```

## Audit

See [`audit/dmoi_pathway_v0.6.md`](audit/dmoi_pathway_v0.6.md) for the
full per-target tables. Full 50-row per-(target, cohort) data lives in
`audit/dmoi_pathway_v0.6_{target}__{cohort}.csv`.

## License attribution

The MSigDB Hallmark gene sets bundled in `data/msigdb/` are
**Copyright (c) 2004-2025 Broad Institute, Inc., Massachusetts Institute
of Technology, and Regents of the University of California**, released
under the
[Creative Commons Attribution 4.0 International License (CC-BY 4.0)](http://creativecommons.org/licenses/by/4.0/).
Citation: Liberzon A, et al. *The Molecular Signatures Database (MSigDB)
hallmark gene set collection.* Cell Systems. 2015;1(6):417-425.
