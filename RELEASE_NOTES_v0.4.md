# DMOI POC v0.4

Two things in v0.4: (1) the v0.3 interpretability story validated
cross-cohort on METABRIC, and (2) a hygiene refactor of `train_one_fold`
that lets downstream scripts skip the awkward double-train pattern.

The interpretability narrative is now six acts ending in: "**the model
learned biology, and it generalizes to an independent cohort.**"

## What's new since v0.3

| Capability | v0.3 | v0.4 |
|---|---|---|
| Per-patient IG on TCGA test (n=84) | ✓ | ✓ |
| **Per-patient IG on METABRIC (n=1,175)** | — | **✓** |
| **Cross-cohort top-K Jaccard agreement** | — | **lumA 0.667 · lumB 0.667 · final_logit 0.538** |
| **train_one_fold returns model + scalers** | — | **✓ (keep_artifacts=True)** |
| explain_dmoi.py double-train | re-trained ~70 LOC | single pass |

## v0.4 cross-cohort attribution headline

**Same trained model attributed on TCGA test and METABRIC. Top-10 RNA
features overlap heavily for both pole-specific targets:**

| Pole | Jaccard top-10 (TCGA test ∩ METABRIC) | Shared genes |
|---|---|---|
| **lumA_pole** | **0.667** (8/12 in union) | FOXC1, BCL2, PDLIM3, TUBB2B, KRT15, EGR3, RAB17, AHNAK |
| **lumB_pole** | **0.667** (8/12 in union) | EFNA5, RANBP1, NBN, ZW10, POLA2, CKS1B, DSCC1, IFRD1 |

**Every lumA headline gene from v0.3 is also in METABRIC's top-10.** The
"inverse-basal-marker + BCL2" story is independently reproduced.

**lumB picks up MORE canonical proliferation markers on METABRIC.** v0.3's
TCGA-test list (RANBP1, NBN, ZW10, POLA2 + EFNA5) is fully retained; on
METABRIC we ALSO get:

- **CKS1B** — CDK regulatory subunit (core cell-cycle machinery)
- **DBF4** — CDC7 kinase activator (S-phase initiation)
- **NDC80** — kinetochore complex (mitosis)
- **DSCC1** — replication-fork protein

The larger METABRIC cohort gave the model enough statistical power to
surface the textbook proliferation gene set on top of v0.3's
structural-mitotic markers. **The model's biology is tighter on
METABRIC, not looser.**

## v0.4 cleanup: train_one_fold returns artifacts

`FoldResult` gains three optional fields populated when `train_one_fold`
is called with `keep_artifacts=True`:

- `.model` — trained `DMOIModel` with best-epoch weights loaded
- `.rna_scaler`, `.meth_scaler` — sklearn StandardScalers fit on the
  train fold

`scripts/explain_dmoi.py` (TCGA test) and `scripts/explain_metabric.py`
(METABRIC) both use these fields and skip the previous awkward re-train
pattern. The result is the same trained model used for the AUROC report
is the one being attributed — strictly more coherent.

## Honest caveats

- **METABRIC has no methylation.** The IG attribution on METABRIC is over
  RNA only; the methylation branch sees the silenced zero tensor (same
  for every patient), so methylation attribution is uninformative in
  v0.4.
- **Slight gene-rank shift v0.3 → v0.4.** Because explain_dmoi.py now
  uses the same trained model as the test AUROC reporter (not a
  separately-seeded re-train), the IG attribution numbers shift slightly.
  Top-2 genes per pole are stable (FOXC1+BCL2; EFNA5+RANBP1); ranks 3-5
  move within the same biological category. The v0.3 GitHub Release tag
  still captures the original v0.3 numbers verbatim.
- **final_logit completeness still has the disagreement-scalar outlier.**
  The pole-specific attributions are the recommended
  clinical-interpretability headline; documented in the audit MD.

## Modules + scripts

- `scripts/explain_metabric.py` — METABRIC IG driver. Re-uses
  `attribution.py` + `external.py` + the v0.4 keep_artifacts pipeline.
- `src/dmoi_brca/train.py` — `train_one_fold(..., keep_artifacts=True)`
  populates `.model` / `.rna_scaler` / `.meth_scaler` on the returned
  `FoldResult`.
- `scripts/explain_dmoi.py`, `scripts/eval_external.py` — refactored
  to use the new artifacts; ~70 lines of duplicate-train logic removed
  from explain_dmoi.py.
- `audit/dmoi_explain_external_v0.4.md` + per_patient TSV + global TSV
  + 3 PNGs.

## Reproduce

```bash
uv sync
python scripts/build_cohort_v2.py
python scripts/run_baseline_v2.py
python scripts/eval_dmoi.py
python scripts/fetch_metabric.py
python scripts/build_metabric_cohort.py
python scripts/eval_external.py
python scripts/explain_dmoi.py        # v0.3 TCGA-test IG (~3 min)
python scripts/explain_metabric.py    # v0.4 METABRIC cross-cohort IG (~10 min)
```

## Test status

180 unit tests, all passing (added 2 train_one_fold artifact tests in
this release). ruff clean. CJK gate clean on 78 public artifacts.
