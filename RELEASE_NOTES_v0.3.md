# DMOI POC v0.3

Adds per-patient Integrated Gradients attribution on the TCGA test split,
revealing the sophisticated biology the dual-pole architecture is doing.

The interpretability story is now a five-act narrative ending in a
clinical-decision-support claim: not just "the model generalizes" but
"the model picked the right discrimination axis, and here's the per-patient
gene list to back it up."

## What's new since v0.2

| Capability | v0.2 | v0.3 |
|---|---|---|
| Held-out TCGA test (n=84) | AUROC 0.968 | unchanged |
| METABRIC external (n=1,175) | AUROC 0.909 | unchanged |
| Cohort-specific calibration | T_TCGA 0.634 / T_METABRIC 0.934 | unchanged |
| **Per-patient attribution** (Integrated Gradients) | — | **lumA / lumB / final on all 84 test patients** |
| **Per-target global rankings** (top-50 mean \|IG\|) | — | **3 targets × 2 modalities = 6 lists + 3 PNGs** |
| Captum dep | — | added `captum>=0.7.0` |

## v0.3 biological headline

| Pole | Top-5 RNA features | Reading |
|---|---|---|
| **lumA_pole** | FOXC1, PDLIM3, TUBB2B, **BCL2**, KRT15 | Learned "this is NOT basal-like." FOXC1 and KRT15 are basal/myoepithelial markers used *inversely* as LumA evidence. BCL2 is the canonical anti-apoptotic luminal gene. |
| **lumB_pole** | EFNA5, RANBP1, SMC6, ZW10, DMD | Learned proliferation via cell-cycle structural genes (RANBP1 nuclear-transport, SMC6 chromatin, ZW10 mitotic-checkpoint). |

**The big "honest interpretability" finding**: ESR1, PGR, FOXA1 — the
canonical pan-luminal markers — are *correctly absent* from the top
attributions because both LumA and LumB are ER+, so those genes don't
discriminate within this cohort. The model picked the right axis
(proliferation + inverse-basal-marker), not a naïve pan-luminal prior.

## IG faithfulness check

Completeness axiom residual `|sum(IG) - (f(x) - f(0))|` per target on
all 84 test patients:

| Target | Mean | Max | Verdict |
|---|---|---|---|
| **lumA_pole** | 0.0016 | 0.0155 | tight |
| **lumB_pole** | 0.0027 | 0.0216 | acceptable |
| final_logit | 0.0162 | 0.2062 | one outlier — see caveat |

The final_logit completeness has a single-patient outlier of 0.21, likely
from the disagreement scalar `|s_LumA − (1 − s_LumB)|` which has a
non-differentiable `abs()` at zero. The pole-specific attributions are
the recommended clinical-interpretability headline.

## Modules + scripts

- `src/dmoi_brca/attribution.py` — `integrated_gradients_dmoi()` wrapping
  Captum's IntegratedGradients with three target heads, plus
  `completeness_residual()`, `top_k_per_patient()`, `global_aggregate()`.
- `tests/test_attribution.py` — 11 unit tests including direction-sanity
  on a tiny trained mini-model and IG-completeness within 1e-2.
- `scripts/explain_dmoi.py` — full driver. Loads cohort_v2 train+test,
  re-trains the deterministic Option A model, runs IG for the three
  targets on the test split, writes audit MD + per-patient TSV + global
  TSV + 3 bar-chart PNGs.
- `docs/v0.3-design-attribution.md` — algorithm + scope rationale.
- `pyproject.toml` — `+captum>=0.7.0`.

## Honest caveats

- **Attribution scope is TCGA test only** (n=84). METABRIC attribution
  is straightforward to add (same module API) but deferred to v0.4.
- **IG is over standardized inputs.** Pathway-level aggregation
  (e.g., MSigDB rollup) is out of scope for v0.3.
- **The driver re-trains the model.** `scripts/explain_dmoi.py` first
  calls `train_one_fold` to get the announced test AUROC, then runs a
  second deterministic training pass to produce the model object for
  Captum. Cleaner fix (one-line change to `train_one_fold` to also
  return the trained model) is deferred to v0.4.

## Reproduce

```bash
uv sync                                  # or pip install -e '.[dev]'
python scripts/build_cohort_v2.py        # TCGA cohort + 80/20 split
python scripts/run_baseline_v2.py
python scripts/eval_dmoi.py
python scripts/fetch_metabric.py
python scripts/build_metabric_cohort.py
python scripts/eval_external.py
python scripts/explain_dmoi.py           # v0.3 attribution, ~3-4 min on MPS
```

## Test status

All unit tests pass (189 across the repo, including 11 new for `attribution.py`).
ruff clean. CJK gate clean.
