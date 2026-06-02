# What is out of scope

This file is **required** in every repo created from the scaffold template.
The CI lint job verifies that this file exists; the PR template references
it as part of the review checklist.

## Why this file exists

A capability-portrait repo's value comes from being *small and complete*. The
single largest risk to that value is the steady accumulation of "while we're
here, let's also..." additions. This file is the anti-scope-creep ledger.
If a PR proposes something on this list, the PR template asks the contributor
to answer one question:

> Why is this still out of scope?

If the answer is good, edit this file in the same PR. If the answer is not
good, the PR doesn't land.

## Default out-of-scope items

(Copy and edit these into the derived repo's `what-is-out-of-scope.md`.)

- **Statistical-power claims**. The demo uses a tiny public subset; effect
  sizes and p-values are illustrative, not conclusive.
- **Full-cohort reproduction**. Adding samples beyond the manifest cap
  requires editing both `data/manifest.yaml` and the README's
  "minimum subset" claim.
- **Multi-cohort meta-analysis**. Out of scope unless this repo's capability
  *is* meta-analysis.
- **Production hardening** (HA, RBAC, multi-tenant). The substrate provides
  the foundation; the capability portrait does not re-implement it.
- **Cost optimization for cloud deployment**. The demo runs on a single
  workstation; cloud cost is by definition out of scope.

## Per-project out-of-scope items

This is `dmoi-brca-poc`'s own list. The capability on display is a
*hypothesis-conditioned multi-omics architecture* (DMOI) and the evidence that
its one architectural commitment is reusable across cohort, task, and split.
Anything that does not serve that single demonstration is out of scope.

### DMOI (`dmoi-brca-poc`)

- **Beating the LogReg AUROC ceiling**. *The flat-concat baseline already
  saturates the within-luminal signal (v0.1, Δ ≈ −0.002); the architecture's
  value is calibration, interpretability, and reusable inductive bias, not a
  higher headline number.*
- **Trainable / learnable pathway attention**. *Falsified across three
  variants in v0.7–v0.8; the pathway view stays post-hoc IG interpretation and
  gene-level commitment remains canonical. Reopening requires new evidence, not
  a new variant.*
- **Additional omics modalities** (CNV, RPPA/proteomics, miRNA, ATAC-seq).
  *The demo is intentionally a two-modality RNA + HM450-methylation portrait;
  more modalities is a different capability.*
- **Methylation branch on METABRIC**. *METABRIC has no HM450 data; the external
  test runs RNA-only with the meth branch silenced, and imputing methylation is
  out of scope.*
- **BRCA subtype axes beyond LumA-vs-LumB and Luminal-vs-Basal** (HER2-enriched,
  normal-like, PAM50 5-class). *Two axes already demonstrate task-reusability;
  the full PAM50 problem is not the point being made.*
- **Survival, outcome, or therapy-response modeling**. *This repo is a subtype
  *classification* portrait; time-to-event and treatment-response live in other
  capability portraits.*
- **Wet-lab or causal validation of the attributed genes/pathways**. *Integrated
  Gradients attributions are interpretive evidence that the model uses sensible
  biology, not mechanistic claims to be experimentally confirmed.*
- **Architecture/hyperparameter search (NAS, sweeps)**. *The v0.7–v0.8
  three-variant experiment already probed the architecture question directly;
  broad automated search is not in scope.*
- ~~**Cross-cohort calibration transfer**~~ — **in scope as of v0.13**
  (`scripts/calibrate_transfer.py`). Finding: the raw model is already
  calibrated on METABRIC and no temperature transfer beats leaving probabilities
  uncalibrated; base-rate correction is the useful cross-cohort adjustment. See
  `audit/dmoi_calibration_transfer_v0.13.md`.

## How to add an item

Open a PR that:

1. Adds the item to the appropriate section above.
2. Adds a one-sentence reason in italics.
3. Links to the PR or issue where the item was originally proposed.

That's it. The friction is intentional.
