# `dmoi-brca-poc` — In-Scope Promotion Plan (v0.13+)

*Drafted 2026-06-02. Scope: decide which currently out-of-scope items to pull
**into** scope next, in what order, and under what acceptance bar. This is a
planning doc, not a commitment — each promotion still goes through the
`what-is-out-of-scope.md` "why is this still out of scope?" gate.*

> **Status — updated 2026-06-02.** Priority 1 (**v0.13**) and Priority 2
> (**v0.14**) are **shipped** (committed + tagged). Results below. The only
> remaining roadmap item is Priority 3 (CNV third modality), now re-evaluated in
> §3 — recommendation: **treat v0.14 as the public-release stopping point and
> defer CNV to a spin-off**, not a v0.15 in this repo.

### Shipped this cycle

| Version | Item | Result | Verdict |
|---|---|---|---|
| **v0.13** | Cross-cohort calibration transfer | raw METABRIC ECE **0.0745** ≈ labelled oracle **0.0738**; naive TCGA-T worsens it (0.1051); no transfer (label-free align, prior-odds, METABRIC-mini) beats leaving probabilities raw | **null result, closed** — don't transfer temperature cross-cohort; base-rate correction is the useful lever (best Brier) |
| **v0.14** | HER2-vs-Luminal third task axis | TCGA 5-fold AUROC **0.892 ± 0.056** / METABRIC **0.893**; all 5 expected per-pole priors hit **5/5 folds** (ER vs PI3K-MTOR-G2M); HER2 top-3 Jaccard **1.0** | **transfers** — v0.6 architecture is task-reusable on a 3rd axis; biology fold-invariant, AUROC band wide by small-n design |

---

## 1. Selection criteria

An out-of-scope item is worth promoting only if it scores well on all four:

1. **Reinforces the thesis.** The repo's value is a *reusable
   hypothesis-conditioned architecture* plus **calibration** and
   **interpretability** — not a higher headline AUROC. A promotion should
   deepen one of those, not chase the number.
2. **Stays "small and complete."** Bounded effort, reproducible on one
   workstation in minutes, no sprawling new subsystem.
3. **Uses data already on hand.** TCGA-BRCA (RNA + HM450) and METABRIC
   (RNA-only) are already wired. New-modality downloads are a cost multiplier.
4. **Low risk to sealed results.** v0.6 (canonical architecture) and the
   v0.9–v0.12-A reusability/split-invariance seals must not be destabilized.

Data already available (no new download): the TCGA clinical matrix already
carries **ER/PR/HER2 status, PAM50 subtype, and OS/PFS**, and METABRIC RNA is
already fetched. That makes a new task axis and calibration work cheap; it does
**not** make new omics modalities cheap.

---

## 2. Verdict at a glance

| # | Out-of-scope item | Verdict | Target |
|---|---|---|---|
| 9 | Cross-cohort calibration transfer | ✅ **Shipped** (null result) | v0.13 ✓ |
| 5 | New subtype/task axis (HER2-enriched vs rest) | ✅ **Shipped** (transfers) | v0.14 ✓ |
| 3 | Additional omics modality (CNV branch) | **Defer → spin-off** (re-evaluated, §3) | not v0.15 |
| 1 | Beat the LogReg AUROC ceiling | Keep out | — |
| 2 | Trainable pathway attention | Keep out (falsified) | — |
| 4 | Methylation branch on METABRIC | Keep out (no HM450 data) | — |
| 6 | Survival / therapy-response modeling | Spin-off, not here | — |
| 7 | Wet-lab / causal validation | Keep out (not computational) | — |
| 8 | Architecture / hyperparameter search | Keep out (low value) | — |

---

## 3. What to start — prioritized

### Priority 1 — Cross-cohort calibration transfer  → **v0.13**

**Why first.** Calibration was the v0.1 win and cross-cohort generalization was
the v0.2/v0.4 win; this item sits exactly at their intersection and is currently
documented as *deliberately left open*. v0.2 already showed naive transfer
fails (TCGA `T=0.634` over-sharpens meth-silenced METABRIC; METABRIC's own
`T=0.934` is correct). Closing that loop honestly is the single highest-value,
lowest-data-cost next step.

**Concrete work**
- Add a `scripts/calibrate_transfer.py` that compares, on METABRIC:
  (a) no calibration, (b) naive TCGA-fit `T`, (c) METABRIC cal-split-fit `T`
  (oracle reference), (d) **a transfer method** — start with a small
  cohort-shift correction (e.g. re-fit `T` on a tiny METABRIC calibration slice,
  or a logit-distribution alignment between cohorts).
- Report ECE / Brier / reliability-curve for each on the same held-out scores.
- Honest framing: if a transferable calibrator only partially closes the gap,
  *that result is the deliverable* (consistent with the repo's "honest scope"
  voice).

**Data / entry points.** Reuses `eval_external.py`, `train_dmoi.py`'s temperature
step, existing METABRIC cohort builders. No new download.

**Acceptance**
- Reliability table across the 4 conditions, on TCGA test and METABRIC.
- Hash-chained NDJSON audit + MLflow run logged; canary still green.
- New `tests/test_calibration_transfer.py`.
- README "four axes" table gains a calibration-transfer row;
  `what-is-out-of-scope.md` updated (item moved out of the out-of-scope list
  with a one-line rationale).

**Effort.** Small–medium (≈1–2 focused sessions).

---

### Priority 2 — A third task axis: HER2-enriched vs rest  → **v0.14**

**Why second.** v0.9–v0.12-A sealed task-reusability on two axes (LumA-vs-LumB,
Luminal-vs-Basal). A *third orthogonal axis* reusing the canonical v0.6
architecture with only new pole-defining Hallmark sets is the cleanest way to
strengthen the reusability claim without touching the architecture.

**Concrete work**
- New cohort builder (`build_cohort_v4.py`) selecting HER2-enriched (PAM50 =
  Her2 and/or HER2+ by clinical status) vs a comparison pole, from the existing
  clinical matrix.
- Define pole priors around the ERBB2/HER2 biology (e.g. an ERBB2-amplicon /
  ER-response contrast) via `build_priors.py`.
- Run the **unchanged** v0.6 architecture; report AUROC + per-pole IG top-k and
  whether expected priors land, mirroring the v0.9/v0.11 reporting.

**Watch-outs**
- HER2-enriched is a **smaller** class → keep the "statistical-power claims"
  item firmly out of scope; frame strictly as *reusability demonstration*, not
  an effect-size result. State n explicitly.
- If the axis is trivially separable (like Luminal-vs-Basal hit AUROC 1.000),
  report it plainly — separability is a property of the data, not a win.

**Acceptance**
- AUROC + per-pole IG table on TCGA; cross-cohort to METABRIC if class sizes
  permit (RNA-only, meth silenced — same pattern as v0.10).
- Audit + MLflow + canary; new `tests/test_cohort_v4.py`.
- README table + `what-is-out-of-scope.md` amended (axis list updated).

**Effort.** Medium (≈2 sessions; mostly cohort + priors definition).

---

### Priority 3 (RE-EVALUATED 2026-06-02) — CNV branch (3rd modality) → **defer to spin-off, not v0.15**

The decision rule was: promote only if, after v0.13–v0.14, the portfolio still
needs a ">2 modality reusability" proof point *and* a reviewer would value it
**in this repo** rather than a new one. Applying it now:

- **The reusability story is already complete.** v0.14 added a third *task* axis
  on top of the v0.9–v0.12-A cohort/task/split seals. The architecture's
  reusability is demonstrated across cohort, task, and split — a reviewer asking
  "does this generalize?" is already answered. CNV would add a *modality* axis,
  which is a different (and bigger) claim.
- **It violates "small and complete."** CNV requires a new TCGA dataset in
  `data/manifest.yaml`, a third hypothesis-conditioned branch + mask, and new
  cross-platform handling — the largest single addition on the list, on a repo
  that is otherwise at a clean stopping point.
- **METABRIC can't follow.** The cross-cohort check that gives v0.10/v0.14 their
  weight needs the external cohort to carry the modality; METABRIC has no
  matched CNV at the same processing, so a CNV branch would be TCGA-only — a
  weaker design than what v0.13/v0.14 set as the bar.

**Decision: do NOT do CNV as v0.15 in this repo.** If the multi-modality claim
is wanted later, it belongs in a **new capability repo** (its own scope note,
its own data layer), not bolted onto `dmoi-brca-poc`. Leave the item in
`what-is-out-of-scope.md` (it already is) with a one-line "spin-off, not here"
rationale.

---

## 4. Keep firmly out (do not start)

- **Beat the LogReg ceiling (1)** — contradicts the thesis; the baseline
  saturating the signal is itself a documented finding.
- **Trainable pathway attention (2)** — falsified across three variants
  (v0.7–v0.8). Reopen only with genuinely new evidence, never another variant.
- **Methylation on METABRIC (4)** — METABRIC has no HM450; impossible without
  imputation, which is out of scope.
- **Survival / therapy-response (6)** — different capability; OS/PFS lives more
  naturally with `tp53-aml-hrd-severity` or a new repo, not here.
- **Wet-lab / causal validation (7)** — not a computational deliverable.
- **NAS / hyperparameter sweeps (8)** — the architecture question was already
  probed directly; broad search is low-signal.

---

## 5. Sequencing & milestones

```
v0.13  Cross-cohort calibration transfer      (Priority 1)  ── ✅ shipped (null result, closed)
   │
v0.14  HER2-vs-Luminal task axis + 5-fold CV   (Priority 2)  ── ✅ shipped (transfers; biology fold-invariant)
   │
SHIP   Public release of dmoi-brca-poc          ── ◀ recommended next: v0.14 is the stopping point
   ┊
(spin-off)  CNV / multi-modality reusability    (Priority 3)  ── NOT v0.15; a new repo if ever
```

**Recommendation: flip the repo public now.** The capability story is complete
— architecture + calibration (incl. the v0.13 cross-cohort calibration finding)
+ interpretability + reusability across cohort, task (now 3 axes), and split.
CNV is the only remaining roadmap item and it has been re-scoped to a spin-off.
The only pre-public chores left are housekeeping, not new experiments:

- Confirm `dmoi-brca-poc` is flipped from Private → Public (the manual GitHub
  step; was "private during dev").
- Optionally fill the `what-is-out-of-scope.md` items that are still template
  defaults (already DMOI-specific as of the v0.13/v0.14 edits).
- Push the v0.13 / v0.14 commits + tags and cut their GitHub Releases.

---

## 6. Definition of Done (every promotion)

Each promoted item is not "done" until all of the following hold:

1. Reproducible entry point (`python scripts/...`) runs on a workstation in
   minutes and is documented in `scripts/README.md`.
2. **Hash-chained NDJSON audit** entries emitted; **MLflow** run tracked;
   **canary** smoke test green.
3. Tests added; `ci.yml` and `english-only` CI both pass.
4. README updated (the relevant results table + reproducibility note).
5. `docs/what-is-out-of-scope.md` edited in the *same change* — the item is
   removed from the out-of-scope list with a one-line "now in scope as of vX"
   note (the file's own rule).
6. Release notes captured via the GitHub Release (the repo's convention — no
   `RELEASE_NOTES_*.md` checked in).

---

## 7. Risks

- ~~**Calibration-transfer underwhelms.**~~ **Resolved (v0.13):** it was a null
  result (raw already calibrated; no transfer helps) — reported honestly and
  closed the open question, exactly the acceptable outcome anticipated here.
- ~~**HER2 axis too small / too easy.**~~ **Resolved (v0.14):** small but
  handled — framed as reusability, n stated, and backed by a 5-fold band
  (0.892 ± 0.056) plus the METABRIC n=224 external. Biology was fold-invariant
  (5/5 priors), not trivially separable.
- ~~**Scope creep via CNV.**~~ **Resolved:** re-evaluated to a spin-off (§3); no
  CNV work enters this repo.
- **Destabilizing sealed results.** New axes/branches are additive; v0.6 stays
  canonical and the v0.9–v0.12-A seals are not re-run or altered.
