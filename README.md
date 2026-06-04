# `dmoi-brca-poc`

![ci](https://github.com/hryankim-architect/dmoi-brca-poc/actions/workflows/ci.yml/badge.svg)

> **One principle, applied here.** Pick the smallest, most interpretable representation that could carry the signal; measure it against an honest baseline; report the verdict faithfully — whether the compact choice wins, ties, or loses. *That last step is why AI safety is needed: knowing a capability is real rather than a flattering benchmark.*
>
> In this repo: **representation** pole-feature views + a 1-scalar disagreement signal, with *parameter-free* attention masks → **baseline** a *trainable* pathway-attention variant → **verdict** the fancy version was *falsified*: the parameter-free design held across cohort, task, and split (richer interface didn't help).

This repo uses TCGA-BRCA RNA-seq + HM450 methylation data, intentionally subsetted to keep evaluation fast and outputs fully reproducible. METABRIC (RNA-only, n=1,175–1,399) serves as the external validation cohort throughout.

**What this shows.** A hypothesis-conditioned multi-omics architecture
(Dialectical Multi-Omics Integration, DMOI) on TCGA-BRCA. Two "pole" perspectives
(LumA-like and LumB-like) each see only their pole-relevant RNA + DNA-methylation
features via Hallmark-gene priors and HM450 cis-mapping. Their predictions are
fused with a disagreement signal exposed to the classifier head. Temperature
scaling on a held-out calibration split delivers well-calibrated
probabilities. **v0.2 adds two external generalization tests**, a held-out
TCGA test split and an independent cohort (METABRIC, RNA-only), and a clean
cohort-specific calibration analysis.

## TL;DR: four axes of reusability, all split-invariant

The v0.6 → v0.14 sequence tests one architectural commitment across multiple
orthogonal axes and an adversarial alternative architecture — including a third
task axis (HER2-vs-Luminal) added in v0.14. Every axis holds,
and the two task axes are sealed as split-invariant under 5-fold CV.

| Axis of reusability | Headline result | Version |
|---|---|---|
| Internal same-task (LumA-vs-LumB, TCGA) | val AUROC **0.9702 ± 0.0122** (5-fold) | v0.6 → v0.12-A |
| Cross-cohort same-task (→ METABRIC) | AUROC **0.9254 ± 0.0052** (5-fold, std = 0.5 pp) | v0.4 → v0.12-A |
| Cross-task same-cohort (Luminal-vs-Basal, TCGA) | AUROC **1.0000 ± 0.0000** (5-fold) | v0.9 → v0.11 |
| Cross-cohort + cross-task (→ METABRIC) | AUROC **0.965**, 8/8 expected priors hit | v0.10 |
| Cross-task #2 + cross-cohort (HER2-vs-Luminal) | TCGA 5-fold AUROC **0.892 ± 0.056** / METABRIC **0.893**; 5/5 expected priors fold-invariant (ER vs PI3K-MTOR-G2M) | v0.14 |
| Calibration transfer | within-cohort ECE **0.138 → 0.077**; cross-cohort: raw METABRIC ECE **0.074** ≈ labelled oracle (no T-transfer beats leaving probabilities raw) | v0.1 / v0.2 / v0.13 |
| Adversarial check | 3-variant experiment **falsified** trainable pathway-attention | v0.7 / v0.8 |

The progression is one arc: a finding, then systematic testing, then a falsified alternative, then generalization across cohort, task, and both at once, and finally confirmation that it holds under split perturbation on every measurable axis. The dense per-version tables and analysis follow below.

**Reproducibility.** `python scripts/eval_dmoi.py` reproduces the full TCGA
evaluation in about 2 minutes on an M-series Mac.
`python scripts/eval_external.py` adds the METABRIC external test
in another ~3 minutes (after a one-time ~690 MB METABRIC download).

**Substrate.** Emits NDJSON audit entries (each entry chained to the previous by hash), tracks MLflow runs, exposes a canary smoke test for `lab_semantic_check.py`.

**Industry context.** A version of this method ran at full cohort scale on proprietary multi-omics data. This lab implementation demonstrates the method and the engineering pipeline, not those production-scale results. See [`docs/what-is-out-of-scope.md`](docs/what-is-out-of-scope.md).

---

## v0.12-A headline result (cross-cohort split-invariance seal on v0.4 / v0.6)

| Metric | DMOI v0.8 |
|---|---|
| 5-fold CV AUROC (TCGA train split, n=333) | 0.954 ± 0.017 |
| **v0.6 reference: TCGA held-out test AUROC** | **0.968** |
| v0.7 Phase A (collapse) TCGA AUROC | 0.957 |
| v0.7.1 Phase B (scalar) TCGA AUROC | 0.960 |
| **v0.8 Variant C (vector, proj_dim=16) TCGA AUROC** | **0.954** |
| **v0.6 reference: METABRIC external AUROC** | **0.909** |
| v0.7 Phase A METABRIC | 0.913 |
| v0.7.1 Phase B METABRIC | 0.898 |
| **v0.8 Variant C METABRIC** | **0.920** (within noise of v0.6) |
| Per-patient IG attribution (v0.3) | lumA / lumB / final logit on TCGA test (n=84) |
| Cross-cohort attribution agreement (v0.4) | Jaccard top-10 = 0.667 lumA + 0.667 lumB on METABRIC vs TCGA test |
| Pathway-level aggregation, 5 sets (v0.5) | lumA loads ESTROGEN_RESPONSE ~300× harder than cell-cycle; lumB loads cell-cycle ~45× harder than ER |
| Full Hallmark catalog rollup, 50 sets (v0.6) | All v0.5 top pathways stay in the top-3 out of 50, on both cohorts |
| Learnable pathway attention v0.7 Phase A | Softmax collapse under standardized inputs (uniform weights, no learning) |
| Learnable pathway attention v0.7.1 Phase B | Collapse fixed, scalar pole feature converged on magnitude-driven basin (WNT/INTERFERON/MTORC1 LumA; KRAS/MTORC1/PEROXISOME LumB) instead of v0.6 IG-derived ER/cell-cycle basin |
| **Learnable pathway attention v0.8 Variant C** | **Vector pole feature (proj_dim=16, 17× more pathway-branch parameters than v0.7.1) converged to the SAME wrong basin as v0.7.1. Top-5 weights identical within sub-percentage-point precision on both poles. Interface dimensionality is not the bottleneck, the gene-level branch fully captures discriminative direction, so the pathway branch can only find magnitude variance.** |
| **Closing conclusion (v0.7+v0.8)** | **3 variants confirmed: gene-level commitment is the right architectural level; pathway view should remain post-hoc interpretation (v0.5/v0.6 IG rollup). v0.6 remains canonical.** |
| **Cross-task generalization (v0.9)** | **Same v0.6 architecture transferred to a new classification axis, TCGA cohort_v3 Luminal (LumA+LumB) vs Basal, with the only changes being the cohort file, the pole-defining Hallmark sets, and the class-positive label. TCGA test AUROC = 1.000 (bacc 0.972). Luminal pole top-3 IG = ER_EARLY + ER_LATE + ANDROGEN_RESPONSE (3/3 hand-picked priors). Basal pole top-5 IG = MYC_V1 + G2M + EMT + E2F + MYC_V2 (5/5 priors). 8 / 8 expected pathways in top-5 with zero architecture changes, framework reusability empirically confirmed.** |
| **Cross-cohort + cross-task generalization (v0.10)** | **Same v0.9 trained model scored on METABRIC Luminal-vs-Basal n=1,384 (Luminal 1,175 + Basal 209) RNA-only with meth silenced + QN to TCGA train: METABRIC AUROC = 0.965 (bacc 0.842). Same per-pole IG top-3 as TCGA cohort_v3 (Luminal: ER_EARLY + ER_LATE + ANDROGEN; Basal: MYC_V1 + G2M + EMT). 8/8 expected pathways in top-5 AND 3/3 + 3/3 top-3 stable between TCGA and METABRIC. Doubly generalized: cohort-invariant AND task-invariant.** |
| **5-fold CV stability seal (v0.11)** | **Same v0.9 cohort_v3 + architecture, 5-fold StratifiedKFold (random_state=42): AUROC mean ± std = 1.0000 ± 0.0000 (every fold AUROC = 1.000), bacc 0.9724 ± 0.0091. Per-fold IG top-5: 3/3 Luminal priors hit 5/5 folds, 4/5 Basal priors hit 5/5 folds, 5th (MYC_TARGETS_V2) hits 4/5. Top-3 pairwise Jaccard = 1.0000 on both poles (every fold picked the same top-3). v0.9 / v0.10 finding is split-invariant, not a lucky single 80/20 split.** |
| **Cross-cohort split-invariance seal (v0.12-A)** | **TCGA cohort_v2 5-fold StratifiedKFold (random_state=42) × full-METABRIC score per fold (re-fit QN per fold, meth silenced). TCGA val AUROC = 0.9702 ± 0.0122 (v0.6 5-fold ref: 0.954 ± 0.017, tighter band, +1.6 pp). METABRIC AUROC = 0.9254 ± 0.0052 (v0.4 single-shot ref: 0.909, +1.6 pp with std = 0.5 pp). Per-fold IG on METABRIC: 2/2 LumA priors hit 5/5 folds; 3/3 LumB priors hit 5/5 folds; LumB top-3 pairwise Jaccard = 1.0000 (LumA Jaccard = 0.7, structurally bounded by only 2 expected priors). v0.4 / v0.6 cross-cohort finding is split-invariant, the cross-cohort metric is essentially deterministic under split perturbation.** |

**The takeaway:**

1. **Baseline saturated the easy signal.** Plain LogReg on
   concat(RNA, methylation) lands at 0.963 AUROC on the 417-patient
   LumA vs LumB cohort. Within-luminal class structure is already strongly
   expressed in the bulk transcriptome; there isn't much room for an
   architectural upgrade to beat it.

2. **Hypothesis-conditioned attention did NOT lift AUROC.** DMOI's primary
   metric ties baseline within noise (Δ ≈ −0.002 on the full v0.1 cohort).
   The dual-perspective architecture's value is **not** in the headline
   number, the LogReg ceiling reflects the structure of the data, not a
   model limitation we can engineer around.

3. **Calibration was the v0.1 win.** Temperature scaling on a held-out
   15% nested calibration split cuts ECE roughly in half on TCGA (0.138 →
   0.077). T < 1, the architecture is *under*-confident, not over-confident.

4. **External generalization was the v0.2 win.** A truly held-out TCGA test
   split (random_state=2024, never seen during model dev) scores AUROC 0.968.
   An independent cohort (METABRIC, n=1,175) with the methylation branch
   silenced (METABRIC has no HM450 data) scores AUROC 0.909. Calibration
   parameters do NOT transfer naively between cohorts: TCGA's T=0.634
   over-sharpens the meth-silenced METABRIC predictions; METABRIC's own
   cal-split-fit T=0.934 is correctly close to 1.0.

   **v0.13 closes this loop.** On a fixed METABRIC eval slice the raw model is
   already calibrated (ECE 0.074, essentially equal to the labelled-oracle
   T's 0.074), and no transfer strategy — naive TCGA-T, label-free logit
   alignment, class-prior odds, or a small labelled METABRIC slice — beats
   leaving the probabilities uncalibrated. The cross-cohort policy is therefore
   to apply *no* temperature; what does help cross-cohort is base-rate
   (prior-odds) correction, which lowers Brier and lifts LumB sensitivity. See
   `audit/dmoi_calibration_transfer_v0.13.md`.

5. **Interpretability was the v0.3 win.** Per-patient Integrated Gradients
   attribution on the TCGA test set reveals the architecture is doing
   sophisticated biology: the LumA pole learned **inverse-basal-marker**
   discrimination (FOXC1, KRT15 used as "this is NOT basal") plus the
   canonical anti-apoptotic luminal gene BCL2; the LumB pole learned
   cell-cycle structural genes (RANBP1, NBN, ZW10, POLA2). Canonical
   pan-luminal markers ESR1/PGR are correctly absent from the top
   attributions because they don't discriminate within the ER+ cohort.

6. **Cross-cohort interpretability was the v0.4 win.** The same IG pipeline
   on METABRIC (n=1,175) shows the lumA and lumB pole biology generalizes
   across cohorts (Jaccard top-10 = 0.667 for both poles vs TCGA test).
   **Every lumA headline gene from v0.3 (FOXC1, BCL2, PDLIM3, TUBB2B,
   KRT15) is also top-10 on METABRIC.** lumB picks up MORE canonical
   proliferation markers on the larger METABRIC cohort (CKS1B, DBF4,
   NDC80, DSCC1 added to v0.3's RANBP1, NBN, ZW10, POLA2).

7. **Pathway-level interpretability is the v0.5 win.** Rolling per-gene
   IG up to the MSigDB Hallmark pathway sets that defined the model's
   architectural priors: the lumA pole loads
   `ESTROGEN_RESPONSE_EARLY/LATE` **~300× harder** than the cell-cycle
   pathways. The lumB pole loads `E2F_TARGETS / G2M_CHECKPOINT /
   MYC_TARGETS_V1` **~45× harder** than ER pathways. Both findings hold
   identically on TCGA test AND METABRIC. **The architectural
   hypothesis-conditioning prior was the right inductive bias and the
   model learned to use it.**

8. **The pathway-level finding survives the catalog-widening sanity
   check, v0.6.** v0.5 rolled IG into the same 5 Hallmark sets that
   were already routed to the pole masks, which raises the obvious
   objection: *did the model only score those pathways highly because
   they were the only ones loaded?* v0.6 answers by loading the full
   **50-set** MSigDB Hallmark v2024.1.Hs catalog (CC-BY 4.0,
   checked into `data/msigdb/`) and re-running the rollup. On both
   TCGA test and METABRIC, **every v0.5 top pathway stays in the
   v0.6 top-3 out of 50**: lumA pole is still ER_EARLY + ER_LATE
   (with `IL2_STAT5_SIGNALING` joining as a known ER co-regulator);
   lumB pole is still `E2F_TARGETS / G2M_CHECKPOINT /
   MYC_TARGETS_V1` (with `MYC_TARGETS_V2` and `MITOTIC_SPINDLE`
   appearing immediately below as additional proliferation programs).
   The 5-set v0.5 finding wasn't an artifact of which sets were
   loaded.

9. **Learnable pathway-pole attention, v0.7.1 two-phase documented
   negative.** v0.7 attempted to replace the hand-picked v0.6 pole
   masks with a `softmax`-normalized learnable attention over all 50
   Hallmark pathways (Variant D from
   [`docs/v0.7-design-pathway-attention.md`](docs/v0.7-design-pathway-attention.md)).
   Two phases:
   * **Phase A** (standardized inputs + tight init): attention
     collapsed to uniform, softmax over zero-centered input plus
     weight decay = self-reinforcing equilibrium of uselessness.
   * **Phase B** (raw inputs + warm init): collapse fixed, but the
     model learned a different basin than v0.6's IG-derived ranking.
     0/3 v0.6 top-3 overlap on both poles; AUROC dropped.
     Diagnosis: scalar `pole_pathway_feat = sum_k w_k ×
     mean_expression_k` only captures pathway-magnitude variance,
     not pathway-direction signal.

   Both phases are recorded in [`audit/dmoi_v0.7.md`](audit/dmoi_v0.7.md).

10. **Variant C confirms gene-level commitment is correct, v0.8
    closure.** v0.8 upgraded the per-pole pathway feature from
    scalar to a 16-dim vector via a learnable
    `Linear(n_pathways, 16)` per pole, giving the classifier head
    32 pathway features (17× more pathway-branch parameters than
    v0.7.1's 2) and a fundamentally richer interface that *should*
    let the model read per-pathway direction signals. Result: the
    same wrong basin. LumA top-5 weights are identical to v0.7.1
    within sub-percentage-point precision (WNT 0.0689 → 0.0685;
    INTERFERON 0.0591 → 0.0594; MTORC1 0.0383 → 0.0384). 0/3 v0.6
    top-3 overlap. AUROC on TCGA dropped further (0.954 vs v0.6's
    0.968); METABRIC moved within noise.

    The 3-variant matched-basin convergence is information-theoretic
    evidence that **gene-level commitment is the right architectural
    level for LumA-vs-LumB**. The gene-level encoder sees ESR1, PGR,
    FOXA1, and the cell-cycle structural genes and resolves the
    decision there. By the time gradient reaches the pathway branch,
    the only signal left to exploit is pathway-*magnitude* variance,
    which the head can grip whether it has 2 features or 32. The
    pathway view's correct architectural role is the v0.5/v0.6
    **post-hoc** IG rollup, not a trainable branch. Full closure
    analysis in [`audit/dmoi_v0.8.md`](audit/dmoi_v0.8.md).

11. **Same architecture, different axis: framework reusability is
    real, v0.9.** The v0.6 / v0.7 / v0.8 trilogy worked exclusively
    on LumA-vs-LumB (within the ER+ luminal subtype). v0.9 transferred
    the same v0.6 architecture to a fundamentally different
    classification axis, cross-lineage **Luminal (LumA + LumB) vs
    Basal**, n=502 dual-modality (Luminal 415, Basal 87), with the
    only changes being (1) the cohort file (`cohort_v3.tsv`), (2) the
    pole-defining Hallmark sets (`POLE_LUMINAL` = ER_EARLY + ER_LATE
    + ANDROGEN_RESPONSE; `POLE_BASAL` = EMT + MYC_TARGETS_V1 +
    G2M_CHECKPOINT), and (3) the class-positive label (`Basal` → 1).
    **Zero changes to model architecture, training loop, fusion,
    attention, encoder, or classifier head.** Result: TCGA test AUROC
    = **1.000** (bacc 0.972, 1 patient misclassified out of 101);
    Luminal pole top-3 IG = `ER_EARLY` + `ER_LATE` + `ANDROGEN_RESPONSE`
    (3/3 hand-picked priors); Basal pole top-5 IG = `MYC_V1` + `G2M`
    + `EMT` + `E2F` + `MYC_V2` (5/5 priors). The 8/8 expected-pathway
    hit rate across both poles with zero architecture changes is the
    strongest possible cross-task generalization signal: **the v0.6
    framework is empirically task-agnostic within DMOI scope.** Full
    write-up in [`audit/dmoi_v0.9.md`](audit/dmoi_v0.9.md).

12. **Doubly generalized: cross-cohort + cross-task, v0.10.** The
    v0.9 trained model was scored on the METABRIC cohort_v3
    (Luminal-vs-Basal, n=1,384: Luminal 1,175 + Basal 209,
    different patient population, Illumina HT-12 v3 microarray
    platform, RNA-only with methylation silenced + quantile-normalized
    to TCGA train RNA per the v0.2 / v0.4 / v0.6 protocol). Result:
    **METABRIC AUROC = 0.965** (bacc 0.842, vs v0.4 LumA-vs-LumB
    cross-cohort reference of 0.909). Per-pole IG top-3 on METABRIC
    is **identical to TCGA cohort_v3**: Luminal pole loads
    `ER_EARLY` + `ER_LATE` + `ANDROGEN_RESPONSE`; Basal pole loads
    `MYC_TARGETS_V1` + `G2M_CHECKPOINT` + `EPITHELIAL_MESENCHYMAL_TRANSITION`.
    8/8 expected priors in METABRIC top-5 AND 3/3 + 3/3 top-3
    stability between TCGA and METABRIC. The v0.6 framework is
    **simultaneously** task-invariant and cohort-invariant within
    DMOI scope. Full write-up in
    [`audit/dmoi_v0.10.md`](audit/dmoi_v0.10.md). The v0.6 → v0.10
    sequence now reads as a complete falsifiable architectural
    inquiry validating four axes of reusability: calibration transfer
    (v0.1 / v0.2), cross-cohort same-task (v0.4: AUROC 0.909, Jaccard
    0.667), cross-task same-cohort (v0.9: AUROC 1.000, 8/8 priors),
    and cross-cohort + cross-task (v0.10: AUROC 0.965, 8/8 priors,
    3/3 + 3/3 top-3 stable). The v0.7 + v0.8 three-variant
    architecture experiment further showed that adding a trainable
    pathway-attention branch is structurally redundant. **Gene-level
    hypothesis-conditioned attention + hand-picked pole priors +
    post-hoc Hallmark IG rollup is empirically the right architectural
    commitment for multi-omics binary subtype classification within
    DMOI scope.**

13. **Stability seal: the four-axis result is split-invariant,
    v0.11.** The natural skeptic's question about v0.9's AUROC =
    1.000 on the single 80/20 split was: *would this hold under a
    different split?* v0.11 ran 5-fold StratifiedKFold
    (random_state=42) on the same TCGA cohort_v3 with identical
    architecture and priors. **Every fold reached AUROC = 1.000**
    (mean ± std = 1.0000 ± 0.0000; bacc 0.9724 ± 0.0091). The
    biology recovery is fold-invariant: 3 / 3 Luminal expected
    priors hit per-fold IG top-5 in 5 of 5 folds; 4 / 5 Basal
    expected priors hit 5 of 5 folds; the 5th
    (`HALLMARK_MYC_TARGETS_V2`) hits 4 of 5. The pairwise mean
    Jaccard of **1.0000** on both poles means every fold selected
    the same top-3 pathways. The gene-level architecture commitment
    plus the Luminal vs Basal lineage signal in cohort_v3 is strong
    enough that the top-3 is essentially fixed under split
    perturbation. **v0.10's cross-cohort + cross-task result was
    not riding a single lucky split.** Full write-up in
    [`audit/dmoi_v0.11.md`](audit/dmoi_v0.11.md).

14. **Cross-cohort split-invariance: v0.4 / v0.6 cross-cohort
    metric is essentially deterministic, v0.12-A.** v0.11 sealed
    the Luminal-vs-Basal task as split-invariant on TCGA cohort_v3
    internally. v0.12-A asks the matching question one task axis
    over: is the v0.4 cross-cohort AUROC = 0.909 (METABRIC
    LumA-vs-LumB single shot) split-invariant, or did it ride a
    lucky TCGA-train split? Protocol: 5-fold StratifiedKFold
    (random_state=42) on TCGA cohort_v2 LumA-vs-LumB (n=417
    dual-modality, LumA 289 + LumB 128); for each fold, train v0.6
    architecture + score TCGA val fold + score full METABRIC
    LumA-vs-LumB (n=1,175, RNA-only, meth silenced, **QN re-fit
    per fold against the fold's TCGA-train RNA distribution**,
    the right thing under proper cross-validation). Result:
    **TCGA val AUROC = 0.9702 ± 0.0122** (+1.6 pp above v0.6
    5-fold reference of 0.954 ± 0.017, with tighter std);
    **METABRIC AUROC = 0.9254 ± 0.0052** (+1.6 pp above v0.4
    single-shot reference of 0.909, with std = 0.5 pp, essentially
    deterministic). Per-fold IG on METABRIC: **2 / 2 LumA expected
    priors (ER_EARLY + ER_LATE) hit top-5 in 5 / 5 folds**; **3 / 3
    LumB expected priors (E2F + G2M + MYC_V1) hit top-5 in 5 / 5
    folds**; LumB top-3 pairwise mean Jaccard = **1.0000** (every
    fold picked the same top-3); LumA top-3 Jaccard = 0.7000
    (structurally bounded, only 2 expected priors out of 50, so
    the 3rd top-3 slot must rotate). **The v0.4 / v0.6 cross-cohort
    capability is split-invariant on every measurable dimension.**
    v0.11 + v0.12-A together cover the internal AND cross-cohort
    variance bands on both task axes (Luminal-vs-Basal AND
    LumA-vs-LumB). Full write-up in
    [`audit/dmoi_v0.12.md`](audit/dmoi_v0.12.md).

---

## Architecture (one paragraph)

Two pole-specific input masks (LumA, LumB) are derived from MSigDB Hallmark
gene sets, `ESTROGEN_RESPONSE_EARLY` + `ESTROGEN_RESPONSE_LATE` for LumA;
`E2F_TARGETS` + `G2M_CHECKPOINT` + `MYC_TARGETS_V1` for LumB, and the HM450
probe-to-gene cis-mapping from UCSC Xena. Each pole branch sees a
hypothesis-attended view of RNA + methylation through an MLP encoder,
ending in a sub-classifier supervised with an auxiliary BCE loss
(`aux_weight=0.3`). The two pole representations are fused and concatenated
with a scalar disagreement = `|s_LumA − (1 − s_LumB)|` before the final
classifier head. Temperature scaling is fit by LBFGS on a held-out 15%
calibration split.

See [`docs/architecture.md`](docs/architecture.md) for the diagram + module map.

---

## Internal results (TCGA cohort_v2, 5-fold CV on the 80% train split)

### 3-way ablation

| Variant | AUROC | BalAcc |
|---|---|---|
| **Option A** (aux BCE + disagreement IN, ships in v0.2) | 0.954 ± 0.021 | 0.861 ± 0.067 |
| Option B (no aux + disagreement IN) | 0.958 ± 0.027 | 0.869 ± 0.066 |
| Ablation (no aux + no disagreement) | 0.961 ± 0.023 | 0.889 ± 0.041 |
| Δ A − B | −0.004 | −0.007 |
| Δ A − Ablation | −0.007 | −0.028 |

Option A pays ~0.007 AUROC for the auxiliary supervision that surfaces the
disagreement signal. The architecture neither helps nor hurts the primary
metric.

### Calibration (5-fold CV)

| T fit on | Mean T | Mean ECE on val | Interpretation |
|---|---|---|---|
| val (optimistic, upper bound) | 0.620 ± 0.252 | 0.111 | T tuned to the same fold ECE is measured on |
| **held-out cal split (held-out, ship)** | **0.673 ± 0.294** | **0.121** | T never saw the val data |

T < 1 means DMOI is **under-confident**: the pole-conditioned architecture
plus class-balanced BCE compress logits toward zero. Calibration *sharpens*.

### Held-out TCGA test (v0.2 Path C, n=84)

The 20% TCGA test split is carved at cohort-construction time with
`random_state=2024` (distinct from the CV seed) and scored *once* by a single
Option A model trained on the full train split for the CV-mean best epoch
(no early stopping, no test-AUC-driven epoch selection).

- AUROC : **0.968** (internal CV mean: 0.954, Δ = +0.014)
- BalAcc : 0.897
- ECE before T-scaling : 0.143
- ECE after T-scaling  : **0.079**  (T=0.634)

Test ≥ CV is unusual but in the right direction, the model isn't overfitting
to the CV folds.

---

## External validation on METABRIC (v0.2 Path A', n=1,175)

| Metric | METABRIC |
|---|---|
| AUROC | **0.9095** |
| BalAcc | 0.788 |
| LumB sensitivity (default 0.5 threshold) | 0.619 |
| LumB specificity | 0.956 |

### Calibration does NOT blindly transfer across cohorts

Reported on the same 85% METABRIC eval slice (n=999); the other 15% (n=176)
is used as the cohort-specific cal slice.

| Calibration | T | ECE on eval slice |
|---|---|---|
| Uncalibrated | 1.000 | 0.0745 |
| T from TCGA cal-split (naive transfer) | 0.634 | 0.1051 |
| **T from METABRIC cal-split (cohort-specific)** | **0.934** | **0.0738** |

TCGA's T was fit on a model with both RNA + methylation; on METABRIC the
methylation branch is silenced, so the logit distribution is different and
TCGA's sharpening T over-corrects. A METABRIC-specific T lands near 1.0
(the meth-silenced model is already nearly well-calibrated) and slightly
improves ECE. **Calibration parameters are cohort/modality specific.**

### LumB sensitivity decomposes into prior shift + modality silencing

The 0.619 / 0.956 sensitivity-specificity asymmetry on METABRIC isn't a
single phenomenon. Two corrections compared on the same 85% eval slice:

| Strategy | LumB sens | LumB spec | BalAcc | F1 LumB |
|---|---|---|---|---|
| Default @0.5 | 0.619 | 0.956 | 0.788 | 0.735 |
| **Bayes prior-adjusted** | **0.691** | 0.933 | **0.812** | **0.772** |
| Tuned threshold (0.425 from cal slice) | 0.656 | 0.943 | 0.799 | 0.754 |

TCGA train has 31% LumB; METABRIC has 40% LumB. Bayes class-prior adjustment
boosts LumB calls without tuning, gaining +0.072 sensitivity and +0.024
BalAcc. The data-tuned threshold lands at 0.425, exactly the direction
Bayes predicts. Both corrections triangulate to the same conclusion: **the
prior shift explains about half of the sensitivity asymmetry. The remainder
is the methylation-silencing residual**, the meth branch normally
contributes positive signal for harder LumB calls, and without it some
borderline cases are unrecoverable from RNA alone.

### What METABRIC validation does and does NOT show

It DOES show that the hypothesis-conditioned RNA encoder generalizes to an
independent cohort across platforms (HiSeq → Illumina HT-12 v3) with
quantile normalization and gene-symbol harmonization (16,890 shared genes
out of 20,530 TCGA / 20,384 METABRIC unique Hugo symbols).

It does NOT validate the dual-modality story, no public BRCA cohort has
paired RNA-seq + HM450 outside TCGA. See
[`docs/v0.2-design-external-validation.md`](docs/v0.2-design-external-validation.md)
for the recon trail.

Full reports: [`audit/dmoi_eval_v0.md`](audit/dmoi_eval_v0.md) (TCGA),
[`audit/dmoi_external_v0.2.md`](audit/dmoi_external_v0.2.md) (METABRIC).

---

## Per-patient attribution (v0.3, TCGA test n=84)

Integrated Gradients (Sundararajan et al. 2017) on each of three model
outputs, baseline = zero in the standardized domain (= train per-feature
mean). 50 Riemann steps per IG run. See
[`docs/v0.3-design-attribution.md`](docs/v0.3-design-attribution.md) for
the algorithm + scope rationale.

### Top-5 global features per pole (mean |IG| across 84 test patients)

| Rank | lumA_pole RNA | mean \|IG\| | lumB_pole RNA | mean \|IG\| |
|---|---|---|---|---|
| 1 | `FOXC1` | 0.0447 | `EFNA5` | 0.0143 |
| 2 | **`BCL2`** | 0.0323 | `RANBP1` | 0.0104 |
| 3 | `PDLIM3` | 0.0319 | `NBN` | 0.0100 |
| 4 | `TUBB2B` | 0.0303 | `ZW10` | 0.0094 |
| 5 | `EGR3` | 0.0263 | `POLA2` | 0.0087 |

### Three biological readings

- **lumA pole learned "this is NOT basal-like" + the canonical luminal gene.**
  FOXC1 (basal/myoepithelial transcription factor) anchors the top spot;
  the LumA pole's strongest discriminative signal is its *low* expression
  in LumA. BCL2, the canonical anti-apoptotic luminal marker, ranks
  second. The rest of the top-5 (PDLIM3, TUBB2B, EGR3) are cytoskeletal /
  early-response markers that distinguish LumA's lower-proliferation
  phenotype from LumB.
- **lumB pole learned cell-cycle + DNA-repair machinery.** RANBP1 (nuclear
  transport during mitosis), NBN (nibrin / DNA damage response), ZW10
  (mitotic checkpoint), and POLA2 (DNA polymerase α subunit) are all
  proliferation- and replication-stress genes. Not the textbook
  MKI67/TOP2A/AURKA but biologically equivalent, many gene proxies
  exist for the proliferation axis and the model picked DNA-damage and
  mitotic-machinery ones.
- **ESR1 / PGR / FOXA1 are correctly absent from the top attributions.**
  Both LumA and LumB are ER+, so the canonical luminal markers don't
  discriminate within this cohort. Their absence here is evidence that
  the model picked the right axis (proliferation + inverse-basal) rather
  than a naïve pan-luminal prior.

### Completeness check (IG faithfulness axiom)

| Target | Mean residual | Max residual | Status |
|---|---|---|---|
| **lumA_pole** | 0.0023 | 0.0182 | tight |
| **lumB_pole** | 0.0022 | 0.0112 | tight |
| final_logit | 0.0205 | 0.3425 | one outlier, likely from the disagreement scalar `\|s_LumA − (1 − s_LumB)\|` which has a non-differentiable `abs()` at 0; the pole-specific attributions are the recommended clinical-interpretability headline |

Full per-patient + global lists:
[`audit/dmoi_explain_v0.3.md`](audit/dmoi_explain_v0.3.md),
[`audit/dmoi_explain_per_patient.tsv`](audit/dmoi_explain_per_patient.tsv)
(5,040 rows = 84 × 3 × 2 × top-10), [`audit/dmoi_explain_global.tsv`](audit/dmoi_explain_global.tsv).

---

## Cross-cohort attribution (v0.4, METABRIC n=1,175)

Same IG pipeline as v0.3, applied to the METABRIC external cohort with
the methylation branch silenced (METABRIC has no HM450). Validates
whether the model's biology, not just its AUROC, generalizes.

### Cross-cohort top-K agreement

| Target | Jaccard top-10 | Jaccard top-50 | Verdict |
|---|---|---|---|
| **lumA_pole** | **0.667** | **0.786** | Strong, biology generalizes |
| **lumB_pole** | **0.667** | 0.538 | Strong, biology generalizes |
| final_logit | 0.538 | 0.724 | Moderate; final logit has the disagreement-scalar instability |

### Shared top-10 lumA pole genes (TCGA test ∩ METABRIC)

`FOXC1`, `BCL2`, `PDLIM3`, `TUBB2B`, `KRT15`, `EGR3`, `RAB17`, `AHNAK`, every lumA
headline gene from the v0.3 TCGA-test attribution also appears in the
METABRIC top-10. The "inverse-basal-marker + BCL2" story is confirmed
on an independent cohort.

### Shared top-10 lumB pole genes + the METABRIC-new ones

Shared with v0.3 TCGA test: `RANBP1`, `NBN`, `ZW10`, `POLA2`, `EFNA5`.

New on METABRIC top-10 (more canonical proliferation markers than v0.3's
list): **`CKS1B`** (CDK regulatory subunit, core cell-cycle), **`DBF4`**
(CDC7 kinase activator, S-phase initiation), **`NDC80`** (kinetochore
complex, mitosis), **`DSCC1`** (replication fork). The larger METABRIC
cohort gave the model enough statistical power to surface the textbook
proliferation gene set on top of v0.3's structural-mitotic markers. The
model's biology is **tighter on METABRIC, not looser**.

### Completeness check on METABRIC

| Target | Mean residual | Max residual |
|---|---|---|
| **lumA_pole** | 0.0015 | 0.0163 |
| **lumB_pole** | 0.0023 | 0.0204 |
| final_logit | 0.0155 | 0.2276 |

Same IG-faithfulness regime as TCGA test. The disagreement-scalar
non-differentiability still produces one final_logit outlier; the
pole-specific attributions remain the recommended interpretability
headline.

Full report: [`audit/dmoi_explain_external_v0.4.md`](audit/dmoi_explain_external_v0.4.md),
[`audit/dmoi_explain_external_per_patient.tsv`](audit/dmoi_explain_external_per_patient.tsv)
(70,500 rows = 1,175 × 3 × 2 × top-10),
[`audit/dmoi_explain_external_global.tsv`](audit/dmoi_explain_external_global.tsv).

---

## Pathway-level aggregation (v0.5, MSigDB Hallmark rollup)

Per-gene IG attributions rolled up to the five MSigDB Hallmark gene sets
that defined the model's architectural priors
(`ESTROGEN_RESPONSE_EARLY`, `ESTROGEN_RESPONSE_LATE` for the LumA pole;
`E2F_TARGETS`, `G2M_CHECKPOINT`, `MYC_TARGETS_V1` for the LumB pole).
Per-pathway score = mean of |IG| across patients and pathway-member
genes that appear in the input feature set.

### Pole-pathway alignment ratio (mean |IG| of expected pathway / mean |IG| of "wrong" pathway)

| Pole | Cohort | Expected pathway (mean \|IG\|) | "Wrong" pathway (mean \|IG\|) | Ratio |
|---|---|---|---|---|
| **lumA_pole** | TCGA test | ESTROGEN_RESPONSE_EARLY (0.00991) | E2F_TARGETS (0.00003) | **~330×** |
| **lumA_pole** | METABRIC | ESTROGEN_RESPONSE_EARLY (0.01076) | E2F_TARGETS (0.00004) | **~270×** |
| **lumB_pole** | TCGA test | G2M_CHECKPOINT (0.00334) | ESTROGEN_RESPONSE_EARLY (0.00008) | **~42×** |
| **lumB_pole** | METABRIC | G2M_CHECKPOINT (0.00362) | ESTROGEN_RESPONSE_EARLY (0.00008) | **~45×** |

The LumA pole loads the estrogen-response pathways ~300× harder than the
cell-cycle pathways the LumB pole was given; the LumB pole loads the
cell-cycle pathways ~45× harder than the estrogen pathways. **The
hypothesis-conditioning prior worked exactly as designed**, and this
holds with the same ratios on a completely independent METABRIC cohort.

### Cross-cohort top-3 pathway agreement

| Target | Top-3 (shared between TCGA test and METABRIC) |
|---|---|
| **lumA_pole** | `ESTROGEN_RESPONSE_EARLY`, `ESTROGEN_RESPONSE_LATE`, `G2M_CHECKPOINT` |
| **lumB_pole** | `E2F_TARGETS`, `G2M_CHECKPOINT`, `MYC_TARGETS_V1` |
| final_logit | `ESTROGEN_RESPONSE_EARLY`, `ESTROGEN_RESPONSE_LATE`, `G2M_CHECKPOINT` |

For both pole-specific targets, every top-3 pathway is shared between
TCGA and METABRIC. The lumA `G2M_CHECKPOINT` entry is a small-magnitude
placeholder (its score is ~75× smaller than the dominant ER pathways),
present in the top-3 only because there are only five pathways in the
aggregation.

### Limitations (v0.5)

Only the 5 Hallmark sets in `priors.py` are aggregated. The full
50-set MSigDB Hallmark catalog would let a wider unsupervised
pathway-discovery test ("are there *other* pathways the model loaded on
that we didn't include in the priors?"). Adding a `gmt`-file loader
+ the full catalog was the v0.6 follow-up below; the v0.5 scope was
"validate the architectural-prior pathways are exactly the ones the
model uses."

Full report: [`audit/dmoi_pathway_v0.5.md`](audit/dmoi_pathway_v0.5.md).

---

## Full Hallmark catalog rollup (v0.6, all 50 MSigDB Hallmark sets)

v0.5 rolled IG into the 5 Hallmark sets that were already routed to the
pole masks. The obvious objection: *did those 5 sets win because they
were the only ones loaded?* v0.6 answers by loading the full 50-set
**MSigDB Hallmark v2024.1.Hs** catalog
([`data/msigdb/h.all.v2024.1.Hs.symbols.gmt`](data/msigdb/h.all.v2024.1.Hs.symbols.gmt),
CC-BY 4.0, ~48 KB, parsed by [`src/dmoi_brca/hallmark.py`](src/dmoi_brca/hallmark.py))
and re-running the same IG rollup on the same TCGA test split + the
same METABRIC cohort.

### v0.5 finding survives the 50-set widening

| Target | TCGA test top-3 of 50 | METABRIC top-3 of 50 | v0.5 top-pathway(s) survive? |
|---|---|---|---|
| **lumA_pole** | `ESTROGEN_RESPONSE_EARLY`, `ESTROGEN_RESPONSE_LATE`, `IL2_STAT5_SIGNALING` | `ESTROGEN_RESPONSE_EARLY`, `ESTROGEN_RESPONSE_LATE`, `IL2_STAT5_SIGNALING` | **Yes**, both ER sets in top-3 on both cohorts |
| **lumB_pole** | `MYC_TARGETS_V1`, `E2F_TARGETS`, `G2M_CHECKPOINT` | `E2F_TARGETS`, `G2M_CHECKPOINT`, `MYC_TARGETS_V1` | **Yes**, same 3 cell-cycle sets dominate both cohorts (rank order swaps within a near-tie) |
| `final_logit` | `ESTROGEN_RESPONSE_EARLY`, `ESTROGEN_RESPONSE_LATE`, `G2M_CHECKPOINT` | `ESTROGEN_RESPONSE_EARLY`, `ESTROGEN_RESPONSE_LATE`, `G2M_CHECKPOINT` | **Yes**, identical top-3 across cohorts |

The 5-set v0.5 finding wasn't an artifact of which sets were loaded.

### New secondary findings (only visible with the full catalog)

- **`IL2_STAT5_SIGNALING` joins lumA top-3** on both cohorts (TCGA test
  mean |IG| 0.00096; METABRIC 0.00101). STAT5 is a known co-regulator
  of `ESR1` transcriptional activity in luminal breast cancer, so this
  reads as a biologically coherent secondary signal rather than noise.
  It's still ~6.6× below ER_EARLY on both cohorts.
- **`MYC_TARGETS_V2` (rank 4) and `MITOTIC_SPINDLE` (rank 5)** join the
  lumB top-5 on both cohorts. Both are additional proliferation
  programs; the model is loading the entire cell-cycle / growth axis,
  not just one pathway.
- **No surprise non-proliferation, non-ER pathway** appears in either
  pole's top-5, the architectural prior catches the biology that's
  actually there.

### Pathway-level pole-specificity (full 50-set view)

| Pole | Cohort | Top pathway (mean \|IG\|) | Top "opposite-program" pathway (mean \|IG\|) | Ratio |
|---|---|---|---|---|
| **lumA_pole** | TCGA test | `ESTROGEN_RESPONSE_EARLY` (0.00629) | `MYC_TARGETS_V1` (in lumA-top-50 lower tier; see CSV) | dominant ER |
| **lumA_pole** | METABRIC  | `ESTROGEN_RESPONSE_EARLY` (0.00681) | same                                                     | dominant ER |
| **lumB_pole** | TCGA test | `MYC_TARGETS_V1` (0.00328)         | `ESTROGEN_RESPONSE_EARLY` (lumB-tier; see CSV)           | dominant cell-cycle |
| **lumB_pole** | METABRIC  | `E2F_TARGETS` (0.00355)            | same                                                     | dominant cell-cycle |

The exact magnitude ratios drift run-to-run (because each version
re-trains a fresh model), but the **structure**, lumA pole stacks ER
at the top, lumB pole stacks cell-cycle at the top, is stable across
v0.5 (5-set) and v0.6 (50-set) on both cohorts.

### Limitations (v0.6)

- 50 Hallmark sets loaded, the full v2024.1.Hs catalog. The C2 curated
  catalog (~5,000 sets) and other MSigDB collections remain out of
  scope.
- Aggregation is still RNA-only. METABRIC has no methylation; even on
  TCGA the methylation features are HM450 probes, not gene symbols, so
  a Hallmark rollup of methylation IG needs a probe → gene crosswalk
  before it would be meaningful.
- The pathway scores are interpretation artifacts. The model still
  attends to genes, not to pathways. Pathway-level *attention* (feeding
  pathway embeddings directly into the model) is the natural v0.7+
  candidate.

Full report: [`audit/dmoi_pathway_v0.6.md`](audit/dmoi_pathway_v0.6.md)
+ six per-(target, cohort) CSVs at
[`audit/dmoi_pathway_v0.6_*.csv`](audit/) (50 rows each, ranked by
mean |IG|).

---

## Learnable pathway-pole attention (v0.7.1, two-phase documented negative)

v0.7 tried to replace v0.6's hand-picked Hallmark pole masks with a
learnable softmax distribution over all 50 Hallmark pathways per pole
(Variant D in
[`docs/v0.7-design-pathway-attention.md`](docs/v0.7-design-pathway-attention.md)).
The architecture-level question: *if we let the model decide which
pathways define each pole, does it rediscover v0.6's ER-for-LumA /
cell-cycle-for-LumB alignment from scratch?*

v0.7.1 answer: **no in two distinct ways, each instructive.**

### Phase A: standardized inputs + tight init (collapse)

| Cohort | AUROC | vs v0.6 |
|---|---|---|
| TCGA test | 0.957 | −0.011 |
| METABRIC  | 0.913 | +0.004 |

Top weights spanned 0.0203 – 0.0205 across all 50 pathways on both
poles, effectively uniform (1/50 = 0.0200). 0/3 v0.6 top-3 made the
Phase A top-3. **Mechanism**: pathway scores were standardized to
zero mean; softmax-uniform attention × zero-centered input = zero
output; head learns to ignore; no gradient back; attention stays
uniform forever, a self-reinforcing equilibrium of uselessness.

### Phase B: raw inputs + warm init (collapse fixed; wrong basin)

| Cohort | AUROC | vs v0.6 | vs Phase A |
|---|---|---|---|
| TCGA test | 0.960 | −0.009 | +0.003 |
| METABRIC  | 0.898 | −0.012 | −0.016 |

Collapse fixed, LumA top weight 0.069 (3.4× uniform), LumB top
weight 0.048 (2.4× uniform). The attention *learns*. But:

| Pole | Phase B top-3 | v0.6 IG top-3 | Shared |
|---|---|---|---|
| LumA | WNT, INTERFERON_ALPHA, MTORC1 | ER_EARLY, ER_LATE, IL2_STAT5 | **0 / 3** |
| LumB | KRAS_UP, MTORC1, PEROXISOME  | E2F, G2M, MYC_V1               | **0 / 3** |

The model learned a completely different basin. AUROC dropped on
both cohorts. **Diagnosis**: the scalar
`pole_pathway_feat = sum_k w_k × mean_expression_k(patient)` only
exposes pathway-*magnitude* variance across patients to the head,
not pathway-*direction* signal. The LumA-vs-LumB discriminative axis
is in direction (ER program up for LumA, cell-cycle program up for
LumB), not in absolute magnitude. So the Phase B attention learns to
concentrate on high-baseline pathways (WNT, MTORC1, KRAS) that have
big patient-to-patient variance but no class-discriminative
direction. AUROC drops because the new branch is competing with the
gene-level branch and adding noise.

### v0.8 Variant C result: same wrong basin, architecture experiment closed

v0.8 ran the planned Variant C upgrade: the per-pole pathway feature
becomes a 16-dim vector via a learnable `Linear(n_pathways, 16)` per
pole. The head now sees 32 pathway features (vs v0.7.1's 2) and the
pathway branch has 17× more parameters (1700 vs 100). The design
hypothesis: a richer interface lets the model read per-pathway
*direction* signals (each pathway has a learned embedding row in the
projection matrix), not just aggregate magnitude.

| Cohort | v0.8 AUROC | v0.7.1 ref | v0.6 ref | Δ vs v0.6 |
|---|---|---|---|---|
| TCGA test | 0.954 | 0.960 | 0.968 | −0.015 |
| METABRIC  | 0.920 | 0.898 | 0.909 | +0.011 |

| Pole | v0.8 top-3 | v0.7.1 top-3 | Same basin? |
|---|---|---|---|
| LumA | WNT, INTERFERON, MTORC1 | WNT, INTERFERON, MTORC1 | **Yes** (top-5 weights identical to v0.7.1 within sub-pp) |
| LumB | MTORC1, KRAS, PEROXISOME | MTORC1, KRAS, PEROXISOME | **Yes** (same set, same order) |

**Variant C with 17× more parameters and a richer interface converged
to the same magnitude-driven basin as v0.7.1.** The interface
dimensionality was not the bottleneck.

### Information-theoretic interpretation

This is the decisive finding of the v0.7 + v0.8 experiment. The
gradient signal reaching the pathway branch is *what the gene-level
branch hasn't already explained*. The gene-level encoder sees ESR1,
PGR, FOXA1, RANBP1, NBN, ZW10, and the rest, and resolves the
LumA-vs-LumB decision there, at the gene level, in the direction
axis. The pathway branch is left to grip whatever residual is left,
and that residual happens to be pathway-magnitude variance (which
pathways have high baseline absolute expression in any given
patient), not pathway-direction signal (which pole-relevant program
is up vs down).

**Whether the head reads one scalar or 32 features per patient does
not change what gradient flows back**, the same magnitude-driven
basin is found either way. The matched-basin convergence is
information-theoretic evidence that gene-level commitment is the
right architectural level for LumA-vs-LumB, and that the pathway
view's correct role is the v0.5/v0.6 post-hoc IG rollup, not a
trainable extension.

### v0.6 remains canonical

The v0.7+v0.8 three-variant experiment now reads as a complete
falsifiable architectural inquiry: tried softmax collapse mode
(failed mechanically), tried scalar pole feature (wrong basin),
tried vector pole feature with 17× more parameters (same wrong
basin). The hypothesis "learnable pathway-pole attention can
replace the v0.6 hand-picked masks" is falsified. The
hypothesis "v0.6's gene-level commitment is the right level"
is supported by 3 independent failures of the alternative.

Full reports: [`audit/dmoi_v0.7.md`](audit/dmoi_v0.7.md) (v0.7.1
two-phase write-up) + [`audit/dmoi_v0.8.md`](audit/dmoi_v0.8.md)
(v0.8 closure analysis).

---

## Cross-task generalization (v0.9, Luminal vs Basal)

The v0.6 / v0.7 / v0.8 trilogy worked exclusively on LumA-vs-LumB.
v0.9 transferred the same v0.6 architecture to **Luminal (LumA +
LumB) vs Basal**, a cross-lineage classification on a new cohort
of 502 dual-modality patients (Luminal 415, Basal 87 per
`PAM50Call_RNAseq`, stratified 80/20 split with `random_state=2024`).

### What changed

- `data/tcga_brca/cohort_v3.tsv`, built by
  [`scripts/build_cohort_v3.py`](scripts/build_cohort_v3.py).
- `POLE_LUMINAL` (ER_EARLY + ER_LATE + ANDROGEN_RESPONSE) +
  `POLE_BASAL` (EMT + MYC_TARGETS_V1 + G2M_CHECKPOINT) added to
  [`src/dmoi_brca/priors.py`](src/dmoi_brca/priors.py).
- `make_pole_masks` (and the underlying mask builders) extended with
  an optional `hallmark_sets=` kwarg so callers can pass the full
  50-set catalog from `load_hallmark_gmt(...)` for pole names that
  aren't in `priors.HALLMARK_SETS`.
- `train_one_fold` and `integrated_gradients_dmoi` accept an optional
  `pole_order` kwarg; default `("LumA", "LumB")` keeps v0.6 backward
  compatibility, v0.9 passes `("Luminal", "Basal")`.
- `DMOIModel.forward` uses `pole_order[0]` / `pole_order[1]` for the
  disagreement scalar and the head input (instead of hardcoded
  `"LumA"` / `"LumB"`).
- `scripts/eval_dmoi_v0.9.py`, new driver. ~250 LOC, clones the
  v0.7 driver pattern but uses cohort_v3 + Luminal/Basal poles +
  Hallmark gmt override.

**Zero changes to the model architecture, training loop, fusion,
attention, encoder, or classifier head.**

### Result

| Metric | DMOI v0.9 (Luminal vs Basal) | v0.6 reference (LumA vs LumB) |
|---|---|---|
| TCGA test AUROC      | **1.000**             | 0.968 |
| TCGA test bacc       | 0.972 (1 of 101 misclassified) | — |
| Train n              | 401 (Luminal 332, Basal 69) | 333 (LumA, LumB stratified) |
| Test n               | 101 (Luminal 83, Basal 18) | 84 |
| Pole RNA mask sizes  | Luminal 371 / Basal 536 (of 20,530) | LumA ~107 / LumB ~189 |

| Pole | v0.9 IG top-5 (by mean \|IG\|) | Expected priors | Hits |
|---|---|---|---|
| **Luminal** | `ESTROGEN_RESPONSE_EARLY`, `ESTROGEN_RESPONSE_LATE`, `ANDROGEN_RESPONSE`, `CHOLESTEROL_HOMEOSTASIS`, `UV_RESPONSE_DN` | ER_EARLY + ER_LATE + ANDROGEN_RESPONSE | **3 / 3** |
| **Basal**   | `MYC_TARGETS_V1`, `G2M_CHECKPOINT`, `EPITHELIAL_MESENCHYMAL_TRANSITION`, `E2F_TARGETS`, `MYC_TARGETS_V2` | EMT + MYC_TARGETS_V1 + G2M_CHECKPOINT + E2F_TARGETS + MYC_TARGETS_V2 | **5 / 5** |

**8 / 8** expected pathways in top-5 per pole, with zero architecture
changes. The `final_logit` top-3, `ESTROGEN_RESPONSE_EARLY`,
`ESTROGEN_RESPONSE_LATE`, `ANDROGEN_RESPONSE`, mirrors the Luminal
pole because Luminal is the majority class; signed-IG analysis
confirms the model is correctly identifying Luminal patients via
high ER expression and Basal patients via high cell-cycle + EMT
expression.

### Reading

This is the strongest possible cross-task generalization signal
within DMOI scope. The v0.6 architecture is empirically task-agnostic:
the same model, training loop, and IG analysis pipeline pick up the
Luminal-vs-Basal biology cleanly when given Luminal-vs-Basal priors
and labels. The 8 / 8 expected-prior hit rate is what makes it
decisive, AUROC = 1.000 alone could be cohort-easy artifact, but
the IG ranking confirms the model is leveraging the wired biology, not
a shortcut.

The v0.7 + v0.8 conclusion ("gene-level commitment is the right
architectural level; learnable pathway branch is structurally
redundant") composes cleanly with v0.9 ("gene-level commitment
generalizes to a different classification axis"). Together, v0.6 →
v0.9 reads as a falsifiable architectural inquiry that **(a)** found a
working framework, **(b)** systematically tested whether a richer
architecture beats it (no, in 3 variants), and **(c)** confirmed
framework reusability on a new task (yes, decisively).

Full report: [`audit/dmoi_v0.9.md`](audit/dmoi_v0.9.md).

---

## Cross-cohort + cross-task generalization (v0.10, METABRIC)

v0.4 / v0.6 already validated cross-cohort generalization on the
LumA-vs-LumB axis (METABRIC RNA-only AUROC 0.909, Jaccard 0.667
gene-level). v0.9 validated cross-task generalization (Luminal-vs-Basal
on TCGA, AUROC 1.000, 8/8 priors). v0.10 composes the two: trains the
v0.9 model on TCGA cohort_v3, scores **METABRIC cohort_v3
(Luminal-vs-Basal, n=1,384, Luminal 1,175 + Basal 209)** with the
same RNA-only + meth-silenced + QN-to-TCGA protocol from v0.2 / v0.4.

### Result

| Cohort | AUROC | bacc | per-pole IG vs priors |
|---|---|---|---|
| TCGA cohort_v3 test (v0.9 reference) | 1.000 | 0.972 | Luminal 3/3 + Basal 5/5 = **8/8** |
| **METABRIC cohort_v3 external (v0.10)** | **0.965** | **0.842** | **Luminal 3/3 + Basal 5/5 = 8/8** |
| v0.4 LumA-vs-LumB METABRIC reference | 0.909 | — | — |

| Pole | TCGA test top-3 | METABRIC top-3 | Shared (n / 3) |
|---|---|---|---|
| **Luminal** | `ESTROGEN_RESPONSE_EARLY`, `ESTROGEN_RESPONSE_LATE`, `ANDROGEN_RESPONSE` | `ESTROGEN_RESPONSE_EARLY`, `ESTROGEN_RESPONSE_LATE`, `ANDROGEN_RESPONSE` | **3 / 3** |
| **Basal**   | `MYC_TARGETS_V1`, `G2M_CHECKPOINT`, `EPITHELIAL_MESENCHYMAL_TRANSITION` | `MYC_TARGETS_V1`, `G2M_CHECKPOINT`, `EPITHELIAL_MESENCHYMAL_TRANSITION` | **3 / 3** |

3 / 3 top-3 stability on both poles between TCGA and METABRIC,
the per-pole biology recovered by the model is cohort-invariant.
The 5.6 pp METABRIC-AUROC lift over the v0.4 LumA-vs-LumB reference
(0.965 vs 0.909) reflects the intrinsically easier separability of
the cross-lineage axis, but the 8/8 expected-prior hit on METABRIC
is the decisive metric, the framework is recovering the wired
biology cleanly across both axes of variation.

### Closure: v0.6 → v0.10 reads as four-axis framework reusability

| Axis | Evidence | Where |
|---|---|---|
| Calibration transfer | Cohort-specific T_TCGA=0.634 vs T_METABRIC=0.934 | v0.1 / v0.2 |
| Cross-cohort same-task | LumA-vs-LumB METABRIC AUROC 0.909, Jaccard 0.667 | v0.4 |
| Cross-task same-cohort | Luminal-vs-Basal TCGA AUROC 1.000, 8/8 priors | v0.9 |
| **Cross-cohort + cross-task** | **Luminal-vs-Basal METABRIC AUROC 0.965, 8/8 priors, 3/3 + 3/3 top-3 stable** | **v0.10** |

The v0.7 + v0.8 three-variant architecture experiment further showed
that adding a trainable pathway-attention branch is structurally
redundant (gene-level commitment captures all discriminative
direction signal). Across v0.6 to v0.10, the result was found, then tested, then generalized across cohort, across task, and across both. Gene-level hypothesis-conditioned attention +
hand-picked pole priors + post-hoc Hallmark IG rollup is empirically
the right architectural commitment for multi-omics binary subtype
classification within DMOI scope.

Full report: [`audit/dmoi_v0.10.md`](audit/dmoi_v0.10.md).

---

## Reproduce

```bash
# 1. Install pinned deps.
uv sync

# 2. (One-time) Fetch TCGA cohort.
python scripts/build_cohort_v2.py     # produces data/tcga_brca/cohort_v2.tsv

# 3. Baseline (LogReg + RF on concat / rna / meth).
python scripts/run_baseline_v2.py     # writes audit/baseline_v2_*

# 4. DMOI 3-way ablation + calibration + held-out TCGA test (Step A).
python scripts/eval_dmoi.py           # ~2 min on Apple Silicon (MPS)

# 5. (One-time) Fetch METABRIC. ~690 MB.
python scripts/fetch_metabric.py
python scripts/build_metabric_cohort.py

# 6. METABRIC external validation + cohort-specific cal + LumB sens analysis.
python scripts/eval_external.py       # ~3 min on MPS

# 7. (v0.3) Per-patient Integrated Gradients attribution on TCGA test (n=84).
python scripts/explain_dmoi.py        # ~3-4 min on MPS
                                      # writes audit/dmoi_explain_v0.3.md
                                      # + per_patient.tsv + global.tsv + 3 PNG plots

# 8. (v0.4) Cross-cohort IG attribution on METABRIC (n=1,175, meth silenced).
python scripts/explain_metabric.py    # ~10 min on MPS
                                      # writes audit/dmoi_explain_external_v0.4.md
                                      # + external_per_patient.tsv + external_global.tsv + 3 PNGs
                                      # depends on audit/dmoi_explain_global.tsv from step 7
                                      # for the cross-cohort Jaccard comparison

# 9. (v0.5) Pathway-level IG aggregation (5 priors-set rollup, both cohorts).
python scripts/aggregate_pathway_ig.py  # ~7 min on MPS
                                        # writes audit/dmoi_pathway_v0.5.md
                                        # with cross-cohort top-3 pathway agreement

# 10. (v0.6) Full 50-set Hallmark catalog rollup (closes the 5-set caveat).
python scripts/aggregate_pathway_ig_full.py  # ~7 min on MPS
                                              # writes audit/dmoi_pathway_v0.6.md
                                              # + 6 per-(target, cohort) CSVs with all 50 sets
                                              # uses data/msigdb/h.all.v2024.1.Hs.symbols.gmt (CC-BY 4.0)

# 11. (v0.7.1) Learnable pathway-pole attention (scalar pole feature, Phase B).
python scripts/eval_dmoi_v0.7.py             # ~7 min on MPS
                                              # writes audit/dmoi_v0.7.md
                                              # Two-phase documented negative

# 12. (v0.8) Variant C, vector pole feature (proj_dim=16, 17x more params).
python scripts/eval_dmoi_v0.8.py             # ~7 min on MPS
                                              # writes audit/dmoi_v0.8.md
                                              # Closes the v0.7+v0.8 architecture experiment:
                                              # same wrong basin as v0.7.1 with richer interface
                                              # => gene-level commitment is the right level

# 13. (v0.9) Cross-task generalization: Luminal vs Basal cohort_v3.
python scripts/build_cohort_v3.py             # builds data/tcga_brca/cohort_v3.tsv
python scripts/eval_dmoi_v0.9.py              # ~7 min on MPS
                                              # writes audit/dmoi_v0.9.md
                                              # AUROC 1.000, 8/8 expected priors in top-5 IG
                                              # framework reusability empirically confirmed

# 14. (v0.10) Cross-cohort + cross-task generalization: METABRIC cohort_v3.
python scripts/build_metabric_cohort_v3.py    # builds data/metabric/cohort_v3.tsv
python scripts/eval_metabric_v0.10.py         # ~10 min on MPS
                                              # writes audit/dmoi_v0.10.md
                                              # METABRIC AUROC 0.965, 8/8 priors, 3/3+3/3 top-3 stable
                                              # framework reusability across BOTH axes
```

Pinned to Python 3.11+, `numpy 2.2`, `scikit-learn 1.7`, `torch 2.x`,
`captum 0.7+` (MPS-supported on Apple Silicon).

---

## Layout

```
src/dmoi_brca/
├── features.py             # cohort + RNA + streaming top-K methylation loader
├── priors.py               # Hallmark gene-set priors per pole
├── hypothesis_attention.py # cis-mapping + PoleMaskSet + make_pole_masks
├── encoder.py              # pole-conditioned MLP encoders
├── fusion.py               # dual-perspective fuser + disagreement scalar
├── dmoi_model.py           # end-to-end DMOIModel
├── train.py                # train_one_fold + run_dmoi_cv (StratifiedKFold)
├── eval.py                 # per-class metrics, ECE, disagreement-vs-misclass
├── calibration.py          # temperature scaling (LBFGS on log_T)
├── attribution.py          # v0.3: Captum-based Integrated Gradients wrapper
├── pathway.py              # v0.5: Hallmark-set pathway aggregation of IG attributions
├── hallmark.py             # v0.6: MSigDB Hallmark gmt-file loader (50 sets, CC-BY 4.0)
├── pathway_attention.py    # v0.7: learnable softmax pathway-pole attention (Variant D)
├── external.py             # v0.2: cross-cohort gene align + QN + meth-silenced helpers
├── cohort.py               # cohort construction + 80/20 train/test split
├── baseline.py             # sklearn baselines (LogReg, RF)
├── audit.py                # NDJSON hash-chained ledger
├── tracking.py             # MLflow run wrapper
└── canary.py               # smoke-test interface for lab_semantic_check.py

scripts/
├── build_cohort_v2.py        # TCGA cohort selection + stratified 80/20 split
├── run_baseline_v2.py        # baseline driver
├── train_dmoi.py             # single-config DMOI driver
├── eval_dmoi.py              # 3-way ablation + calibration + held-out TCGA test
├── fetch_metabric.py         # v0.2: cBioPortal LFS download (~690 MB)
├── build_metabric_cohort.py  # v0.2: filter to LumA/LumB
├── eval_external.py          # v0.2: cross-cohort eval + cal-transfer + LumB sens
├── explain_dmoi.py           # v0.3: per-patient IG attribution + audit MD (TCGA test)
├── explain_metabric.py       # v0.4: cross-cohort IG attribution (METABRIC, meth silenced)
├── aggregate_pathway_ig.py        # v0.5: Hallmark pathway rollup driver (5 priors-sets)
├── aggregate_pathway_ig_full.py   # v0.6: full 50-set Hallmark catalog rollup driver
├── eval_dmoi_v0.7.py              # v0.7.1 Phase B: scalar pole feature driver
├── eval_dmoi_v0.8.py              # v0.8: vector pole feature (Variant C, proj_dim=16) driver
├── build_cohort_v3.py             # v0.9: TCGA Luminal-vs-Basal cohort builder
├── eval_dmoi_v0.9.py              # v0.9: cross-task generalization driver (TCGA)
├── build_metabric_cohort_v3.py    # v0.10: METABRIC Luminal-vs-Basal cohort builder
├── eval_metabric_v0.10.py         # v0.10: cross-cohort + cross-task generalization driver
└── check_english_only.py     # CJK gate run by the pre-commit hook
```

---

## What's out of scope for v0.10

See [`docs/what-is-out-of-scope.md`](docs/what-is-out-of-scope.md) for the
full list. Key items still deliberately deferred after v0.8:

- **Multi-modal external validation.** No public BRCA cohort outside TCGA
  has paired RNA-seq + HM450, see the v0.2 design doc for the recon.
- **Other pole hypotheses** (ER−/HER2+, basal vs claudin-low).
- **Variant E / auxiliary direction supervision.** Force the
  pathway-attention to match the v0.6 IG-derived ranking via an
  auxiliary loss. This would defeat the "does the model find it on
  its own" question that v0.7+v0.8 explicitly tested, so it's
  out of scope here.
- **Variant A / pathway-only model** (replace genes with 50 pathway
  means). Interesting ablation; AUROC drop almost certain
  (~0.95→~0.90); deferred to v0.9+ as a sanity check on the v0.8
  conclusion.
- **proj_dim hyperparameter sweep.** v0.8 used proj_dim=16; the
  matched-basin convergence across 2 ↔ 32 feature interfaces is
  robust evidence that intermediate proj_dim values would not change
  the conclusion.
- **Other MSigDB collections.** Hallmark v2024.1.Hs (50 sets) only.
  C2 curated (~5,000 sets) and other collections are not included.
- **Methylation pathway rollup.** The Hallmark aggregation is
  RNA-only; HM450 probes need a probe → gene crosswalk first.
- **Counterfactual explanations** ("what would need to change to flip the
  prediction"), adversarial-style, much heavier than IG.
- **Nested CV for hyperparameter tuning**. `calibration_frac=0.15` is a
  fixed choice carried over from Guo et al., not swept.

---

## License

MIT. See [`LICENSE`](LICENSE).
