# `dmoi-brca-poc`

> **Capability portrait, not a research result.** Public data is intentionally
> subsetted to keep the demo small and reproducible on a single workstation.

**What this shows.** A hypothesis-conditioned multi-omics architecture
(Dialectical Multi-Omics Integration, DMOI) on TCGA-BRCA. Two "pole" perspectives
(LumA-like and LumB-like) each see only their pole-relevant RNA + DNA-methylation
features via Hallmark-gene priors and HM450 cis-mapping. Their predictions are
fused with a disagreement signal exposed to the classifier head. Temperature
scaling on a held-out calibration split delivers honest, well-calibrated
probabilities. **v0.2 adds two external generalization tests** — a held-out
TCGA test split and an independent cohort (METABRIC, RNA-only) — and a clean
cohort-specific calibration analysis.

**Reproducibility.** `python scripts/eval_dmoi.py` reproduces the full TCGA
evaluation in about 2 minutes on an M-series Mac.
`python scripts/eval_external.py` adds the METABRIC external test
in another ~3 minutes (after a one-time ~690 MB METABRIC download).

**Substrate.** Emits hash-chained NDJSON audit entries, tracks MLflow runs,
exposes a canary smoke test for `lab_semantic_check.py`.

**Production framing.** A version of this method ran at full cohort scale on
proprietary multi-omics data during my time in industry. The lab version here
proves the *method* and the *engineering*, not the result. See
[`docs/what-is-out-of-scope.md`](docs/what-is-out-of-scope.md).

---

## v0.7 headline result (Phase A — honest negative)

| Metric | DMOI v0.7 (Phase A) |
|---|---|
| 5-fold CV AUROC (TCGA train split, n=333) | 0.954 ± 0.017 |
| **v0.6 reference: TCGA held-out test AUROC** | **0.968** |
| **v0.7 Phase A: TCGA held-out test AUROC** | **0.957** (Δ −0.011 vs v0.6) |
| **v0.6 reference: METABRIC external AUROC** | **0.909** |
| **v0.7 Phase A: METABRIC external AUROC** | **0.913** (Δ +0.004 vs v0.6) |
| Per-patient IG attribution (v0.3) | lumA / lumB / final logit on TCGA test (n=84) |
| Cross-cohort attribution agreement (v0.4) | Jaccard top-10 = 0.667 lumA + 0.667 lumB on METABRIC vs TCGA test |
| Pathway-level aggregation, 5 sets (v0.5) | lumA loads ESTROGEN_RESPONSE ~300× harder than cell-cycle; lumB loads cell-cycle ~45× harder than ER |
| Full Hallmark catalog rollup, 50 sets (v0.6) | All v0.5 top pathways stay in the top-3 out of 50, on both cohorts. The 5-set rollup wasn't an artifact |
| **Pathway-pole attention, learnable (v0.7 Phase A)** | **Attention collapsed to uniform (top-5 weights 0.0203–0.0205 vs uniform 0.0200). 0/3 top-3 pathway overlap with v0.6 IG on either pole. AUROC roughly neutral. Documented failure mode of softmax-over-standardized-inputs; Phase B retries with raw expression + warmer init.** |

**The honest takeaway, in nine acts:**

1. **Baseline saturated the easy signal.** Plain LogReg on
   concat(RNA, methylation) lands at 0.963 AUROC on the 417-patient
   LumA vs LumB cohort. Within-luminal class structure is already strongly
   expressed in the bulk transcriptome; there isn't much room for an
   architectural upgrade to beat it.

2. **Hypothesis-conditioned attention did NOT lift AUROC.** DMOI's primary
   metric ties baseline within noise (Δ ≈ −0.002 on the full v0.1 cohort).
   The dual-perspective architecture's value is **not** in the headline
   number — the LogReg ceiling reflects the structure of the data, not a
   model limitation we can engineer around.

3. **Calibration was the v0.1 win.** Temperature scaling on a held-out
   15% nested calibration split cuts ECE roughly in half on TCGA (0.138 →
   0.077). T < 1 — the architecture is *under*-confident, not over-confident.

4. **External generalization was the v0.2 win.** A truly held-out TCGA test
   split (random_state=2024, never seen during model dev) scores AUROC 0.968.
   An independent cohort (METABRIC, n=1,175) with the methylation branch
   silenced (METABRIC has no HM450 data) scores AUROC 0.909. Calibration
   parameters do NOT transfer naively between cohorts: TCGA's T=0.634
   over-sharpens the meth-silenced METABRIC predictions; METABRIC's own
   cal-split-fit T=0.934 is correctly close to 1.0.

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
   check — v0.6.** v0.5 rolled IG into the same 5 Hallmark sets that
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

9. **Learnable pathway-pole attention collapsed to uniform under naive
   Variant D — v0.7 Phase A honest negative.** v0.7 attempted to
   replace the hand-picked v0.6 pole masks with a `softmax`-normalized
   learnable attention over all 50 Hallmark pathways (Variant D from
   [`docs/v0.7-design-pathway-attention.md`](docs/v0.7-design-pathway-attention.md)).
   After 15 epochs of training the learned attention weights spanned
   0.0203–0.0205 across all 50 pathways and both poles — effectively
   uniform (1/50 = 0.0200). Zero of the v0.6 top-3 pathways made the
   v0.7 top-3 on either pole. TCGA AUROC slipped 1.1pp to 0.957;
   METABRIC AUROC moved up 0.4pp to 0.913 (roughly neutral).
   Mechanism diagnosed in [`audit/dmoi_v0.7.md`](audit/dmoi_v0.7.md):
   pathway scores were standardized to zero mean, so uniform attention
   produced a near-zero per-pole feature, so the classifier head
   learned to ignore that input, so gradient stopped flowing back to
   the attention logits — a self-reinforcing equilibrium of
   uselessness. v0.6 remains the canonical architecture. **Phase B
   (v0.7.1) retries with raw (un-standardized) pathway expression and
   warmer init to test whether the architecture can learn the
   alignment when the gradient signal isn't crushed.**

---

## Architecture (one paragraph)

Two pole-specific input masks (LumA, LumB) are derived from MSigDB Hallmark
gene sets — `ESTROGEN_RESPONSE_EARLY` + `ESTROGEN_RESPONSE_LATE` for LumA;
`E2F_TARGETS` + `G2M_CHECKPOINT` + `MYC_TARGETS_V1` for LumB — and the HM450
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
| **held-out cal split (honest, ship)** | **0.673 ± 0.294** | **0.121** | T never saw the val data |

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

Test ≥ CV is unusual but in the right direction — the model isn't overfitting
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
is the methylation-silencing residual** — the meth branch normally
contributes positive signal for harder LumB calls, and without it some
borderline cases are unrecoverable from RNA alone.

### What METABRIC validation does and does NOT show

It DOES show that the hypothesis-conditioned RNA encoder generalizes to an
independent cohort across platforms (HiSeq → Illumina HT-12 v3) with
quantile normalization and gene-symbol harmonization (16,890 shared genes
out of 20,530 TCGA / 20,384 METABRIC unique Hugo symbols).

It does NOT validate the dual-modality story — no public BRCA cohort has
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
  in LumA. BCL2 — the canonical anti-apoptotic luminal marker — ranks
  second. The rest of the top-5 (PDLIM3, TUBB2B, EGR3) are cytoskeletal /
  early-response markers that distinguish LumA's lower-proliferation
  phenotype from LumB.
- **lumB pole learned cell-cycle + DNA-repair machinery.** RANBP1 (nuclear
  transport during mitosis), NBN (nibrin / DNA damage response), ZW10
  (mitotic checkpoint), and POLA2 (DNA polymerase α subunit) are all
  proliferation- and replication-stress genes. Not the textbook
  MKI67/TOP2A/AURKA but biologically equivalent — many gene proxies
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
| final_logit | 0.0205 | 0.3425 | one outlier — likely from the disagreement scalar `\|s_LumA − (1 − s_LumB)\|` which has a non-differentiable `abs()` at 0; the pole-specific attributions are the recommended clinical-interpretability headline |

Full per-patient + global lists:
[`audit/dmoi_explain_v0.3.md`](audit/dmoi_explain_v0.3.md),
[`audit/dmoi_explain_per_patient.tsv`](audit/dmoi_explain_per_patient.tsv)
(5,040 rows = 84 × 3 × 2 × top-10), [`audit/dmoi_explain_global.tsv`](audit/dmoi_explain_global.tsv).

---

## Cross-cohort attribution (v0.4, METABRIC n=1,175)

Same IG pipeline as v0.3, applied to the METABRIC external cohort with
the methylation branch silenced (METABRIC has no HM450). Validates
whether the model's biology — not just its AUROC — generalizes.

### Cross-cohort top-K agreement

| Target | Jaccard top-10 | Jaccard top-50 | Verdict |
|---|---|---|---|
| **lumA_pole** | **0.667** | **0.786** | Strong — biology generalizes |
| **lumB_pole** | **0.667** | 0.538 | Strong — biology generalizes |
| final_logit | 0.538 | 0.724 | Moderate; final logit has the disagreement-scalar instability |

### Shared top-10 lumA pole genes (TCGA test ∩ METABRIC)

`FOXC1`, `BCL2`, `PDLIM3`, `TUBB2B`, `KRT15`, `EGR3`, `RAB17`, `AHNAK` — every lumA
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
placeholder (its score is ~75× smaller than the dominant ER pathways) —
present in the top-3 only because there are only five pathways in the
aggregation.

### Honest scope (v0.5)

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
| **lumA_pole** | `ESTROGEN_RESPONSE_EARLY`, `ESTROGEN_RESPONSE_LATE`, `IL2_STAT5_SIGNALING` | `ESTROGEN_RESPONSE_EARLY`, `ESTROGEN_RESPONSE_LATE`, `IL2_STAT5_SIGNALING` | **Yes** — both ER sets in top-3 on both cohorts |
| **lumB_pole** | `MYC_TARGETS_V1`, `E2F_TARGETS`, `G2M_CHECKPOINT` | `E2F_TARGETS`, `G2M_CHECKPOINT`, `MYC_TARGETS_V1` | **Yes** — same 3 cell-cycle sets dominate both cohorts (rank order swaps within a near-tie) |
| `final_logit` | `ESTROGEN_RESPONSE_EARLY`, `ESTROGEN_RESPONSE_LATE`, `G2M_CHECKPOINT` | `ESTROGEN_RESPONSE_EARLY`, `ESTROGEN_RESPONSE_LATE`, `G2M_CHECKPOINT` | **Yes** — identical top-3 across cohorts |

The 5-set v0.5 finding wasn't an artifact of which sets were loaded.

### New honest secondary findings (only visible with the full catalog)

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
  pole's top-5 — the architectural prior catches the biology that's
  actually there.

### Pathway-level pole-specificity (full 50-set view)

| Pole | Cohort | Top pathway (mean \|IG\|) | Top "opposite-program" pathway (mean \|IG\|) | Ratio |
|---|---|---|---|---|
| **lumA_pole** | TCGA test | `ESTROGEN_RESPONSE_EARLY` (0.00629) | `MYC_TARGETS_V1` (in lumA-top-50 lower tier; see CSV) | dominant ER |
| **lumA_pole** | METABRIC  | `ESTROGEN_RESPONSE_EARLY` (0.00681) | same                                                     | dominant ER |
| **lumB_pole** | TCGA test | `MYC_TARGETS_V1` (0.00328)         | `ESTROGEN_RESPONSE_EARLY` (lumB-tier; see CSV)           | dominant cell-cycle |
| **lumB_pole** | METABRIC  | `E2F_TARGETS` (0.00355)            | same                                                     | dominant cell-cycle |

The exact magnitude ratios drift run-to-run (because each version
re-trains a fresh model), but the **structure** — lumA pole stacks ER
at the top, lumB pole stacks cell-cycle at the top — is stable across
v0.5 (5-set) and v0.6 (50-set) on both cohorts.

### Honest scope (v0.6)

- 50 Hallmark sets loaded — the full v2024.1.Hs catalog. The C2 curated
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

## Learnable pathway-pole attention (v0.7 Phase A — honest negative)

v0.7 tried to replace v0.6's hand-picked Hallmark pole masks with a
learnable softmax distribution over all 50 Hallmark pathways per pole
(Variant D in
[`docs/v0.7-design-pathway-attention.md`](docs/v0.7-design-pathway-attention.md)).
The architecture-level question: *if we let the model decide which
pathways define each pole, does it rediscover v0.6's ER-for-LumA /
cell-cycle-for-LumB alignment from scratch?*

Phase A answer: no, but for an instructive mechanical reason.

### Result table

| Cohort | v0.7 Phase A AUROC | v0.6 reference | Δ |
|---|---|---|---|
| TCGA held-out test | 0.957 | 0.968 | −0.011 |
| METABRIC external | 0.913 | 0.909 | +0.004 |

| Pole | v0.7 Phase A learned top-3 | v0.6 IG-derived top-3 | Shared |
|---|---|---|---|
| LumA | INTERFERON_ALPHA, MTORC1, WNT | ER_EARLY, ER_LATE, IL2_STAT5 | **0 / 3** |
| LumB | KRAS_UP, MTORC1, PEROXISOME | E2F, G2M, MYC_V1 | **0 / 3** |

### Mechanism (why collapse?)

The learned attention weights spanned 0.0203–0.0205 across all 50
pathways on both poles — effectively uniform (1/50 = 0.0200). The
"top-3" displayed above are picked among near-tied weights and carry
no signal. The cause is mechanical:

1. v0.7 standardizes per-patient pathway-expression scores (mean → 0,
   std → 1) before feeding them into `PathwayPoleAttention`.
2. With softmax-uniform attention weights, the pole's pathway feature
   becomes `sum_k (1/50) * standardized_score[k]`, which averages to
   ~0 across a batch because standardized scores are zero-centered.
3. The classifier head receives a near-zero pole_pathway_feat for
   every patient. It learns to ignore that input.
4. No downstream signal → no gradient back to `attn_logits` → with
   `wd=1e-4` pulling logits toward zero, the attention stays uniform
   forever.

A self-reinforcing equilibrium of uselessness. The TCGA AUROC drop
(~1.1pp) is consistent with the head having two extra noisy input
features to learn around; METABRIC's 0.4pp move is within noise.

### v0.7.1 (Phase B) — the planned retry

Two small fixes:

- **Drop the StandardScaler on pathway scores.** Use raw mean
  expression. Different Hallmark pathways have different baseline
  means; uniform-attention output is no longer zero-centered, so the
  head has a signal to grip.
- **Bump `PathwayPoleAttention.init_std` from 0.01 to 0.5.** Start the
  attention asymmetrically so gradient has a direction to follow on
  epoch 1.

If Phase B also collapses, the next step is the Variant C upgrade —
project the per-pole pathway feature from a scalar to a vector so the
head has a richer interface into the pathway branch.

### v0.6 remains canonical

Regardless of Phase B outcome, v0.6 (gene-level pole masks + post-hoc
Hallmark IG rollup) remains the canonical DMOI architecture and the
canonical interpretability story. Phase A's negative finding is an
architecture-experiment lesson; it does not invalidate v0.5 / v0.6.

Full report: [`audit/dmoi_v0.7.md`](audit/dmoi_v0.7.md).

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

# 11. (v0.7 Phase A) Learnable pathway-pole attention (architecture extension).
python scripts/eval_dmoi_v0.7.py             # ~7 min on MPS
                                              # writes audit/dmoi_v0.7.md
                                              # Phase A documented softmax-attention collapse;
                                              # Phase B (v0.7.1) re-runs with raw inputs + warmer init.
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
├── eval_dmoi_v0.7.py              # v0.7 Phase A: learnable pathway-pole attention driver
└── check_english_only.py     # CJK gate enforced pre-push
```

---

## What's out of scope for v0.7

See [`docs/what-is-out-of-scope.md`](docs/what-is-out-of-scope.md) for the
full list. Key items still deliberately deferred after v0.7 Phase A:

- **Multi-modal external validation.** No public BRCA cohort outside TCGA
  has paired RNA-seq + HM450 — see the v0.2 design doc for the recon.
- **Other pole hypotheses** (ER−/HER2+, basal vs claudin-low).
- **Variant C / vector-valued per-pole pathway feature.** v0.7 Phase A
  uses a scalar per pole; if Phase B also collapses, the next step is
  projecting the pole pathway feature from scalar → vector to give the
  head a richer interface.
- **Other MSigDB collections.** Hallmark v2024.1.Hs (50 sets) only.
  C2 curated (~5,000 sets) and other collections are not included.
- **Methylation pathway rollup.** The Hallmark aggregation is
  RNA-only; the HM450 probes need a probe → gene crosswalk before a
  methylation pathway view is meaningful.
- **Counterfactual explanations** ("what would need to change to flip the
  prediction") — adversarial-style, much heavier than IG.
- **Nested CV for hyperparameter tuning**. `calibration_frac=0.15` is a
  fixed choice carried over from Guo et al., not swept.

---

## License

MIT. See [`LICENSE`](LICENSE).
