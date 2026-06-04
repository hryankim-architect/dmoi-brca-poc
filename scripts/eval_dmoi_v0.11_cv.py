#!/usr/bin/env python3
"""DMOI v0.11: 5-fold CV stability check for v0.9 Luminal-vs-Basal finding.

v0.9 reported TCGA test AUROC 1.000 + 8/8 expected Hallmark priors in
per-pole IG top-5 on a single 80/20 train/test split. v0.10 reported
METABRIC AUROC 0.965 on a single trained model. Both finding rest on
one fold of TCGA cohort_v3.

The natural skeptic's question: is AUROC = 1.000 a stable feature of
the framework, or a lucky split? v0.11 answers by running 5-fold
StratifiedKFold CV on cohort_v3 (random_state=42, matching the v0.0
baseline CV protocol) and reporting:

  - Per-fold AUROC + balanced accuracy + variance band.
  - Per-fold per-pole IG top-5 with 8/8-priors hit count.
  - Cross-fold pathway stability: how often each of the 8 expected
    priors makes the per-fold pole top-5 (frequency / 5).
  - Cross-fold top-3 Jaccard for both poles.

Scope:
  - Same v0.6 architecture, same v0.9 priors, same hyperparameters.
    Only the train/val split changes per fold.
  - cohort_v3 = 502 patients (Luminal 415, Basal 87). With 5 folds
    each val fold has ~17 Basal patients -- AUROC variance is wider
    than the original 80/20 v0.9 split's n=18 Basal test.
  - No METABRIC scoring in this run; v0.10 already covered cross-
    cohort. v0.11 is purely a v0.9 stability check on TCGA.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone  # noqa: UP017
from pathlib import Path

UTC = timezone.utc  # noqa: UP017

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dmoi_brca.attribution import integrated_gradients_dmoi  # noqa: E402
from dmoi_brca.features import load_features  # noqa: E402
from dmoi_brca.hallmark import load_hallmark_gmt  # noqa: E402
from dmoi_brca.hypothesis_attention import (  # noqa: E402
    load_hm450_cis_mapping,
    make_pole_masks,
)
from dmoi_brca.pathway import pathway_aggregate, rank_pathways  # noqa: E402
from dmoi_brca.priors import POLE_BASAL, POLE_LUMINAL  # noqa: E402
from dmoi_brca.train import aggregate_fold_results, run_dmoi_cv  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TCGA = REPO / "data" / "tcga_brca"
AUDIT = REPO / "audit"
HALLMARK_GMT = REPO / "data" / "msigdb" / "h.all.v2024.1.Hs.symbols.gmt"

POLE_ORDER = ("Luminal", "Basal")
POSITIVE_LABEL = "Basal"
COHORT_TSV_NAME = "cohort_v3.tsv"

# Reuse v0.9 final hyperparameters exactly.
FINAL_KWARGS = dict(
    latent_dim=128, rna_hidden=(1024, 256), meth_hidden=(512,),
    fuse_hidden=(128,), fuse_out=64, head_hidden=32, dropout=0.3,
    batch_size=64, lr=1e-4, weight_decay=1e-4,
    seed=42, device="auto", verbose=False,
    use_disagreement=True, aux_weight=0.3,
    # CV-aware: pick best epoch (val fold is a real val fold here, NOT
    # held-out test as in v0.9 single-split). pick_best_epoch=True is
    # the standard CV protocol matching v0.0 baseline.
    pick_best_epoch=True,
)
N_EPOCHS = 15
N_IG_STEPS = 50

EXPECTED_LUMINAL_TOP = {
    "HALLMARK_ESTROGEN_RESPONSE_EARLY",
    "HALLMARK_ESTROGEN_RESPONSE_LATE",
    "HALLMARK_ANDROGEN_RESPONSE",
}
EXPECTED_BASAL_TOP = {
    "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",
    "HALLMARK_MYC_TARGETS_V1",
    "HALLMARK_G2M_CHECKPOINT",
    "HALLMARK_E2F_TARGETS",
    "HALLMARK_MYC_TARGETS_V2",
}


def _per_fold_ig_rollup(
    result, feats_rna, feats_meth, feats_rna_features,
    val_idx, hallmark, device,
) -> dict[str, list]:
    """Compute Luminal_pole + Basal_pole IG Hallmark rollup on val fold."""
    rna_val_raw = feats_rna[val_idx]
    meth_val_raw = feats_meth[val_idx]
    rna_val_std = result.rna_scaler.transform(rna_val_raw).astype(np.float32)
    meth_val_std = result.meth_scaler.transform(meth_val_raw).astype(np.float32)
    out: dict[str, list] = {}
    for tname, attr_t in (("Luminal_pole", "lumA_pole"), ("Basal_pole", "lumB_pole")):
        attr = integrated_gradients_dmoi(
            result.model, rna_val_std, meth_val_std,
            target=attr_t, n_steps=N_IG_STEPS, device=str(device),
            pole_order=POLE_ORDER,
        )
        scores = pathway_aggregate(
            attr.rna_attribution, feats_rna_features, hallmark,
        )
        out[tname] = scores
    return out


def main() -> int:
    cohort_tsv = TCGA / COHORT_TSV_NAME
    rna_gz = TCGA / "HiSeqV2.gz"
    meth_gz = TCGA / "HumanMethylation450.gz"
    probemap = TCGA / "hm450_probemap.tsv"

    for p in (cohort_tsv, rna_gz, meth_gz, probemap, HALLMARK_GMT):
        if not p.exists():
            sys.stderr.write(f"ERROR: missing input {p}\n")
            return 1

    print("=== DMOI v0.11: 5-fold CV stability check for v0.9 finding ===")

    # --- Hallmark catalog ---
    print(f"\n--- Loading Hallmark catalog: {HALLMARK_GMT.name} ---")
    hallmark: dict[str, list[str]] = load_hallmark_gmt(HALLMARK_GMT)
    print(f"  {len(hallmark)} Hallmark sets loaded.")

    # --- Load all 502 TCGA cohort_v3 patients (no train/test split) ---
    print(f"\n--- Loading TCGA features from {COHORT_TSV_NAME} ---")
    feats_all = load_features(
        cohort_tsv=cohort_tsv, rna_gz=rna_gz, meth_gz=meth_gz,
        meth_topk=10_000, dual_modality_only=True,
        positive_label=POSITIVE_LABEL,
    )
    print(f"  cohort: n={len(feats_all.sample_ids)} "
          f"(Basal={int(feats_all.y.sum())}, "
          f"Luminal={len(feats_all.y) - int(feats_all.y.sum())})")

    # --- Pole masks ---
    print("\n--- Building Luminal / Basal pole masks ---")
    cis = load_hm450_cis_mapping(probemap)
    pole_masks = make_pole_masks(
        feats_all.rna_features, feats_all.meth_features, cis,
        {"Luminal": POLE_LUMINAL, "Basal": POLE_BASAL},
        hallmark_sets={k: list(v) for k, v in hallmark.items()},
    )
    for pname, p in pole_masks.items():
        print(f"  {pname:10s} {p.summary()}")

    # --- 5-fold CV ---
    print("\n--- Running 5-fold CV ---")
    # Note: FINAL_KWARGS contains verbose=False (for train_one_fold's per-epoch
    # silence); run_dmoi_cv inherits that same verbose value via the kwarg
    # spread, so the between-fold prints are suppressed. Our explicit
    # per-fold loop below prints AUROC + bacc + priors-hit so progress is
    # visible without epoch spam.
    # run_dmoi_cv supplies rna_dim/meth_dim/fold automatically per fold;
    # we only pass the v0.9-specific tuning kwargs + pole_order overrides.
    results = run_dmoi_cv(
        rna=feats_all.rna, meth=feats_all.meth, y=feats_all.y,
        pole_masks=pole_masks,
        n_splits=5, random_state=42,
        n_epochs=N_EPOCHS, patience=N_EPOCHS + 1,
        keep_artifacts=True,
        pole_order=POLE_ORDER,
        **FINAL_KWARGS,
    )

    # --- Aggregate AUROC + bacc ---
    agg = aggregate_fold_results(results)
    print(f"\n--- 5-fold CV aggregate ---")
    print(f"  AUROC : mean={agg['auc_mean']:.4f}  std={agg['auc_std']:.4f}")
    print(f"  bacc  : mean={agg['bacc_mean']:.4f} std={agg['bacc_std']:.4f}")

    # --- Per-fold IG + Hallmark rollup + priors hit count ---
    print("\n--- Per-fold IG + Hallmark rollup + priors-hit check ---")
    from sklearn.model_selection import StratifiedKFold  # noqa: E402
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_splits = list(skf.split(np.zeros(len(feats_all.y)), feats_all.y))
    per_fold_ig: list[dict] = []
    per_fold_hits: list[tuple[int, int]] = []
    per_fold_top3_luminal: list[set] = []
    per_fold_top3_basal: list[set] = []
    import torch  # noqa: E402
    for fi, (result, (_tr_idx, te_idx)) in enumerate(zip(results, fold_splits, strict=False), start=1):
        if result.model is None:
            sys.stderr.write(f"ERROR: fold {fi} returned no model.\n")
            return 1
        device = next(result.model.parameters()).device
        rollup = _per_fold_ig_rollup(
            result, feats_all.rna, feats_all.meth, feats_all.rna_features,
            te_idx, hallmark, device,
        )
        per_fold_ig.append(rollup)
        lum_top5 = {
            s.pathway_name for s in
            rank_pathways(rollup["Luminal_pole"], by="mean_abs_ig")[:5]
        }
        bas_top5 = {
            s.pathway_name for s in
            rank_pathways(rollup["Basal_pole"], by="mean_abs_ig")[:5]
        }
        lum_top3 = {
            s.pathway_name for s in
            rank_pathways(rollup["Luminal_pole"], by="mean_abs_ig")[:3]
        }
        bas_top3 = {
            s.pathway_name for s in
            rank_pathways(rollup["Basal_pole"], by="mean_abs_ig")[:3]
        }
        lum_hits = len(lum_top5 & EXPECTED_LUMINAL_TOP)
        bas_hits = len(bas_top5 & EXPECTED_BASAL_TOP)
        per_fold_hits.append((lum_hits, bas_hits))
        per_fold_top3_luminal.append(lum_top3)
        per_fold_top3_basal.append(bas_top3)
        print(f"  fold {fi}: AUROC={result.best_val_auc:.4f} bacc={result.best_val_bacc:.4f} "
              f"Lum-hits={lum_hits}/3 Bas-hits={bas_hits}/5")

    # --- Cross-fold pathway-frequency table ---
    # How often does each expected prior appear in per-fold pole top-5?
    print("\n--- Cross-fold pathway frequency (out of 5 folds) ---")
    lum_freq: dict[str, int] = dict.fromkeys(EXPECTED_LUMINAL_TOP, 0)
    bas_freq: dict[str, int] = dict.fromkeys(EXPECTED_BASAL_TOP, 0)
    for rollup in per_fold_ig:
        lum_top5 = {
            s.pathway_name for s in
            rank_pathways(rollup["Luminal_pole"], by="mean_abs_ig")[:5]
        }
        bas_top5 = {
            s.pathway_name for s in
            rank_pathways(rollup["Basal_pole"], by="mean_abs_ig")[:5]
        }
        for p in EXPECTED_LUMINAL_TOP:
            if p in lum_top5:
                lum_freq[p] += 1
        for p in EXPECTED_BASAL_TOP:
            if p in bas_top5:
                bas_freq[p] += 1
    print("  Luminal expected priors (freq / 5):")
    for p, c in sorted(lum_freq.items(), key=lambda x: -x[1]):
        print(f"    {c}/5  {p}")
    print("  Basal expected priors (freq / 5):")
    for p, c in sorted(bas_freq.items(), key=lambda x: -x[1]):
        print(f"    {c}/5  {p}")

    # --- Cross-fold top-3 stability (Jaccard pairwise) ---
    print("\n--- Cross-fold top-3 stability ---")
    n_folds = len(results)

    def _mean_pairwise_jaccard(setlist: list[set]) -> float:
        if n_folds < 2:
            return 0.0
        jsum = 0.0
        npairs = 0
        for i in range(n_folds):
            for j in range(i + 1, n_folds):
                a, b = setlist[i], setlist[j]
                if not a and not b:
                    continue
                jsum += len(a & b) / len(a | b)
                npairs += 1
        return jsum / npairs if npairs else 0.0

    lum_j = _mean_pairwise_jaccard(per_fold_top3_luminal)
    bas_j = _mean_pairwise_jaccard(per_fold_top3_basal)
    print(f"  Luminal top-3 mean pairwise Jaccard : {lum_j:.4f}")
    print(f"  Basal   top-3 mean pairwise Jaccard : {bas_j:.4f}")

    # --- Audit MD ---
    AUDIT.mkdir(exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    md_path = AUDIT / "dmoi_v0.11.md"

    fold_rows = []
    for i, (result, (lum_hits, bas_hits)) in enumerate(
        zip(results, per_fold_hits, strict=False), start=1,
    ):
        fold_rows.append(
            f"| {i} | {result.best_val_auc:.4f} | {result.best_val_bacc:.4f} | "
            f"{lum_hits} / 3 | {bas_hits} / 5 | {result.best_epoch} |"
        )

    md_path.write_text(
        "# DMOI v0.11 -- 5-fold CV stability check for v0.9 Luminal-vs-Basal\n\n"
        f"Generated: {ts}\n\n"
        "## Setup\n\n"
        "- Architecture: v0.6 base (same as v0.9 / v0.10), n_pathways=0.\n"
        "- Cohort: TCGA cohort_v3 (Luminal+Basal dual-modality, "
        f"n={len(feats_all.sample_ids)}).\n"
        "- Split: 5-fold StratifiedKFold (random_state=42, matches the v0.0 "
        "baseline CV protocol).\n"
        f"- Epochs: {N_EPOCHS}, optimizer: AdamW(lr=1e-4, wd=1e-4), "
        "BCEWithLogitsLoss + aux=0.3, pick_best_epoch=True "
        "(standard CV protocol with a real val fold).\n"
        "- v0.6 / v0.9 single-split references: TCGA cohort_v2 5-fold "
        "(LumA-vs-LumB) was 0.954 ± 0.017; TCGA cohort_v3 80/20 single "
        "split (Luminal-vs-Basal, v0.9) was AUROC 1.000 / bacc 0.972.\n\n"
        "## Aggregate AUROC + bacc (5-fold)\n\n"
        f"| Metric | mean | std |\n|---|---|---|\n"
        f"| AUROC | **{agg['auc_mean']:.4f}** | {agg['auc_std']:.4f} |\n"
        f"| bacc  | **{agg['bacc_mean']:.4f}** | {agg['bacc_std']:.4f} |\n\n"
        "## Per-fold table\n\n"
        "| Fold | AUROC | bacc | Luminal IG top-5 ∩ priors | Basal IG top-5 ∩ priors | best epoch |\n"
        "|---|---|---|---|---|---|\n"
        + "\n".join(fold_rows) + "\n\n"
        "## Cross-fold pathway frequency (5-fold stability of v0.9 priors hit)\n\n"
        "### Luminal pole -- frequency in per-fold top-5 (out of 5 folds)\n\n"
        "| Pathway | Frequency |\n|---|---|\n"
        + "\n".join(
            f"| `{p}` | {c} / 5 |"
            for p, c in sorted(lum_freq.items(), key=lambda x: -x[1])
        )
        + "\n\n"
        "### Basal pole -- frequency in per-fold top-5 (out of 5 folds)\n\n"
        "| Pathway | Frequency |\n|---|---|\n"
        + "\n".join(
            f"| `{p}` | {c} / 5 |"
            for p, c in sorted(bas_freq.items(), key=lambda x: -x[1])
        )
        + "\n\n"
        "## Cross-fold top-3 stability (pairwise mean Jaccard)\n\n"
        f"- Luminal pole top-3 mean pairwise Jaccard : **{lum_j:.4f}**\n"
        f"- Basal   pole top-3 mean pairwise Jaccard : **{bas_j:.4f}**\n\n"
        "Jaccard of 1.0 means every fold picked the same top-3.\n"
        "Jaccard of 0.5 means top-3 sets overlap in 2 of 3 pathways "
        "(or, equivalently, 2 of 4 in symmetric-difference terms).\n\n"
        "## Reading\n\n"
        "v0.11 quantifies the natural skeptic's question about v0.9's "
        "AUROC = 1.000 on the single 80/20 split:\n\n"
        f"- If AUROC mean is >= 0.99 and std <= 0.015, the v0.9 finding "
        "is decisively stable.\n"
        f"- If priors-hit frequency is 5/5 for all 8 expected pathways, "
        "v0.9's per-pole biology recovery is fold-invariant.\n"
        f"- If top-3 Jaccard is >= 0.8, the cohort_v3 / Luminal-vs-Basal "
        "task is structurally easy enough that the top-3 is essentially "
        "fixed -- consistent with the cohort_v3 LogReg baseline being "
        "near-saturated and the gene-level architecture commitment "
        "carrying that signal cleanly.\n\n"
        "## Scope\n\n"
        "- Same architecture and priors as v0.9. Only the train/val "
        "split changes across folds.\n"
        "- pick_best_epoch=True is the standard CV protocol; v0.9 used "
        "pick_best_epoch=False because val was a held-out test split.\n"
        "- Each val fold has ~17 Basal patients; AUROC variance is "
        "wider than the v0.9 single-split test (n=18 Basal) but the "
        "variance band is the actual deliverable.\n"
        "- No METABRIC scoring here; v0.10 already validated cross-\n"
        "  cohort. v0.11 is purely a v0.9 TCGA stability check.\n\n"
        "## Reproduce\n\n"
        "```bash\n"
        "python scripts/eval_dmoi_v0.11_cv.py\n"
        "```\n",
    )
    print(f"\nWrote {md_path}")

    print("\n=== DMOI v0.11 summary ===")
    print(f"  AUROC mean ± std : {agg['auc_mean']:.4f} ± {agg['auc_std']:.4f} "
          f"(v0.9 single-split: 1.000)")
    print(f"  bacc  mean ± std : {agg['bacc_mean']:.4f} ± {agg['bacc_std']:.4f} "
          f"(v0.9 single-split: 0.972)")
    n_priors_5of5_lum = sum(1 for c in lum_freq.values() if c == 5)
    n_priors_5of5_bas = sum(1 for c in bas_freq.values() if c == 5)
    print(f"  Luminal priors 5/5 in top-5 (out of 3): {n_priors_5of5_lum} / 3")
    print(f"  Basal   priors 5/5 in top-5 (out of 5): {n_priors_5of5_bas} / 5")
    print(f"  Luminal top-3 mean pairwise Jaccard : {lum_j:.4f}")
    print(f"  Basal   top-3 mean pairwise Jaccard : {bas_j:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
