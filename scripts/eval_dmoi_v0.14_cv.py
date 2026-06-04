#!/usr/bin/env python3
"""DMOI v0.14 CV: 5-fold stability check for the HER2-vs-Luminal finding.

eval_dmoi_v0.14.py reported TCGA held-out test AUROC 0.891 on a single 80/20
split with only 12 HER2 test patients, plus METABRIC external AUROC 0.893.
The single TCGA split is small; this script reports a 5-fold StratifiedKFold
band (random_state=42) so the TCGA headline is a mean±std, not one fold.

Reports, mirroring v0.11:
  - Per-fold AUROC + balanced accuracy + variance band.
  - Per-fold per-pole IG top-5 expected-priors hit count
    (Luminal: ER_EARLY/LATE = 2; HER2: PI3K_AKT_MTOR/MTORC1/G2M = 3).
  - Cross-fold pathway frequency (how often each prior makes per-fold top-5).
  - Cross-fold top-3 Jaccard for both poles.

Scope:
  - Same v0.6 architecture, same v0.14 priors, same hyperparameters; only the
    train/val split changes per fold.
  - cohort_v4 = 436 dual-modality patients (Luminal 378, HER2 58). With 5 folds
    each val fold has ~12 HER2 patients -- AUROC variance is wide; the band IS
    the deliverable. METABRIC (n=224 HER2, eval_dmoi_v0.14.py) carries the
    cross-cohort weight.
  - No METABRIC scoring here -- this is purely a TCGA stability check.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone  # noqa: UP017
from pathlib import Path

UTC = timezone.utc  # noqa: UP017

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from dmoi_brca import audit  # noqa: E402
from dmoi_brca.attribution import integrated_gradients_dmoi  # noqa: E402
from dmoi_brca.features import load_features  # noqa: E402
from dmoi_brca.hallmark import load_hallmark_gmt  # noqa: E402
from dmoi_brca.hypothesis_attention import (  # noqa: E402
    load_hm450_cis_mapping,
    make_pole_masks,
)
from dmoi_brca.pathway import pathway_aggregate, rank_pathways  # noqa: E402
from dmoi_brca.priors import POLE_HER2, POLE_LUMINAL_ER  # noqa: E402
from dmoi_brca.train import aggregate_fold_results, run_dmoi_cv  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TCGA = REPO / "data" / "tcga_brca"
AUDIT = REPO / "audit"
HALLMARK_GMT = REPO / "data" / "msigdb" / "h.all.v2024.1.Hs.symbols.gmt"

POLE_ORDER = ("Luminal", "HER2")
POSITIVE_LABEL = "HER2"
COHORT_TSV_NAME = "cohort_v4.tsv"
JOB_ID = "dmoi-her2-axis-cv-v0.14"

FINAL_KWARGS = dict(
    latent_dim=128, rna_hidden=(1024, 256), meth_hidden=(512,),
    fuse_hidden=(128,), fuse_out=64, head_hidden=32, dropout=0.3,
    batch_size=64, lr=1e-4, weight_decay=1e-4,
    seed=42, device="auto", verbose=False,
    use_disagreement=True, aux_weight=0.3,
    pick_best_epoch=True,  # standard CV protocol (real val fold)
)
N_EPOCHS = 15
N_IG_STEPS = 50

EXPECTED_LUMINAL_TOP = {
    "HALLMARK_ESTROGEN_RESPONSE_EARLY",
    "HALLMARK_ESTROGEN_RESPONSE_LATE",
}
EXPECTED_HER2_TOP = {
    "HALLMARK_PI3K_AKT_MTOR_SIGNALING",
    "HALLMARK_MTORC1_SIGNALING",
    "HALLMARK_G2M_CHECKPOINT",
}


def _per_fold_ig_rollup(
    result, feats_rna, feats_meth, feats_rna_features,
    val_idx, hallmark, device,
) -> dict[str, list]:
    rna_val_std = result.rna_scaler.transform(feats_rna[val_idx]).astype(np.float32)
    meth_val_std = result.meth_scaler.transform(feats_meth[val_idx]).astype(np.float32)
    out: dict[str, list] = {}
    for tname, attr_t in (("Luminal_pole", "lumA_pole"), ("HER2_pole", "lumB_pole")):
        attr = integrated_gradients_dmoi(
            result.model, rna_val_std, meth_val_std,
            target=attr_t, n_steps=N_IG_STEPS, device=str(device),
            pole_order=POLE_ORDER,
        )
        out[tname] = pathway_aggregate(
            attr.rna_attribution, feats_rna_features, hallmark,
        )
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

    print("=== DMOI v0.14 CV: 5-fold stability for HER2-vs-Luminal ===")

    print(f"\n--- Loading Hallmark catalog: {HALLMARK_GMT.name} ---")
    hallmark: dict[str, list[str]] = load_hallmark_gmt(HALLMARK_GMT)
    print(f"  {len(hallmark)} Hallmark sets loaded.")

    print(f"\n--- Loading TCGA features from {COHORT_TSV_NAME} ---")
    feats_all = load_features(
        cohort_tsv=cohort_tsv, rna_gz=rna_gz, meth_gz=meth_gz,
        meth_topk=10_000, dual_modality_only=True,
        positive_label=POSITIVE_LABEL,
    )
    print(f"  cohort: n={len(feats_all.sample_ids)} "
          f"(HER2={int(feats_all.y.sum())}, "
          f"Luminal={len(feats_all.y) - int(feats_all.y.sum())})")

    print("\n--- Building Luminal / HER2 pole masks ---")
    cis = load_hm450_cis_mapping(probemap)
    pole_masks = make_pole_masks(
        feats_all.rna_features, feats_all.meth_features, cis,
        {"Luminal": POLE_LUMINAL_ER, "HER2": POLE_HER2},
        hallmark_sets={k: list(v) for k, v in hallmark.items()},
    )
    for pname, pmask in pole_masks.items():
        print(f"  {pname:10s} {pmask.summary()}")

    print("\n--- Running 5-fold CV ---")
    results = run_dmoi_cv(
        rna=feats_all.rna, meth=feats_all.meth, y=feats_all.y,
        pole_masks=pole_masks,
        n_splits=5, random_state=42,
        n_epochs=N_EPOCHS, patience=N_EPOCHS + 1,
        keep_artifacts=True,
        pole_order=POLE_ORDER,
        **FINAL_KWARGS,
    )

    agg = aggregate_fold_results(results)
    print("\n--- 5-fold CV aggregate ---")
    print(f"  AUROC : mean={agg['auc_mean']:.4f}  std={agg['auc_std']:.4f}")
    print(f"  bacc  : mean={agg['bacc_mean']:.4f} std={agg['bacc_std']:.4f}")

    print("\n--- Per-fold IG + Hallmark rollup + priors-hit check ---")
    from sklearn.model_selection import StratifiedKFold  # noqa: PLC0415
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_splits = list(skf.split(np.zeros(len(feats_all.y)), feats_all.y))
    per_fold_ig: list[dict] = []
    per_fold_hits: list[tuple[int, int]] = []
    per_fold_top3_luminal: list[set] = []
    per_fold_top3_her2: list[set] = []
    for fi, (result, (_tr_idx, te_idx)) in enumerate(
        zip(results, fold_splits, strict=False), start=1,
    ):
        if result.model is None:
            sys.stderr.write(f"ERROR: fold {fi} returned no model.\n")
            return 1
        device = next(result.model.parameters()).device
        rollup = _per_fold_ig_rollup(
            result, feats_all.rna, feats_all.meth, feats_all.rna_features,
            te_idx, hallmark, device,
        )
        per_fold_ig.append(rollup)
        lum_top5 = {s.pathway_name for s in
                    rank_pathways(rollup["Luminal_pole"], by="mean_abs_ig")[:5]}
        her2_top5 = {s.pathway_name for s in
                     rank_pathways(rollup["HER2_pole"], by="mean_abs_ig")[:5]}
        per_fold_top3_luminal.append({s.pathway_name for s in
                                      rank_pathways(rollup["Luminal_pole"], by="mean_abs_ig")[:3]})
        per_fold_top3_her2.append({s.pathway_name for s in
                                   rank_pathways(rollup["HER2_pole"], by="mean_abs_ig")[:3]})
        lum_hits = len(lum_top5 & EXPECTED_LUMINAL_TOP)
        her2_hits = len(her2_top5 & EXPECTED_HER2_TOP)
        per_fold_hits.append((lum_hits, her2_hits))
        print(f"  fold {fi}: AUROC={result.best_val_auc:.4f} "
              f"bacc={result.best_val_bacc:.4f} "
              f"Lum-hits={lum_hits}/2 HER2-hits={her2_hits}/3")

    print("\n--- Cross-fold pathway frequency (out of 5 folds) ---")
    lum_freq: dict[str, int] = dict.fromkeys(EXPECTED_LUMINAL_TOP, 0)
    her2_freq: dict[str, int] = dict.fromkeys(EXPECTED_HER2_TOP, 0)
    for rollup in per_fold_ig:
        lum_top5 = {s.pathway_name for s in
                    rank_pathways(rollup["Luminal_pole"], by="mean_abs_ig")[:5]}
        her2_top5 = {s.pathway_name for s in
                     rank_pathways(rollup["HER2_pole"], by="mean_abs_ig")[:5]}
        for p in EXPECTED_LUMINAL_TOP:
            if p in lum_top5:
                lum_freq[p] += 1
        for p in EXPECTED_HER2_TOP:
            if p in her2_top5:
                her2_freq[p] += 1
    print("  Luminal expected priors (freq / 5):")
    for p, c in sorted(lum_freq.items(), key=lambda x: -x[1]):
        print(f"    {c}/5  {p}")
    print("  HER2 expected priors (freq / 5):")
    for p, c in sorted(her2_freq.items(), key=lambda x: -x[1]):
        print(f"    {c}/5  {p}")

    print("\n--- Cross-fold top-3 stability ---")
    n_folds = len(results)

    def _mean_pairwise_jaccard(setlist: list[set]) -> float:
        if n_folds < 2:
            return 0.0
        jsum, npairs = 0.0, 0
        for i in range(n_folds):
            for j in range(i + 1, n_folds):
                a, b = setlist[i], setlist[j]
                if not a and not b:
                    continue
                jsum += len(a & b) / len(a | b)
                npairs += 1
        return jsum / npairs if npairs else 0.0

    lum_j = _mean_pairwise_jaccard(per_fold_top3_luminal)
    her2_j = _mean_pairwise_jaccard(per_fold_top3_her2)
    print(f"  Luminal top-3 mean pairwise Jaccard : {lum_j:.4f}")
    print(f"  HER2    top-3 mean pairwise Jaccard : {her2_j:.4f}")

    AUDIT.mkdir(exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    md_path = AUDIT / "dmoi_v0.14_cv.md"
    fold_rows = []
    for i, (result, (lum_hits, her2_hits)) in enumerate(
        zip(results, per_fold_hits, strict=False), start=1,
    ):
        fold_rows.append(
            f"| {i} | {result.best_val_auc:.4f} | {result.best_val_bacc:.4f} | "
            f"{lum_hits} / 2 | {her2_hits} / 3 | {result.best_epoch} |"
        )

    md_path.write_text(
        "# DMOI v0.14 CV -- 5-fold stability check for HER2-vs-Luminal\n\n"
        f"Generated: {ts}\n\n"
        "## Setup\n\n"
        "- Architecture: v0.6 base (same as v0.14 single-split), n_pathways=0.\n"
        "- Cohort: TCGA cohort_v4 (HER2 + Luminal dual-modality, "
        f"n={len(feats_all.sample_ids)}; HER2={int(feats_all.y.sum())}, "
        f"Luminal={len(feats_all.y) - int(feats_all.y.sum())}).\n"
        "- Split: 5-fold StratifiedKFold (random_state=42), pick_best_epoch=True.\n"
        "- POLE_HER2 = PI3K_AKT_MTOR + MTORC1 + G2M_CHECKPOINT; "
        "POLE_LUMINAL_ER = ER_EARLY + ER_LATE.\n"
        "- v0.14 single-split references: TCGA test AUROC 0.891 / bacc 0.849 "
        "(n_test=88, HER2=12); METABRIC external AUROC 0.893.\n\n"
        "## Aggregate AUROC + bacc (5-fold)\n\n"
        "| Metric | mean | std |\n|---|---|---|\n"
        f"| AUROC | **{agg['auc_mean']:.4f}** | {agg['auc_std']:.4f} |\n"
        f"| bacc  | **{agg['bacc_mean']:.4f}** | {agg['bacc_std']:.4f} |\n\n"
        "## Per-fold table\n\n"
        "| Fold | AUROC | bacc | Luminal IG top-5 ∩ priors | HER2 IG top-5 ∩ priors | best epoch |\n"
        "|---|---|---|---|---|---|\n"
        + "\n".join(fold_rows) + "\n\n"
        "## Cross-fold pathway frequency\n\n"
        "### Luminal pole -- frequency in per-fold top-5 (out of 5)\n\n"
        "| Pathway | Frequency |\n|---|---|\n"
        + "\n".join(f"| `{p}` | {c} / 5 |"
                    for p, c in sorted(lum_freq.items(), key=lambda x: -x[1]))
        + "\n\n"
        "### HER2 pole -- frequency in per-fold top-5 (out of 5)\n\n"
        "| Pathway | Frequency |\n|---|---|\n"
        + "\n".join(f"| `{p}` | {c} / 5 |"
                    for p, c in sorted(her2_freq.items(), key=lambda x: -x[1]))
        + "\n\n"
        "## Cross-fold top-3 stability (pairwise mean Jaccard)\n\n"
        f"- Luminal pole top-3 mean pairwise Jaccard : **{lum_j:.4f}**\n"
        f"- HER2    pole top-3 mean pairwise Jaccard : **{her2_j:.4f}**\n\n"
        "## Scope\n\n"
        "- Same architecture and priors as v0.14 single-split. Only the "
        "train/val split changes across folds.\n"
        "- HER2 is the small class (~12 per val fold); the AUROC band is wider "
        "than a larger-cohort axis would give -- that width is the honest "
        "deliverable, and METABRIC (n=224 HER2) carries the cross-cohort weight.\n"
        "- No METABRIC scoring here; eval_dmoi_v0.14.py covers cross-cohort. "
        "v0.14 CV is purely a TCGA stability check.\n\n"
        "## Reproduce\n\n"
        "```bash\n"
        "python scripts/build_cohort_v4.py     # if cohort_v4.tsv not built\n"
        "python scripts/eval_dmoi_v0.14_cv.py\n"
        "```\n",
    )
    print(f"\nWrote {md_path}")

    audit.emit(
        "her2_axis_cv_v0.14", JOB_ID,
        fields={
            "auc_mean": agg["auc_mean"], "auc_std": agg["auc_std"],
            "bacc_mean": agg["bacc_mean"], "bacc_std": agg["bacc_std"],
            "n_cohort": int(len(feats_all.sample_ids)),
            "luminal_top3_jaccard": lum_j, "her2_top3_jaccard": her2_j,
        },
    )

    print("\n=== DMOI v0.14 CV summary ===")
    print(f"  AUROC mean ± std : {agg['auc_mean']:.4f} ± {agg['auc_std']:.4f} "
          f"(v0.14 single-split: 0.891)")
    print(f"  bacc  mean ± std : {agg['bacc_mean']:.4f} ± {agg['bacc_std']:.4f}")
    print(f"  Luminal priors 5/5 in top-5 (out of 2): "
          f"{sum(1 for c in lum_freq.values() if c == 5)} / 2")
    print(f"  HER2    priors 5/5 in top-5 (out of 3): "
          f"{sum(1 for c in her2_freq.values() if c == 5)} / 3")
    print(f"  Luminal top-3 Jaccard : {lum_j:.4f}   HER2 top-3 Jaccard : {her2_j:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
