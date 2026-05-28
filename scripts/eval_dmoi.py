#!/usr/bin/env python3
"""Day-4 driver: full DMOI evaluation + ablation.

Runs 5-fold CV twice on cohort_v2:
  1. Full DMOI model (with disagreement scalar in classifier head)
  2. Ablation: same model with disagreement removed from classifier head

For each fold of run #1, also computes:
  - Per-class precision / recall / F1 (focuses on LumB minority)
  - Expected Calibration Error (ECE) with 10 reliability bins
  - Disagreement-vs-misclassification analysis (the key test of DMOI's
    Option-B thesis)

Writes:
  audit/dmoi_eval_v0.md          (aggregate + ablation + disagreement analysis)
  audit/dmoi_eval_per_fold.tsv   (per-fold detail for reproducibility)
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone  # noqa: UP017
from pathlib import Path

UTC = timezone.utc  # noqa: UP017

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from dmoi_brca.eval import (  # noqa: E402
    aggregate_cross_fold,
    build_fold_eval_bundle,
    concat_fold_predictions,
    confusion_matrix_table,
)
from dmoi_brca.features import load_features  # noqa: E402
from dmoi_brca.hypothesis_attention import (  # noqa: E402
    load_hm450_cis_mapping,
    make_pole_masks,
)
from dmoi_brca.priors import POLE_LUMA, POLE_LUMB  # noqa: E402
from dmoi_brca.train import aggregate_fold_results, run_dmoi_cv  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "tcga_brca"
AUDIT = REPO / "audit"

# Shared CV + model hyperparameters (mirror Day-3 train_dmoi.py).
COMMON_KWARGS = dict(
    n_splits=5,
    random_state=42,
    latent_dim=128,
    rna_hidden=(1024, 256),
    meth_hidden=(512,),
    fuse_hidden=(128,),
    fuse_out=64,
    head_hidden=32,
    dropout=0.3,
    n_epochs=50,
    batch_size=64,
    lr=1e-4,
    weight_decay=1e-4,
    patience=10,
    seed=42,
    device="auto",
    verbose=False,
)


def main() -> int:
    cohort_tsv = DATA / "cohort_v2.tsv"
    rna_gz = DATA / "HiSeqV2.gz"
    meth_gz = DATA / "HumanMethylation450.gz"
    probemap = DATA / "hm450_probemap.tsv"

    for p in (cohort_tsv, rna_gz, meth_gz, probemap):
        if not p.exists():
            sys.stderr.write(f"ERROR: missing input {p}\n")
            return 1

    print("=== Day-4: DMOI full evaluation + ablation ===")
    print("\n--- Loading features ---")
    feats = load_features(
        cohort_tsv=cohort_tsv, rna_gz=rna_gz, meth_gz=meth_gz,
        meth_topk=10_000, dual_modality_only=True, positive_label="LumB",
    )
    n = len(feats.sample_ids)
    print(f"  Cohort v2 dual-modality: {n} patients "
          f"(LumA={int((feats.y == 0).sum())}, LumB={int((feats.y == 1).sum())})")

    print("\n--- Building pole masks ---")
    cis = load_hm450_cis_mapping(probemap)
    pole_masks = make_pole_masks(
        feats.rna_features, feats.meth_features, cis,
        {"LumA": POLE_LUMA, "LumB": POLE_LUMB},
    )

    # --- Full DMOI (with disagreement) ---
    print("\n--- Run 1/2: Full DMOI (disagreement IN) ---")
    full_results = run_dmoi_cv(
        rna=feats.rna, meth=feats.meth, y=feats.y,
        pole_masks=pole_masks, use_disagreement=True, **COMMON_KWARGS,
    )
    full_agg_train = aggregate_fold_results(full_results)

    # --- Ablation (no disagreement) ---
    print("\n--- Run 2/2: Ablation (disagreement OUT) ---")
    ablation_results = run_dmoi_cv(
        rna=feats.rna, meth=feats.meth, y=feats.y,
        pole_masks=pole_masks, use_disagreement=False, **COMMON_KWARGS,
    )
    ablation_agg_train = aggregate_fold_results(ablation_results)

    # --- Build per-fold eval bundles for the full DMOI run ---
    print("\n--- Analytical eval ---")
    bundles = []
    for r in full_results:
        if r.val_labels is None or r.val_proba is None or r.val_disagreement is None:
            sys.stderr.write(f"WARN: fold {r.fold} missing val arrays; skipping eval bundle\n")
            continue
        bundles.append(build_fold_eval_bundle(
            fold=r.fold,
            labels=r.val_labels,
            proba=r.val_proba,
            disagreement=r.val_disagreement,
        ))

    eval_agg = aggregate_cross_fold(bundles)
    pooled_labels, pooled_proba, pooled_dis = concat_fold_predictions(bundles)
    pooled_pred = (pooled_proba >= 0.5).astype(np.int64)
    pooled_cm = confusion_matrix_table(pooled_labels, pooled_pred)

    # --- Per-fold TSV ---
    AUDIT.mkdir(exist_ok=True)
    per_fold = AUDIT / "dmoi_eval_per_fold.tsv"
    with per_fold.open("w") as f:
        f.write(
            "fold\tauc_full\tauc_ablation\tbacc_full\tbacc_ablation\t"
            "f1_lumA\tf1_lumB\tece\tdis_auc\tdis_r\tdis_p\tn_test\tn_pos_test\n",
        )
        for full_r, abl_r, b in zip(full_results, ablation_results, bundles, strict=True):
            f.write(
                f"{b.fold}\t{full_r.best_val_auc:.4f}\t{abl_r.best_val_auc:.4f}\t"
                f"{full_r.best_val_bacc:.4f}\t{abl_r.best_val_bacc:.4f}\t"
                f"{b.per_class['LumA'].f1:.4f}\t{b.per_class['LumB'].f1:.4f}\t"
                f"{b.calibration.ece:.4f}\t"
                f"{b.disagreement_report.auc_dis_predicts_misclass:.4f}\t"
                f"{b.disagreement_report.point_biserial_r:.4f}\t"
                f"{b.disagreement_report.point_biserial_p:.4f}\t"
                f"{b.n_test}\t{int((b.labels == 1).sum())}\n",
            )
    print(f"Wrote {per_fold}")

    # --- Aggregate disagreement signal ---
    dis_aucs = [b.disagreement_report.auc_dis_predicts_misclass for b in bundles]
    dis_rs = [b.disagreement_report.point_biserial_r for b in bundles]
    dis_ps = [b.disagreement_report.point_biserial_p for b in bundles]
    n_info_folds = sum(1 for b in bundles if b.disagreement_report.is_informative)

    delta_auc = full_agg_train["auc_mean"] - ablation_agg_train["auc_mean"]
    delta_bacc = full_agg_train["bacc_mean"] - ablation_agg_train["bacc_mean"]

    # --- Audit MD ---
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    informative_verdict = (
        f"**{n_info_folds}/{len(bundles)} folds** show statistically informative "
        "disagreement (mean dis on misclass > mean dis on correct AND p < 0.05). "
    )
    if n_info_folds >= 3:
        verdict_paragraph = (
            informative_verdict +
            "DMOI's Option-B thesis (disagreement is INFORMATIVE rather than a "
            "regularization target) is **empirically supported** on cohort_v2: "
            "high-disagreement cases are disproportionately the misclassified ones, "
            "which are biologically the LumA/LumB borderline tumors where the "
            "two pole perspectives genuinely disagree.\n"
        )
    elif n_info_folds >= 1:
        verdict_paragraph = (
            informative_verdict +
            "**Partial support** for DMOI's Option-B thesis. Disagreement is "
            "informative on some folds but not consistently. v0.2 should consider "
            "Option A (auxiliary BCE on sub-classifier scores) as an alternative.\n"
        )
    else:
        verdict_paragraph = (
            informative_verdict +
            "**Thesis NOT supported** on this run: disagreement is not statistically "
            "elevated on misclassified cases. The signal is noise on cohort_v2. "
            "v0.2 should either drop disagreement or move to Option A.\n"
        )

    summary_md = AUDIT / "dmoi_eval_v0.md"
    summary_md.write_text(
        "# DMOI Full Evaluation + Ablation (Day-4)\n\n"
        f"Generated: {ts}\n\n"
        f"## Cohort v2\n\n"
        f"- Dual-modality patients: **{n}** (LumA "
        f"{int((feats.y == 0).sum())} / LumB {int((feats.y == 1).sum())})\n\n"
        "## Headline metrics (Full DMOI, 5-fold CV)\n\n"
        f"- **AUROC** : {full_agg_train['auc_mean']:.4f} ± {full_agg_train['auc_std']:.4f}\n"
        f"- **BalAcc** : {full_agg_train['bacc_mean']:.4f} ± {full_agg_train['bacc_std']:.4f}\n"
        f"- **F1 LumA** : {eval_agg['f1_LumA_mean']:.4f} ± {eval_agg['f1_LumA_std']:.4f}\n"
        f"- **F1 LumB** : {eval_agg['f1_LumB_mean']:.4f} ± {eval_agg['f1_LumB_std']:.4f}  "
        f"← minority class\n"
        f"- **ECE** : {eval_agg['ece_mean']:.4f} ± {eval_agg['ece_std']:.4f}  "
        "(lower = better calibrated)\n"
        f"- **Disagreement AUC for misclass** : "
        f"{eval_agg['auc_dis_predicts_misclass_mean']:.4f} ± "
        f"{eval_agg['auc_dis_predicts_misclass_std']:.4f}  "
        "(0.5 = no signal, 1.0 = perfect)\n\n"
        "## Ablation: disagreement IN vs OUT of classifier head\n\n"
        "Both runs use the same encoder + attention + fuser; only the final\n"
        "ClassifierHead differs. The flag `use_disagreement` toggles whether\n"
        "the scalar disagreement value is concatenated alongside [z_LumA, z_LumB].\n\n"
        "| Variant | AUROC | BalAcc |\n"
        "|---|---|---|\n"
        f"| Full DMOI (disagreement IN) | "
        f"{full_agg_train['auc_mean']:.4f} ± {full_agg_train['auc_std']:.4f} | "
        f"{full_agg_train['bacc_mean']:.4f} ± {full_agg_train['bacc_std']:.4f} |\n"
        f"| Ablation (disagreement OUT) | "
        f"{ablation_agg_train['auc_mean']:.4f} ± {ablation_agg_train['auc_std']:.4f} | "
        f"{ablation_agg_train['bacc_mean']:.4f} ± {ablation_agg_train['bacc_std']:.4f} |\n"
        f"| **Δ (full − ablation)** | **{delta_auc:+.4f}** | **{delta_bacc:+.4f}** |\n\n"
        "Interpretation:\n"
        f"- If Δ AUROC > +0.005 and the disagreement-vs-misclass AUC is > 0.6, "
        "the disagreement feature is empirically useful and DMOI's Option-B "
        "thesis is supported.\n"
        "- If Δ AUROC ≈ 0 (within 1 std), the disagreement feature is **redundant** "
        "with what the pole-fused latents already encode.\n\n"
        "## Disagreement-vs-misclassification analysis\n\n"
        f"- Mean disagreement AUC for predicting misclass: "
        f"{eval_agg['auc_dis_predicts_misclass_mean']:.4f}\n"
        f"- Per-fold AUCs: {dis_aucs}\n"
        f"- Point-biserial correlation r per fold: {[f'{r:+.3f}' for r in dis_rs]}\n"
        f"- Point-biserial p per fold: {[f'{p:.4f}' for p in dis_ps]}\n\n"
        f"{verdict_paragraph}\n"
        "## Pooled OOF confusion matrix (all 5 folds concatenated)\n\n"
        f"|       | pred LumA | pred LumB |\n"
        f"|-------|-----------|-----------|\n"
        f"| true LumA | {pooled_cm['tn']} | {pooled_cm['fp']} |\n"
        f"| true LumB | {pooled_cm['fn']} | {pooled_cm['tp']} |\n\n"
        f"Pooled accuracy: "
        f"{(pooled_cm['tn'] + pooled_cm['tp']) / sum(pooled_cm.values()):.4f}  ·  "
        f"LumB sensitivity: {pooled_cm['tp'] / max(pooled_cm['tp'] + pooled_cm['fn'], 1):.4f}  ·  "
        f"LumB specificity: {pooled_cm['tn'] / max(pooled_cm['tn'] + pooled_cm['fp'], 1):.4f}\n\n"
        "## Reproduce\n\n"
        "```bash\n"
        "python scripts/eval_dmoi.py\n"
        "```\n",
    )
    print(f"Wrote {summary_md}")

    print("\n=== Day-4 summary ===")
    print(f"  Full DMOI    AUROC : {full_agg_train['auc_mean']:.4f}  "
          f"BalAcc : {full_agg_train['bacc_mean']:.4f}")
    print(f"  Ablation     AUROC : {ablation_agg_train['auc_mean']:.4f}  "
          f"BalAcc : {ablation_agg_train['bacc_mean']:.4f}")
    print(f"  Δ AUROC (full − ablation) : {delta_auc:+.4f}")
    print(f"  Δ BalAcc (full − ablation) : {delta_bacc:+.4f}")
    print(f"  F1 LumB (minority) : {eval_agg['f1_LumB_mean']:.4f} ± "
          f"{eval_agg['f1_LumB_std']:.4f}")
    print(f"  ECE : {eval_agg['ece_mean']:.4f} ± {eval_agg['ece_std']:.4f}")
    print(f"  Disagreement AUC for misclass : "
          f"{eval_agg['auc_dis_predicts_misclass_mean']:.4f}")
    print(f"  Informative-disagreement folds : {n_info_folds}/{len(bundles)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
