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

from dmoi_brca.calibration import calibrate_fold  # noqa: E402
from dmoi_brca.eval import (  # noqa: E402
    aggregate_cross_fold,
    build_fold_eval_bundle,
    compute_calibration,
    concat_fold_predictions,
    confusion_matrix_table,
)
from dmoi_brca.features import FeatureMatrices, load_features  # noqa: E402
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
    print("\n--- Loading features (full cohort, will slice by split) ---")
    feats_all = load_features(
        cohort_tsv=cohort_tsv, rna_gz=rna_gz, meth_gz=meth_gz,
        meth_topk=10_000, dual_modality_only=True, positive_label="LumB",
    )
    # Slice into train/test using the cohort TSV's split column. Loading the
    # full cohort once means train and test share the SAME top-K methylation
    # probe selection (the alternative — loading each split separately —
    # would pick different probes per split and break the model).
    import pandas as pd  # noqa: PLC0415
    cohort_df = pd.read_csv(cohort_tsv, sep="\t")
    cohort_df = cohort_df[cohort_df["has_rna"] & cohort_df["has_meth"]].copy()
    if "split" not in cohort_df.columns:
        sys.stderr.write(
            "ERROR: cohort_v2.tsv missing `split` column. "
            "Regenerate via `python scripts/build_cohort_v2.py` to add the 80/20 split.\n",
        )
        return 1
    sample_to_split = dict(zip(cohort_df["sample_id"], cohort_df["split"], strict=False))
    train_idx = np.array([
        i for i, sid in enumerate(feats_all.sample_ids)
        if sample_to_split.get(sid) == "train"
    ])
    test_idx = np.array([
        i for i, sid in enumerate(feats_all.sample_ids)
        if sample_to_split.get(sid) == "test"
    ])

    def _slice(idx: np.ndarray) -> FeatureMatrices:
        return FeatureMatrices(
            sample_ids=[feats_all.sample_ids[i] for i in idx],
            y=feats_all.y[idx],
            rna=feats_all.rna[idx],
            meth=feats_all.meth[idx],
            rna_features=feats_all.rna_features,
            meth_features=feats_all.meth_features,
        )

    feats = _slice(train_idx)
    feats_test = _slice(test_idx)
    n = len(feats.sample_ids)
    n_test = len(feats_test.sample_ids)
    print(f"  Cohort v2 train split: {n} patients "
          f"(LumA={int((feats.y == 0).sum())}, LumB={int((feats.y == 1).sum())})")
    print(f"  Cohort v2 test  split: {n_test} patients "
          f"(LumA={int((feats_test.y == 0).sum())}, "
          f"LumB={int((feats_test.y == 1).sum())})")

    print("\n--- Building pole masks ---")
    cis = load_hm450_cis_mapping(probemap)
    pole_masks = make_pole_masks(
        feats.rna_features, feats.meth_features, cis,
        {"LumA": POLE_LUMA, "LumB": POLE_LUMB},
    )

    # --- Option A: aux BCE supervision on sub-classifiers + disagreement IN ---
    # calibration_frac=0.15: hold out 15% of each train fold (stratified on y)
    # to fit temperature on, so the calibrated-ECE we report on val is honest
    # (not optimistic). Optimistic T-on-val is still computed below for the
    # upper-bound comparison.
    print("\n--- Run 1/3: Option A (aux BCE on sub-clf + disagreement IN) ---")
    optionA_results = run_dmoi_cv(
        rna=feats.rna, meth=feats.meth, y=feats.y,
        pole_masks=pole_masks, use_disagreement=True,
        aux_weight=0.3, calibration_frac=0.15, **COMMON_KWARGS,
    )
    optionA_agg_train = aggregate_fold_results(optionA_results)

    # --- Option B: original v0.1 — no aux, disagreement IN ---
    print("\n--- Run 2/3: Option B (no aux + disagreement IN) [v0.1 baseline] ---")
    optionB_results = run_dmoi_cv(
        rna=feats.rna, meth=feats.meth, y=feats.y,
        pole_masks=pole_masks, use_disagreement=True,
        aux_weight=0.0, **COMMON_KWARGS,
    )
    optionB_agg_train = aggregate_fold_results(optionB_results)

    # --- Ablation: no aux, no disagreement input ---
    print("\n--- Run 3/3: Ablation (no aux + disagreement OUT) ---")
    ablation_results = run_dmoi_cv(
        rna=feats.rna, meth=feats.meth, y=feats.y,
        pole_masks=pole_masks, use_disagreement=False,
        aux_weight=0.0, **COMMON_KWARGS,
    )
    ablation_agg_train = aggregate_fold_results(ablation_results)

    # Use Option A as the "full DMOI" canonical run going forward.
    full_results = optionA_results
    full_agg_train = optionA_agg_train

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

    # --- Temperature scaling calibration on Option A val logits ---
    # CAVEAT: T is fit on the same val fold we then measure ECE on. This is
    # optimistic — it sets an upper bound on what post-hoc calibration can
    # buy. v0.2+ should fit T on a nested calibration split carved out of
    # train. See `src/dmoi_brca/calibration.py` module docstring.
    print("\n--- Temperature scaling (Option A) ---")
    calib_fits: list = []
    calibrated_probas: list[np.ndarray] = []
    calibrated_eces: list[float] = []
    for r in optionA_results:
        if r.val_logits is None or r.val_labels is None:
            sys.stderr.write(
                f"WARN: fold {r.fold} missing val_logits or val_labels; "
                "skipping calibration\n",
            )
            calib_fits.append(None)
            calibrated_probas.append(np.zeros(0))
            calibrated_eces.append(float("nan"))
            continue
        calibrated, fit = calibrate_fold(r.val_logits, r.val_labels)
        cal_report = compute_calibration(r.val_labels, calibrated, n_bins=10)
        calib_fits.append(fit)
        calibrated_probas.append(calibrated)
        calibrated_eces.append(float(cal_report.ece))
        print(
            f"  fold {r.fold}: T={fit.temperature:.3f}  "
            f"NLL {fit.nll_before:.4f} -> {fit.nll_after:.4f}  "
            f"ECE_cal={cal_report.ece:.4f}",
        )

    valid_fits = [f for f in calib_fits if f is not None]
    valid_cal_eces = np.array(
        [e for e in calibrated_eces if not np.isnan(e)],
    )
    uncal_eces = np.array([b.calibration.ece for b in bundles])
    mean_T = float(np.mean([f.temperature for f in valid_fits])) if valid_fits else float("nan")
    std_T = (
        float(np.std([f.temperature for f in valid_fits], ddof=1))
        if len(valid_fits) > 1
        else 0.0
    )
    mean_ece_uncal = float(uncal_eces.mean())
    mean_ece_cal = float(valid_cal_eces.mean()) if len(valid_cal_eces) > 0 else float("nan")
    ece_reduction = mean_ece_uncal - mean_ece_cal

    # --- Nested temperature scaling: T fit on held-out calibration split ---
    # Option A was trained with calibration_frac=0.15, so each fold has a
    # cal_logits/cal_labels pair that the model never trained on. Fit T on
    # those, then apply to val_logits and recompute ECE on val. This is the
    # honest, non-optimistic number.
    print("\n--- Nested temperature scaling (Option A, T fit on held-out cal split) ---")
    nested_fits: list = []
    nested_eces_cal: list[float] = []
    for r in optionA_results:
        if (
            r.cal_logits is None or r.cal_labels is None
            or r.val_logits is None or r.val_labels is None
        ):
            sys.stderr.write(
                f"WARN: fold {r.fold} missing nested cal arrays; "
                "skipping nested calibration\n",
            )
            nested_fits.append(None)
            nested_eces_cal.append(float("nan"))
            continue
        from dmoi_brca.calibration import apply_temperature, fit_temperature  # noqa: PLC0415
        fit = fit_temperature(r.cal_logits, r.cal_labels)
        cal_val_proba = apply_temperature(r.val_logits, fit.temperature)
        rep = compute_calibration(r.val_labels, cal_val_proba, n_bins=10)
        nested_fits.append(fit)
        nested_eces_cal.append(float(rep.ece))
        print(
            f"  fold {r.fold}: n_cal={r.n_cal}  T_nested={fit.temperature:.3f}  "
            f"ECE_val_nested_cal={rep.ece:.4f}",
        )
    valid_nested_fits = [f for f in nested_fits if f is not None]
    valid_nested_eces = np.array(
        [e for e in nested_eces_cal if not np.isnan(e)],
    )
    mean_T_nested = (
        float(np.mean([f.temperature for f in valid_nested_fits]))
        if valid_nested_fits
        else float("nan")
    )
    std_T_nested = (
        float(np.std([f.temperature for f in valid_nested_fits], ddof=1))
        if len(valid_nested_fits) > 1
        else 0.0
    )
    mean_ece_nested = (
        float(valid_nested_eces.mean()) if len(valid_nested_eces) > 0 else float("nan")
    )
    nested_reduction = mean_ece_uncal - mean_ece_nested

    # --- Held-out TCGA test scoring (v0.2 Path C) ---
    # Train one Option A model on the FULL train split (no CV) for the mean
    # CV best epoch, score the test split exactly once. patience > n_epochs
    # disables early stopping, and pick_best_epoch=False stops the loop from
    # selecting the val-AUC-maximizing epoch (val == test here would leak).
    print("\n--- Held-out test scoring (TCGA 80/20) ---")
    mean_best_epoch = int(round(np.mean([r.best_epoch for r in optionA_results])))
    print(f"  Training final Option A on full train split for "
          f"{mean_best_epoch} epochs (CV mean best_epoch); no early stop, no peek.")
    from dmoi_brca.train import train_one_fold  # noqa: PLC0415

    test_result = train_one_fold(
        rna_train=feats.rna, meth_train=feats.meth, y_train=feats.y,
        rna_val=feats_test.rna, meth_val=feats_test.meth, y_val=feats_test.y,
        pole_masks=pole_masks,
        fold=0,
        rna_dim=feats.rna.shape[1], meth_dim=feats.meth.shape[1],
        latent_dim=128, rna_hidden=(1024, 256), meth_hidden=(512,),
        fuse_hidden=(128,), fuse_out=64, head_hidden=32,
        dropout=0.3, n_epochs=mean_best_epoch, batch_size=64,
        lr=1e-4, weight_decay=1e-4,
        patience=mean_best_epoch + 1,  # disable early stopping
        seed=42, device="auto", verbose=False,
        use_disagreement=True, aux_weight=0.3,
        calibration_frac=0.15,
        pick_best_epoch=False,  # no peek at test for best-epoch selection
    )
    print(f"  Test AUROC : {test_result.best_val_auc:.4f}")
    print(f"  Test BalAcc: {test_result.best_val_bacc:.4f}")

    test_labels = test_result.val_labels
    test_proba = test_result.val_proba
    test_logits = test_result.val_logits
    test_ece_uncal = compute_calibration(test_labels, test_proba, n_bins=10).ece

    # Apply T fit on the cal-split inside the final training to test logits.
    from dmoi_brca.calibration import apply_temperature, fit_temperature  # noqa: PLC0415

    test_T_used = float("nan")
    test_ece_cal = float("nan")
    if test_result.cal_logits is not None and test_result.cal_labels is not None:
        test_fit = fit_temperature(test_result.cal_logits, test_result.cal_labels)
        test_T_used = test_fit.temperature
        test_proba_cal = apply_temperature(test_logits, test_T_used)
        test_ece_cal = compute_calibration(test_labels, test_proba_cal, n_bins=10).ece
        print(f"  Test T (cal-split, n_cal={test_result.n_cal}) : {test_T_used:.3f}")
        print(f"  Test ECE before -> after : {test_ece_uncal:.4f} -> {test_ece_cal:.4f}")
    else:
        print("  WARN: no cal split returned; skipping test calibration.")

    test_pred = (test_proba >= 0.5).astype(np.int64)
    test_cm = confusion_matrix_table(test_labels, test_pred)

    # --- Per-fold TSV (3-way: Option A / Option B / Ablation) ---
    AUDIT.mkdir(exist_ok=True)
    per_fold = AUDIT / "dmoi_eval_per_fold.tsv"
    with per_fold.open("w") as f:
        f.write(
            "fold\tauc_A\tauc_B\tauc_ablation\tbacc_A\tbacc_B\tbacc_ablation\t"
            "f1_lumA_optA\tf1_lumB_optA\tece_optA\ttemperature_optA\tece_cal_optA\t"
            "temperature_nested\tece_cal_nested\t"
            "dis_auc_optA\tdis_r_optA\tdis_p_optA\tn_test\tn_pos_test\n",
        )
        for a_r, b_r, abl_r, eb, fit, ece_cal, nfit, ece_nest in zip(
            optionA_results, optionB_results, ablation_results, bundles,
            calib_fits, calibrated_eces, nested_fits, nested_eces_cal,
            strict=True,
        ):
            T_str = f"{fit.temperature:.4f}" if fit is not None else "nan"
            ece_cal_str = f"{ece_cal:.4f}" if not np.isnan(ece_cal) else "nan"
            T_nest_str = f"{nfit.temperature:.4f}" if nfit is not None else "nan"
            ece_nest_str = f"{ece_nest:.4f}" if not np.isnan(ece_nest) else "nan"
            f.write(
                f"{eb.fold}\t{a_r.best_val_auc:.4f}\t{b_r.best_val_auc:.4f}\t"
                f"{abl_r.best_val_auc:.4f}\t"
                f"{a_r.best_val_bacc:.4f}\t{b_r.best_val_bacc:.4f}\t"
                f"{abl_r.best_val_bacc:.4f}\t"
                f"{eb.per_class['LumA'].f1:.4f}\t{eb.per_class['LumB'].f1:.4f}\t"
                f"{eb.calibration.ece:.4f}\t{T_str}\t{ece_cal_str}\t"
                f"{T_nest_str}\t{ece_nest_str}\t"
                f"{eb.disagreement_report.auc_dis_predicts_misclass:.4f}\t"
                f"{eb.disagreement_report.point_biserial_r:.4f}\t"
                f"{eb.disagreement_report.point_biserial_p:.4f}\t"
                f"{eb.n_test}\t{int((eb.labels == 1).sum())}\n",
            )
    print(f"Wrote {per_fold}")

    # --- Aggregate disagreement signal ---
    dis_aucs = [b.disagreement_report.auc_dis_predicts_misclass for b in bundles]
    dis_rs = [b.disagreement_report.point_biserial_r for b in bundles]
    dis_ps = [b.disagreement_report.point_biserial_p for b in bundles]
    n_info_folds = sum(1 for b in bundles if b.disagreement_report.is_informative)

    delta_auc_A_vs_abl = optionA_agg_train["auc_mean"] - ablation_agg_train["auc_mean"]
    delta_bacc_A_vs_abl = optionA_agg_train["bacc_mean"] - ablation_agg_train["bacc_mean"]
    delta_auc_A_vs_B = optionA_agg_train["auc_mean"] - optionB_agg_train["auc_mean"]
    delta_bacc_A_vs_B = optionA_agg_train["bacc_mean"] - optionB_agg_train["bacc_mean"]

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
        "## 3-way ablation: Option A vs Option B vs no-disagreement\n\n"
        "Three architectural variants. All share the same encoder + attention\n"
        "+ fuser; only the loss and ClassifierHead input vary:\n\n"
        "- **Option A** (v0.2 candidate): aux BCE supervision on sub-classifiers\n"
        "  with weight 0.3; disagreement scalar included as classifier-head input.\n"
        "- **Option B** (v0.1 baseline): NO aux supervision; disagreement IN.\n"
        "- **Ablation**: NO aux + disagreement OUT.\n\n"
        "| Variant | AUROC | BalAcc |\n"
        "|---|---|---|\n"
        f"| Option A (aux + disagreement IN) | "
        f"{optionA_agg_train['auc_mean']:.4f} ± {optionA_agg_train['auc_std']:.4f} | "
        f"{optionA_agg_train['bacc_mean']:.4f} ± {optionA_agg_train['bacc_std']:.4f} |\n"
        f"| Option B (no aux + disagreement IN) | "
        f"{optionB_agg_train['auc_mean']:.4f} ± {optionB_agg_train['auc_std']:.4f} | "
        f"{optionB_agg_train['bacc_mean']:.4f} ± {optionB_agg_train['bacc_std']:.4f} |\n"
        f"| Ablation (no aux + disagreement OUT) | "
        f"{ablation_agg_train['auc_mean']:.4f} ± {ablation_agg_train['auc_std']:.4f} | "
        f"{ablation_agg_train['bacc_mean']:.4f} ± {ablation_agg_train['bacc_std']:.4f} |\n"
        f"| **Δ A − B** | **{delta_auc_A_vs_B:+.4f}** | **{delta_bacc_A_vs_B:+.4f}** |\n"
        f"| **Δ A − Ablation** | **{delta_auc_A_vs_abl:+.4f}** | **{delta_bacc_A_vs_abl:+.4f}** |\n\n"
        "Interpretation:\n"
        "- **Δ A − B**: does the auxiliary supervision on sub-classifiers add value?\n"
        "  + If > +0.005: aux supervision sharpens the disagreement signal — Option A wins.\n"
        "  + If ≈ 0: aux didn't help meaningfully; v0.1 (Option B) was already near-optimal.\n"
        "  + If < 0: aux supervision over-constrained the sub-classifiers; revisit weight.\n"
        "- **Δ A − Ablation**: is the dual-perspective architecture (with supervised sub-clfs)\n"
        "  better than dropping the disagreement / sub-clf branch entirely?\n\n"
        "## Disagreement-vs-misclassification analysis\n\n"
        f"- Mean disagreement AUC for predicting misclass: "
        f"{eval_agg['auc_dis_predicts_misclass_mean']:.4f}\n"
        f"- Per-fold AUCs: {dis_aucs}\n"
        f"- Point-biserial correlation r per fold: {[f'{r:+.3f}' for r in dis_rs]}\n"
        f"- Point-biserial p per fold: {[f'{p:.4f}' for p in dis_ps]}\n\n"
        f"{verdict_paragraph}\n"
        "## Temperature scaling calibration (Option A)\n\n"
        "Single-parameter post-hoc calibration via Guo et al. 2017:\n"
        "`calibrated_proba = sigmoid(logits / T)`. T fit by LBFGS on the\n"
        "BCE NLL of each fold's val logits.\n\n"
        "**Caveat (v0.1):** T is fit on the same val fold we measure ECE on,\n"
        "which is **optimistic** — it's an upper bound on what post-hoc\n"
        "calibration can buy with this architecture on this cohort.\n"
        "v0.2+ should fit T on a nested calibration split carved out of\n"
        "the train fold.\n\n"
        f"- Mean T : **{mean_T:.3f} ± {std_T:.3f}**  "
        "(T > 1 = overconfident; T = 1 = already calibrated)\n"
        f"- Per-fold T : "
        f"{[f'{f.temperature:.3f}' if f is not None else 'nan' for f in calib_fits]}\n"
        f"- Per-fold ECE (uncalibrated) : "
        f"{[f'{e:.4f}' for e in uncal_eces.tolist()]}\n"
        f"- Per-fold ECE (T-calibrated) : "
        f"{[f'{e:.4f}' if not np.isnan(e) else 'nan' for e in calibrated_eces]}\n"
        f"- **Mean ECE before → after** : **{mean_ece_uncal:.4f} → {mean_ece_cal:.4f}**\n"
        f"- **Δ ECE (improvement)** : **{ece_reduction:+.4f}**\n\n"
        "## Temperature scaling calibration — nested split (Option A, honest)\n\n"
        "Each fold also held out 15% of its train data (stratified on y) as a\n"
        "calibration split that the model never trained on. T is fit on those\n"
        "cal logits and applied to val. This is the **honest** number — no\n"
        "double-dipping between fit and evaluation.\n\n"
        f"- Cal split fraction : **15%** of each train fold (stratified)\n"
        f"- Mean T (nested) : **{mean_T_nested:.3f} ± {std_T_nested:.3f}**\n"
        f"- Per-fold T (nested) : "
        f"{[f'{f.temperature:.3f}' if f is not None else 'nan' for f in nested_fits]}\n"
        f"- Per-fold ECE on val (T fit on cal split) : "
        f"{[f'{e:.4f}' if not np.isnan(e) else 'nan' for e in nested_eces_cal]}\n"
        f"- **Mean ECE before → after (nested)** : "
        f"**{mean_ece_uncal:.4f} → {mean_ece_nested:.4f}**\n"
        f"- **Δ ECE (honest improvement)** : **{nested_reduction:+.4f}**\n\n"
        "Comparison:\n\n"
        "| T fit on | Mean T | Mean ECE on val | Notes |\n"
        "|---|---|---|---|\n"
        f"| val (optimistic) | {mean_T:.3f} | {mean_ece_cal:.4f} | upper bound — T tuned to the same fold |\n"
        f"| held-out cal split (honest) | {mean_T_nested:.3f} | {mean_ece_nested:.4f} | what generalizes |\n\n"
        "## Held-out TCGA test (v0.2 Path C, 80/20 split)\n\n"
        f"A {n_test}-patient stratified test split was carved at cohort "
        "construction time with random_state=2024 (distinct from the CV "
        f"seed). It is scored **once** by a single Option A model trained on "
        f"the full train split ({n} patients) for "
        f"{mean_best_epoch} epochs (CV mean best epoch; no early stopping; "
        "`pick_best_epoch=False` so val/test AUC does not select an epoch).\n\n"
        f"- **Test AUROC** : **{test_result.best_val_auc:.4f}** "
        f"(internal CV mean: {optionA_agg_train['auc_mean']:.4f})\n"
        f"- **Test BalAcc**: {test_result.best_val_bacc:.4f}\n"
        f"- **Test ECE before T-scaling** : {test_ece_uncal:.4f}\n"
        f"- **Test ECE after T-scaling**  : {test_ece_cal:.4f}  "
        f"(T={test_T_used:.3f} fit on a 15% cal split of train)\n\n"
        "| | pred LumA | pred LumB |\n"
        "|---|---|---|\n"
        f"| true LumA | {test_cm['tn']} | {test_cm['fp']} |\n"
        f"| true LumB | {test_cm['fn']} | {test_cm['tp']} |\n\n"
        f"Test accuracy: "
        f"{(test_cm['tn'] + test_cm['tp']) / max(sum(test_cm.values()), 1):.4f}  "
        f"·  LumB sensitivity: "
        f"{test_cm['tp'] / max(test_cm['tp'] + test_cm['fn'], 1):.4f}  "
        f"·  LumB specificity: "
        f"{test_cm['tn'] / max(test_cm['tn'] + test_cm['fp'], 1):.4f}\n\n"
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

    print("\n=== Day-4 v0.2 summary (3-way ablation) ===")
    print(f"  Option A (aux+dis IN) AUROC : {optionA_agg_train['auc_mean']:.4f}  "
          f"BalAcc : {optionA_agg_train['bacc_mean']:.4f}")
    print(f"  Option B (no aux+dis IN) AUROC : {optionB_agg_train['auc_mean']:.4f}  "
          f"BalAcc : {optionB_agg_train['bacc_mean']:.4f}")
    print(f"  Ablation (no aux+dis OUT) AUROC : {ablation_agg_train['auc_mean']:.4f}  "
          f"BalAcc : {ablation_agg_train['bacc_mean']:.4f}")
    print(f"  Δ AUROC (Option A − Option B) : {delta_auc_A_vs_B:+.4f}")
    print(f"  Δ AUROC (Option A − Ablation) : {delta_auc_A_vs_abl:+.4f}")
    print(f"  Δ BalAcc (Option A − Ablation) : {delta_bacc_A_vs_abl:+.4f}")
    print(f"  F1 LumB (minority) : {eval_agg['f1_LumB_mean']:.4f} ± "
          f"{eval_agg['f1_LumB_std']:.4f}")
    print(f"  ECE : {eval_agg['ece_mean']:.4f} ± {eval_agg['ece_std']:.4f}")
    print(f"  T (optimistic, val-fit) : {mean_T:.3f} ± {std_T:.3f}  "
          f"-> ECE val {mean_ece_cal:.4f}")
    print(f"  T (nested, cal-split-fit) : {mean_T_nested:.3f} ± {std_T_nested:.3f}  "
          f"-> ECE val {mean_ece_nested:.4f}")
    print(f"  Disagreement AUC for misclass : "
          f"{eval_agg['auc_dis_predicts_misclass_mean']:.4f}")
    print(f"  Informative-disagreement folds : {n_info_folds}/{len(bundles)}")
    print(f"\n  === Held-out TCGA test (n={n_test}) ===")
    print(f"  Test AUROC : {test_result.best_val_auc:.4f}  "
          f"(internal CV mean: {optionA_agg_train['auc_mean']:.4f}, "
          f"Δ = {test_result.best_val_auc - optionA_agg_train['auc_mean']:+.4f})")
    print(f"  Test BalAcc: {test_result.best_val_bacc:.4f}")
    print(f"  Test ECE   : {test_ece_uncal:.4f} -> {test_ece_cal:.4f} "
          f"(T={test_T_used:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
