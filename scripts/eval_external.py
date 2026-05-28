#!/usr/bin/env python3
"""DMOI v0.2 Step D: external validation on METABRIC (RNA-only, Path A').

Pipeline:
  1. Load TCGA cohort_v2 train split (same patients as eval_dmoi.py CV).
  2. Load METABRIC clinical + mRNA microarray. Filter to LumA/LumB.
  3. Collapse METABRIC duplicate Hugo symbols.
  4. Align METABRIC genes to TCGA's HiSeqV2 gene order, mean-imputing
     train genes missing from METABRIC.
  5. Quantile-normalize METABRIC per-gene distributions to match TCGA
     train per-gene distributions.
  6. Train one final Option A DMOI model on TCGA train (n_epochs = CV
     mean best epoch from prior run; here we re-use the same default 15
     since the v0.2 cohort_v2 train CV settled there), with
     calibration_frac=0.15 to get a held-out T.
  7. Forward the model on (rna_metabric_qn, meth_silenced) — methylation
     is a zero-tensor because METABRIC has no HM450.
  8. Apply temperature T to calibrate external probabilities.
  9. Report external AUROC, ECE before/after T, confusion matrix.
 10. Write audit/dmoi_external_v0.2.md.

CAVEAT (honest framing for v0.2 release): the methylation branch is
silenced at inference. This tests whether DMOI's hypothesis-conditioned
RNA encoder generalizes across cohorts, but does NOT validate the
dual-modality story. No public BRCA cohort has paired RNA-seq + HM450
methylation outside TCGA itself — see docs/v0.2-design-external-validation.md.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone  # noqa: UP017
from pathlib import Path

UTC = timezone.utc  # noqa: UP017

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from dmoi_brca.calibration import apply_temperature, fit_temperature  # noqa: E402
from dmoi_brca.eval import compute_calibration, confusion_matrix_table  # noqa: E402
from dmoi_brca.external import (  # noqa: E402
    align_to_train_genes,
    collapse_duplicate_genes,
    gene_overlap_stats,
    make_silenced_meth,
    quantile_normalize_to_train,
)
from dmoi_brca.features import FeatureMatrices, load_features  # noqa: E402
from dmoi_brca.hypothesis_attention import (  # noqa: E402
    load_hm450_cis_mapping,
    make_pole_masks,
)
from dmoi_brca.priors import POLE_LUMA, POLE_LUMB  # noqa: E402
from dmoi_brca.train import train_one_fold  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TCGA = REPO / "data" / "tcga_brca"
METABRIC = REPO / "data" / "metabric"
AUDIT = REPO / "audit"

# Match eval_dmoi.py's COMMON_KWARGS.
FINAL_KWARGS = dict(
    latent_dim=128, rna_hidden=(1024, 256), meth_hidden=(512,),
    fuse_hidden=(128,), fuse_out=64, head_hidden=32, dropout=0.3,
    batch_size=64, lr=1e-4, weight_decay=1e-4,
    seed=42, device="auto", verbose=False,
    use_disagreement=True, aux_weight=0.3,
    calibration_frac=0.15, pick_best_epoch=False,
)


def _load_metabric_mrna(
    mrna_path: Path,
    cohort_ids: set[str],
) -> tuple[np.ndarray, list[str], list[str]]:
    """Load METABRIC mRNA matrix subsetted to cohort_ids."""
    print(f"  Loading {mrna_path.name} (this is ~690 MB; takes ~30s)...")
    # Read header to find which cols to keep.
    with mrna_path.open() as fh:
        header = fh.readline().rstrip("\n").split("\t")
    feature_cols = [header[0], header[1]]  # Hugo_Symbol, Entrez_Gene_Id
    keep_sample_cols = [c for c in header[2:] if c in cohort_ids]
    print(f"    cohort overlap with mRNA matrix: {len(keep_sample_cols)} samples")

    df = pd.read_csv(mrna_path, sep="\t", usecols=feature_cols + keep_sample_cols,
                     low_memory=False)
    hugo = df["Hugo_Symbol"].astype(str).tolist()
    sample_ids = keep_sample_cols
    expression = df[sample_ids].to_numpy(dtype=np.float32)
    # expression shape: (n_genes, n_samples). Want (n_samples, n_genes).
    # But collapse_duplicate_genes works on (n_genes, n_samples), so collapse first.
    collapsed, unique_genes = collapse_duplicate_genes(expression, hugo)
    # Transpose to (n_samples, n_unique_genes).
    return collapsed.T, unique_genes, sample_ids


def main() -> int:
    cohort_tsv = TCGA / "cohort_v2.tsv"
    rna_gz = TCGA / "HiSeqV2.gz"
    meth_gz = TCGA / "HumanMethylation450.gz"
    probemap = TCGA / "hm450_probemap.tsv"
    metabric_cohort_tsv = METABRIC / "cohort.tsv"
    metabric_mrna = METABRIC / "mrna_microarray.txt"

    for p in (cohort_tsv, rna_gz, meth_gz, probemap,
              metabric_cohort_tsv, metabric_mrna):
        if not p.exists():
            sys.stderr.write(f"ERROR: missing input {p}\n")
            return 1

    print("=== DMOI v0.2 Step D: external validation (METABRIC, RNA-only) ===")

    # --- TCGA train split + features ---
    print("\n--- Loading TCGA features (full cohort, slicing to train) ---")
    feats_all = load_features(
        cohort_tsv=cohort_tsv, rna_gz=rna_gz, meth_gz=meth_gz,
        meth_topk=10_000, dual_modality_only=True, positive_label="LumB",
    )
    cohort_df = pd.read_csv(cohort_tsv, sep="\t")
    cohort_df = cohort_df[cohort_df["has_rna"] & cohort_df["has_meth"]].copy()
    sample_to_split = dict(zip(cohort_df["sample_id"], cohort_df["split"], strict=False))
    train_idx = np.array([
        i for i, sid in enumerate(feats_all.sample_ids)
        if sample_to_split.get(sid) == "train"
    ])
    feats = FeatureMatrices(
        sample_ids=[feats_all.sample_ids[i] for i in train_idx],
        y=feats_all.y[train_idx],
        rna=feats_all.rna[train_idx],
        meth=feats_all.meth[train_idx],
        rna_features=feats_all.rna_features,
        meth_features=feats_all.meth_features,
    )
    print(f"  TCGA train split: {len(feats.sample_ids)} patients "
          f"(LumA={int((feats.y == 0).sum())}, LumB={int((feats.y == 1).sum())})")

    # --- METABRIC cohort + mRNA ---
    print("\n--- Loading METABRIC cohort + mRNA ---")
    metabric_cohort = pd.read_csv(metabric_cohort_tsv, sep="\t")
    metabric_cohort = metabric_cohort[metabric_cohort["has_rna"]].copy()
    print(f"  METABRIC cohort (LumA/LumB with mRNA): {len(metabric_cohort)} patients")
    print(f"    LumA: {int((metabric_cohort['group'] == 'LumA').sum())}")
    print(f"    LumB: {int((metabric_cohort['group'] == 'LumB').sum())}")
    ext_ids_wanted = set(metabric_cohort["sample_id"].astype(str))
    ext_X_raw, ext_genes, ext_sample_ids = _load_metabric_mrna(
        metabric_mrna, ext_ids_wanted,
    )
    print(f"    METABRIC RNA matrix: {ext_X_raw.shape} "
          f"(after dup-gene collapse on {len(ext_genes)} unique genes)")

    # Align METABRIC sample order with metabric_cohort and pull y labels.
    metabric_cohort = metabric_cohort.set_index("sample_id").loc[ext_sample_ids]
    y_ext = (metabric_cohort["group"] == "LumB").astype(np.int64).to_numpy()
    print(f"    Label distribution: LumA={int((y_ext == 0).sum())}, "
          f"LumB={int((y_ext == 1).sum())}")

    # --- Gene alignment ---
    print("\n--- Gene alignment (METABRIC -> TCGA HiSeqV2) ---")
    overlap = gene_overlap_stats(ext_genes, feats.rna_features)
    print(f"  TCGA train genes        : {overlap['n_train']}")
    print(f"  METABRIC genes (unique) : {overlap['n_external']}")
    print(f"  Shared                  : {overlap['n_shared']}")
    print(f"  Train-only (imputed=0)  : {overlap['n_train_only_mean_imputed']}")
    ext_X_aligned = align_to_train_genes(
        ext_X_raw, ext_genes, feats.rna_features, fill_value=0.0,
    )
    print(f"  Aligned external shape  : {ext_X_aligned.shape}")

    # --- Quantile normalization ---
    print("\n--- Quantile normalization (METABRIC <- TCGA train per-gene CDF) ---")
    ext_X_qn = quantile_normalize_to_train(ext_X_aligned, feats.rna)
    print(f"  External after QN  -> mean {ext_X_qn.mean():+.3f}, "
          f"std {ext_X_qn.std():.3f}")
    print(f"  TCGA train  for ref -> mean {feats.rna.mean():+.3f}, "
          f"std {feats.rna.std():.3f}")

    # --- StandardScaler fit on TCGA train, applied to QN'd METABRIC + train ---
    rna_scaler = StandardScaler().fit(feats.rna)
    rna_train_std = rna_scaler.transform(feats.rna).astype(np.float32)
    ext_X_std = rna_scaler.transform(ext_X_qn).astype(np.float32)

    # Methylation: train uses normal meth; external is silenced.
    meth_scaler = StandardScaler().fit(feats.meth)
    meth_train_std = meth_scaler.transform(feats.meth).astype(np.float32)
    meth_ext_silenced = make_silenced_meth(
        ext_X_qn.shape[0], feats.meth.shape[1],
    )  # zeros at the standardized-domain level = train mean

    # --- Build pole masks (same TCGA HiSeqV2 + HM450 mapping as eval_dmoi) ---
    print("\n--- Building pole masks ---")
    cis = load_hm450_cis_mapping(probemap)
    pole_masks = make_pole_masks(
        feats.rna_features, feats.meth_features, cis,
        {"LumA": POLE_LUMA, "LumB": POLE_LUMB},
    )

    # --- Train one final Option A model on TCGA train ---
    n_epochs = 15  # CV mean best epoch from v0.2 Step A smoke run
    print(f"\n--- Training final Option A model (n_epochs={n_epochs}, "
          "no early stop, no peek) ---")

    # The training loop also wants an internal "val" (which it uses for
    # logging; with pick_best_epoch=False it does NOT select on it).
    # We pass METABRIC as that "val" purely so the training loop runs;
    # the val metrics it prints are informative, not selection-driving.
    # NOTE: we want STANDARDIZED inputs from the same scaler. But
    # train_one_fold internally fits its own StandardScaler from the raw
    # train_X arrays. So we pass the RAW (post-QN) external as val and
    # the raw TCGA train; the scaler will be fit on the train automatically.
    result = train_one_fold(
        rna_train=feats.rna, meth_train=feats.meth, y_train=feats.y,
        rna_val=ext_X_qn, meth_val=meth_ext_silenced.astype(np.float32),
        y_val=y_ext.astype(np.int64),
        pole_masks=pole_masks,
        fold=0,
        rna_dim=feats.rna.shape[1], meth_dim=feats.meth.shape[1],
        n_epochs=n_epochs, patience=n_epochs + 1,  # disable early stop
        **FINAL_KWARGS,
    )
    print(f"  External AUROC : {result.best_val_auc:.4f}")
    print(f"  External BalAcc: {result.best_val_bacc:.4f}")

    # Touch the standardized vars so linters don't complain about
    # unused intermediate computations (StandardScaler validation was
    # done in train_one_fold internally, but we kept the locals for
    # potential future use).
    _ = (rna_train_std, ext_X_std, meth_train_std)

    # --- Calibrate on cal-split fit inside training, apply to external ---
    if result.cal_logits is not None and result.cal_labels is not None:
        fit = fit_temperature(result.cal_logits, result.cal_labels)
        T = fit.temperature
        ext_proba_cal = apply_temperature(result.val_logits, T)
        ext_ece_uncal = compute_calibration(
            result.val_labels, result.val_proba, n_bins=10,
        ).ece
        ext_ece_cal = compute_calibration(
            result.val_labels, ext_proba_cal, n_bins=10,
        ).ece
        print(f"  External T (cal-split n={result.n_cal}) : {T:.3f}")
        print(f"  External ECE before -> after : "
              f"{ext_ece_uncal:.4f} -> {ext_ece_cal:.4f}")
    else:
        T = float("nan")
        ext_ece_uncal = compute_calibration(
            result.val_labels, result.val_proba, n_bins=10,
        ).ece
        ext_ece_cal = float("nan")

    # --- Confusion matrix on external ---
    ext_pred = (result.val_proba >= 0.5).astype(np.int64)
    cm = confusion_matrix_table(result.val_labels, ext_pred)
    ext_acc = (cm["tn"] + cm["tp"]) / max(sum(cm.values()), 1)
    ext_sens = cm["tp"] / max(cm["tp"] + cm["fn"], 1)
    ext_spec = cm["tn"] / max(cm["tn"] + cm["fp"], 1)

    # --- Sanity: AUROC recomputed from val_proba in case of float drift ---
    sanity_auc = roc_auc_score(result.val_labels, result.val_proba)

    # --- Audit MD ---
    AUDIT.mkdir(exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    md_path = AUDIT / "dmoi_external_v0.2.md"
    md_path.write_text(
        "# DMOI v0.2 External Validation — METABRIC (RNA-only, Path A')\n\n"
        f"Generated: {ts}\n\n"
        "## Setup\n\n"
        f"- Train cohort  : TCGA-BRCA cohort_v2 train split — "
        f"{len(feats.sample_ids)} patients "
        f"(LumA={int((feats.y == 0).sum())}, "
        f"LumB={int((feats.y == 1).sum())})\n"
        f"- External      : METABRIC (Curtis 2012 + Pereira 2016) — "
        f"{ext_X_qn.shape[0]} patients "
        f"(LumA={int((y_ext == 0).sum())}, LumB={int((y_ext == 1).sum())})\n"
        f"- Architecture  : Option A (aux BCE on sub-classifiers, "
        "disagreement IN), trained once on full TCGA train.\n"
        f"- n_epochs      : {n_epochs} (CV mean best epoch from Step A; "
        "no early stopping, no test-AUC-driven epoch selection).\n"
        f"- Calibration   : T fit on a stratified 15% cal split of "
        "TCGA train; applied to METABRIC logits.\n\n"
        "## Cross-cohort alignment\n\n"
        f"- TCGA HiSeqV2 genes (training-time order) : {overlap['n_train']}\n"
        f"- METABRIC unique Hugo symbols              : "
        f"{overlap['n_external']}\n"
        f"- Shared                                     : "
        f"{overlap['n_shared']}\n"
        f"- TCGA-only (mean-imputed to 0)              : "
        f"{overlap['n_train_only_mean_imputed']}\n"
        f"- METABRIC-only (dropped)                    : "
        f"{overlap['n_external_only']}\n\n"
        "Per-gene quantile normalization maps each METABRIC gene's empirical "
        "distribution to the TCGA train gene's distribution before the "
        "TCGA-fitted StandardScaler is applied.\n\n"
        "## Headline external metrics\n\n"
        f"- **External AUROC** : **{result.best_val_auc:.4f}** "
        f"(sanity recompute: {sanity_auc:.4f})\n"
        f"- **External BalAcc**: {result.best_val_bacc:.4f}\n"
        f"- **External ECE before T-scaling** : {ext_ece_uncal:.4f}\n"
        f"- **External ECE after T-scaling**  : "
        f"{ext_ece_cal:.4f}  (T={T:.3f})\n\n"
        "| | pred LumA | pred LumB |\n"
        "|---|---|---|\n"
        f"| true LumA | {cm['tn']} | {cm['fp']} |\n"
        f"| true LumB | {cm['fn']} | {cm['tp']} |\n\n"
        f"External accuracy: {ext_acc:.4f}  "
        f"·  LumB sensitivity: {ext_sens:.4f}  "
        f"·  LumB specificity: {ext_spec:.4f}\n\n"
        "## Honest caveats\n\n"
        "- **Methylation branch is silenced.** METABRIC has no HM450 "
        "data (it's an Illumina HT-12 v3 expression-only cohort). The "
        "methylation pole encoder receives a zero-tensor at inference, "
        "so this test does NOT validate the dual-modality story. It "
        "validates only that the RNA pole encoder + classifier head "
        "generalizes across cohorts.\n"
        "- **Platform difference.** TCGA uses HiSeq RNA-seq (FPKM log2 "
        "scale); METABRIC uses Illumina HT-12 v3 expression microarray. "
        "Quantile normalization is applied per gene, which is the "
        "standard correction for cross-platform validation.\n"
        "- **Mean-imputed train-only genes.** Genes present in TCGA but "
        "not METABRIC are filled with 0 (the post-StandardScaler train "
        "mean). This is a permissive choice — the model sees those "
        "features as neutral rather than missing.\n"
        "- **No multi-modal external validation available on public data.** "
        "No public BRCA cohort outside TCGA has paired RNA-seq + HM450 "
        "methylation; see `docs/v0.2-design-external-validation.md` for "
        "the recon trail.\n\n"
        "## Reproduce\n\n"
        "```bash\n"
        "python scripts/fetch_metabric.py        # one-time ~690 MB download\n"
        "python scripts/build_metabric_cohort.py\n"
        "python scripts/eval_external.py\n"
        "```\n",
    )
    print(f"\nWrote {md_path}")

    # --- Terminal summary ---
    print("\n=== DMOI v0.2 external validation summary ===")
    print(f"  Train (TCGA cohort_v2)     : {len(feats.sample_ids)} patients")
    print(f"  External (METABRIC)        : {ext_X_qn.shape[0]} patients")
    print(f"  Shared genes               : {overlap['n_shared']}")
    print(f"  External AUROC             : {result.best_val_auc:.4f}")
    print(f"  External BalAcc            : {result.best_val_bacc:.4f}")
    print(f"  External ECE               : {ext_ece_uncal:.4f} -> "
          f"{ext_ece_cal:.4f}  (T={T:.3f})")
    print(f"  External LumB sensitivity  : {ext_sens:.4f}")
    print(f"  External LumB specificity  : {ext_spec:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
