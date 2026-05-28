#!/usr/bin/env python3
"""DMOI v0.12-A: TCGA cohort_v2 5-fold CV x full-METABRIC scoring per fold.

v0.11 sealed the v0.9 / v0.10 four-axis closure as split-invariant on
TCGA cohort_v3 (Luminal-vs-Basal): every fold reached AUROC = 1.000
under 5-fold StratifiedKFold. v0.4 reported METABRIC LumA-vs-LumB
AUROC = 0.909 as the cross-cohort same-task headline -- but that was
a single trained model on a single TCGA train split scored against
METABRIC. The natural skeptic's question:

  Is the v0.4 cross-cohort AUROC = 0.909 split-invariant, or did it
  ride a lucky TCGA-train split?

v0.12-A answers by running 5-fold StratifiedKFold on TCGA cohort_v2
LumA-vs-LumB and, for each fold, scoring the full METABRIC
LumA-vs-LumB cohort with the fold's QN scaler + meth-silenced. The
deliverable is a paired variance band:

  - Per-fold TCGA val AUROC (the internal-stability band, pairs with
    v0.11's TCGA cohort_v3 finding -- 5-axis closure on the
    LumA-vs-LumB task this time).
  - Per-fold METABRIC AUROC (the cross-cohort-stability band, the new
    headline).
  - Per-fold per-pole IG Hallmark rollup on METABRIC + cross-fold
    top-3 Jaccard.

v0.11 (Luminal-vs-Basal) + v0.12-A (LumA-vs-LumB) together cover the
internal AND cross-cohort variance bands on both task axes. The v0.6
single-split numbers (TCGA test 0.968; METABRIC 0.909) get replaced
with split-distributed bands on both metrics simultaneously.

Honest scope:
  - Same architecture, same priors, same hyperparameters as v0.6.
    Only the train/val split changes per fold.
  - Each fold's METABRIC score uses that fold's TCGA-train RNA
    distribution as the QN reference (re-fit per fold; the right
    thing under proper cross-validation).
  - 5 folds means 5 re-fits of the QN scaler -- minor overhead.
  - cohort_v2 has 417 dual-modality patients (LumA 290 + LumB 127);
    each val fold has ~25 LumB patients. METABRIC LumA-vs-LumB
    n=1175 (LumA 700 + LumB 475).
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
from dmoi_brca.external import (  # noqa: E402
    align_to_train_genes,
    collapse_duplicate_genes,
    make_silenced_meth,
    quantile_normalize_to_train,
)
from dmoi_brca.features import load_features  # noqa: E402
from dmoi_brca.hallmark import load_hallmark_gmt  # noqa: E402
from dmoi_brca.hypothesis_attention import (  # noqa: E402
    load_hm450_cis_mapping,
    make_pole_masks,
)
from dmoi_brca.pathway import pathway_aggregate, rank_pathways  # noqa: E402
from dmoi_brca.priors import POLE_LUMA, POLE_LUMB  # noqa: E402
from dmoi_brca.train import train_one_fold  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TCGA = REPO / "data" / "tcga_brca"
METABRIC = REPO / "data" / "metabric"
AUDIT = REPO / "audit"
HALLMARK_GMT = REPO / "data" / "msigdb" / "h.all.v2024.1.Hs.symbols.gmt"

POLE_ORDER = ("LumA", "LumB")
POSITIVE_LABEL = "LumB"  # LumB = 1 (minority within luminal)
TCGA_COHORT_TSV_NAME = "cohort_v2.tsv"
METABRIC_COHORT_TSV_NAME = "cohort.tsv"

# Reuse v0.6 / v0.9 base hyperparameters exactly.
FINAL_KWARGS = dict(
    latent_dim=128, rna_hidden=(1024, 256), meth_hidden=(512,),
    fuse_hidden=(128,), fuse_out=64, head_hidden=32, dropout=0.3,
    batch_size=64, lr=1e-4, weight_decay=1e-4,
    seed=42, device="auto", verbose=False,
    use_disagreement=True, aux_weight=0.3,
    calibration_frac=0.15,
    # CV-aware: pick best epoch (val fold is a real val fold here).
    pick_best_epoch=True,
)
N_EPOCHS = 15
N_IG_STEPS = 50

# v0.5 / v0.6 expected priors per pole (LumA-vs-LumB task)
EXPECTED_LUMA_TOP = {
    "HALLMARK_ESTROGEN_RESPONSE_EARLY",
    "HALLMARK_ESTROGEN_RESPONSE_LATE",
}
EXPECTED_LUMB_TOP = {
    "HALLMARK_E2F_TARGETS",
    "HALLMARK_G2M_CHECKPOINT",
    "HALLMARK_MYC_TARGETS_V1",
}


def _load_metabric_mrna(mrna_path: Path, cohort_ids: set[str]):
    """Reuse the v0.2 / v0.4 / v0.10 METABRIC loader pattern."""
    with mrna_path.open() as fh:
        header = fh.readline().rstrip("\n").split("\t")
    keep = [c for c in header[2:] if c in cohort_ids]
    df = pd.read_csv(
        mrna_path, sep="\t",
        usecols=[header[0], header[1]] + keep, low_memory=False,
    )
    hugo = df["Hugo_Symbol"].astype(str).tolist()
    expression = df[keep].to_numpy(dtype=np.float32)
    collapsed, unique_genes = collapse_duplicate_genes(expression, hugo)
    return collapsed.T, unique_genes, keep


def _score_metabric_per_fold(
    result, ext_X_qn_template_rna_train, ext_X_aligned,
    meth_silenced, y_ext,
):
    """Score full METABRIC with this fold's QN + scalers."""
    import torch  # noqa: E402
    # QN to THIS fold's TCGA-train RNA (re-fit per fold)
    ext_X_qn = quantile_normalize_to_train(
        ext_X_aligned, ext_X_qn_template_rna_train,
    )
    # Standardize via this fold's scalers
    ext_rna_std = result.rna_scaler.transform(ext_X_qn).astype(np.float32)
    ext_meth_std = result.meth_scaler.transform(meth_silenced).astype(np.float32)
    device = next(result.model.parameters()).device
    result.model.eval()
    with torch.no_grad():
        out = result.model(
            torch.from_numpy(ext_rna_std).to(device),
            torch.from_numpy(ext_meth_std).to(device),
        )
        proba = torch.sigmoid(out["logits"]).cpu().numpy()
    auc = float(roc_auc_score(y_ext, proba))
    bacc = float(balanced_accuracy_score(y_ext, (proba >= 0.5).astype(int)))
    return auc, bacc, ext_rna_std, ext_meth_std


def _per_fold_ig_metabric(
    result, ext_rna_std, ext_meth_std, rna_features, hallmark, device,
):
    """Per-fold IG Hallmark rollup on METABRIC (LumA + LumB poles)."""
    out: dict[str, list] = {}
    for tname, attr_t in (("LumA_pole", "lumA_pole"), ("LumB_pole", "lumB_pole")):
        attr = integrated_gradients_dmoi(
            result.model, ext_rna_std, ext_meth_std,
            target=attr_t, n_steps=N_IG_STEPS, device=str(device),
            pole_order=POLE_ORDER,
        )
        scores = pathway_aggregate(
            attr.rna_attribution, rna_features, hallmark,
        )
        out[tname] = scores
    return out


def main() -> int:
    tcga_cohort_tsv = TCGA / TCGA_COHORT_TSV_NAME
    rna_gz = TCGA / "HiSeqV2.gz"
    meth_gz = TCGA / "HumanMethylation450.gz"
    probemap = TCGA / "hm450_probemap.tsv"
    metabric_cohort_tsv = METABRIC / METABRIC_COHORT_TSV_NAME
    metabric_mrna = METABRIC / "mrna_microarray.txt"

    for p in (tcga_cohort_tsv, rna_gz, meth_gz, probemap, HALLMARK_GMT,
              metabric_cohort_tsv, metabric_mrna):
        if not p.exists():
            sys.stderr.write(f"ERROR: missing input {p}\n")
            return 1

    print("=== DMOI v0.12-A: TCGA cohort_v2 5-fold CV x full-METABRIC per fold ===")

    # --- Hallmark catalog ---
    print(f"\n--- Loading Hallmark catalog: {HALLMARK_GMT.name} ---")
    hallmark: dict[str, list[str]] = load_hallmark_gmt(HALLMARK_GMT)
    print(f"  {len(hallmark)} Hallmark sets loaded.")

    # --- TCGA cohort_v2 features (dual-modality only) ---
    print(f"\n--- Loading TCGA features from {TCGA_COHORT_TSV_NAME} ---")
    feats_all = load_features(
        cohort_tsv=tcga_cohort_tsv, rna_gz=rna_gz, meth_gz=meth_gz,
        meth_topk=10_000, dual_modality_only=True,
        positive_label=POSITIVE_LABEL,
    )
    print(f"  cohort_v2 dual-modality: n={len(feats_all.sample_ids)} "
          f"(LumB={int(feats_all.y.sum())}, "
          f"LumA={len(feats_all.y) - int(feats_all.y.sum())})")

    # --- METABRIC cohort + mRNA (load once; QN re-fit per fold) ---
    print(f"\n--- Loading METABRIC cohort from {METABRIC_COHORT_TSV_NAME} ---")
    metabric_cohort = pd.read_csv(metabric_cohort_tsv, sep="\t")
    metabric_cohort = metabric_cohort[metabric_cohort["has_rna"]].copy()
    # Restrict to LumA + LumB (drop any other groups if present)
    metabric_cohort = metabric_cohort[
        metabric_cohort["group"].isin(["LumA", "LumB"])
    ].copy()
    ext_ids_wanted = set(metabric_cohort["sample_id"].astype(str))
    ext_X_raw, ext_genes, ext_sample_ids = _load_metabric_mrna(
        metabric_mrna, ext_ids_wanted,
    )
    ext_cohort = metabric_cohort.set_index("sample_id").loc[ext_sample_ids]
    y_ext = ext_cohort["group"].map({"LumA": 0, "LumB": 1}).to_numpy().astype(np.int64)
    print(f"  METABRIC LumA/LumB with mRNA: {ext_X_raw.shape[0]} patients "
          f"(LumA={int((y_ext == 0).sum())}, LumB={int((y_ext == 1).sum())})")

    # --- Align METABRIC RNA to TCGA train gene order (once; QN per fold) ---
    print("\n--- Aligning METABRIC RNA to TCGA gene order ---")
    ext_X_aligned = align_to_train_genes(
        ext_X_raw, ext_genes, feats_all.rna_features, fill_value=0.0,
    )
    meth_ext_silenced = make_silenced_meth(
        ext_X_aligned.shape[0], feats_all.meth.shape[1],
    )

    # --- Pole masks (LumA / LumB) ---
    print("\n--- Building LumA / LumB pole masks ---")
    cis = load_hm450_cis_mapping(probemap)
    pole_masks = make_pole_masks(
        feats_all.rna_features, feats_all.meth_features, cis,
        {"LumA": POLE_LUMA, "LumB": POLE_LUMB},
        hallmark_sets={k: list(v) for k, v in hallmark.items()},
    )
    for pname, p in pole_masks.items():
        print(f"  {pname:10s} {p.summary()}")

    # --- 5-fold CV ---
    print("\n--- Running 5-fold StratifiedKFold (random_state=42) ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_splits = list(skf.split(np.zeros(len(feats_all.y)), feats_all.y))

    per_fold_rows: list[dict] = []
    per_fold_top3_luma: list[set] = []
    per_fold_top3_lumb: list[set] = []
    per_fold_metab_ig: list[dict] = []

    import torch  # noqa: E402
    for fi, (tr_idx, te_idx) in enumerate(fold_splits, start=1):
        rna_tr = feats_all.rna[tr_idx]
        meth_tr = feats_all.meth[tr_idx]
        y_tr = feats_all.y[tr_idx]
        rna_te = feats_all.rna[te_idx]
        meth_te = feats_all.meth[te_idx]
        y_te = feats_all.y[te_idx]

        print(f"\n--- Fold {fi} : train n={len(tr_idx)} "
              f"(LumB={int(y_tr.sum())}) | val n={len(te_idx)} "
              f"(LumB={int(y_te.sum())}) ---")

        result = train_one_fold(
            rna_train=rna_tr, meth_train=meth_tr, y_train=y_tr,
            rna_val=rna_te, meth_val=meth_te, y_val=y_te,
            pole_masks=pole_masks,
            fold=fi - 1,
            rna_dim=rna_tr.shape[1], meth_dim=meth_tr.shape[1],
            n_epochs=N_EPOCHS, patience=N_EPOCHS + 1,
            keep_artifacts=True,
            pole_order=POLE_ORDER,
            **FINAL_KWARGS,
        )
        if result.model is None or result.rna_scaler is None:
            sys.stderr.write(f"ERROR: fold {fi} returned no artifacts.\n")
            return 1

        tcga_val_auc = result.best_val_auc
        tcga_val_bacc = result.best_val_bacc
        print(f"  TCGA val   AUROC={tcga_val_auc:.4f} bacc={tcga_val_bacc:.4f} "
              f"(best epoch {result.best_epoch})")

        # METABRIC score per fold (re-fit QN on this fold's TCGA-train RNA)
        metab_auc, metab_bacc, ext_rna_std, ext_meth_std = _score_metabric_per_fold(
            result, rna_tr, ext_X_aligned, meth_ext_silenced, y_ext,
        )
        print(f"  METABRIC   AUROC={metab_auc:.4f} bacc={metab_bacc:.4f}")

        # Per-fold IG on METABRIC
        device = next(result.model.parameters()).device
        rollup = _per_fold_ig_metabric(
            result, ext_rna_std, ext_meth_std,
            feats_all.rna_features, hallmark, device,
        )
        per_fold_metab_ig.append(rollup)

        luma_top5 = {
            s.pathway_name for s in
            rank_pathways(rollup["LumA_pole"], by="mean_abs_ig")[:5]
        }
        lumb_top5 = {
            s.pathway_name for s in
            rank_pathways(rollup["LumB_pole"], by="mean_abs_ig")[:5]
        }
        luma_top3 = {
            s.pathway_name for s in
            rank_pathways(rollup["LumA_pole"], by="mean_abs_ig")[:3]
        }
        lumb_top3 = {
            s.pathway_name for s in
            rank_pathways(rollup["LumB_pole"], by="mean_abs_ig")[:3]
        }
        luma_hits = len(luma_top5 & EXPECTED_LUMA_TOP)
        lumb_hits = len(lumb_top5 & EXPECTED_LUMB_TOP)
        per_fold_top3_luma.append(luma_top3)
        per_fold_top3_lumb.append(lumb_top3)

        print(f"  METAB IG   LumA-hits={luma_hits}/2  LumB-hits={lumb_hits}/3")

        per_fold_rows.append(dict(
            fold=fi,
            tcga_auc=tcga_val_auc, tcga_bacc=tcga_val_bacc,
            metab_auc=metab_auc, metab_bacc=metab_bacc,
            luma_hits=luma_hits, lumb_hits=lumb_hits,
            best_epoch=result.best_epoch,
        ))

    # --- Aggregates ---
    tcga_aucs = np.array([r["tcga_auc"] for r in per_fold_rows])
    tcga_baccs = np.array([r["tcga_bacc"] for r in per_fold_rows])
    metab_aucs = np.array([r["metab_auc"] for r in per_fold_rows])
    metab_baccs = np.array([r["metab_bacc"] for r in per_fold_rows])

    print("\n--- 5-fold CV aggregates ---")
    print(f"  TCGA val   AUROC : {tcga_aucs.mean():.4f} +/- {tcga_aucs.std():.4f}")
    print(f"  TCGA val   bacc  : {tcga_baccs.mean():.4f} +/- {tcga_baccs.std():.4f}")
    print(f"  METABRIC   AUROC : {metab_aucs.mean():.4f} +/- {metab_aucs.std():.4f}")
    print(f"  METABRIC   bacc  : {metab_baccs.mean():.4f} +/- {metab_baccs.std():.4f}")

    # --- Cross-fold pathway frequency on METABRIC ---
    print("\n--- Cross-fold METABRIC pathway frequency (out of 5 folds) ---")
    luma_freq: dict[str, int] = dict.fromkeys(EXPECTED_LUMA_TOP, 0)
    lumb_freq: dict[str, int] = dict.fromkeys(EXPECTED_LUMB_TOP, 0)
    for rollup in per_fold_metab_ig:
        luma_top5 = {
            s.pathway_name for s in
            rank_pathways(rollup["LumA_pole"], by="mean_abs_ig")[:5]
        }
        lumb_top5 = {
            s.pathway_name for s in
            rank_pathways(rollup["LumB_pole"], by="mean_abs_ig")[:5]
        }
        for p in EXPECTED_LUMA_TOP:
            if p in luma_top5:
                luma_freq[p] += 1
        for p in EXPECTED_LUMB_TOP:
            if p in lumb_top5:
                lumb_freq[p] += 1
    print("  LumA expected priors (freq / 5):")
    for p, c in sorted(luma_freq.items(), key=lambda x: -x[1]):
        print(f"    {c}/5  {p}")
    print("  LumB expected priors (freq / 5):")
    for p, c in sorted(lumb_freq.items(), key=lambda x: -x[1]):
        print(f"    {c}/5  {p}")

    # --- Cross-fold top-3 stability (pairwise mean Jaccard) on METABRIC ---
    n_folds = len(per_fold_rows)

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

    luma_j = _mean_pairwise_jaccard(per_fold_top3_luma)
    lumb_j = _mean_pairwise_jaccard(per_fold_top3_lumb)
    print("\n--- METABRIC cross-fold top-3 stability ---")
    print(f"  LumA top-3 mean pairwise Jaccard : {luma_j:.4f}")
    print(f"  LumB top-3 mean pairwise Jaccard : {lumb_j:.4f}")

    # --- Audit MD ---
    AUDIT.mkdir(exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    md_path = AUDIT / "dmoi_v0.12.md"

    fold_table_rows = [
        f"| {r['fold']} | {r['tcga_auc']:.4f} | {r['tcga_bacc']:.4f} | "
        f"{r['metab_auc']:.4f} | {r['metab_bacc']:.4f} | "
        f"{r['luma_hits']} / 2 | {r['lumb_hits']} / 3 | {r['best_epoch']} |"
        for r in per_fold_rows
    ]

    md_path.write_text(
        "# DMOI v0.12-A -- TCGA cohort_v2 5-fold CV x full-METABRIC per fold\n\n"
        f"Generated: {ts}\n\n"
        "## Setup\n\n"
        "- Architecture: v0.6 base (same as v0.7-A baseline / v0.11), "
        "n_pathways=0.\n"
        "- POLE_LUMA = ER_EARLY + ER_LATE\n"
        "- POLE_LUMB = E2F_TARGETS + G2M_CHECKPOINT + MYC_TARGETS_V1\n"
        f"- TCGA cohort: cohort_v2 dual-modality, n={len(feats_all.sample_ids)} "
        f"(LumB={int(feats_all.y.sum())}, "
        f"LumA={len(feats_all.y) - int(feats_all.y.sum())}).\n"
        f"- METABRIC cohort: LumA + LumB with mRNA, "
        f"n={ext_X_aligned.shape[0]} "
        f"(LumA={int((y_ext == 0).sum())}, LumB={int((y_ext == 1).sum())}).\n"
        "- Split: 5-fold StratifiedKFold (random_state=42, matches v0.11 / "
        "v0.0 baseline CV protocol).\n"
        f"- Epochs: {N_EPOCHS}, optimizer: AdamW(lr=1e-4, wd=1e-4), "
        "BCEWithLogitsLoss + aux=0.3, pick_best_epoch=True.\n"
        "- METABRIC QN: re-fit per fold against the fold's TCGA train RNA "
        "distribution (correct cross-validation protocol; v0.4 single-shot "
        "fit once on the full TCGA train).\n"
        "- METABRIC meth branch silenced (no HM450 in METABRIC, same as "
        "v0.4 / v0.10).\n\n"
        "## Aggregate variance bands (5-fold)\n\n"
        "| Metric | mean | std | Reference |\n"
        "|---|---|---|---|\n"
        f"| TCGA val AUROC | **{tcga_aucs.mean():.4f}** | "
        f"{tcga_aucs.std():.4f} | v0.6 5-fold ref: 0.954 +/- 0.017 |\n"
        f"| TCGA val bacc  | {tcga_baccs.mean():.4f} | "
        f"{tcga_baccs.std():.4f} |  |\n"
        f"| METABRIC AUROC | **{metab_aucs.mean():.4f}** | "
        f"{metab_aucs.std():.4f} | v0.4 single-shot ref: 0.909 |\n"
        f"| METABRIC bacc  | {metab_baccs.mean():.4f} | "
        f"{metab_baccs.std():.4f} |  |\n\n"
        "## Per-fold table\n\n"
        "| Fold | TCGA val AUROC | TCGA val bacc | METABRIC AUROC | METABRIC bacc | "
        "LumA IG hits | LumB IG hits | best epoch |\n"
        "|---|---|---|---|---|---|---|---|\n"
        + "\n".join(fold_table_rows) + "\n\n"
        "## Cross-fold METABRIC pathway frequency\n\n"
        "### LumA pole -- frequency in per-fold top-5 (out of 5 folds)\n\n"
        "| Pathway | Frequency |\n|---|---|\n"
        + "\n".join(
            f"| `{p}` | {c} / 5 |"
            for p, c in sorted(luma_freq.items(), key=lambda x: -x[1])
        )
        + "\n\n"
        "### LumB pole -- frequency in per-fold top-5 (out of 5 folds)\n\n"
        "| Pathway | Frequency |\n|---|---|\n"
        + "\n".join(
            f"| `{p}` | {c} / 5 |"
            for p, c in sorted(lumb_freq.items(), key=lambda x: -x[1])
        )
        + "\n\n"
        "## METABRIC cross-fold top-3 stability (pairwise mean Jaccard)\n\n"
        f"- LumA pole top-3 mean pairwise Jaccard : **{luma_j:.4f}**\n"
        f"- LumB pole top-3 mean pairwise Jaccard : **{lumb_j:.4f}**\n\n"
        "Jaccard of 1.0 means every fold picked the same top-3 pathways on "
        "METABRIC.\n\n"
        "## Reading\n\n"
        "v0.12-A pairs with v0.11. v0.11 showed that the v0.9 / v0.10 four-axis "
        "closure on the Luminal-vs-Basal task is split-invariant on TCGA "
        "cohort_v3. v0.12-A asks the same question one task axis over -- the "
        "v0.4 / v0.6 LumA-vs-LumB cohort_v2 narrative -- AND adds the new "
        "cross-cohort variance band.\n\n"
        "- If TCGA val AUROC is within v0.6 5-fold ref band (0.954 +/- 0.017), "
        "the cohort_v2 internal stability is reproduced.\n"
        "- If METABRIC AUROC is around v0.4 single-shot ref (0.909) with low "
        "std, the cross-cohort metric is split-invariant: the v0.4 0.909 was "
        "not a lucky TCGA train split.\n"
        "- If METABRIC LumA / LumB priors-hit frequency is >= 4 / 5 on the "
        "expected pathways AND top-3 Jaccard is high, the v0.5 / v0.6 / v0.10 "
        "cross-cohort biology (ER for LumA, cell-cycle for LumB) is also "
        "split-invariant.\n\n"
        "## Honest scope\n\n"
        "- Same architecture, same priors, same hyperparameters as v0.6.\n"
        "- pick_best_epoch=True is the standard CV protocol.\n"
        "- Each fold has ~25 LumB patients in TCGA val. AUROC variance on "
        "the TCGA val side is wider than v0.6's single-test (n=27 LumB).\n"
        "- METABRIC scoring per fold uses re-fit QN on the fold's TCGA "
        "train RNA -- the right thing under proper CV. v0.4 single-shot "
        "fit QN once on the full TCGA train (not directly comparable to a "
        "single fold's METABRIC AUROC; the 5-fold band IS the comparison).\n"
        "- METABRIC LumA n=700, LumB n=475 (LumB-majority within Luminal\n"
        "  selection; opposite of TCGA cohort_v2 LumA-majority).\n\n"
        "## Reproduce\n\n"
        "```bash\n"
        "python scripts/eval_dmoi_v0.12_cv.py\n"
        "```\n",
    )
    print(f"\nWrote {md_path}")

    print("\n=== DMOI v0.12-A summary ===")
    print(f"  TCGA val   AUROC mean+/-std : "
          f"{tcga_aucs.mean():.4f} +/- {tcga_aucs.std():.4f} "
          "(v0.6 5-fold ref: 0.954 +/- 0.017)")
    print(f"  METABRIC   AUROC mean+/-std : "
          f"{metab_aucs.mean():.4f} +/- {metab_aucs.std():.4f} "
          "(v0.4 single-shot ref: 0.909)")
    n_5_5_luma = sum(1 for c in luma_freq.values() if c == 5)
    n_5_5_lumb = sum(1 for c in lumb_freq.values() if c == 5)
    print(f"  METABRIC LumA priors 5/5 in top-5 (out of 2): {n_5_5_luma} / 2")
    print(f"  METABRIC LumB priors 5/5 in top-5 (out of 3): {n_5_5_lumb} / 3")
    print(f"  METABRIC LumA top-3 mean pairwise Jaccard : {luma_j:.4f}")
    print(f"  METABRIC LumB top-3 mean pairwise Jaccard : {lumb_j:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
