#!/usr/bin/env python3
"""DMOI v0.9: Luminal-vs-Basal cross-task generalization of v0.6 architecture.

The v0.6 architectural commitment (gene-level hypothesis-conditioned attention
+ pole-specific Hallmark gene-set priors + dual-perspective fusion) was tested
on TCGA-BRCA LumA-vs-LumB. v0.7+v0.8 confirmed that the v0.6 commitment is the
right architectural level for that task. v0.9 asks: does the same architecture
transfer to a different classification axis -- Luminal (LumA+LumB) vs Basal --
with the only change being the pole-defining Hallmark sets?

If AUROC is competitive and per-pole IG surfaces biologically coherent pathways
(Luminal -> ER + Androgen response; Basal -> EMT + cell-cycle / MYC), the v0.6
framework reads as task-agnostic. If AUROC drops or the IG ranking is not
biologically interpretable, the architecture is task-specific and the v0.6
LumA-vs-LumB success is partly cohort + task luck.

Pipeline:
  1. Load cohort_v3.tsv (Luminal=415, Basal=87 dual-modality; 401 train / 101 test).
  2. Load full 50-set Hallmark catalog (so POLE_LUMINAL / POLE_BASAL can reference
     HALLMARK_ANDROGEN_RESPONSE and HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION).
  3. Build pole masks using `make_pole_masks(..., hallmark_sets=hallmark_full)`.
  4. Train DMOI v0.6 architecture (n_pathways=0, no pathway-attention branch)
     with `keep_artifacts=True` on TCGA train.
  5. Score TCGA test, compute IG attribution for both poles + final logit, roll
     up to the full 50-set Hallmark catalog (same protocol as v0.6).
  6. Write audit/dmoi_v0.9.md.

Honest scope:
  - Same architecture as v0.6 (no model changes); only cohort + Hallmark pole
    priors change.
  - Class imbalance is 4.8:1 (Luminal majority) -- pos_weight handles it but
    the AUROC headline is the right metric (balanced-accuracy is reported too).
  - Test set has 18 Basal patients -- AUROC variance is higher than v0.6's
    n=84 test split; we don't over-interpret 0.005-level differences.
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
from dmoi_brca.features import FeatureMatrices, load_features  # noqa: E402
from dmoi_brca.hallmark import load_hallmark_gmt  # noqa: E402
from dmoi_brca.hypothesis_attention import (  # noqa: E402
    load_hm450_cis_mapping,
    make_pole_masks,
)
from dmoi_brca.pathway import pathway_aggregate, rank_pathways  # noqa: E402
from dmoi_brca.priors import POLE_BASAL, POLE_LUMINAL  # noqa: E402
from dmoi_brca.train import train_one_fold  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TCGA = REPO / "data" / "tcga_brca"
AUDIT = REPO / "audit"
HALLMARK_GMT = REPO / "data" / "msigdb" / "h.all.v2024.1.Hs.symbols.gmt"

POLE_ORDER = ("Luminal", "Basal")
POSITIVE_LABEL = "Basal"  # Basal = 1 (minority class)
COHORT_TSV_NAME = "cohort_v3.tsv"

# v0.6 hyperparameters carry over unchanged so the only varying factor is
# cohort + pole priors -- this isolates the cross-task generalization question.
FINAL_KWARGS = dict(
    latent_dim=128, rna_hidden=(1024, 256), meth_hidden=(512,),
    fuse_hidden=(128,), fuse_out=64, head_hidden=32, dropout=0.3,
    batch_size=64, lr=1e-4, weight_decay=1e-4,
    seed=42, device="auto", verbose=False,
    use_disagreement=True, aux_weight=0.3,
    calibration_frac=0.15, pick_best_epoch=False,
)
N_EPOCHS = 15
N_IG_STEPS = 50


def main() -> int:
    cohort_tsv = TCGA / COHORT_TSV_NAME
    rna_gz = TCGA / "HiSeqV2.gz"
    meth_gz = TCGA / "HumanMethylation450.gz"
    probemap = TCGA / "hm450_probemap.tsv"

    for p in (cohort_tsv, rna_gz, meth_gz, probemap, HALLMARK_GMT):
        if not p.exists():
            sys.stderr.write(f"ERROR: missing input {p}\n")
            return 1

    print("=== DMOI v0.9: Luminal-vs-Basal cross-task generalization ===")

    # --- Hallmark catalog (needed for POLE_LUMINAL/POLE_BASAL) ---
    print(f"\n--- Loading Hallmark catalog: {HALLMARK_GMT.name} ---")
    hallmark_full: dict[str, list[str]] = load_hallmark_gmt(HALLMARK_GMT)
    print(f"  {len(hallmark_full)} Hallmark sets loaded.")

    # --- TCGA features (cohort_v3) ---
    print(f"\n--- Loading TCGA features from {COHORT_TSV_NAME} ---")
    feats_all = load_features(
        cohort_tsv=cohort_tsv, rna_gz=rna_gz, meth_gz=meth_gz,
        meth_topk=10_000, dual_modality_only=True,
        positive_label=POSITIVE_LABEL,
    )
    cohort_df = pd.read_csv(cohort_tsv, sep="\t")
    cohort_df = cohort_df[cohort_df["has_rna"] & cohort_df["has_meth"]].copy()
    sample_to_split = dict(zip(
        cohort_df["sample_id"], cohort_df["split"], strict=False,
    ))
    train_idx = np.array([
        i for i, sid in enumerate(feats_all.sample_ids)
        if sample_to_split.get(sid) == "train"
    ])
    test_idx = np.array([
        i for i, sid in enumerate(feats_all.sample_ids)
        if sample_to_split.get(sid) == "test"
    ])

    def _slice(idx):
        return FeatureMatrices(
            sample_ids=[feats_all.sample_ids[i] for i in idx],
            y=feats_all.y[idx],
            rna=feats_all.rna[idx], meth=feats_all.meth[idx],
            rna_features=feats_all.rna_features,
            meth_features=feats_all.meth_features,
        )

    feats = _slice(train_idx)
    feats_test = _slice(test_idx)
    print(f"  TCGA train: n={len(feats.sample_ids)} "
          f"(Basal={int(feats.y.sum())}, Luminal={len(feats.y)-int(feats.y.sum())})")
    print(f"  TCGA test:  n={len(feats_test.sample_ids)} "
          f"(Basal={int(feats_test.y.sum())}, Luminal={len(feats_test.y)-int(feats_test.y.sum())})")

    # --- Pole masks (gmt override required for POLE_LUMINAL/POLE_BASAL) ---
    print("\n--- Building Luminal / Basal pole masks ---")
    cis = load_hm450_cis_mapping(probemap)
    pole_masks = make_pole_masks(
        feats.rna_features, feats.meth_features, cis,
        {"Luminal": POLE_LUMINAL, "Basal": POLE_BASAL},
        hallmark_sets={k: list(v) for k, v in hallmark_full.items()},
    )
    for pname, p in pole_masks.items():
        print(f"  {pname:10s} {p.summary()}")

    # --- Train DMOI v0.6 architecture on TCGA train ---
    print("\n--- Training v0.9 model (v0.6 architecture, new poles) ---")
    result = train_one_fold(
        rna_train=feats.rna, meth_train=feats.meth, y_train=feats.y,
        rna_val=feats_test.rna, meth_val=feats_test.meth,
        y_val=feats_test.y,
        pole_masks=pole_masks,
        fold=0,
        rna_dim=feats.rna.shape[1], meth_dim=feats.meth.shape[1],
        n_epochs=N_EPOCHS, patience=N_EPOCHS + 1,
        keep_artifacts=True,
        pole_order=POLE_ORDER,
        **FINAL_KWARGS,
    )
    print(f"  TCGA test AUROC : {result.best_val_auc:.4f} "
          f"(v0.6 LumA-vs-LumB ref: 0.9682)")
    print(f"  TCGA test bacc  : {result.best_val_bacc:.4f}")
    if result.model is None or result.rna_scaler is None or result.meth_scaler is None:
        sys.stderr.write("ERROR: train_one_fold returned no artifacts.\n")
        return 1

    # --- IG attribution + Hallmark rollup ---
    AUDIT.mkdir(exist_ok=True)
    import torch  # noqa: E402
    device = next(result.model.parameters()).device
    # integrated_gradients_dmoi expects numpy arrays (it wraps them in
    # torch.from_numpy internally), so we keep the standardized matrices
    # as numpy here -- don't pre-convert to torch tensors.
    rna_test_std = result.rna_scaler.transform(feats_test.rna).astype(np.float32)
    meth_test_std = result.meth_scaler.transform(feats_test.meth).astype(np.float32)

    pathway_results: dict[str, list] = {}
    print("\n--- IG + Hallmark pathway rollup per target ---")
    for target_name in ("Luminal_pole", "Basal_pole", "final_logit"):
        # attribution.py expects target names "lumA_pole" / "lumB_pole" /
        # "final_logit" mapped to pole_order indices 0 / 1. Translate.
        if target_name == "Luminal_pole":
            attr_target = "lumA_pole"  # pole index 0
        elif target_name == "Basal_pole":
            attr_target = "lumB_pole"  # pole index 1
        else:
            attr_target = "final_logit"
        print(f"\n  --- {target_name} on TCGA test ---")
        attr = integrated_gradients_dmoi(
            result.model, rna_test_std, meth_test_std,
            target=attr_target, n_steps=N_IG_STEPS, device=str(device),
            pole_order=POLE_ORDER,
        )
        scores = pathway_aggregate(
            attr.rna_attribution, feats.rna_features, hallmark_full,
        )
        pathway_results[target_name] = scores
        top = rank_pathways(scores, by="mean_abs_ig")[:5]
        for s in top:
            print(f"    {s.pathway_name:45s}  mean|IG| {s.mean_abs_ig:.5f}  "
                  f"signed_mean {s.signed_mean:+.5f}  "
                  f"({s.n_pathway_genes_in_inputs} genes in inputs)")

    # --- Cross-pole sanity: does Luminal pole load ER, Basal pole load EMT/MYC? ---
    print("\n--- Cross-pole biology sanity check ---")
    expected_luminal_top = {
        "HALLMARK_ESTROGEN_RESPONSE_EARLY",
        "HALLMARK_ESTROGEN_RESPONSE_LATE",
        "HALLMARK_ANDROGEN_RESPONSE",
    }
    expected_basal_top = {
        "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",
        "HALLMARK_MYC_TARGETS_V1",
        "HALLMARK_G2M_CHECKPOINT",
        "HALLMARK_E2F_TARGETS",
        "HALLMARK_MYC_TARGETS_V2",
    }
    lum_top5 = {
        s.pathway_name
        for s in rank_pathways(pathway_results["Luminal_pole"], by="mean_abs_ig")[:5]
    }
    bas_top5 = {
        s.pathway_name
        for s in rank_pathways(pathway_results["Basal_pole"], by="mean_abs_ig")[:5]
    }
    lum_match = sorted(lum_top5 & expected_luminal_top)
    bas_match = sorted(bas_top5 & expected_basal_top)
    print(f"  Luminal pole top-5 expected-set match : {len(lum_match)} / 3 -- {lum_match}")
    print(f"  Basal   pole top-5 expected-set match : {len(bas_match)} / 5 -- {bas_match}")

    # --- Audit MD ---
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    md_path = AUDIT / "dmoi_v0.9.md"

    def _rank_table(target: str) -> str:
        ranked = rank_pathways(pathway_results[target], by="mean_abs_ig")[:10]
        rows = [
            "| Rank | Pathway | mean \\|IG\\| | signed_mean |",
            "|---|---|---|---|",
        ]
        for i, s in enumerate(ranked, 1):
            rows.append(
                f"| {i} | `{s.pathway_name}` | {s.mean_abs_ig:.5f} | "
                f"{s.signed_mean:+.5f} |",
            )
        return "\n".join(rows)

    md_path.write_text(
        "# DMOI v0.9 -- Luminal-vs-Basal cross-task generalization\n\n"
        f"Generated: {ts}\n\n"
        "## Setup\n\n"
        f"- Architecture: v0.6 base (no model changes; n_pathways=0). "
        "Only cohort and pole-defining Hallmark sets differ from v0.6.\n"
        f"- Pole pair: Luminal (LumA + LumB) vs Basal (PAM50call_RNAseq).\n"
        f"- POLE_LUMINAL = ESTROGEN_RESPONSE_EARLY + LATE + ANDROGEN_RESPONSE.\n"
        f"- POLE_BASAL = EPITHELIAL_MESENCHYMAL_TRANSITION + MYC_TARGETS_V1 + "
        "G2M_CHECKPOINT.\n"
        f"- Train cohort: TCGA cohort_v3 train split, n={len(feats.sample_ids)} "
        f"(Basal={int(feats.y.sum())}, Luminal={len(feats.y)-int(feats.y.sum())}).\n"
        f"- TCGA test:    n={len(feats_test.sample_ids)} "
        f"(Basal={int(feats_test.y.sum())}, Luminal={len(feats_test.y)-int(feats_test.y.sum())}).\n"
        f"- Epochs: {N_EPOCHS}, optimizer: AdamW(lr=1e-4, wd=1e-4), "
        f"BCEWithLogitsLoss + aux=0.3, pick_best_epoch=False.\n\n"
        "## Headline AUROC\n\n"
        f"| Metric | DMOI v0.9 |\n|---|---|\n"
        f"| TCGA held-out test AUROC | **{result.best_val_auc:.4f}** |\n"
        f"| TCGA held-out test bacc  | {result.best_val_bacc:.4f} |\n"
        f"| v0.6 LumA-vs-LumB ref AUROC | 0.9682 |\n\n"
        "## Per-pole IG top-10 pathways\n\n"
        "### Luminal pole\n\n" + _rank_table("Luminal_pole") + "\n\n"
        "### Basal pole\n\n" + _rank_table("Basal_pole") + "\n\n"
        "### final_logit\n\n" + _rank_table("final_logit") + "\n\n"
        "## Cross-pole biology sanity check\n\n"
        f"Expected Luminal-pole top-5 to include some of "
        f"{{ER_EARLY, ER_LATE, ANDROGEN_RESPONSE}}.\n"
        f"Expected Basal-pole top-5 to include some of "
        "{EMT, MYC_TARGETS_V1, G2M_CHECKPOINT, E2F_TARGETS, MYC_TARGETS_V2}.\n\n"
        f"- Luminal pole top-5 ∩ expected = {len(lum_match)} / 3 : "
        f"{', '.join(f'`{p}`' for p in lum_match) or '(none)'}\n"
        f"- Basal pole top-5 ∩ expected = {len(bas_match)} / 5 : "
        f"{', '.join(f'`{p}`' for p in bas_match) or '(none)'}\n\n"
        "## Honest scope\n\n"
        "- Same architecture as v0.6 (no model changes); only cohort and "
        "pole priors change. Cross-task generalization is the only thing "
        "under test.\n"
        f"- Class imbalance is 4.8:1 (Luminal majority); test set has "
        f"{int(feats_test.y.sum())} Basal patients. AUROC variance is higher "
        "than v0.6's n=84 LumA-vs-LumB test.\n"
        "- No METABRIC external validation in v0.9 -- METABRIC's "
        "Luminal-vs-Basal subset would require a parallel cohort builder + "
        "ER/PAM50 mapping; deferred to v0.10+ if v0.9 transfers.\n\n"
        "## Reproduce\n\n"
        "```bash\n"
        "python scripts/build_cohort_v3.py    # builds data/tcga_brca/cohort_v3.tsv\n"
        "python scripts/eval_dmoi_v0.9.py     # ~7 min on MPS\n"
        "```\n",
    )
    print(f"\nWrote {md_path}")

    print("\n=== DMOI v0.9 summary ===")
    print(f"  TCGA AUROC: {result.best_val_auc:.4f} "
          f"(v0.6 LumA-vs-LumB ref: 0.9682)")
    print(f"  TCGA bacc : {result.best_val_bacc:.4f}")
    print(f"  Luminal pole top-5 expected-match: {len(lum_match)} / 3")
    print(f"  Basal pole top-5 expected-match  : {len(bas_match)} / 5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
