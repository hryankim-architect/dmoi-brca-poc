#!/usr/bin/env python3
"""DMOI v0.14: HER2-vs-Luminal cross-task + cross-cohort generalization.

Third classification axis for the task-reusability claim (after LumA-vs-LumB in
v0.6 and Luminal-vs-Basal in v0.9/v0.10). The v0.6 architecture is unchanged;
only the cohort (cohort_v4), the pole-defining Hallmark priors
(POLE_HER2 / POLE_LUMINAL_ER), and the positive label change.

Structure mirrors eval_metabric_v0.10.py: train once on TCGA cohort_v4 train,
report TCGA held-out test AUROC, then score the METABRIC external cohort
(RNA-only, meth silenced) and roll up per-pole IG to the 50-set Hallmark catalog.

Limitations:
  - HER2+ is the small TCGA class (~58 dual-modality). The single-split TCGA
    number is noisy; the METABRIC external (Her2 n≈224) carries the statistical
    weight. A 5-fold CV variant is the natural follow-up.
  - HER2 definition differs slightly across cohorts: TCGA = clinical HER2+
    (HER2_Final_Status); METABRIC = PAM50 'Her2' (CLAUDIN_SUBTYPE). Recorded.
  - METABRIC has no HM450 methylation -> meth branch silenced + QN to TCGA RNA.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone  # noqa: UP017
from pathlib import Path

UTC = timezone.utc  # noqa: UP017

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, roc_auc_score  # noqa: E402

from dmoi_brca import audit, tracking  # noqa: E402
from dmoi_brca.attribution import integrated_gradients_dmoi  # noqa: E402
from dmoi_brca.external import (  # noqa: E402
    align_to_train_genes,
    collapse_duplicate_genes,
    make_silenced_meth,
    quantile_normalize_to_train,
)
from dmoi_brca.features import FeatureMatrices, load_features  # noqa: E402
from dmoi_brca.hallmark import load_hallmark_gmt  # noqa: E402
from dmoi_brca.hypothesis_attention import (  # noqa: E402
    load_hm450_cis_mapping,
    make_pole_masks,
)
from dmoi_brca.pathway import pathway_aggregate, rank_pathways  # noqa: E402
from dmoi_brca.priors import POLE_HER2, POLE_LUMINAL_ER  # noqa: E402
from dmoi_brca.train import train_one_fold  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TCGA = REPO / "data" / "tcga_brca"
METABRIC = REPO / "data" / "metabric"
AUDIT = REPO / "audit"
HALLMARK_GMT = REPO / "data" / "msigdb" / "h.all.v2024.1.Hs.symbols.gmt"

POLE_ORDER = ("Luminal", "HER2")
POSITIVE_LABEL = "HER2"  # HER2 = 1 (minority class)
TCGA_COHORT_TSV_NAME = "cohort_v4.tsv"
METABRIC_COHORT_TSV_NAME = "cohort_v4.tsv"
JOB_ID = "dmoi-her2-axis-v0.14"

# v0.6 hyperparameters carry over unchanged.
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

EXPECTED_LUMINAL_TOP = {
    "HALLMARK_ESTROGEN_RESPONSE_EARLY",
    "HALLMARK_ESTROGEN_RESPONSE_LATE",
}
EXPECTED_HER2_TOP = {
    "HALLMARK_PI3K_AKT_MTOR_SIGNALING",
    "HALLMARK_MTORC1_SIGNALING",
    "HALLMARK_G2M_CHECKPOINT",
}


def _load_metabric_mrna(mrna_path: Path, cohort_ids: set[str]):
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

    print("=== DMOI v0.14: HER2-vs-Luminal cross-task + cross-cohort ===")

    print(f"\n--- Loading Hallmark catalog: {HALLMARK_GMT.name} ---")
    hallmark_full: dict[str, list[str]] = load_hallmark_gmt(HALLMARK_GMT)
    print(f"  {len(hallmark_full)} Hallmark sets loaded.")

    print(f"\n--- Loading TCGA features from {TCGA_COHORT_TSV_NAME} ---")
    feats_all = load_features(
        cohort_tsv=tcga_cohort_tsv, rna_gz=rna_gz, meth_gz=meth_gz,
        meth_topk=10_000, dual_modality_only=True,
        positive_label=POSITIVE_LABEL,
    )
    cohort_df = pd.read_csv(tcga_cohort_tsv, sep="\t")
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
          f"(HER2={int(feats.y.sum())}, "
          f"Luminal={len(feats.y) - int(feats.y.sum())})")
    print(f"  TCGA test:  n={len(feats_test.sample_ids)} "
          f"(HER2={int(feats_test.y.sum())}, "
          f"Luminal={len(feats_test.y) - int(feats_test.y.sum())})")

    print(f"\n--- Loading METABRIC cohort from {METABRIC_COHORT_TSV_NAME} ---")
    metabric_cohort = pd.read_csv(metabric_cohort_tsv, sep="\t")
    metabric_cohort = metabric_cohort[metabric_cohort["has_rna"]].copy()
    ext_ids_wanted = set(metabric_cohort["sample_id"].astype(str))
    ext_X_raw, ext_genes, ext_sample_ids = _load_metabric_mrna(
        metabric_mrna, ext_ids_wanted,
    )
    ext_cohort = metabric_cohort.set_index("sample_id").loc[ext_sample_ids]
    y_ext = ext_cohort["group"].map({"Luminal": 0, "HER2": 1}).to_numpy().astype(np.int64)
    print(f"  METABRIC HER2/Luminal with mRNA: {ext_X_raw.shape[0]} patients "
          f"(Luminal={int((y_ext == 0).sum())}, HER2={int((y_ext == 1).sum())})")

    print("\n--- Aligning METABRIC RNA to TCGA train gene order + QN ---")
    ext_X_aligned = align_to_train_genes(
        ext_X_raw, ext_genes, feats.rna_features, fill_value=0.0,
    )
    ext_X_qn = quantile_normalize_to_train(ext_X_aligned, feats.rna)
    meth_ext_silenced = make_silenced_meth(ext_X_qn.shape[0], feats.meth.shape[1])

    print("\n--- Building Luminal / HER2 pole masks ---")
    cis = load_hm450_cis_mapping(probemap)
    pole_masks = make_pole_masks(
        feats.rna_features, feats.meth_features, cis,
        {"Luminal": POLE_LUMINAL_ER, "HER2": POLE_HER2},
        hallmark_sets={k: list(v) for k, v in hallmark_full.items()},
    )
    for pname, pmask in pole_masks.items():
        print(f"  {pname:10s} {pmask.summary()}")

    print("\n--- Training v0.14 model (v0.6 architecture, HER2 poles) ---")
    result = train_one_fold(
        rna_train=feats.rna, meth_train=feats.meth, y_train=feats.y,
        rna_val=feats_test.rna, meth_val=feats_test.meth, y_val=feats_test.y,
        pole_masks=pole_masks, fold=0,
        rna_dim=feats.rna.shape[1], meth_dim=feats.meth.shape[1],
        n_epochs=N_EPOCHS, patience=N_EPOCHS + 1,
        keep_artifacts=True, pole_order=POLE_ORDER,
        **FINAL_KWARGS,
    )
    if result.model is None or result.rna_scaler is None or result.meth_scaler is None:
        sys.stderr.write("ERROR: train_one_fold returned no artifacts.\n")
        return 1
    tcga_test_auc = result.best_val_auc
    tcga_test_bacc = result.best_val_bacc
    print(f"  TCGA test AUROC : {tcga_test_auc:.4f}")
    print(f"  TCGA test bacc  : {tcga_test_bacc:.4f}")

    print("\n--- Scoring METABRIC (RNA-only, meth silenced) ---")
    import torch  # noqa: PLC0415
    device = next(result.model.parameters()).device
    ext_rna_std = result.rna_scaler.transform(ext_X_qn).astype(np.float32)
    ext_meth_std = result.meth_scaler.transform(meth_ext_silenced).astype(np.float32)
    result.model.eval()
    with torch.no_grad():
        ext_out = result.model(
            torch.from_numpy(ext_rna_std).to(device),
            torch.from_numpy(ext_meth_std).to(device),
        )
        ext_proba = torch.sigmoid(ext_out["logits"]).cpu().numpy()
    metab_auc = float(roc_auc_score(y_ext, ext_proba))
    metab_bacc = float(balanced_accuracy_score(y_ext, (ext_proba >= 0.5).astype(int)))
    print(f"  METABRIC AUROC : {metab_auc:.4f}")
    print(f"  METABRIC bacc  : {metab_bacc:.4f}")

    AUDIT.mkdir(exist_ok=True)
    print("\n--- IG + Hallmark pathway rollup on METABRIC ---")
    pathway_results: dict[str, list] = {}
    for target_name in ("Luminal_pole", "HER2_pole", "final_logit"):
        if target_name == "Luminal_pole":
            attr_target = "lumA_pole"  # pole index 0
        elif target_name == "HER2_pole":
            attr_target = "lumB_pole"  # pole index 1
        else:
            attr_target = "final_logit"
        print(f"\n  --- {target_name} on METABRIC ---")
        attr = integrated_gradients_dmoi(
            result.model, ext_rna_std, ext_meth_std,
            target=attr_target, n_steps=N_IG_STEPS, device=str(device),
            pole_order=POLE_ORDER,
        )
        scores = pathway_aggregate(
            attr.rna_attribution, feats.rna_features, hallmark_full,
        )
        pathway_results[target_name] = scores
        for s in rank_pathways(scores, by="mean_abs_ig")[:5]:
            print(f"    {s.pathway_name:45s}  mean|IG| {s.mean_abs_ig:.5f}  "
                  f"signed_mean {s.signed_mean:+.5f}  "
                  f"({s.n_pathway_genes_in_inputs} genes)")

    lum_top5 = {s.pathway_name for s in
                rank_pathways(pathway_results["Luminal_pole"], by="mean_abs_ig")[:5]}
    her2_top5 = {s.pathway_name for s in
                 rank_pathways(pathway_results["HER2_pole"], by="mean_abs_ig")[:5]}
    lum_match = sorted(lum_top5 & EXPECTED_LUMINAL_TOP)
    her2_match = sorted(her2_top5 & EXPECTED_HER2_TOP)
    print("\n--- Cross-pole biology sanity check (METABRIC) ---")
    print(f"  Luminal pole top-5 ∩ expected : {len(lum_match)} / 2 -- {lum_match}")
    print(f"  HER2    pole top-5 ∩ expected : {len(her2_match)} / 3 -- {her2_match}")

    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    md_path = AUDIT / "dmoi_v0.14.md"

    def _rank_table(target: str) -> str:
        ranked = rank_pathways(pathway_results[target], by="mean_abs_ig")[:10]
        rows = ["| Rank | Pathway | mean \\|IG\\| | signed_mean |", "|---|---|---|---|"]
        for i, s in enumerate(ranked, 1):
            rows.append(f"| {i} | `{s.pathway_name}` | {s.mean_abs_ig:.5f} | {s.signed_mean:+.5f} |")
        return "\n".join(rows)

    md_path.write_text(
        "# DMOI v0.14 -- HER2-vs-Luminal cross-task + cross-cohort generalization\n\n"
        f"Generated: {ts}\n\n"
        "## Setup\n\n"
        "- Architecture: v0.6 base (no model changes; n_pathways=0). Only cohort "
        "and pole-defining Hallmark sets differ.\n"
        "- Pole pair: HER2 (clinical HER2+) vs Luminal (LumA+LumB).\n"
        "- POLE_HER2 = PI3K_AKT_MTOR_SIGNALING + MTORC1_SIGNALING + G2M_CHECKPOINT.\n"
        "- POLE_LUMINAL_ER = ESTROGEN_RESPONSE_EARLY + ESTROGEN_RESPONSE_LATE.\n"
        f"- TCGA train: cohort_v4 train, n={len(feats.sample_ids)} "
        f"(HER2={int(feats.y.sum())}, Luminal={len(feats.y) - int(feats.y.sum())}).\n"
        f"- TCGA test:  n={len(feats_test.sample_ids)} "
        f"(HER2={int(feats_test.y.sum())}, Luminal={len(feats_test.y) - int(feats_test.y.sum())}).\n"
        f"- METABRIC external: n={ext_X_qn.shape[0]} "
        f"(Luminal={int((y_ext == 0).sum())}, HER2={int((y_ext == 1).sum())}). "
        "RNA-only + meth silenced + QN to TCGA train RNA.\n\n"
        "## Headline AUROC\n\n"
        "| Cohort | AUROC | bacc |\n|---|---|---|\n"
        f"| TCGA held-out test | **{tcga_test_auc:.4f}** | {tcga_test_bacc:.4f} |\n"
        f"| METABRIC external  | **{metab_auc:.4f}** | {metab_bacc:.4f} |\n\n"
        "## Per-pole IG top-10 pathways (METABRIC)\n\n"
        "### Luminal pole\n\n" + _rank_table("Luminal_pole") + "\n\n"
        "### HER2 pole\n\n" + _rank_table("HER2_pole") + "\n\n"
        "### final_logit\n\n" + _rank_table("final_logit") + "\n\n"
        "## Cross-pole biology sanity check (METABRIC)\n\n"
        "Expected Luminal-pole top-5 to include {ER_EARLY, ER_LATE}.\n"
        "Expected HER2-pole top-5 to include {PI3K_AKT_MTOR, MTORC1, G2M_CHECKPOINT}.\n\n"
        f"- Luminal pole top-5 ∩ expected = {len(lum_match)} / 2 : "
        f"{', '.join(f'`{p}`' for p in lum_match) or '(none)'}\n"
        f"- HER2 pole top-5 ∩ expected = {len(her2_match)} / 3 : "
        f"{', '.join(f'`{p}`' for p in her2_match) or '(none)'}\n\n"
        "## Limitations\n\n"
        f"- HER2+ is the small TCGA class (train HER2={int(feats.y.sum())}, "
        f"test HER2={int(feats_test.y.sum())}). The single-split TCGA AUROC is "
        "noisy; the METABRIC external (HER2 n="
        f"{int((y_ext == 1).sum())}) carries the statistical weight. A 5-fold CV "
        "variant is the natural follow-up before quoting a TCGA headline.\n"
        "- Definitional difference across cohorts: TCGA HER2 = clinical HER2+ "
        "(HER2_Final_Status); METABRIC HER2 = PAM50 'Her2' (CLAUDIN_SUBTYPE). "
        "Recorded as a cross-cohort caveat.\n"
        "- METABRIC has no HM450 methylation -> meth branch silenced + per-gene "
        "QN to TCGA train RNA (v0.2/v0.4/v0.6/v0.10 protocol).\n"
        "- Same architecture as v0.6/v0.9 (no model changes); only cohort + pole "
        "priors change. This is a reusability demonstration, not a powered result.\n\n"
        "## Reproduce\n\n"
        "```bash\n"
        "python scripts/build_cohort_v4.py            # TCGA cohort_v4\n"
        "python scripts/build_metabric_cohort_v4.py   # METABRIC cohort_v4\n"
        "python scripts/eval_dmoi_v0.14.py\n"
        "```\n",
    )
    print(f"\nWrote {md_path}")

    # Substrate: audit ledger + MLflow (best-effort).
    audit.emit(
        "her2_axis_v0.14", JOB_ID,
        fields={
            "tcga_test_auroc": tcga_test_auc, "tcga_test_bacc": tcga_test_bacc,
            "metabric_auroc": metab_auc, "metabric_bacc": metab_bacc,
            "n_tcga_train": int(len(feats.sample_ids)),
            "n_metabric": int(ext_X_qn.shape[0]),
            "luminal_pole_match": len(lum_match), "her2_pole_match": len(her2_match),
        },
    )
    try:
        if tracking.is_enabled():
            with tracking.run("v0.14-her2-axis", experiment="dmoi-brca"):
                tracking.log_params({"positive_label": POSITIVE_LABEL, "n_epochs": N_EPOCHS})
                tracking.log_metrics({
                    "tcga_test_auroc": tcga_test_auc, "metabric_auroc": metab_auc,
                    "metabric_bacc": metab_bacc,
                })
                tracking.log_artifact(str(md_path))
    except Exception as exc:  # noqa: BLE001 — tracking must never be pipeline-fatal
        print(f"  (MLflow logging skipped: {exc})")

    print("\n=== DMOI v0.14 summary ===")
    print(f"  TCGA  AUROC: {tcga_test_auc:.4f}  bacc: {tcga_test_bacc:.4f} "
          f"(n_test={len(feats_test.sample_ids)}, HER2={int(feats_test.y.sum())})")
    print(f"  METAB AUROC: {metab_auc:.4f}  bacc: {metab_bacc:.4f} "
          f"(HER2={int((y_ext == 1).sum())})")
    print(f"  Luminal pole top-5 ∩ expected : {len(lum_match)} / 2")
    print(f"  HER2    pole top-5 ∩ expected : {len(her2_match)} / 3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
