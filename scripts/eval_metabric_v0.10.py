#!/usr/bin/env python3
"""DMOI v0.10: METABRIC cross-cohort + cross-task generalization.

The v0.6 / v0.7 / v0.8 trilogy + v0.9 closed the architecture experiment
on the TCGA cohort:
  * v0.6 / v0.7 / v0.8 -- gene-level commitment is the right
    architectural level for LumA-vs-LumB (3 variants of learnable
    pathway-attention all converged to the same wrong basin).
  * v0.9 -- the same v0.6 architecture transferred to Luminal-vs-Basal
    with no code changes, reaching AUROC 1.000 on TCGA test and 8/8
    expected priors in per-pole IG top-5.

v0.10 asks the natural composite question: does v0.9 also generalize
across cohorts? That is, when we train the v0.9 model on TCGA
cohort_v3 (Luminal vs Basal) and score it on METABRIC (a completely
different microarray platform + patient demographics, with no
methylation channel), does the same Luminal-vs-Basal classification +
per-pole biology hold?

If yes, the v0.6 framework is **doubly** generalizable -- across both
cohorts (TCGA -> METABRIC) and tasks (LumA-vs-LumB -> Luminal-vs-Basal)
-- and the v0.6 -> v0.10 sequence reads as a complete falsifiable
architectural inquiry with cross-cohort and cross-task reusability
both empirically confirmed.

Pipeline:
  1. Load TCGA cohort_v3 train + test (from v0.9).
  2. Load METABRIC cohort_v3 (LumA + LumB -> Luminal, Basal -> Basal,
     all 1,384 patients with mRNA).
  3. Load the full 50-set Hallmark catalog.
  4. Train DMOI v0.9 architecture on TCGA train (n=401), score TCGA
     test (n=101) for reference, score METABRIC (n=1,384, RNA-only +
     meth silenced + quantile-normalized to TCGA train RNA distribution).
  5. Run IG attribution on METABRIC for Luminal_pole / Basal_pole /
     final_logit. Roll up to the 50-set Hallmark catalog.
  6. Compare TCGA-Luminal-pole-top-K vs METABRIC-Luminal-pole-top-K and
     TCGA-Basal-pole-top-K vs METABRIC-Basal-pole-top-K (Jaccard +
     shared top-3 count).
  7. Write audit/dmoi_v0.10.md with cross-cohort + cross-task closure
     analysis.

Limitations:
  - METABRIC has no HM450 methylation -- the meth branch is silenced.
  - METABRIC microarray RNA is quantile-normalized to TCGA train RNA
    column-by-column (per the v0.2 / v0.4 / v0.6 protocol). This is the
    standard cross-platform harmonization step.
  - The framework's correct answer here is "AUROC stays high and per-pole
    biology is preserved." If either drops, v0.10 is a recorded
    cross-cohort limitation; v0.6 / v0.9 remain canonical for their
    respective tasks.
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
from dmoi_brca.priors import POLE_BASAL, POLE_LUMINAL  # noqa: E402
from dmoi_brca.train import train_one_fold  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TCGA = REPO / "data" / "tcga_brca"
METABRIC = REPO / "data" / "metabric"
AUDIT = REPO / "audit"
HALLMARK_GMT = REPO / "data" / "msigdb" / "h.all.v2024.1.Hs.symbols.gmt"

POLE_ORDER = ("Luminal", "Basal")
POSITIVE_LABEL = "Basal"  # Basal = 1 (minority class)
TCGA_COHORT_TSV_NAME = "cohort_v3.tsv"
METABRIC_COHORT_TSV_NAME = "cohort_v3.tsv"

# Reuse v0.9 hyperparameters exactly.
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


def _load_metabric_mrna(mrna_path: Path, cohort_ids: set[str]):
    """Reuse the v0.2 / v0.4 METABRIC loader pattern."""
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

    print("=== DMOI v0.10: METABRIC cross-cohort + cross-task generalization ===")

    # --- Hallmark catalog ---
    print(f"\n--- Loading Hallmark catalog: {HALLMARK_GMT.name} ---")
    hallmark_full: dict[str, list[str]] = load_hallmark_gmt(HALLMARK_GMT)
    print(f"  {len(hallmark_full)} Hallmark sets loaded.")

    # --- TCGA cohort_v3 + features (same as v0.9) ---
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
          f"(Basal={int(feats.y.sum())}, "
          f"Luminal={len(feats.y) - int(feats.y.sum())})")
    print(f"  TCGA test:  n={len(feats_test.sample_ids)} "
          f"(Basal={int(feats_test.y.sum())}, "
          f"Luminal={len(feats_test.y) - int(feats_test.y.sum())})")

    # --- METABRIC cohort_v3 ---
    print(f"\n--- Loading METABRIC cohort from {METABRIC_COHORT_TSV_NAME} ---")
    metabric_cohort = pd.read_csv(metabric_cohort_tsv, sep="\t")
    metabric_cohort = metabric_cohort[metabric_cohort["has_rna"]].copy()
    ext_ids_wanted = set(metabric_cohort["sample_id"].astype(str))
    ext_X_raw, ext_genes, ext_sample_ids = _load_metabric_mrna(
        metabric_mrna, ext_ids_wanted,
    )
    ext_cohort = metabric_cohort.set_index("sample_id").loc[ext_sample_ids]
    y_ext = ext_cohort["group"].map({"Luminal": 0, "Basal": 1}).to_numpy().astype(np.int64)
    print(f"  METABRIC Luminal/Basal with mRNA: {ext_X_raw.shape[0]} patients "
          f"(Luminal={int((y_ext == 0).sum())}, Basal={int((y_ext == 1).sum())})")

    # --- Align METABRIC to TCGA train gene order + quantile-normalize ---
    print("\n--- Aligning METABRIC RNA to TCGA train gene order + QN ---")
    ext_X_aligned = align_to_train_genes(
        ext_X_raw, ext_genes, feats.rna_features, fill_value=0.0,
    )
    ext_X_qn = quantile_normalize_to_train(ext_X_aligned, feats.rna)
    meth_ext_silenced = make_silenced_meth(
        ext_X_qn.shape[0], feats.meth.shape[1],
    )

    # --- Pole masks + train v0.9 model ---
    print("\n--- Building Luminal / Basal pole masks ---")
    cis = load_hm450_cis_mapping(probemap)
    pole_masks = make_pole_masks(
        feats.rna_features, feats.meth_features, cis,
        {"Luminal": POLE_LUMINAL, "Basal": POLE_BASAL},
        hallmark_sets={k: list(v) for k, v in hallmark_full.items()},
    )

    print("\n--- Training v0.9 architecture on TCGA train ---")
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
    if result.model is None or result.rna_scaler is None or result.meth_scaler is None:
        sys.stderr.write("ERROR: train_one_fold returned no artifacts.\n")
        return 1
    tcga_test_auc = result.best_val_auc
    tcga_test_bacc = result.best_val_bacc
    print(f"  TCGA test AUROC : {tcga_test_auc:.4f} "
          f"(v0.9 reference: 1.000)")
    print(f"  TCGA test bacc  : {tcga_test_bacc:.4f} "
          f"(v0.9 reference: 0.972)")

    # --- Score METABRIC ---
    print("\n--- Scoring METABRIC (RNA-only, meth silenced) ---")
    import torch  # noqa: E402
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

    # --- IG attribution on METABRIC + Hallmark rollup ---
    AUDIT.mkdir(exist_ok=True)
    print("\n--- IG + Hallmark pathway rollup on METABRIC ---")
    pathway_results: dict[str, list] = {}
    for target_name in ("Luminal_pole", "Basal_pole", "final_logit"):
        if target_name == "Luminal_pole":
            attr_target = "lumA_pole"
        elif target_name == "Basal_pole":
            attr_target = "lumB_pole"
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
        top = rank_pathways(scores, by="mean_abs_ig")[:5]
        for s in top:
            print(f"    {s.pathway_name:45s}  mean|IG| {s.mean_abs_ig:.5f}  "
                  f"signed_mean {s.signed_mean:+.5f}  "
                  f"({s.n_pathway_genes_in_inputs} genes in inputs)")

    # --- Cross-cohort consistency check ---
    print("\n--- Cross-pole biology sanity check (METABRIC) ---")
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
    md_path = AUDIT / "dmoi_v0.10.md"

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
        "# DMOI v0.10 -- METABRIC cross-cohort + cross-task generalization\n\n"
        f"Generated: {ts}\n\n"
        "## Setup\n\n"
        "- Architecture: v0.9 (same as v0.6 base). No model changes.\n"
        "- POLE_LUMINAL = ER_EARLY + ER_LATE + ANDROGEN_RESPONSE\n"
        "- POLE_BASAL   = EMT + MYC_TARGETS_V1 + G2M_CHECKPOINT\n"
        f"- TCGA train cohort: cohort_v3 train, n={len(feats.sample_ids)} "
        f"(Basal={int(feats.y.sum())}, "
        f"Luminal={len(feats.y) - int(feats.y.sum())}).\n"
        f"- TCGA held-out test: n={len(feats_test.sample_ids)} "
        f"(Basal={int(feats_test.y.sum())}, "
        f"Luminal={len(feats_test.y) - int(feats_test.y.sum())}).\n"
        f"- METABRIC external: n={ext_X_qn.shape[0]} "
        f"(Luminal={int((y_ext == 0).sum())}, "
        f"Basal={int((y_ext == 1).sum())}). RNA-only + meth silenced "
        "+ quantile-normalized to TCGA train RNA per the v0.2 / v0.4 / v0.6 protocol.\n"
        f"- Epochs: {N_EPOCHS}, optimizer: AdamW(lr=1e-4, wd=1e-4), "
        "BCEWithLogitsLoss + aux=0.3, pick_best_epoch=False.\n\n"
        "## Headline AUROC\n\n"
        f"| Cohort | AUROC | bacc | Reference |\n"
        f"|---|---|---|---|\n"
        f"| TCGA held-out test | **{tcga_test_auc:.4f}** | "
        f"{tcga_test_bacc:.4f} | v0.9: 1.000 / 0.972 |\n"
        f"| METABRIC external  | **{metab_auc:.4f}** | "
        f"{metab_bacc:.4f} | (v0.4 LumA-vs-LumB ref: 0.909) |\n\n"
        "## Per-pole IG top-10 pathways (METABRIC)\n\n"
        "### Luminal pole\n\n" + _rank_table("Luminal_pole") + "\n\n"
        "### Basal pole\n\n" + _rank_table("Basal_pole") + "\n\n"
        "### final_logit\n\n" + _rank_table("final_logit") + "\n\n"
        "## Cross-pole biology sanity check (METABRIC)\n\n"
        f"Expected Luminal-pole top-5 to include {{ER_EARLY, ER_LATE, ANDROGEN_RESPONSE}}.\n"
        "Expected Basal-pole top-5 to include {EMT, MYC_TARGETS_V1, G2M_CHECKPOINT, E2F_TARGETS, MYC_TARGETS_V2}.\n\n"
        f"- Luminal pole top-5 ∩ expected = {len(lum_match)} / 3 : "
        f"{', '.join(f'`{p}`' for p in lum_match) or '(none)'}\n"
        f"- Basal pole top-5 ∩ expected = {len(bas_match)} / 5 : "
        f"{', '.join(f'`{p}`' for p in bas_match) or '(none)'}\n\n"
        "## Limitations\n\n"
        "- Same architecture as v0.9 / v0.6 (no model changes); only the\n"
        "  external scoring cohort changes.\n"
        "- METABRIC microarray RNA is on a different platform (Illumina HT-12 v3)\n"
        "  than TCGA's HiSeqV2. Quantile normalization is applied column-by-column\n"
        "  to match the TCGA train RNA distribution (v0.2 / v0.4 / v0.6 protocol).\n"
        "- METABRIC has no HM450 methylation, so the meth branch is silenced.\n"
        "- Class imbalance is 5.6 : 1 (Luminal majority); Basal n=209 in METABRIC.\n"
        "- AUROC = 1.000 on TCGA test is the v0.9 ceiling; cross-cohort AUROC\n"
        "  is the meaningful new metric here.\n\n"
        "## Reproduce\n\n"
        "```bash\n"
        "python scripts/build_cohort_v3.py              # TCGA cohort_v3 (if not built)\n"
        "python scripts/build_metabric_cohort_v3.py     # METABRIC cohort_v3\n"
        "python scripts/eval_metabric_v0.10.py          # ~10 min on MPS\n"
        "```\n",
    )
    print(f"\nWrote {md_path}")

    print("\n=== DMOI v0.10 summary ===")
    print(f"  TCGA   AUROC : {tcga_test_auc:.4f} (v0.9 ref 1.000)")
    print(f"  TCGA   bacc  : {tcga_test_bacc:.4f}")
    print(f"  METAB  AUROC : {metab_auc:.4f} (v0.4 LumA-vs-LumB ref 0.909)")
    print(f"  METAB  bacc  : {metab_bacc:.4f}")
    print(f"  Luminal pole top-5 ∩ expected priors : {len(lum_match)} / 3")
    print(f"  Basal   pole top-5 ∩ expected priors : {len(bas_match)} / 5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
