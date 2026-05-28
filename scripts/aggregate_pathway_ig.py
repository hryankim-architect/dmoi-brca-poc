#!/usr/bin/env python3
"""DMOI v0.5: pathway-level IG aggregation on TCGA test + METABRIC.

Rolls per-gene IG attributions up to MSigDB Hallmark pathway scores.
The v0.3+v0.4 finding was at the gene level (LumA picked FOXC1/BCL2,
LumB picked RANBP1/NBN/ZW10/POLA2); v0.5 expresses the same finding at
the pathway level (LumA loads ESTROGEN_RESPONSE; LumB loads
E2F_TARGETS / G2M_CHECKPOINT / MYC_TARGETS).

Pipeline:
  1. Load TCGA cohort_v2 train split + METABRIC LumA/LumB cohort.
  2. Train ONE Option A model on TCGA train (keep_artifacts=True).
  3. Run IG on the TCGA test split for the lumA + lumB pole targets.
  4. Run IG on METABRIC (meth silenced) for the same targets.
  5. Aggregate per-gene IG to per-pathway scores via
     `dmoi_brca.pathway.pathway_aggregate` using priors.HALLMARK_SETS.
  6. Write audit/dmoi_pathway_v0.5.md with cross-cohort comparison.

Honest scope: only the 5 Hallmark sets in priors.py are aggregated
(ESTROGEN_RESPONSE_EARLY + LATE; E2F_TARGETS; G2M_CHECKPOINT;
MYC_TARGETS_V1). The full 50-set MSigDB Hallmark catalog is out of
scope for v0.5 — keeping the dependency surface tight.
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
from dmoi_brca.features import FeatureMatrices, load_features  # noqa: E402
from dmoi_brca.hypothesis_attention import (  # noqa: E402
    load_hm450_cis_mapping,
    make_pole_masks,
)
from dmoi_brca.pathway import pathway_aggregate, rank_pathways  # noqa: E402
from dmoi_brca.priors import HALLMARK_SETS, POLE_LUMA, POLE_LUMB  # noqa: E402
from dmoi_brca.train import train_one_fold  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TCGA = REPO / "data" / "tcga_brca"
METABRIC = REPO / "data" / "metabric"
AUDIT = REPO / "audit"

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


def _scores_table_md(target_name: str, scores_by_cohort: dict[str, list]) -> str:
    """Side-by-side TCGA test vs METABRIC pathway rankings table."""
    rows = []
    for pathway in HALLMARK_SETS:
        tcga = next(
            (s for s in scores_by_cohort["TCGA test"] if s.pathway_name == pathway),
            None,
        )
        metab = next(
            (s for s in scores_by_cohort["METABRIC"] if s.pathway_name == pathway),
            None,
        )
        tcga_str = (
            f"{tcga.mean_abs_ig:.5f} (signed_mean {tcga.signed_mean:+.5f})"
            if tcga else "—"
        )
        metab_str = (
            f"{metab.mean_abs_ig:.5f} (signed_mean {metab.signed_mean:+.5f})"
            if metab else "—"
        )
        rows.append(
            f"| `{pathway}` | "
            f"{tcga.n_pathway_genes_in_inputs if tcga else 0} | "
            f"{tcga_str} | {metab_str} |",
        )
    head = (
        "| Pathway | TCGA genes in inputs | TCGA test mean \\|IG\\| | "
        "METABRIC mean \\|IG\\| |\n|---|---|---|---|"
    )
    return f"### {target_name}\n\n{head}\n" + "\n".join(rows)


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

    print("=== DMOI v0.5: pathway-level IG aggregation ===")

    # --- TCGA cohort + train/test slice ---
    print("\n--- Loading TCGA features ---")
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
    print(f"  TCGA train: {len(feats.sample_ids)}, test: {len(feats_test.sample_ids)}")

    # --- METABRIC ---
    print("\n--- Loading METABRIC ---")
    metabric_cohort = pd.read_csv(metabric_cohort_tsv, sep="\t")
    metabric_cohort = metabric_cohort[metabric_cohort["has_rna"]].copy()
    ext_ids_wanted = set(metabric_cohort["sample_id"].astype(str))
    ext_X_raw, ext_genes, ext_sample_ids = _load_metabric_mrna(
        metabric_mrna, ext_ids_wanted,
    )
    # ext_sample_ids order is the order rows come out of _load_metabric_mrna;
    # we don't need y_ext for IG attribution (it's input-only), so we skip
    # building the label vector here to keep the script lean.
    _ = metabric_cohort.set_index("sample_id").loc[ext_sample_ids]
    print(f"  METABRIC LumA/LumB with mRNA: {ext_X_raw.shape[0]} patients")

    # --- Align + QN METABRIC ---
    ext_X_aligned = align_to_train_genes(
        ext_X_raw, ext_genes, feats.rna_features, fill_value=0.0,
    )
    ext_X_qn = quantile_normalize_to_train(ext_X_aligned, feats.rna)
    meth_ext_silenced_raw = make_silenced_meth(
        ext_X_qn.shape[0], feats.meth.shape[1],
    )

    # --- Pole masks + train model ---
    print("\n--- Training final Option A model (keep_artifacts=True) ---")
    cis = load_hm450_cis_mapping(probemap)
    pole_masks = make_pole_masks(
        feats.rna_features, feats.meth_features, cis,
        {"LumA": POLE_LUMA, "LumB": POLE_LUMB},
    )
    result = train_one_fold(
        rna_train=feats.rna, meth_train=feats.meth, y_train=feats.y,
        rna_val=feats_test.rna, meth_val=feats_test.meth,
        y_val=feats_test.y,
        pole_masks=pole_masks,
        fold=0,
        rna_dim=feats.rna.shape[1], meth_dim=feats.meth.shape[1],
        n_epochs=N_EPOCHS, patience=N_EPOCHS + 1,
        keep_artifacts=True,
        **FINAL_KWARGS,
    )
    print(f"  TCGA test AUROC : {result.best_val_auc:.4f}")
    if result.model is None or result.rna_scaler is None or result.meth_scaler is None:
        sys.stderr.write("ERROR: train_one_fold returned no artifacts.\n")
        return 1
    model = result.model
    model.eval()

    # --- Standardize cohorts the way the model saw them ---
    rna_tcga_test_std = result.rna_scaler.transform(feats_test.rna).astype(np.float32)
    meth_tcga_test_std = result.meth_scaler.transform(feats_test.meth).astype(np.float32)
    rna_metab_std = result.rna_scaler.transform(ext_X_qn).astype(np.float32)
    meth_metab_std = result.meth_scaler.transform(meth_ext_silenced_raw).astype(np.float32)

    # --- IG + pathway aggregation per target per cohort ---
    AUDIT.mkdir(exist_ok=True)
    cohort_inputs = {
        "TCGA test": (rna_tcga_test_std, meth_tcga_test_std),
        "METABRIC":  (rna_metab_std, meth_metab_std),
    }
    pathway_results: dict[str, dict[str, list]] = {}  # [target][cohort] -> [PathwayScore]
    for target_name in ("lumA_pole", "lumB_pole", "final_logit"):
        pathway_results[target_name] = {}
        for cohort_name, (rna_x, meth_x) in cohort_inputs.items():
            print(f"\n--- IG + pathway rollup: {target_name} on {cohort_name} ---")
            attr = integrated_gradients_dmoi(
                model, rna_x, meth_x,
                target=target_name, n_steps=N_IG_STEPS, device="cpu",
            )
            scores = pathway_aggregate(
                attr.rna_attribution, feats.rna_features, HALLMARK_SETS,
            )
            pathway_results[target_name][cohort_name] = scores
            top = rank_pathways(scores, by="mean_abs_ig")[:5]
            for s in top:
                print(f"  {s.pathway_name:40s}  "
                      f"mean|IG| {s.mean_abs_ig:.5f}  "
                      f"signed_mean {s.signed_mean:+.5f}  "
                      f"({s.n_pathway_genes_in_inputs} genes in inputs)")

    # --- Cross-cohort comparison: does the dominant pathway match? ---
    print("\n--- Cross-cohort pathway agreement ---")
    cross_cohort: dict[str, dict] = {}
    for target_name in ("lumA_pole", "lumB_pole", "final_logit"):
        tcga_top = [
            s.pathway_name
            for s in rank_pathways(
                pathway_results[target_name]["TCGA test"], by="mean_abs_ig",
            )[:3]
        ]
        metab_top = [
            s.pathway_name
            for s in rank_pathways(
                pathway_results[target_name]["METABRIC"], by="mean_abs_ig",
            )[:3]
        ]
        shared = sorted(set(tcga_top) & set(metab_top))
        cross_cohort[target_name] = {
            "tcga_top3": tcga_top,
            "metab_top3": metab_top,
            "shared": shared,
        }
        print(f"  {target_name}: TCGA top-3 = {tcga_top}")
        print(f"  {' ' * 14} METABRIC top-3 = {metab_top}")
        print(f"  {' ' * 14} shared = {shared}")

    # --- Audit MD ---
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    md_path = AUDIT / "dmoi_pathway_v0.5.md"

    def _shared_md(target_name):
        c = cross_cohort[target_name]
        return (
            f"- TCGA test top-3 pathways: {', '.join(f'`{p}`' for p in c['tcga_top3'])}\n"
            f"- METABRIC top-3 pathways : {', '.join(f'`{p}`' for p in c['metab_top3'])}\n"
            f"- Shared : {', '.join(f'`{p}`' for p in c['shared']) or '_(none)_'}\n"
        )

    md_path.write_text(
        "# DMOI v0.5 — Pathway-level IG aggregation (MSigDB Hallmark)\n\n"
        f"Generated: {ts}\n\n"
        "## Setup\n\n"
        f"- Train cohort     : TCGA cohort_v2 train split, n={len(feats.sample_ids)}\n"
        f"- TCGA test cohort : n={len(feats_test.sample_ids)} (AUROC "
        f"{result.best_val_auc:.4f})\n"
        f"- METABRIC cohort  : n={ext_X_qn.shape[0]} (RNA-only, meth silenced)\n"
        f"- Pathway sets     : {len(HALLMARK_SETS)} MSigDB Hallmark sets "
        "from `priors.py` (ESTROGEN_RESPONSE_EARLY/LATE, E2F_TARGETS, "
        "G2M_CHECKPOINT, MYC_TARGETS_V1)\n"
        f"- Aggregation      : per-pathway `mean |IG|`, `sum_signed`, "
        "`signed_mean` over per-patient × per-gene attributions\n\n"
        "## Cross-cohort pathway agreement (top-3 per target)\n\n"
        "### lumA_pole\n\n" + _shared_md("lumA_pole") + "\n"
        "### lumB_pole\n\n" + _shared_md("lumB_pole") + "\n"
        "### final_logit\n\n" + _shared_md("final_logit") + "\n"
        "## Detailed scores per pathway × cohort\n\n"
        + _scores_table_md("lumA_pole", pathway_results["lumA_pole"]) + "\n\n"
        + _scores_table_md("lumB_pole", pathway_results["lumB_pole"]) + "\n\n"
        + _scores_table_md("final_logit", pathway_results["final_logit"]) + "\n\n"
        "## Reading\n\n"
        "- `mean |IG|` — how loudly the pathway speaks (magnitude).\n"
        "- `signed_mean` — direction (positive = pushes toward LumB; "
        "negative = pushes toward LumA for the final logit; for the pole "
        "scores, positive = pushes toward 'this is the pole's class').\n"
        "- A pathway with high `mean |IG|` but `signed_mean ≈ 0` means the "
        "pathway has both pro- and anti- genes that roughly cancel — the "
        "pathway is important but ambiguous in direction.\n\n"
        "## Honest scope\n\n"
        "- Only 5 Hallmark sets are loaded (the ones already in "
        "`priors.py` for the pole masks). The full 50-set MSigDB Hallmark "
        "catalog is out of scope for v0.5 — keeping the dependency "
        "surface tight. Future work: add a `gmt`-file loader and roll "
        "up the full Hallmark catalog (or even C2 curated pathways).\n"
        "- Aggregation is over the RNA modality only. METABRIC's "
        "methylation branch is silenced; even on TCGA the methylation "
        "pathway aggregation isn't meaningful because the meth features "
        "are HM450 probes, not gene symbols.\n\n"
        "## Reproduce\n\n"
        "```bash\n"
        "python scripts/aggregate_pathway_ig.py\n"
        "```\n",
    )
    print(f"\nWrote {md_path}")

    print("\n=== DMOI v0.5 pathway aggregation summary ===")
    for target_name in ("lumA_pole", "lumB_pole", "final_logit"):
        c = cross_cohort[target_name]
        print(f"  {target_name:14s} TCGA-test top : {c['tcga_top3']}")
        print(f"  {' ' * 14} METABRIC top  : {c['metab_top3']}")
        print(f"  {' ' * 14} shared        : {c['shared']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
