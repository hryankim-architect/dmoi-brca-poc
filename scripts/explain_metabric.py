#!/usr/bin/env python3
"""DMOI v0.4: cross-cohort Integrated Gradients attribution on METABRIC.

Re-runs the v0.3 attribution pipeline on the METABRIC cohort
(n=1,175) with the methylation branch silenced. Same architecture, same
trained model, same three targets (final_logit / lumA_pole / lumB_pole) —
only the inputs differ.

Pipeline:
  1. Load TCGA cohort_v2 train split (n=333).
  2. Load METABRIC + dedup-collapse-align-QN per `dmoi_brca.external`.
  3. Train ONE Option A model on TCGA train with keep_artifacts=True so
     we get the trained model + fitted StandardScalers back from the
     same FoldResult (v0.4 cleanup).
  4. Standardize METABRIC inputs with the train-fitted scalers, plus
     a meth-silenced zero tensor of train_meth shape, then standardize
     it too so the model sees what it actually trained on.
  5. Run IG for each of three targets, 50 steps.
  6. Compute global top-50 + per-patient top-10 + completeness residuals.
  7. Compare TCGA-test and METABRIC top-K agreement (Jaccard on the
     top-10 set per pole) — the cross-cohort interpretability headline.
  8. Audit MD: audit/dmoi_explain_external_v0.4.md with the same shape
     as dmoi_explain_v0.3.md plus a cross-cohort comparison section.

Honest scope: METABRIC has no HM450. Methylation branch is silenced at
inference (the dual-modality story is NOT validated cross-cohort; only
the RNA pole encoder's per-patient explanations are).
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone  # noqa: UP017
from pathlib import Path

UTC = timezone.utc  # noqa: UP017

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dmoi_brca.attribution import (  # noqa: E402
    completeness_residual,
    global_aggregate,
    integrated_gradients_dmoi,
    top_k_per_patient,
)
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
K_PER_PATIENT = 10
K_GLOBAL = 50


def _load_metabric_mrna(
    mrna_path: Path,
    cohort_ids: set[str],
) -> tuple[np.ndarray, list[str], list[str]]:
    """Load METABRIC mRNA matrix subset to cohort_ids."""
    with mrna_path.open() as fh:
        header = fh.readline().rstrip("\n").split("\t")
    keep = [c for c in header[2:] if c in cohort_ids]
    print(f"    METABRIC mRNA matrix: {len(keep)} cohort samples in header")
    df = pd.read_csv(
        mrna_path, sep="\t",
        usecols=[header[0], header[1]] + keep, low_memory=False,
    )
    hugo = df["Hugo_Symbol"].astype(str).tolist()
    expression = df[keep].to_numpy(dtype=np.float32)
    collapsed, unique_genes = collapse_duplicate_genes(expression, hugo)
    return collapsed.T, unique_genes, keep


def _try_plot(global_agg, target_name, out_path, top_n: int = 20) -> bool:
    try:
        import matplotlib  # noqa: PLC0415
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, modality in zip(axes, ("rna", "meth"), strict=False):
        rows = global_agg[modality][:top_n][::-1]
        names = [r[0] for r in rows]
        vals = [r[1] for r in rows]
        ax.barh(range(len(rows)), vals)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("mean |IG attribution|")
        ax.set_title(f"METABRIC {target_name} — top {top_n} {modality}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return True


def _load_tcga_top_features(
    tsv_path: Path, target_name: str, modality: str, k: int,
) -> list[str]:
    """Pull TCGA-test top-K RNA features for a target from v0.3 audit TSV."""
    if not tsv_path.exists():
        return []
    df = pd.read_csv(tsv_path, sep="\t")
    sub = df[(df["target"] == target_name) & (df["modality"] == modality)]
    return sub.sort_values("rank").head(k)["feature"].tolist()


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

    print("=== DMOI v0.4: METABRIC per-patient IG attribution ===")

    # --- TCGA train split + features (same as eval_external.py) ---
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
    print(f"  TCGA train: {len(feats.sample_ids)} (LumA "
          f"{int((feats.y == 0).sum())} / LumB {int((feats.y == 1).sum())})")

    # --- METABRIC cohort + mRNA ---
    print("\n--- Loading METABRIC cohort + mRNA ---")
    metabric_cohort = pd.read_csv(metabric_cohort_tsv, sep="\t")
    metabric_cohort = metabric_cohort[metabric_cohort["has_rna"]].copy()
    print(f"  METABRIC (LumA/LumB + mRNA): {len(metabric_cohort)} patients "
          f"(LumA={int((metabric_cohort['group'] == 'LumA').sum())}, "
          f"LumB={int((metabric_cohort['group'] == 'LumB').sum())})")
    ext_ids_wanted = set(metabric_cohort["sample_id"].astype(str))
    ext_X_raw, ext_genes, ext_sample_ids = _load_metabric_mrna(
        metabric_mrna, ext_ids_wanted,
    )
    metabric_cohort = metabric_cohort.set_index("sample_id").loc[ext_sample_ids]
    y_ext = (metabric_cohort["group"] == "LumB").astype(np.int64).to_numpy()

    # --- Gene alignment + QN ---
    print("\n--- Gene alignment + QN ---")
    overlap = gene_overlap_stats(ext_genes, feats.rna_features)
    print(f"  shared genes: {overlap['n_shared']} / {overlap['n_train']} TCGA")
    ext_X_aligned = align_to_train_genes(
        ext_X_raw, ext_genes, feats.rna_features, fill_value=0.0,
    )
    ext_X_qn = quantile_normalize_to_train(ext_X_aligned, feats.rna)
    meth_ext_silenced_raw = make_silenced_meth(
        ext_X_qn.shape[0], feats.meth.shape[1],
    )
    print(f"  METABRIC after QN: shape {ext_X_qn.shape}")

    # --- Build pole masks ---
    print("\n--- Building pole masks ---")
    cis = load_hm450_cis_mapping(probemap)
    pole_masks = make_pole_masks(
        feats.rna_features, feats.meth_features, cis,
        {"LumA": POLE_LUMA, "LumB": POLE_LUMB},
    )

    # --- Train model with keep_artifacts (single pass, v0.4 cleanup) ---
    print(f"\n--- Training final Option A on TCGA train "
          f"(n_epochs={N_EPOCHS}, keep_artifacts=True) ---")
    result = train_one_fold(
        rna_train=feats.rna, meth_train=feats.meth, y_train=feats.y,
        rna_val=ext_X_qn, meth_val=meth_ext_silenced_raw.astype(np.float32),
        y_val=y_ext.astype(np.int64),
        pole_masks=pole_masks,
        fold=0,
        rna_dim=feats.rna.shape[1], meth_dim=feats.meth.shape[1],
        n_epochs=N_EPOCHS, patience=N_EPOCHS + 1,
        keep_artifacts=True,
        **FINAL_KWARGS,
    )
    print(f"  External AUROC : {result.best_val_auc:.4f}")
    if result.model is None or result.rna_scaler is None or result.meth_scaler is None:
        sys.stderr.write("ERROR: train_one_fold returned no artifacts.\n")
        return 1
    model = result.model
    model.eval()

    # --- Standardize METABRIC inputs the same way the model saw them ---
    ext_rna_std = result.rna_scaler.transform(ext_X_qn).astype(np.float32)
    ext_meth_std = result.meth_scaler.transform(meth_ext_silenced_raw).astype(np.float32)

    # --- Run IG for three targets ---
    AUDIT.mkdir(exist_ok=True)
    per_patient_rows: list[dict] = []
    global_rows: list[dict] = []
    completeness_by_target: dict[str, np.ndarray] = {}
    global_agg_cache: dict[str, dict] = {}

    for target_name in ("final_logit", "lumA_pole", "lumB_pole"):
        print(f"\n--- IG on {target_name} ---")
        attr = integrated_gradients_dmoi(
            model, ext_rna_std, ext_meth_std,
            target=target_name, n_steps=N_IG_STEPS, device="cpu",
        )
        residuals = completeness_residual(attr)
        completeness_by_target[target_name] = residuals
        print(f"  completeness: mean {residuals.mean():.5f}, "
              f"max {residuals.max():.5f}")

        per_patient = top_k_per_patient(
            attr, feats.rna_features, feats.meth_features, k=K_PER_PATIENT,
        )
        for i, row in enumerate(per_patient):
            sid = ext_sample_ids[i]
            for rank, (feature, value, inp) in enumerate(row["topk_rna"], start=1):
                per_patient_rows.append({
                    "sample_id": sid, "y_true": int(y_ext[i]),
                    "target": target_name, "modality": "rna", "rank": rank,
                    "feature": feature, "attribution": value, "input_value": inp,
                    "target_score": row["target_score"],
                })
            for rank, (feature, value, inp) in enumerate(row["topk_meth"], start=1):
                per_patient_rows.append({
                    "sample_id": sid, "y_true": int(y_ext[i]),
                    "target": target_name, "modality": "meth", "rank": rank,
                    "feature": feature, "attribution": value, "input_value": inp,
                    "target_score": row["target_score"],
                })

        agg = global_aggregate(
            attr, feats.rna_features, feats.meth_features, top_k=K_GLOBAL,
        )
        global_agg_cache[target_name] = agg
        for modality, rows in agg.items():
            for rank, (feature, mean_abs) in enumerate(rows, start=1):
                global_rows.append({
                    "target": target_name, "modality": modality,
                    "rank": rank, "feature": feature, "mean_abs_attr": mean_abs,
                })
        png_path = AUDIT / f"dmoi_explain_external_global_{target_name}.png"
        if _try_plot(agg, target_name, png_path):
            print(f"  wrote {png_path}")

    pp_tsv = AUDIT / "dmoi_explain_external_per_patient.tsv"
    pd.DataFrame(per_patient_rows).to_csv(pp_tsv, sep="\t", index=False)
    print(f"\nWrote {pp_tsv}  ({len(per_patient_rows)} rows)")

    g_tsv = AUDIT / "dmoi_explain_external_global.tsv"
    pd.DataFrame(global_rows).to_csv(g_tsv, sep="\t", index=False)
    print(f"Wrote {g_tsv}  ({len(global_rows)} rows)")

    # --- Cross-cohort comparison vs TCGA-test (v0.3) ---
    print("\n--- Cross-cohort comparison vs TCGA test (v0.3) ---")
    tcga_tsv = AUDIT / "dmoi_explain_global.tsv"
    cross_cohort: dict[str, dict] = {}
    for target_name in ("final_logit", "lumA_pole", "lumB_pole"):
        tcga_top = _load_tcga_top_features(tcga_tsv, target_name, "rna", K_GLOBAL)
        metab_top = [
            f for f, _ in global_agg_cache[target_name]["rna"][:K_GLOBAL]
        ]
        if not tcga_top:
            cross_cohort[target_name] = {
                "tcga_top10": [], "metab_top10": metab_top[:10],
                "jaccard_top10": float("nan"), "jaccard_top50": float("nan"),
                "shared_top10": [],
            }
            continue
        s_tcga_10, s_metab_10 = set(tcga_top[:10]), set(metab_top[:10])
        s_tcga_50, s_metab_50 = set(tcga_top), set(metab_top)
        jaccard_10 = (
            len(s_tcga_10 & s_metab_10) / max(len(s_tcga_10 | s_metab_10), 1)
        )
        jaccard_50 = (
            len(s_tcga_50 & s_metab_50) / max(len(s_tcga_50 | s_metab_50), 1)
        )
        shared_10 = sorted(s_tcga_10 & s_metab_10)
        cross_cohort[target_name] = {
            "tcga_top10": tcga_top[:10],
            "metab_top10": metab_top[:10],
            "jaccard_top10": jaccard_10,
            "jaccard_top50": jaccard_50,
            "shared_top10": shared_10,
        }
        print(f"  {target_name}: Jaccard top-10 = {jaccard_10:.3f}, "
              f"top-50 = {jaccard_50:.3f}  ·  shared top-10 genes: "
              f"{shared_10 or '<none>'}")

    # --- Audit MD ---
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    summary = AUDIT / "dmoi_explain_external_v0.4.md"

    def _top_md(target: str, modality: str, k: int = 10) -> str:
        sub = pd.DataFrame(global_rows)
        sub = sub[(sub["target"] == target) & (sub["modality"] == modality)].head(k)
        rows = "\n".join(
            f"| {int(r['rank'])} | `{r['feature']}` | {r['mean_abs_attr']:.5f} |"
            for _, r in sub.iterrows()
        )
        return ("| Rank | Feature | mean |IG| |\n|---|---|---|\n" + rows)

    def _cmp_md(target: str) -> str:
        c = cross_cohort[target]
        if not c["tcga_top10"]:
            return ("_(TCGA test attribution TSV not found — run "
                    "`scripts/explain_dmoi.py` first to enable the comparison.)_")
        rows = "\n".join(
            f"| {i + 1} | `{tcga_g}` | `{metab_g}` |"
            for i, (tcga_g, metab_g) in enumerate(
                zip(c["tcga_top10"], c["metab_top10"], strict=False),
            )
        )
        shared = ", ".join(f"`{g}`" for g in c["shared_top10"]) or "_(none)_"
        return (
            f"- Jaccard(top-10) = **{c['jaccard_top10']:.3f}** · "
            f"Jaccard(top-50) = **{c['jaccard_top50']:.3f}**\n"
            f"- Shared top-10 genes: {shared}\n\n"
            "| Rank | TCGA test top-10 | METABRIC top-10 |\n"
            "|---|---|---|\n" + rows
        )

    completeness_lines = "\n".join(
        f"- **{name}**: mean {r.mean():.5f}, max {r.max():.5f}"
        for name, r in completeness_by_target.items()
    )

    summary.write_text(
        "# DMOI v0.4 — METABRIC per-patient Integrated Gradients attribution\n\n"
        f"Generated: {ts}\n\n"
        "## Setup\n\n"
        f"- Train cohort  : TCGA cohort_v2 train split, n={len(feats.sample_ids)}\n"
        f"- External      : METABRIC LumA/LumB with mRNA, n={ext_X_qn.shape[0]} "
        f"(LumA={int((y_ext == 0).sum())}, LumB={int((y_ext == 1).sum())})\n"
        f"- Architecture  : Option A (aux BCE + disagreement), 15 epochs, "
        "no peek, cal_frac=0.15\n"
        f"- External AUROC: {result.best_val_auc:.4f}\n"
        f"- Methylation   : silenced (METABRIC has no HM450)\n"
        f"- IG steps      : {N_IG_STEPS}\n\n"
        "## Completeness check\n\n"
        f"{completeness_lines}\n\n"
        "## Global top-10 RNA features per target (METABRIC)\n\n"
        f"### final_logit\n\n{_top_md('final_logit', 'rna')}\n\n"
        f"### lumA_pole\n\n{_top_md('lumA_pole', 'rna')}\n\n"
        f"### lumB_pole\n\n{_top_md('lumB_pole', 'rna')}\n\n"
        "## Cross-cohort comparison vs TCGA test (v0.3)\n\n"
        "The interpretability headline: do the same genes dominate when "
        "the trained model attributes on a completely different cohort? "
        "If yes, the biology the model learned generalizes; if no, the "
        "v0.3 finding was cohort-specific.\n\n"
        f"### final_logit\n\n{_cmp_md('final_logit')}\n\n"
        f"### lumA_pole\n\n{_cmp_md('lumA_pole')}\n\n"
        f"### lumB_pole\n\n{_cmp_md('lumB_pole')}\n\n"
        "## Honest caveats\n\n"
        "- **Methylation silenced.** METABRIC has no HM450 data, so the "
        "methylation branch receives a fixed zero (raw-domain) tensor, "
        "which after the train-fitted StandardScaler becomes a fixed "
        "`-mean/std` per probe. The lumA/lumB attribution focuses on the "
        "RNA branch only; the methylation attribution column is reported "
        "for completeness but is uninformative (all patients see the same "
        "meth input).\n"
        "- **Cross-cohort gene set differs slightly.** "
        f"{overlap['n_shared']} of TCGA's {overlap['n_train']} RNA genes "
        "are shared with METABRIC; the remainder are mean-imputed to "
        "zero in METABRIC. A gene that is informative on TCGA but absent "
        "from METABRIC cannot appear in METABRIC attributions.\n"
        "- **Quantile normalization is applied per gene.** METABRIC's "
        "per-gene empirical distribution is mapped to TCGA train's "
        "per-gene CDF before standardization, so each gene's attribution "
        "is computed on inputs that match the training distribution.\n\n"
        "## Reproduce\n\n"
        "```bash\n"
        "python scripts/explain_dmoi.py        # TCGA test attribution (v0.3)\n"
        "python scripts/explain_metabric.py    # METABRIC external attribution (v0.4)\n"
        "```\n",
    )
    print(f"\nWrote {summary}")

    # --- Terminal summary ---
    print("\n=== DMOI v0.4 METABRIC attribution summary ===")
    print(f"  Train (TCGA cohort_v2 train)   : {len(feats.sample_ids)} patients")
    print(f"  External (METABRIC)            : {ext_X_qn.shape[0]} patients")
    print(f"  External AUROC                 : {result.best_val_auc:.4f}")
    for name, r in completeness_by_target.items():
        print(f"  {name:14s} completeness   : "
              f"mean {r.mean():.5f}, max {r.max():.5f}")
    print("\n  --- Cross-cohort agreement (TCGA test vs METABRIC top-10) ---")
    for target_name in ("final_logit", "lumA_pole", "lumB_pole"):
        c = cross_cohort[target_name]
        if c["tcga_top10"]:
            print(f"    {target_name:14s} Jaccard top-10 = "
                  f"{c['jaccard_top10']:.3f}, top-50 = "
                  f"{c['jaccard_top50']:.3f}, shared = {c['shared_top10']}")
        else:
            print(f"    {target_name:14s} (TCGA TSV missing)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
