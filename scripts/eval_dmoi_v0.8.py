#!/usr/bin/env python3
"""DMOI v0.8 Variant C: scalar -> vector pole feature.

v0.7.1 confirmed that a scalar pole_pathway_feat captures only pathway-
*magnitude* variance, not the per-patient *direction* signal that
discriminates LumA from LumB. The model learned (top weights 3.4x
uniform) but chose pathways with high baseline expression magnitude
(WNT, MTORC1, KRAS) rather than the discriminative ER / cell-cycle
programs that v0.6 IG identified.

v0.8 Variant C upgrades the pathway branch with a learnable per-pole
linear projection: each pole gets nn.Linear(n_pathways, proj_dim) that
maps the attention-gated pathway-score vector to a `proj_dim`-vector.
Each pathway thus has a learnable embedding (the row of the per-pole
projection matrix), and the head reads `n_poles * proj_dim` features
instead of just `n_poles` scalars.

Pipeline:
  1. Load TCGA cohort_v2 train / test split + METABRIC LumA/LumB cohort.
  2. Load full 50-set Hallmark catalog from data/msigdb/.
  3. Train DMOIModel(n_pathways=50, pathway_proj_dim=16) on TCGA train
     with pick_best_epoch=False (test set is held out, no val peeking).
  4. Score TCGA test + METABRIC.
  5. Extract `model.pathway_attention.attn_weights` and check whether
     the v0.6 IG-derived top-3 pathways now make the v0.8 top-3.
  6. Write audit/dmoi_v0.8.md.

The success criterion stays the same as v0.7.1: AUROC neutral and
learned attention reproduces the v0.6 ranking. The Variant C
information-flow argument is: a vector interface gives the head room
to combine per-pathway direction signals; if the learned attention
still misses the v0.6 ranking, that means the gene-level branch is
already capturing all the discriminative direction signal that the
pathway view could ever surface.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone  # noqa: UP017
from pathlib import Path

UTC = timezone.utc  # noqa: UP017

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

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
from dmoi_brca.pathway_attention import (  # noqa: E402
    compute_pathway_expression_scores,
)
from dmoi_brca.priors import POLE_LUMA, POLE_LUMB  # noqa: E402
from dmoi_brca.train import train_one_fold  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TCGA = REPO / "data" / "tcga_brca"
METABRIC = REPO / "data" / "metabric"
AUDIT = REPO / "audit"
HALLMARK_GMT = REPO / "data" / "msigdb" / "h.all.v2024.1.Hs.symbols.gmt"

FINAL_KWARGS = dict(
    latent_dim=128, rna_hidden=(1024, 256), meth_hidden=(512,),
    fuse_hidden=(128,), fuse_out=64, head_hidden=32, dropout=0.3,
    batch_size=64, lr=1e-4, weight_decay=1e-4,
    seed=42, device="auto", verbose=False,
    use_disagreement=True, aux_weight=0.3,
    calibration_frac=0.15, pick_best_epoch=False,
)
N_EPOCHS = 15
PATHWAY_PROJ_DIM = 16  # v0.8 Variant C: per-pole vector projection dim

# v0.6 finding (post-hoc IG rollup top-3) for comparison against the
# v0.7 learned attention. From audit/dmoi_pathway_v0.6.md, both cohorts:
V06_TOP3_BY_POLE: dict[str, tuple[str, ...]] = {
    "LumA": (
        "HALLMARK_ESTROGEN_RESPONSE_EARLY",
        "HALLMARK_ESTROGEN_RESPONSE_LATE",
        "HALLMARK_IL2_STAT5_SIGNALING",
    ),
    "LumB": (
        "HALLMARK_E2F_TARGETS",
        "HALLMARK_G2M_CHECKPOINT",
        "HALLMARK_MYC_TARGETS_V1",
    ),
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


def _score_cohort(
    model,
    rna_scaler,
    meth_scaler,
    rna_raw: np.ndarray,
    meth_raw: np.ndarray,
    rna_feature_names: list[str],
    pathways: dict[str, list[str]],
    device,
) -> tuple[np.ndarray, np.ndarray]:
    """Standardize gene/meth + compute raw pathway scores + score. Returns (proba, logits).

    v0.7.1 Phase B: pathway scores are computed from raw RNA expression
    (not standardized) and fed to the model unscaled. See train.py for
    the matching change.
    """
    rna = rna_scaler.transform(rna_raw).astype(np.float32)
    meth = meth_scaler.transform(meth_raw).astype(np.float32)
    path, _ = compute_pathway_expression_scores(
        rna_raw, rna_feature_names, pathways,
    )
    path = path.astype(np.float32)
    x_rna = torch.from_numpy(rna).to(device)
    x_meth = torch.from_numpy(meth).to(device)
    x_path = torch.from_numpy(path).to(device)
    model.eval()
    with torch.no_grad():
        out = model(x_rna, x_meth, x_path)
        proba = torch.sigmoid(out["logits"]).cpu().numpy()
        logits = out["logits"].cpu().numpy()
    return proba, logits


def main() -> int:
    cohort_tsv = TCGA / "cohort_v2.tsv"
    rna_gz = TCGA / "HiSeqV2.gz"
    meth_gz = TCGA / "HumanMethylation450.gz"
    probemap = TCGA / "hm450_probemap.tsv"
    metabric_cohort_tsv = METABRIC / "cohort.tsv"
    metabric_mrna = METABRIC / "mrna_microarray.txt"

    for p in (cohort_tsv, rna_gz, meth_gz, probemap,
              metabric_cohort_tsv, metabric_mrna, HALLMARK_GMT):
        if not p.exists():
            sys.stderr.write(f"ERROR: missing input {p}\n")
            return 1

    print("=== DMOI v0.8 Variant C: vector-per-pole pathway feature ===")
    print(f"    proj_dim = {PATHWAY_PROJ_DIM} (per-pole vector projection)")

    # --- Hallmark catalog ---
    print(f"\n--- Loading Hallmark catalog: {HALLMARK_GMT.name} ---")
    hallmark: dict[str, list[str]] = load_hallmark_gmt(HALLMARK_GMT)
    list(hallmark.keys())
    n_pathways = len(hallmark)
    print(f"  {n_pathways} Hallmark sets loaded.")

    # --- TCGA cohort + train/test slice ---
    print("\n--- Loading TCGA features ---")
    feats_all = load_features(
        cohort_tsv=cohort_tsv, rna_gz=rna_gz, meth_gz=meth_gz,
        meth_topk=10_000, dual_modality_only=True, positive_label="LumB",
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
    print(f"  TCGA train: {len(feats.sample_ids)}, test: {len(feats_test.sample_ids)}")

    # --- METABRIC ---
    print("\n--- Loading METABRIC ---")
    metabric_cohort = pd.read_csv(metabric_cohort_tsv, sep="\t")
    metabric_cohort = metabric_cohort[metabric_cohort["has_rna"]].copy()
    ext_ids_wanted = set(metabric_cohort["sample_id"].astype(str))
    ext_X_raw, ext_genes, ext_sample_ids = _load_metabric_mrna(
        metabric_mrna, ext_ids_wanted,
    )
    ext_cohort = metabric_cohort.set_index("sample_id").loc[ext_sample_ids]
    y_ext = ext_cohort["group"].map({"LumA": 0, "LumB": 1}).to_numpy().astype(np.int64)
    print(f"  METABRIC LumA/LumB with mRNA: {ext_X_raw.shape[0]} patients "
          f"(LumA={int((y_ext == 0).sum())}, LumB={int((y_ext == 1).sum())})")

    # --- Align + QN METABRIC ---
    ext_X_aligned = align_to_train_genes(
        ext_X_raw, ext_genes, feats.rna_features, fill_value=0.0,
    )
    ext_X_qn = quantile_normalize_to_train(ext_X_aligned, feats.rna)
    meth_ext_silenced = make_silenced_meth(
        ext_X_qn.shape[0], feats.meth.shape[1],
    )

    # --- Pole masks + train v0.7 model with pathway-pole attention ---
    print("\n--- Training v0.7 model (n_pathways=50, keep_artifacts=True) ---")
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
        pathway_genes=hallmark,
        rna_feature_names=feats.rna_features,
        pathway_proj_dim=PATHWAY_PROJ_DIM,  # v0.8 Variant C
        **FINAL_KWARGS,
    )
    if result.model is None or result.rna_scaler is None or result.meth_scaler is None:
        sys.stderr.write("ERROR: train_one_fold returned no artifacts.\n")
        return 1
    if result.pathway_names is None:
        sys.stderr.write("ERROR: train_one_fold returned no pathway names.\n")
        return 1
    print(f"  v0.8 TCGA test AUROC : {result.best_val_auc:.4f} "
          f"(v0.6 ref 0.9682; v0.7.1 ref 0.9595)")

    # --- Score METABRIC ---
    print("\n--- Scoring METABRIC (RNA-only, meth silenced + QN) ---")
    device = next(result.model.parameters()).device
    metab_proba, _ = _score_cohort(
        result.model,
        result.rna_scaler, result.meth_scaler,
        ext_X_qn, meth_ext_silenced,
        list(feats.rna_features), hallmark,
        device,
    )
    metab_auc = float(roc_auc_score(y_ext, metab_proba))
    print(f"  v0.8 METABRIC AUROC : {metab_auc:.4f} "
          f"(v0.6 ref 0.9091; v0.7.1 ref 0.8976)")

    # --- Extract learned pathway-pole attention ---
    print("\n--- Learned pathway-pole attention ---")
    pa = result.model.pathway_attention
    if pa is None:
        sys.stderr.write("ERROR: pathway_attention is None.\n")
        return 1
    learned_top: dict[str, list[tuple[str, float]]] = pa.top_k_pathways(
        result.pathway_names, k=10,
    )
    for pole, top in learned_top.items():
        print(f"  {pole}:")
        for rank, (pname, w) in enumerate(top[:5], 1):
            print(f"    {rank}. {pname:45s}  weight={w:.4f}")

    # --- Compare learned top-3 to v0.6 hand-picked / IG-derived top-3 ---
    print("\n--- v0.6 vs v0.7 top-3 pathway agreement ---")
    agreement: dict[str, dict] = {}
    for pole in ("LumA", "LumB"):
        learned_top3 = {p for p, _ in learned_top[pole][:3]}
        v06_top3 = set(V06_TOP3_BY_POLE[pole])
        shared = sorted(learned_top3 & v06_top3)
        agreement[pole] = {
            "learned_top3": sorted(learned_top3),
            "v06_top3": sorted(v06_top3),
            "shared": shared,
            "n_shared": len(shared),
        }
        print(f"  {pole}: learned={sorted(learned_top3)}")
        print(f"  {' ' * (len(pole) + 2)} v0.6 IG={sorted(v06_top3)}")
        print(f"  {' ' * (len(pole) + 2)} shared (n={len(shared)})={shared}")

    # --- Audit MD ---
    AUDIT.mkdir(exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    md_path = AUDIT / "dmoi_v0.8.md"

    auroc_delta_tcga = result.best_val_auc - 0.9682
    auroc_delta_metab = metab_auc - 0.9091
    auroc_verdict = (
        "AUROC HELD" if abs(auroc_delta_tcga) <= 0.005 and abs(auroc_delta_metab) <= 0.01
        else ("AUROC LIFTED" if auroc_delta_tcga > 0.005 else "AUROC DROPPED")
    )

    def _rank_table(pole: str) -> str:
        rows = [
            "| Rank | Pathway | softmax weight |",
            "|---|---|---|",
        ]
        for i, (pname, w) in enumerate(learned_top[pole][:10], 1):
            rows.append(f"| {i} | `{pname}` | {w:.4f} |")
        return "\n".join(rows)

    md_path.write_text(
        "# DMOI v0.8 -- Pathway-pole attention (Variant C: vector-per-pole)\n\n"
        f"Generated: {ts} (proj_dim={PATHWAY_PROJ_DIM})\n\n"
        "## Setup\n\n"
        f"- Architecture: v0.6 base + `PathwayPoleAttention(n_pathways=50, "
        f"proj_dim={PATHWAY_PROJ_DIM})`. Per-pole softmax over the full "
        "v2024.1.Hs Hallmark catalog, then a learnable "
        f"`Linear(n_pathways={n_pathways}, proj_dim={PATHWAY_PROJ_DIM})` per "
        "pole that maps the attention-gated pathway-score vector to a "
        f"{PATHWAY_PROJ_DIM}-dim feature. The head sees "
        f"`n_poles * proj_dim` = {2 * PATHWAY_PROJ_DIM} pathway features.\n"
        f"- Train cohort: TCGA cohort_v2 train split, n={len(feats.sample_ids)}.\n"
        f"- TCGA test:    n={len(feats_test.sample_ids)} (held out, scored once).\n"
        f"- METABRIC:     n={ext_X_qn.shape[0]} (RNA-only + meth silenced + QN).\n"
        f"- Epochs: {N_EPOCHS}, optimizer: AdamW(lr=1e-4, wd=1e-4), "
        f"BCEWithLogitsLoss + aux=0.3, pick_best_epoch=False (no val peeking).\n\n"
        "## Headline AUROC\n\n"
        f"| Cohort | v0.8 AUROC | v0.6 reference | v0.7.1 reference | Delta vs v0.6 |\n"
        f"|---|---|---|---|---|\n"
        f"| TCGA held-out test | {result.best_val_auc:.4f} | 0.9682 | 0.9595 | "
        f"{auroc_delta_tcga:+.4f} |\n"
        f"| METABRIC external  | {metab_auc:.4f} | 0.9091 | 0.8976 | "
        f"{auroc_delta_metab:+.4f} |\n\n"
        f"**Verdict**: {auroc_verdict}.\n\n"
        "v0.7.1 (scalar pole feature) captured pathway *magnitude* variance "
        "only and chose the wrong basin. v0.8 Variant C upgrades the per-pole "
        f"feature from scalar to a {PATHWAY_PROJ_DIM}-vector via a learnable "
        "per-pole linear projection. The head now reads per-pathway direction "
        "signals (each pathway has a learned embedding row in the projection "
        "matrix), so the architecture-level question is: with this richer "
        "interface, does the model finally find the v0.6 IG-derived "
        "ER-for-LumA / cell-cycle-for-LumB ranking?\n\n"
        "## Learned pathway-pole attention (top 10 per pole)\n\n"
        "Note: with Variant C, the `softmax weight` is no longer the only "
        "signal between attention and head -- the per-pole projection row "
        "absorbs much of the direction-encoding. The softmax distribution "
        "still indicates which pathways the model considers each pole's "
        "primary inputs, but with a vector interface the model has another "
        "knob to turn.\n\n"
        "### LumA\n\n" + _rank_table("LumA") + "\n\n"
        "### LumB\n\n" + _rank_table("LumB") + "\n\n"
        "## v0.6 (post-hoc IG) vs v0.8 (learned attention) -- top-3 agreement\n\n"
        "v0.6 top-3 per pole comes from `audit/dmoi_pathway_v0.6.md` "
        "(50-set Hallmark IG rollup, identical on TCGA test + METABRIC).\n\n"
        "| Pole | v0.8 learned top-3 | v0.6 IG top-3 | Shared (n / 3) |\n"
        "|---|---|---|---|\n"
        + "\n".join(
            f"| **{pole}** | "
            f"{', '.join(f'`{p}`' for p in agreement[pole]['learned_top3'])} | "
            f"{', '.join(f'`{p}`' for p in agreement[pole]['v06_top3'])} | "
            f"{agreement[pole]['n_shared']} / 3 |"
            for pole in ("LumA", "LumB")
        )
        + "\n\n"
        "## Reading\n\n"
        "- `softmax weight` -- each pole's attention weights sum to 1.0 across the "
        "50 pathways. Weight = 0.02 means \"uniform\" (1/50). Anything above 0.05 "
        "is a meaningful preference; above 0.20 is strong concentration.\n"
        "- Agreement count is informational, not a hypothesis test.\n"
        f"- The per-pole projection adds n_poles * n_pathways * proj_dim = "
        f"{2 * n_pathways * PATHWAY_PROJ_DIM} parameters over v0.7.1.\n\n"
        "## Limitations\n\n"
        "- Single-fold final-model run (no CV). The v0.8 Variant C "
        "architecture diff is what's under test.\n"
        f"- proj_dim={PATHWAY_PROJ_DIM} is a chosen hyperparameter; not swept.\n"
        "- Gene-level interpretation (v0.3 / v0.4 IG) is unaffected -- the "
        "gene-level branch is unchanged from v0.6.\n\n"
        "## Reproduce\n\n"
        "```bash\n"
        "python scripts/eval_dmoi_v0.8.py\n"
        "```\n",
    )
    print(f"\nWrote {md_path}")

    print("\n=== DMOI v0.8 summary ===")
    print(f"  TCGA AUROC: {result.best_val_auc:.4f} (v0.6: 0.9682, Δ={auroc_delta_tcga:+.4f}; v0.7.1: 0.9595)")
    print(f"  METAB AUROC: {metab_auc:.4f} (v0.6: 0.9091, Δ={auroc_delta_metab:+.4f}; v0.7.1: 0.8976)")
    print(f"  Verdict: {auroc_verdict}")
    for pole in ("LumA", "LumB"):
        print(f"  {pole}: shared top-3 with v0.6 IG = {agreement[pole]['n_shared']} / 3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
