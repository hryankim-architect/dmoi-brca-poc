#!/usr/bin/env python3
"""DMOI v0.6: full 50-set MSigDB Hallmark pathway-level IG aggregation.

This is the v0.5 rollup widened from the 5 pole-relevant Hallmark sets
to the full 50-set MSigDB Hallmark catalog (v2024.1.Hs). The v0.5
audit asked the obvious question: 'is the top-pathway finding just
an artifact of which sets you chose to load?'. v0.6 answers it by
loading every Hallmark set and re-running the same IG -> pathway
rollup on TCGA test + METABRIC.

Pipeline:
  1. Load TCGA cohort_v2 train split + METABRIC LumA/LumB cohort.
  2. Train ONE Option A model on TCGA train (keep_artifacts=True).
  3. Run IG on the TCGA test split for lumA / lumB / final_logit targets.
  4. Run IG on METABRIC (meth silenced) for the same targets.
  5. Load full 50-set Hallmark catalog from
     `data/msigdb/h.all.v2024.1.Hs.symbols.gmt`.
  6. Aggregate per-gene IG to per-pathway scores via
     `dmoi_brca.pathway.pathway_aggregate`.
  7. Write `audit/dmoi_pathway_v0.6.md` with top-10 tables, full
     50-row CSV per (target, cohort), and a v0.5-survives section.

Honest scope: aggregation is still RNA-only (METABRIC has no
methylation; even on TCGA the meth features are HM450 probes, not
gene symbols).
"""
from __future__ import annotations

import csv
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
from dmoi_brca.hallmark import load_hallmark_gmt  # noqa: E402
from dmoi_brca.hypothesis_attention import (  # noqa: E402
    load_hm450_cis_mapping,
    make_pole_masks,
)
from dmoi_brca.pathway import pathway_aggregate, rank_pathways  # noqa: E402
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
N_IG_STEPS = 50

# The v0.5 finding to verify survives the 50-set widening.
V05_TOP_BY_TARGET: dict[str, tuple[str, ...]] = {
    "lumA_pole": (
        "HALLMARK_ESTROGEN_RESPONSE_EARLY",
        "HALLMARK_ESTROGEN_RESPONSE_LATE",
    ),
    "lumB_pole": (
        "HALLMARK_G2M_CHECKPOINT",
        "HALLMARK_E2F_TARGETS",
        "HALLMARK_MYC_TARGETS_V1",
    ),
    "final_logit": (
        "HALLMARK_ESTROGEN_RESPONSE_EARLY",
        "HALLMARK_ESTROGEN_RESPONSE_LATE",
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


def _top_k_md(target: str, scores_by_cohort: dict[str, list], k: int = 10) -> str:
    """Side-by-side top-K table for a single target."""
    tcga_ranked = rank_pathways(scores_by_cohort["TCGA test"], by="mean_abs_ig")[:k]
    metab_ranked = rank_pathways(scores_by_cohort["METABRIC"], by="mean_abs_ig")[:k]
    rows = ["| Rank | TCGA test pathway | mean \\|IG\\| | METABRIC pathway | mean \\|IG\\| |",
            "|---|---|---|---|---|"]
    for i in range(k):
        t = tcga_ranked[i] if i < len(tcga_ranked) else None
        m = metab_ranked[i] if i < len(metab_ranked) else None
        t_str = f"`{t.pathway_name}` | {t.mean_abs_ig:.5f}" if t else "— | —"
        m_str = f"`{m.pathway_name}` | {m.mean_abs_ig:.5f}" if m else "— | —"
        rows.append(f"| {i+1} | {t_str} | {m_str} |")
    return f"### {target} — top {k}\n\n" + "\n".join(rows)


def _write_full_csv(target: str, cohort: str, scores: list) -> Path:
    """One row per pathway, full 50-set view, for a single (target, cohort)."""
    safe = f"{target}__{cohort.replace(' ', '_')}"
    out = AUDIT / f"dmoi_pathway_v0.6_{safe}.csv"
    ranked = rank_pathways(scores, by="mean_abs_ig")
    with out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "rank", "pathway_name", "n_genes_total", "n_genes_in_inputs",
            "mean_abs_ig", "sum_signed", "signed_mean",
        ])
        for rank, s in enumerate(ranked, 1):
            writer.writerow([
                rank, s.pathway_name,
                s.n_pathway_genes_total, s.n_pathway_genes_in_inputs,
                f"{s.mean_abs_ig:.6f}",
                f"{s.sum_signed:.6f}",
                f"{s.signed_mean:.6f}",
            ])
    return out


def _v05_survives_check(
    target: str,
    pathway_results: dict[str, dict[str, list]],
) -> tuple[str, bool]:
    """Did the v0.5 top pathways stay in the v0.6 (50-set) top-3?"""
    expected = set(V05_TOP_BY_TARGET[target])
    out_lines = []
    all_survive = True
    for cohort in ("TCGA test", "METABRIC"):
        top3 = {
            s.pathway_name
            for s in rank_pathways(
                pathway_results[target][cohort], by="mean_abs_ig",
            )[:3]
        }
        survivors = expected & top3
        missing = expected - top3
        survived = bool(survivors) and not missing
        if missing:
            all_survive = False
        out_lines.append(
            f"- **{cohort}** top-3 (of 50) = "
            f"{', '.join(f'`{p}`' for p in sorted(top3))}. "
            f"v0.5 finding "
            f"({', '.join(f'`{p}`' for p in sorted(expected))}): "
            f"{'all present' if not missing else 'MISSING ' + ', '.join(f'`{p}`' for p in sorted(missing))}.",
        )
    return "\n".join(out_lines), all_survive


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

    print("=== DMOI v0.6: full-Hallmark pathway-level IG aggregation ===")

    # --- Full Hallmark catalog ---
    print(f"\n--- Loading Hallmark catalog: {HALLMARK_GMT.name} ---")
    hallmark_full: dict[str, list[str]] = load_hallmark_gmt(HALLMARK_GMT)
    print(f"  {len(hallmark_full)} Hallmark sets loaded.")

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
    pathway_results: dict[str, dict[str, list]] = {}
    csv_paths: list[Path] = []
    for target_name in ("lumA_pole", "lumB_pole", "final_logit"):
        pathway_results[target_name] = {}
        for cohort_name, (rna_x, meth_x) in cohort_inputs.items():
            print(f"\n--- IG + pathway rollup (50 sets): {target_name} on {cohort_name} ---")
            attr = integrated_gradients_dmoi(
                model, rna_x, meth_x,
                target=target_name, n_steps=N_IG_STEPS, device="cpu",
            )
            scores = pathway_aggregate(
                attr.rna_attribution, feats.rna_features, hallmark_full,
            )
            pathway_results[target_name][cohort_name] = scores
            top = rank_pathways(scores, by="mean_abs_ig")[:5]
            for s in top:
                print(f"  {s.pathway_name:45s}  "
                      f"mean|IG| {s.mean_abs_ig:.5f}  "
                      f"signed_mean {s.signed_mean:+.5f}  "
                      f"({s.n_pathway_genes_in_inputs} genes in inputs)")
            csv_paths.append(_write_full_csv(target_name, cohort_name, scores))

    # --- v0.5-survives check + cross-cohort top-3 ---
    print("\n--- v0.5-finding survives 50-set widening? ---")
    survives: dict[str, tuple[str, bool]] = {}
    for target_name in ("lumA_pole", "lumB_pole", "final_logit"):
        block, ok = _v05_survives_check(target_name, pathway_results)
        survives[target_name] = (block, ok)
        print(f"  {target_name:14s} survived={ok}")
        for line in block.splitlines():
            print(f"    {line}")

    cross_cohort: dict[str, dict] = {}
    for target_name in ("lumA_pole", "lumB_pole", "final_logit"):
        tcga_top = [
            s.pathway_name for s in rank_pathways(
                pathway_results[target_name]["TCGA test"], by="mean_abs_ig",
            )[:3]
        ]
        metab_top = [
            s.pathway_name for s in rank_pathways(
                pathway_results[target_name]["METABRIC"], by="mean_abs_ig",
            )[:3]
        ]
        shared = sorted(set(tcga_top) & set(metab_top))
        cross_cohort[target_name] = {
            "tcga_top3": tcga_top, "metab_top3": metab_top, "shared": shared,
        }

    # --- Audit MD ---
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    md_path = AUDIT / "dmoi_pathway_v0.6.md"

    def _shared_md(target_name):
        c = cross_cohort[target_name]
        return (
            f"- TCGA test top-3 (of 50): {', '.join(f'`{p}`' for p in c['tcga_top3'])}\n"
            f"- METABRIC top-3 (of 50) : {', '.join(f'`{p}`' for p in c['metab_top3'])}\n"
            f"- Shared : {', '.join(f'`{p}`' for p in c['shared']) or '_(none)_'}\n"
        )

    all_survived = all(ok for (_block, ok) in survives.values())
    survives_status = (
        "All v0.5 top pathways are still in the v0.6 (50-set) top-3 on "
        "both cohorts. The 5-set rollup wasn't an artifact of which sets "
        "were loaded — the same pathways win out of 50."
        if all_survived else
        "Some v0.5 top pathways did NOT remain in the v0.6 top-3 on every "
        "cohort. See the per-target block below."
    )

    md_path.write_text(
        "# DMOI v0.6 — Full 50-set Hallmark IG aggregation\n\n"
        f"Generated: {ts}\n\n"
        "## Setup\n\n"
        f"- Train cohort     : TCGA cohort_v2 train split, n={len(feats.sample_ids)}\n"
        f"- TCGA test cohort : n={len(feats_test.sample_ids)} "
        f"(AUROC {result.best_val_auc:.4f})\n"
        f"- METABRIC cohort  : n={ext_X_qn.shape[0]} (RNA-only, meth silenced)\n"
        f"- Pathway catalog  : {len(hallmark_full)} MSigDB Hallmark v2024.1.Hs sets "
        "loaded from `data/msigdb/h.all.v2024.1.Hs.symbols.gmt`\n"
        "- Aggregation      : per-pathway `mean |IG|`, `sum_signed`, "
        "`signed_mean` over per-patient × per-gene attributions\n\n"
        "## v0.5 finding survives the 50-set widening?\n\n"
        f"{survives_status}\n\n"
        "### lumA_pole\n\n" + survives["lumA_pole"][0] + "\n\n"
        "### lumB_pole\n\n" + survives["lumB_pole"][0] + "\n\n"
        "### final_logit\n\n" + survives["final_logit"][0] + "\n\n"
        "## Cross-cohort top-3 (of 50)\n\n"
        "### lumA_pole\n\n" + _shared_md("lumA_pole") + "\n"
        "### lumB_pole\n\n" + _shared_md("lumB_pole") + "\n"
        "### final_logit\n\n" + _shared_md("final_logit") + "\n"
        "## Top-10 pathways per target × cohort\n\n"
        + _top_k_md("lumA_pole", pathway_results["lumA_pole"]) + "\n\n"
        + _top_k_md("lumB_pole", pathway_results["lumB_pole"]) + "\n\n"
        + _top_k_md("final_logit", pathway_results["final_logit"]) + "\n\n"
        "## Full 50-row tables\n\n"
        "Full per-pathway tables (one CSV per (target, cohort) combination, "
        "ranked by `mean |IG|`):\n\n"
        + "\n".join(f"- `{p.relative_to(REPO)}`" for p in csv_paths)
        + "\n\n"
        "## Reading\n\n"
        "- `mean |IG|` — how loudly the pathway speaks (magnitude).\n"
        "- `signed_mean` — direction (positive = pushes toward LumB; "
        "negative = pushes toward LumA for the final logit; for the pole "
        "scores, positive = pushes toward 'this is the pole's class').\n"
        "- A pathway with high `mean |IG|` but `signed_mean ≈ 0` means "
        "the pathway has both pro- and anti- genes that roughly cancel.\n\n"
        "## Honest scope\n\n"
        "- 50 Hallmark sets loaded — the entire Hallmark v2024.1.Hs "
        "catalog. v0.6 closes the v0.5 caveat ('did you only load the "
        "5 sets that work?'). The C2 curated catalog (~5,000 sets) "
        "remains out of scope.\n"
        "- Aggregation is over the RNA modality only. METABRIC's "
        "methylation branch is silenced; even on TCGA the meth features "
        "are HM450 probes, not gene symbols, so a Hallmark rollup of "
        "meth IG would need a probe -> gene crosswalk.\n"
        "- The pathway scores are interpretation artifacts, not "
        "training signals. The model still attends to genes, not to "
        "pathways. Pathway-level *attention* (vs aggregation) is the "
        "natural v0.7+ candidate.\n\n"
        "## Reproduce\n\n"
        "```bash\n"
        "python scripts/aggregate_pathway_ig_full.py\n"
        "```\n",
    )
    print(f"\nWrote {md_path}")
    print(f"Wrote {len(csv_paths)} per-(target, cohort) CSVs in {AUDIT}/")

    print("\n=== DMOI v0.6 pathway aggregation summary ===")
    print(f"v0.5 survives 50-set widening on all 3 targets: {all_survived}")
    for target_name in ("lumA_pole", "lumB_pole", "final_logit"):
        c = cross_cohort[target_name]
        print(f"  {target_name:14s} TCGA-test top : {c['tcga_top3']}")
        print(f"  {' ' * 14} METABRIC top  : {c['metab_top3']}")
        print(f"  {' ' * 14} shared        : {c['shared']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
