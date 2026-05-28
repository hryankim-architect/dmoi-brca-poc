#!/usr/bin/env python3
"""Day-3 driver: 5-fold CV training of DMOIModel on cohort_v2.

Reads:
  data/tcga_brca/cohort_v2.tsv         (Day-5A LumA/LumB split)
  data/tcga_brca/HiSeqV2.gz            (RNA-seq matrix)
  data/tcga_brca/HumanMethylation450.gz (meth matrix)
  data/tcga_brca/hm450_probemap.tsv    (Day-1 Step 2 cis-mapping)

Writes:
  audit/dmoi_per_fold.tsv              (per-fold metrics for comparison)
  audit/dmoi_results.md                (aggregate + saturation check)

Reproduce:
  python scripts/train_dmoi.py
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone  # noqa: UP017
from pathlib import Path

UTC = timezone.utc  # noqa: UP017

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


from dmoi_brca.features import load_features  # noqa: E402
from dmoi_brca.hypothesis_attention import (  # noqa: E402
    load_hm450_cis_mapping,
    make_pole_masks,
    summarize_mask_coverage,
)
from dmoi_brca.priors import POLE_LUMA, POLE_LUMB  # noqa: E402
from dmoi_brca.saturation import check_saturation  # noqa: E402
from dmoi_brca.train import aggregate_fold_results, run_dmoi_cv  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "tcga_brca"
AUDIT = REPO / "audit"


def _read_rna_feature_genes(rna_gz: Path) -> list[str]:
    """Read the gene symbols (first column of the Xena RNA matrix)."""
    import gzip

    genes: list[str] = []
    with gzip.open(rna_gz, "rt") as fh:
        fh.readline()  # header
        for line in fh:
            sym = line.split("\t", 1)[0]
            genes.append(sym)
    return genes


def main() -> int:
    cohort_tsv = DATA / "cohort_v2.tsv"
    rna_gz = DATA / "HiSeqV2.gz"
    meth_gz = DATA / "HumanMethylation450.gz"
    probemap = DATA / "hm450_probemap.tsv"

    for p in (cohort_tsv, rna_gz, meth_gz, probemap):
        if not p.exists():
            sys.stderr.write(f"ERROR: missing input {p}\n")
            return 1

    print("=== Day-3: DMOI training on cohort_v2 (LumA vs LumB) ===")

    # 1. Load features (same loader path as baseline_v2 for direct comparison).
    print("\n--- Feature load ---")
    feats = load_features(
        cohort_tsv=cohort_tsv,
        rna_gz=rna_gz,
        meth_gz=meth_gz,
        meth_topk=10_000,
        dual_modality_only=True,
        positive_label="LumB",   # LumB=1, LumA=0 — matches baseline_v2 convention
    )
    n = len(feats.sample_ids)
    print(f"\nDual-modality v2 cohort: {n} patients "
          f"(LumA={(feats.y == 0).sum()}, LumB={(feats.y == 1).sum()})")

    # 2. Build pole masks. RNA features = Xena gene symbols (first col).
    print("\n--- Pole masks ---")
    rna_feature_genes = feats.rna_features
    meth_feature_probes = feats.meth_features
    print(f"  RNA features (genes): {len(rna_feature_genes)}")
    print(f"  Meth features (probes): {len(meth_feature_probes)}")

    cis = load_hm450_cis_mapping(probemap)
    pole_masks = make_pole_masks(
        rna_feature_genes,
        meth_feature_probes,
        cis,
        {"LumA": POLE_LUMA, "LumB": POLE_LUMB},
    )
    print(summarize_mask_coverage(list(pole_masks.values())))

    # 3. 5-fold CV (same StratifiedKFold(random_state=42) as baseline_v2).
    print("\n--- 5-fold CV training ---")
    results = run_dmoi_cv(
        rna=feats.rna,
        meth=feats.meth,
        y=feats.y,
        pole_masks=pole_masks,
        n_splits=5,
        random_state=42,
        # Conservative defaults for the small cohort.
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
        device="auto",   # MPS on Mac, CPU fallback
        verbose=True,
    )

    # 4. Aggregate + per-fold TSV.
    agg = aggregate_fold_results(results)
    AUDIT.mkdir(exist_ok=True)
    per_fold = AUDIT / "dmoi_per_fold.tsv"
    with per_fold.open("w") as f:
        f.write(
            "fold\tbest_val_auc\tbest_val_bacc\tbest_epoch\t"
            "n_train\tn_test\tn_pos_train\tn_pos_test\truntime_sec\n",
        )
        for r in results:
            f.write(
                f"{r.fold}\t{r.best_val_auc:.4f}\t{r.best_val_bacc:.4f}\t"
                f"{r.best_epoch}\t{r.n_train}\t{r.n_test}\t"
                f"{r.n_pos_train}\t{r.n_pos_test}\t{r.runtime_seconds:.1f}\n",
            )
    print(f"\nWrote {per_fold}")

    # 5. Saturation check via Lσ substrate primitive (eat own dogfood).
    auc_means = [r.best_val_auc for r in results]
    sat = check_saturation(
        auc_means,
        metric="AUROC",
        candidate_causes=(
            "PAM50 LumA/LumB labels are RNA-derived; if pole-conditioned encoder "
            "captures the same signal, AUROC may saturate.",
            "Cohort size (417) may be small enough that 5-fold CV with strong "
            "regularization still over-fits the val set.",
        ),
    )

    # 6. Audit MD.
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    mask_lines = []
    for m in pole_masks.values():
        mask_lines.append(
            f"- **{m.pole_name}**: rna on {m.n_rna_on}/{m.rna_mask.numel()}, "
            f"meth on {m.n_meth_on}/{m.meth_mask.numel()}",
        )
    summary_md = AUDIT / "dmoi_results.md"
    summary_md.write_text(
        "# DMOI Training Results (Day-3 smoke run)\n\n"
        f"Generated: {ts}\n\n"
        f"## Cohort\n\n"
        f"- Dual-modality v2 patients: **{n}**\n"
        f"- LumA: {int((feats.y == 0).sum())} ({(feats.y == 0).mean()*100:.1f}%)\n"
        f"- LumB: {int((feats.y == 1).sum())} ({(feats.y == 1).mean()*100:.1f}%)\n\n"
        "## Pole masks (input gating)\n\n"
        + "\n".join(mask_lines) + "\n\n"
        "## Training config\n\n"
        "- Latent dim: 128\n"
        "- RNA encoder: 20530 → 1024 → 256 → 128\n"
        "- Meth encoder: 10000 → 512 → 128\n"
        "- Fuser per pole: 256 → 128 → 64; sub-classifier 64 → 1\n"
        "- Head: [z_LumA, z_LumB, disagreement] → 32 → 1\n"
        "- AdamW lr=1e-4 weight_decay=1e-4; batch=64; up to 50 epochs;\n"
        "  early stop on val AUROC, patience=10\n"
        "- BCEWithLogitsLoss + pos_weight = n_LumA / n_LumB (class-balanced)\n"
        "- StratifiedKFold(n_splits=5, shuffle=True, random_state=42) "
        "[same as baseline_v2]\n\n"
        "## Aggregate (mean ± std across 5 folds)\n\n"
        f"- **AUROC**: {agg['auc_mean']:.4f} ± {agg['auc_std']:.4f}\n"
        f"- **Balanced accuracy**: {agg['bacc_mean']:.4f} ± {agg['bacc_std']:.4f}\n"
        f"- Best epoch (mean / max): {agg['epoch_mean']:.1f} / {agg['epoch_max']:.0f}\n"
        f"- Total runtime: {agg['runtime_sec_total']:.1f} s\n\n"
        "## Head-to-head vs baseline_v2\n\n"
        "Baseline_v2 best concat configuration (LogReg, 5-fold same folds):\n"
        "- baseline concat LogReg: AUROC **0.963 ± 0.015**, BalAcc 0.892 ± 0.037\n"
        "- baseline meth   LogReg: AUROC **0.880 ± 0.030**, BalAcc 0.763 ± 0.060\n"
        "- baseline rna    LogReg: AUROC **0.961 ± 0.015**, BalAcc 0.891 ± 0.020\n\n"
        f"DMOI vs concat LogReg: ΔAUROC = {agg['auc_mean'] - 0.963:+.4f}\n"
        f"DMOI vs meth  LogReg: ΔAUROC = {agg['auc_mean'] - 0.880:+.4f}\n\n"
        f"{sat.audit_section()}\n"
        "## Reproduce\n\n"
        "```bash\n"
        "python scripts/train_dmoi.py\n"
        "```\n",
    )
    print(f"Wrote {summary_md}")

    print("\n=== Day-3 summary ===")
    print(f"  DMOI AUROC: {agg['auc_mean']:.4f} ± {agg['auc_std']:.4f}")
    print(f"  DMOI BalAcc: {agg['bacc_mean']:.4f} ± {agg['bacc_std']:.4f}")
    print(f"  Best epoch (mean): {agg['epoch_mean']:.1f}")
    print(f"  Runtime total: {agg['runtime_sec_total']:.1f} s")
    if sat.is_saturated:
        print("\n  ⚠ Saturation detected — see audit/dmoi_results.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
