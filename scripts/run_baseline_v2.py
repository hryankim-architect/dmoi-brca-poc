#!/usr/bin/env python3
"""Day-5A driver: sanity baseline on cohort v2 (LumA vs LumB).

If this saturates at AUC=1.0 again, the re-scope failed and we need a third target.
Expected: AUC ~0.70-0.85 on single-omic, modest concat gain.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone  # noqa: UP017
from pathlib import Path

UTC = timezone.utc  # noqa: UP017

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from dmoi_brca.baseline import aggregate, run_cv  # noqa: E402
from dmoi_brca.features import load_features  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "tcga_brca"
AUDIT = REPO / "audit"


def main() -> int:
    cohort_tsv = DATA / "cohort_v2.tsv"
    rna_gz = DATA / "HiSeqV2.gz"
    meth_gz = DATA / "HumanMethylation450.gz"

    for p in (cohort_tsv, rna_gz, meth_gz):
        if not p.exists():
            sys.stderr.write(f"ERROR: missing {p}\n")
            sys.stderr.write("Run scripts/build_cohort_v2.py first.\n")
            return 1

    print("=== Day-5A baseline (cohort v2: LumA vs LumB) ===")
    feats = load_features(
        cohort_tsv=cohort_tsv,
        rna_gz=rna_gz,
        meth_gz=meth_gz,
        meth_topk=10_000,
        dual_modality_only=True,
        positive_label="LumB",
    )

    n = len(feats.sample_ids)
    print(f"\nDual-modality v2 cohort: {n} patients "
          f"(LumA={(feats.y == 0).sum()}, LumB={(feats.y == 1).sum()})")
    print(f"  RNA features: {feats.rna.shape[1]}")
    print(f"  Meth features: {feats.meth.shape[1]}")

    feature_sets = {
        "rna": feats.rna,
        "meth": feats.meth,
        "concat": np.concatenate([feats.rna, feats.meth], axis=1),
    }

    print("\n=== Day-5A baseline: 5-fold CV ===")
    results = run_cv(feature_sets, feats.y, n_splits=5, random_state=42)
    agg = aggregate(results)

    AUDIT.mkdir(exist_ok=True)
    per_fold = AUDIT / "baseline_v2_per_fold.tsv"
    with per_fold.open("w") as f:
        f.write("fold\tfeature_set\tmodel\tauc\tbacc\tn_train\tn_test\tn_pos_train\tn_pos_test\n")
        for r in results:
            f.write(f"{r.fold}\t{r.feature_set}\t{r.model}\t{r.auc:.4f}\t"
                    f"{r.bacc:.4f}\t{r.n_train}\t{r.n_test}\t"
                    f"{r.n_pos_train}\t{r.n_pos_test}\n")
    print(f"\nWrote {per_fold}")

    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    summary_md = AUDIT / "baseline_v2_results.md"
    rows = []
    for (fset, model), stats in sorted(agg.items()):
        rows.append(
            f"| {fset} | {model} | "
            f"{stats['auc_mean']:.4f} ± {stats['auc_std']:.4f} | "
            f"{stats['bacc_mean']:.4f} ± {stats['bacc_std']:.4f} |"
        )
    table = "\n".join(rows)

    auc_means = [stats["auc_mean"] for stats in agg.values()]
    has_headroom = any(0.55 <= a <= 0.95 for a in auc_means)
    if has_headroom:
        scope = (
            "## Headroom confirmed\n\n"
            "At least one baseline configuration lands in the 0.55-0.95 AUROC range,\n"
            "leaving room for Week-2 hypothesis-conditioning to show a meaningful gain.\n"
            "The LumA vs LumB re-scope is validated.\n"
        )
    elif max(auc_means) > 0.95:
        scope = (
            "## SATURATION AGAIN\n\n"
            "Baselines exceed 0.95 AUROC. LumA vs LumB also too easy on dual-omic.\n"
            "Need a third target — survival prediction or cross-modal regression.\n"
        )
    else:
        scope = (
            "## Below chance / too noisy\n\n"
            "All baselines below 0.55 AUROC — model isn't learning. Check labels,\n"
            "class balance, and feature scaling before proceeding.\n"
        )

    summary_md.write_text(
        "# DMOI POC Baseline v2 Results (Day-5A)\n\n"
        f"Generated: {ts}\n\n"
        "## Target\n\n"
        "Within-luminal LumA vs LumB (Week-2 re-scope).\n"
        "Discriminating axis: proliferation rate.\n\n"
        f"## Cohort\n\n"
        f"- Dual-modality patients: **{n}**\n"
        f"- LumA: {int((feats.y == 0).sum())} ({(feats.y == 0).mean()*100:.1f}%)\n"
        f"- LumB: {int((feats.y == 1).sum())} ({(feats.y == 1).mean()*100:.1f}%)\n\n"
        "## Features\n\n"
        f"- RNA-seq: {feats.rna.shape[1]} genes\n"
        f"- Methylation (HM450): {feats.meth.shape[1]} probes (top-variance from 485k)\n"
        f"- Concat: {feats.rna.shape[1] + feats.meth.shape[1]} features\n\n"
        "## Results (mean ± std, 5-fold CV)\n\n"
        "| Feature set | Model | AUROC | Balanced accuracy |\n"
        "|---|---|---|---|\n"
        f"{table}\n\n"
        f"{scope}\n"
        "## Reproduce\n\n"
        "```bash\n"
        "python scripts/build_cohort_v2.py\n"
        "python scripts/run_baseline_v2.py\n"
        "```\n"
    )
    print(f"Wrote {summary_md}")

    print("\n=== Day-5A summary (mean ± std, 5-fold CV) ===")
    print(f"  {'feature_set':<10}  {'model':<8}  {'AUROC':<20}  {'BalAcc':<20}")
    for (fset, model), stats in sorted(agg.items()):
        print(f"  {fset:<10}  {model:<8}  "
              f"{stats['auc_mean']:.4f} ± {stats['auc_std']:.4f}    "
              f"{stats['bacc_mean']:.4f} ± {stats['bacc_std']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
