#!/usr/bin/env python3
"""Day-4 driver: load 395-patient dual-modality features + run baseline CV.

Outputs:
  audit/baseline_results.md     (committed, aggregate AUC + balanced accuracy)
  audit/baseline_per_fold.tsv   (committed, per-fold details for reproducibility)
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone  # noqa: UP017 -- UTC alias for py<3.11
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
    cohort_tsv = DATA / "cohort.tsv"
    rna_gz = DATA / "HiSeqV2.gz"
    meth_gz = DATA / "HumanMethylation450.gz"

    for p in (cohort_tsv, rna_gz, meth_gz):
        if not p.exists():
            sys.stderr.write(f"ERROR: missing {p}\n")
            sys.stderr.write("Run scripts/build_cohort.py first.\n")
            return 1

    print("=== Day-4 baseline: feature load ===")
    feats = load_features(
        cohort_tsv=cohort_tsv,
        rna_gz=rna_gz,
        meth_gz=meth_gz,
        meth_topk=10_000,
        dual_modality_only=True,
    )

    n = len(feats.sample_ids)
    print(f"\nDual-modality cohort: {n} patients "
          f"(H+={(feats.y == 0).sum()}, H-={(feats.y == 1).sum()})")
    print(f"  RNA features: {feats.rna.shape[1]}")
    print(f"  Meth features: {feats.meth.shape[1]}")

    feature_sets = {
        "rna": feats.rna,
        "meth": feats.meth,
        "concat": np.concatenate([feats.rna, feats.meth], axis=1),
    }

    print("\n=== Day-4 baseline: 5-fold CV ===")
    results = run_cv(feature_sets, feats.y, n_splits=5, random_state=42)
    agg = aggregate(results)

    # Per-fold dump.
    AUDIT.mkdir(exist_ok=True)
    per_fold = AUDIT / "baseline_per_fold.tsv"
    with per_fold.open("w") as f:
        f.write("fold\tfeature_set\tmodel\tauc\tbacc\tn_train\tn_test\tn_pos_train\tn_pos_test\n")
        for r in results:
            f.write(f"{r.fold}\t{r.feature_set}\t{r.model}\t{r.auc:.4f}\t"
                    f"{r.bacc:.4f}\t{r.n_train}\t{r.n_test}\t"
                    f"{r.n_pos_train}\t{r.n_pos_test}\n")
    print(f"\nWrote {per_fold}")

    # Aggregate audit MD.
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    summary_md = AUDIT / "baseline_results.md"
    rows = []
    for (fset, model), stats in sorted(agg.items()):
        rows.append(
            f"| {fset} | {model} | "
            f"{stats['auc_mean']:.4f} ± {stats['auc_std']:.4f} | "
            f"{stats['bacc_mean']:.4f} ± {stats['bacc_std']:.4f} |"
        )
    table = "\n".join(rows)

    # Detect saturation: if every model hits AUC >= 0.99, the task is too easy.
    auc_means = [stats["auc_mean"] for stats in agg.values()]
    is_saturated = all(a >= 0.99 for a in auc_means)
    if is_saturated:
        honest_scope = (
            "## Saturation finding (honest)\n\n"
            "Every (feature_set, model) combination hits AUROC >= 0.99 in 5-fold CV.\n"
            "This is **not** a successful baseline — it means the H+ (luminal) vs\n"
            "H- (basal/TN) task is **too easy** for a DMOI POC discrimination target:\n\n"
            "- The PAM50 labels (LumA/LumB/Basal) used to define the poles are themselves\n"
            "  derived from RNA-seq, so RNA-based classification is partially circular.\n"
            "- The two poles are biologically very distinct cell-of-origin states.\n"
            "  Single-modality discrimination has saturated decades of literature.\n"
            "- Without baseline headroom, the Week-2 DMOI hypothesis-conditioned encoder\n"
            "  cannot demonstrate value on this task.\n\n"
            "**Honest next step**: re-scope the Week-2 discrimination target to a harder\n"
            "task on the same cohort. Candidates:\n\n"
            "- Within-luminal LumA vs LumB (PAM50 mRNA_nature2012 sub-call).\n"
            "- 5-year overall survival / progression-free survival prediction.\n"
            "- Response to neoadjuvant chemotherapy on the basal subset.\n"
            "- Methylation-only prediction of an *RNA-derived* signature score where\n"
            "  the cross-modal task is non-trivial.\n\n"
            "The Day-4 deliverable is therefore **a negative finding + scope decision**\n"
            "rather than a comparison number to beat. The substrate, cohort, and\n"
            "feature pipeline are validated; the Week-2 target needs adjustment.\n"
        )
    else:
        honest_scope = (
            "## Honest scope\n\n"
            "These are simple sklearn baselines on the dual-modality cohort, not the\n"
            "upstream MGDMCL contrastive framework. The point is to record a non-trivial\n"
            "comparison anchor before the Week-2 DMOI hypothesis-conditioned encoder\n"
            "runs against the same cohort + same CV folds.\n"
        )

    summary_md.write_text(
        "# DMOI POC Baseline Results (Day-4)\n\n"
        f"Generated: {ts}\n\n"
        f"## Cohort\n\n"
        f"- Dual-modality patients: **{n}**\n"
        f"- H+ luminal: {int((feats.y == 0).sum())} "
        f"({(feats.y == 0).mean()*100:.1f}%)\n"
        f"- H- basal/TN: {int((feats.y == 1).sum())} "
        f"({(feats.y == 1).mean()*100:.1f}%)\n\n"
        "## Features\n\n"
        f"- RNA-seq (HiSeqV2): {feats.rna.shape[1]} genes (all retained)\n"
        f"- Methylation (HM450): {feats.meth.shape[1]} probes "
        f"(top-variance filter from 485,577)\n"
        f"- Concatenated (early fusion): {feats.rna.shape[1] + feats.meth.shape[1]} features\n\n"
        "## Models\n\n"
        "- `logreg`: L2 LogisticRegression, class_weight='balanced', max_iter=2000\n"
        "- `rf`: RandomForest, 300 trees, class_weight='balanced'\n"
        "- StandardScaler upstream of both\n"
        "- StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n\n"
        "## Results (mean ± std across 5 folds)\n\n"
        "| Feature set | Model | AUROC | Balanced accuracy |\n"
        "|---|---|---|---|\n"
        f"{table}\n\n"
        f"{honest_scope}\n"
        "## Reproduce\n\n"
        "```bash\n"
        "python scripts/build_cohort.py     # if cohort.tsv missing\n"
        "python scripts/run_baseline.py\n"
        "```\n"
    )
    print(f"Wrote {summary_md}")

    # Print aggregate table to terminal too.
    print("\n=== Day-4 baseline summary (mean ± std, 5-fold CV) ===")
    print(f"  {'feature_set':<10}  {'model':<8}  {'AUROC':<20}  {'BalAcc':<20}")
    for (fset, model), stats in sorted(agg.items()):
        print(f"  {fset:<10}  {model:<8}  "
              f"{stats['auc_mean']:.4f} ± {stats['auc_std']:.4f}    "
              f"{stats['bacc_mean']:.4f} ± {stats['bacc_std']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
