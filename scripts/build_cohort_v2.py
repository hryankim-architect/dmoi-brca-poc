#!/usr/bin/env python3
"""Day-5A driver: build cohort v2 (LumA vs LumB) for the Week-2 re-scoped target.

Both poles are ER+; the discriminating axis is proliferation rate. Baseline
sklearn classifiers should land in the 0.7-0.85 AUC range — non-trivial,
literature-consistent, gives DMOI hypothesis-conditioning real headroom.

Reads:
  data/tcga_brca/BRCA_clinicalMatrix.tsv     (Xena phenotype matrix)
  data/tcga_brca/HiSeqV2.gz                  (RNA-seq sample IDs, header row)
  data/tcga_brca/HumanMethylation450.gz      (HM450 sample IDs, header row)

Writes:
  data/tcga_brca/cohort_v2.tsv               (gitignored — local artifact)
  audit/cohort_v2_summary.md                 (committed — n counts only, no PHI)
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone  # noqa: UP017
from pathlib import Path

UTC = timezone.utc  # noqa: UP017

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dmoi_brca.cohort import (  # noqa: E402
    assign_lumab_group,
    build_cohort,
    load_clinical,
    read_sample_ids_from_xena,
    train_test_split_cohort,
)

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "tcga_brca"
AUDIT = REPO / "audit"


def main() -> int:
    clinical_path = DATA / "BRCA_clinicalMatrix.tsv"
    rna_path = DATA / "HiSeqV2.gz"
    meth_path = DATA / "HumanMethylation450.gz"

    for p in (clinical_path, rna_path, meth_path):
        if not p.exists():
            sys.stderr.write(f"ERROR: missing input {p}\n")
            return 1

    print(f"Loading clinical from {clinical_path.name}...")
    clinical = load_clinical(clinical_path)
    print(f"  {len(clinical)} clinical rows")

    print(f"Reading RNA-seq sample IDs from {rna_path.name}...")
    rna_ids = read_sample_ids_from_xena(rna_path)
    print(f"  {len(rna_ids)} RNA-seq samples")

    print(f"Reading HM450 sample IDs from {meth_path.name}...")
    meth_ids = read_sample_ids_from_xena(meth_path)
    print(f"  {len(meth_ids)} HM450 samples")

    print("Building cohort v2 (LumA vs LumB)...")
    cohort, summary = build_cohort(
        clinical, rna_ids, meth_ids,
        assigner=assign_lumab_group,
        label_a="LumA", label_b="LumB",
    )

    print("Adding stratified 80/20 train/test split on dual-modality patients...")
    cohort = train_test_split_cohort(
        cohort, test_frac=0.2, random_state=2024,
        stratify_col="group", dual_modality_only=True,
    )
    n_train = int((cohort["split"] == "train").sum())
    n_test = int((cohort["split"] == "test").sum())
    n_train_lumA = int(((cohort["split"] == "train") & (cohort["group"] == "LumA")).sum())
    n_train_lumB = int(((cohort["split"] == "train") & (cohort["group"] == "LumB")).sum())
    n_test_lumA = int(((cohort["split"] == "test") & (cohort["group"] == "LumA")).sum())
    n_test_lumB = int(((cohort["split"] == "test") & (cohort["group"] == "LumB")).sum())

    out = DATA / "cohort_v2.tsv"
    cohort.to_csv(out, sep="\t", index=False)
    print(f"\nWrote {out} ({len(cohort)} rows)")
    print(f"  LumA:              {summary.n_luminal_h_plus}")
    print(f"  LumB:              {summary.n_basal_h_minus}")
    print(f"  both modalities:   {summary.n_both_modalities}")
    print(f"  RNA-only:          {summary.n_rna_only}")
    print(f"  meth-only:         {summary.n_meth_only}")
    print(f"  train split:       {n_train} (LumA={n_train_lumA}, LumB={n_train_lumB})")
    print(f"  test  split:       {n_test} (LumA={n_test_lumA}, LumB={n_test_lumB})")

    AUDIT.mkdir(exist_ok=True)
    summary_md = AUDIT / "cohort_v2_summary.md"
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    summary_md.write_text(
        "# DMOI POC Cohort v2 Summary (Day-5A — Week-2 re-scope)\n\n"
        f"Generated: {ts}\n\n"
        "## Rationale\n\n"
        "Day-4 baseline saturated at AUROC=1.0 on H+ luminal vs H- basal (cohort v1).\n"
        "Re-scoped Week-2 target to **within-luminal LumA vs LumB** — both poles ER+,\n"
        "discriminating axis is proliferation rate (LumB high Ki67/cell cycle).\n"
        "Literature baseline AUC ~0.70-0.85 on single-omic, much harder.\n\n"
        "## Inputs\n\n"
        f"- Clinical matrix: `{clinical_path.name}` ({len(clinical)} rows)\n"
        f"- RNA-seq samples (HiSeqV2): {len(rna_ids)}\n"
        f"- HM450 methylation samples: {len(meth_ids)}\n\n"
        "## Cohort v2 splits\n\n"
        "| Pole | Definition | n |\n"
        "|---|---|---|\n"
        f"| LumA | PAM50 = LumA (low proliferation, ER+) | "
        f"{summary.n_luminal_h_plus} |\n"
        f"| LumB | PAM50 = LumB (high proliferation, ER+) | "
        f"{summary.n_basal_h_minus} |\n"
        f"| **Total** | | **{len(cohort)}** |\n\n"
        "## Modality coverage\n\n"
        f"- Both RNA + methylation: {summary.n_both_modalities} "
        f"(DMOI dual-modality v2 training set)\n"
        f"- RNA only: {summary.n_rna_only}\n"
        f"- Methylation only: {summary.n_meth_only}\n\n"
        "## Train / test split (v0.2)\n\n"
        "Stratified 80/20 holdout on dual-modality patients, "
        "random_state=2024 (distinct from the CV seed 42). The test split is\n"
        "carved at cohort-construction time and only scored once at the end of\n"
        "each evaluation run — no model selection, no early stopping, no\n"
        "calibration fitting against it.\n\n"
        "| Split | LumA | LumB | Total |\n"
        "|---|---|---|---|\n"
        f"| train | {n_train_lumA} | {n_train_lumB} | {n_train} |\n"
        f"| test  | {n_test_lumA} | {n_test_lumB} | {n_test} |\n\n"
        "## Reproduce\n\n"
        "```bash\n"
        "python scripts/build_cohort_v2.py\n"
        "```\n"
    )
    print(f"Wrote {summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
