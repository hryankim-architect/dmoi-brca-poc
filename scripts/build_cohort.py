#!/usr/bin/env python3
"""Day-3 driver: build DMOI POC cohort split from downloaded TCGA-BRCA data.

Reads:
  data/tcga_brca/BRCA_clinicalMatrix.tsv     (Xena phenotype matrix)
  data/tcga_brca/HiSeqV2.gz                  (RNA-seq sample IDs, header row)
  data/tcga_brca/HumanMethylation450.gz      (HM450 sample IDs, header row)

Writes:
  data/tcga_brca/cohort.tsv                  (gitignored — local artifact)
  audit/cohort_summary.md                    (committed — n counts only, no PHI)

Usage:
  python scripts/build_cohort.py
"""
from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dmoi_brca.cohort import (  # noqa: E402
    build_cohort,
    load_clinical,
    read_sample_ids_from_xena,
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
            sys.stderr.write("Run scripts/download_tcga_brca.sh first.\n")
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

    print("Building cohort (H+ luminal vs H- basal/TN)...")
    cohort, summary = build_cohort(clinical, rna_ids, meth_ids)

    out = DATA / "cohort.tsv"
    cohort.to_csv(out, sep="\t", index=False)
    print(f"\nWrote {out} ({len(cohort)} rows)")
    print(f"  H+ luminal:        {summary.n_luminal_h_plus}")
    print(f"  H- basal/TN:       {summary.n_basal_h_minus}")
    print(f"  both modalities:   {summary.n_both_modalities}")
    print(f"  RNA-only:          {summary.n_rna_only}")
    print(f"  meth-only:         {summary.n_meth_only}")

    AUDIT.mkdir(exist_ok=True)
    summary_md = AUDIT / "cohort_summary.md"
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    summary_md.write_text(
        "# DMOI POC Cohort Summary (Day-3)\n\n"
        f"Generated: {ts}\n\n"
        "## Inputs\n\n"
        f"- Clinical matrix: `{clinical_path.name}` ({len(clinical)} rows)\n"
        f"- RNA-seq samples (HiSeqV2): {len(rna_ids)}\n"
        f"- HM450 methylation samples: {len(meth_ids)}\n\n"
        "## Cohort splits\n\n"
        "| Pole | Definition | n |\n"
        "|---|---|---|\n"
        f"| H+ (luminal) | PAM50 in {{LumA, LumB}} AND ER positive | "
        f"{summary.n_luminal_h_plus} |\n"
        f"| H- (basal/TN) | PAM50 = Basal AND ER/PR/HER2 all negative | "
        f"{summary.n_basal_h_minus} |\n"
        f"| **Total** | | **{len(cohort)}** |\n\n"
        "## Modality coverage\n\n"
        f"- Both RNA + methylation: {summary.n_both_modalities} (DMOI dual-modality training set)\n"
        f"- RNA only: {summary.n_rna_only}\n"
        f"- Methylation only: {summary.n_meth_only}\n\n"
        "## Reproduce\n\n"
        "```bash\n"
        "make data           # or bash scripts/download_tcga_brca.sh\n"
        "python scripts/build_cohort.py\n"
        "```\n\n"
        "## Notes\n\n"
        "- PAM50 source: `PAM50Call_RNAseq` (primary, ~956/1247 coverage) with "
        "fallback to `PAM50_mRNA_nature2012` (long-form labels normalized).\n"
        "- Other PAM50 subtypes (Her2-enriched, Normal-like) are excluded from "
        "the POC — DMOI POC contrasts the H+ vs H- poles only.\n"
        "- `cohort.tsv` lives under `data/` and is gitignored (sample IDs are "
        "TCGA barcodes, derived from open-tier data but kept out of git per the "
        "scaffold's data-handling convention).\n"
    )
    print(f"Wrote {summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
