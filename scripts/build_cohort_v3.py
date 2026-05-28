#!/usr/bin/env python3
"""v0.9: build TCGA-BRCA Luminal-vs-Basal dual-modality cohort (cohort_v3.tsv).

v0.6/v0.8 worked with `cohort_v2.tsv` (LumA-vs-LumB, n=417 dual-modality).
v0.9 changes the classification axis from within-luminal (ER+ subtypes) to
cross-lineage (luminal vs basal). The architecture, attention layer, fusion,
and training loop are all unchanged from v0.6 -- only the cohort and the
pole-defining Hallmark priors change. The v0.9 experiment tests whether the
v0.6 architectural commitment transfers to a different classification axis.

Output: `data/tcga_brca/cohort_v3.tsv` with columns

    sample_id  group  has_rna  has_meth  split

where:
    group ∈ {"Luminal", "Basal"}  (Luminal = LumA or LumB per PAM50call_RNAseq)
    has_rna  = True iff sample_id is in HiSeqV2.gz header
    has_meth = True iff sample_id is in HumanMethylation450.gz header
    split    ∈ {"train", "test"} for dual-modality samples (80/20 stratified
              on group, random_state=2024 -- same as v0.2 cohort_v2 split
              protocol). Single-modality samples have split=NaN.

The PAM50 source is column `PAM50Call_RNAseq` from `BRCA_clinicalMatrix.tsv`.

Expected counts (from recon):

    PAM50         dual-modality
    --------      -------------
    LumA          288
    LumB          127
    Basal          87
    --> Luminal   415  (LumA + LumB)
    --> Basal      87
    Total         502  (80/20 -> ~401 train / ~101 test)

This is roughly 1.2x the size of cohort_v2's 417 LumA+LumB.
"""
from __future__ import annotations

import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[1]
TCGA = REPO / "data" / "tcga_brca"
CLINICAL = TCGA / "BRCA_clinicalMatrix.tsv"
RNA_GZ = TCGA / "HiSeqV2.gz"
METH_GZ = TCGA / "HumanMethylation450.gz"
OUT = TCGA / "cohort_v3.tsv"

RANDOM_STATE = 2024
TEST_FRACTION = 0.20


def _gz_header_samples(path: Path) -> set[str]:
    """First-line sample ID set from a tab-gz expression matrix."""
    with gzip.open(path, "rt") as fh:
        header = fh.readline().rstrip("\n").split("\t")
    return set(header[1:])  # column 0 is gene/probe id


def main() -> int:
    for p in (CLINICAL, RNA_GZ, METH_GZ):
        if not p.exists():
            sys.stderr.write(f"ERROR: missing input {p}\n")
            return 1

    print(f"=== build_cohort_v3.py (Luminal-vs-Basal) ===")

    print(f"--- reading {CLINICAL.name} ---")
    clinical = pd.read_csv(CLINICAL, sep="\t", low_memory=False)
    # Some TCGA clinical exports use "sampleID" or "sample_id" -- this file
    # uses the former. Validate.
    if "sampleID" not in clinical.columns:
        sys.stderr.write(
            f"ERROR: 'sampleID' column not in {CLINICAL.name}.\n",
        )
        return 1
    if "PAM50Call_RNAseq" not in clinical.columns:
        sys.stderr.write(
            "ERROR: 'PAM50Call_RNAseq' column not in clinical.\n",
        )
        return 1
    cl = clinical[["sampleID", "PAM50Call_RNAseq"]].dropna(
        subset=["PAM50Call_RNAseq"],
    ).copy()
    cl = cl.rename(columns={"sampleID": "sample_id"})

    # Assign Luminal vs Basal.
    def _group(pam50: str) -> str | None:
        if pam50 in ("LumA", "LumB"):
            return "Luminal"
        if pam50 == "Basal":
            return "Basal"
        return None

    cl["group"] = cl["PAM50Call_RNAseq"].map(_group)
    cl = cl.dropna(subset=["group"])
    print(f"  {len(cl)} samples with PAM50 ∈ {{LumA, LumB, Basal}}")

    print(f"--- scanning {RNA_GZ.name} header for sample IDs ---")
    rna_samples = _gz_header_samples(RNA_GZ)
    print(f"  RNA samples: {len(rna_samples)}")

    print(f"--- scanning {METH_GZ.name} header for sample IDs ---")
    meth_samples = _gz_header_samples(METH_GZ)
    print(f"  meth samples: {len(meth_samples)}")

    cl["has_rna"] = cl["sample_id"].isin(rna_samples)
    cl["has_meth"] = cl["sample_id"].isin(meth_samples)
    dual = cl[cl["has_rna"] & cl["has_meth"]].copy()
    print(f"\n  dual-modality samples per group:")
    for grp, count in dual["group"].value_counts().items():
        print(f"    {grp:10s} {count}")

    # Stratified 80/20 split on group, same random_state as v0.2 cohort_v2.
    print(f"\n--- stratified 80/20 split (random_state={RANDOM_STATE}) ---")
    train_ids, test_ids = train_test_split(
        dual["sample_id"].to_numpy(),
        test_size=TEST_FRACTION,
        random_state=RANDOM_STATE,
        stratify=dual["group"].to_numpy(),
    )
    train_set = set(train_ids.tolist())
    test_set = set(test_ids.tolist())

    cl["split"] = ""
    cl.loc[cl["sample_id"].isin(train_set), "split"] = "train"
    cl.loc[cl["sample_id"].isin(test_set), "split"] = "test"

    print(f"  train n={len(train_set)} (per-group: "
          f"{dual[dual['sample_id'].isin(train_set)]['group'].value_counts().to_dict()})")
    print(f"  test  n={len(test_set)} (per-group: "
          f"{dual[dual['sample_id'].isin(test_set)]['group'].value_counts().to_dict()})")

    out_df = cl[["sample_id", "group", "has_rna", "has_meth", "split"]].copy()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT, sep="\t", index=False)
    print(f"\nWrote {OUT} ({len(out_df)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
