#!/usr/bin/env python3
"""v0.14: build TCGA-BRCA HER2-vs-Luminal dual-modality cohort (cohort_v4.tsv).

Third classification axis for the task-reusability claim (after LumA-vs-LumB in
v0.6 and Luminal-vs-Basal in v0.9). The architecture, attention layer, fusion,
and training loop are all unchanged from v0.6 — only the cohort, the
pole-defining Hallmark priors (POLE_HER2 / POLE_LUMINAL_ER), and the positive
label change.

HER2 definition: clinical HER2+ (`HER2_Final_Status_nature2012 == "Positive"`),
chosen over the PAM50∩clinical intersection because it yields a usable
dual-modality n (~58 vs 17). Luminal = PAM50 ∈ {LumA, LumB} that are NOT
clinically HER2+ (kept disjoint from the HER2 class).

Output: `data/tcga_brca/cohort_v4.tsv` with columns
    sample_id  group  has_rna  has_meth  split
where group ∈ {"HER2", "Luminal"}, has_* mark modality presence, and split ∈
{"train", "test"} is a stratified 80/20 on dual-modality samples
(random_state=2024, the cohort_v2/v3 protocol). Single-modality rows: split="".

Scope: HER2+ is the small class (~58 dual-modality). v0.14 is a
reusability demonstration, not a powered effect-size result — eval reports a
5-fold band and leans on the METABRIC external (n≈224) for statistical weight.
"""
from __future__ import annotations

import gzip
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[1]
TCGA = REPO / "data" / "tcga_brca"
CLINICAL = TCGA / "BRCA_clinicalMatrix.tsv"
RNA_GZ = TCGA / "HiSeqV2.gz"
METH_GZ = TCGA / "HumanMethylation450.gz"
OUT = TCGA / "cohort_v4.tsv"

RANDOM_STATE = 2024
TEST_FRACTION = 0.20


def _gz_header_samples(path: Path) -> set[str]:
    with gzip.open(path, "rt") as fh:
        header = fh.readline().rstrip("\n").split("\t")
    return set(header[1:])


def main() -> int:
    for p in (CLINICAL, RNA_GZ, METH_GZ):
        if not p.exists():
            sys.stderr.write(f"ERROR: missing input {p}\n")
            return 1

    print("=== build_cohort_v4.py (HER2-vs-Luminal) ===")
    clinical = pd.read_csv(CLINICAL, sep="\t", low_memory=False)
    for col in ("sampleID", "PAM50Call_RNAseq", "HER2_Final_Status_nature2012"):
        if col not in clinical.columns:
            sys.stderr.write(f"ERROR: '{col}' column not in clinical.\n")
            return 1
    cl = clinical[
        ["sampleID", "PAM50Call_RNAseq", "HER2_Final_Status_nature2012"]
    ].rename(columns={"sampleID": "sample_id"}).copy()

    def _group(row) -> str | None:
        if row["HER2_Final_Status_nature2012"] == "Positive":
            return "HER2"
        if row["PAM50Call_RNAseq"] in ("LumA", "LumB"):
            return "Luminal"  # clinically non-positive luminal -> disjoint
        return None

    cl["group"] = cl.apply(_group, axis=1)
    cl = cl.dropna(subset=["group"])
    print(f"  HER2 (clinical+): {int((cl['group'] == 'HER2').sum())}")
    print(f"  Luminal (PAM50 LumA/LumB, non-HER2+): {int((cl['group'] == 'Luminal').sum())}")

    rna_samples = _gz_header_samples(RNA_GZ)
    meth_samples = _gz_header_samples(METH_GZ)
    cl["has_rna"] = cl["sample_id"].isin(rna_samples)
    cl["has_meth"] = cl["sample_id"].isin(meth_samples)
    dual = cl[cl["has_rna"] & cl["has_meth"]].copy()
    print("\n  dual-modality samples per group:")
    for grp, count in dual["group"].value_counts().items():
        print(f"    {grp:10s} {count}")

    print(f"\n--- stratified 80/20 split (random_state={RANDOM_STATE}) ---")
    train_ids, test_ids = train_test_split(
        dual["sample_id"].to_numpy(),
        test_size=TEST_FRACTION,
        random_state=RANDOM_STATE,
        stratify=dual["group"].to_numpy(),
    )
    train_set, test_set = set(train_ids.tolist()), set(test_ids.tolist())
    cl["split"] = ""
    cl.loc[cl["sample_id"].isin(train_set), "split"] = "train"
    cl.loc[cl["sample_id"].isin(test_set), "split"] = "test"
    print(f"  train n={len(train_set)} "
          f"({dual[dual['sample_id'].isin(train_set)]['group'].value_counts().to_dict()})")
    print(f"  test  n={len(test_set)} "
          f"({dual[dual['sample_id'].isin(test_set)]['group'].value_counts().to_dict()})")

    out_df = cl[["sample_id", "group", "has_rna", "has_meth", "split"]].copy()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT, sep="\t", index=False)
    print(f"\nWrote {OUT} ({len(out_df)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
