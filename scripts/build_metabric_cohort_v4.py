#!/usr/bin/env python3
"""v0.14: build METABRIC HER2-vs-Luminal cohort (cohort_v4.tsv) for external eval.

Clone of build_metabric_cohort.py extended to the HER2 axis. METABRIC has no
HM450 methylation, so this cohort is RNA-only (meth silenced at inference, the
v0.10 pattern).

HER2 in METABRIC is taken from CLAUDIN_SUBTYPE == "Her2" (n≈224). Note this is a
PAM50-style intrinsic-subtype call, a slight definitional difference from TCGA's
clinical HER2+ (HER2_Final_Status). The eval audit doc records this caveat.
Luminal = CLAUDIN_SUBTYPE ∈ {LumA, LumB} (n≈1,175).

Reads:
  data/metabric/clinical_patient.txt   (commented header + PATIENT_ID + CLAUDIN_SUBTYPE)
  data/metabric/mrna_microarray.txt    (header row only — confirm sample presence)
Writes:
  data/metabric/cohort_v4.tsv          (gitignored — sample_id, group, has_rna, has_meth)
  audit/metabric_cohort_v4_summary.md  (committed — n counts only, no PHI)
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone  # noqa: UP017
from pathlib import Path

import pandas as pd

UTC = timezone.utc  # noqa: UP017
REPO = Path(__file__).resolve().parents[1]
METABRIC = REPO / "data" / "metabric"
AUDIT = REPO / "audit"


def _read_clinical(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", comment="#")


def _read_mrna_samples(path: Path) -> list[str]:
    with path.open() as fh:
        header = fh.readline().rstrip("\n").split("\t")
    return header[2:]  # cols 0,1 are Hugo_Symbol, Entrez_Gene_Id


def main() -> int:
    clin_path = METABRIC / "clinical_patient.txt"
    mrna_path = METABRIC / "mrna_microarray.txt"
    for p in (clin_path, mrna_path):
        if not p.exists():
            sys.stderr.write(f"ERROR: missing input {p}\nRun fetch_metabric.py first.\n")
            return 1

    print("=== build_metabric_cohort_v4.py (HER2-vs-Luminal) ===")
    clinical = _read_clinical(clin_path)
    if "CLAUDIN_SUBTYPE" not in clinical.columns:
        sys.stderr.write(
            f"ERROR: CLAUDIN_SUBTYPE missing. Found: {list(clinical.columns)[:10]}...\n")
        return 1

    def _group(subtype: str) -> str | None:
        if subtype == "Her2":
            return "HER2"
        if subtype in ("LumA", "LumB"):
            return "Luminal"
        return None

    clinical["group"] = clinical["CLAUDIN_SUBTYPE"].map(_group)
    sub = clinical.dropna(subset=["group"]).copy()
    print(f"  HER2 (CLAUDIN Her2): {int((sub['group'] == 'HER2').sum())}")
    print(f"  Luminal (LumA/LumB): {int((sub['group'] == 'Luminal').sum())}")

    mrna_set = set(_read_mrna_samples(mrna_path))
    sub["has_rna"] = sub["PATIENT_ID"].isin(mrna_set)
    with_rna = sub[sub["has_rna"]]
    print(f"  with mRNA: HER2={int((with_rna['group'] == 'HER2').sum())}, "
          f"Luminal={int((with_rna['group'] == 'Luminal').sum())}")

    cohort = pd.DataFrame({
        "sample_id": sub["PATIENT_ID"],
        "group": sub["group"],
        "has_rna": sub["has_rna"],
        "has_meth": False,  # METABRIC has no HM450
    })
    out = METABRIC / "cohort_v4.tsv"
    cohort.to_csv(out, sep="\t", index=False)
    print(f"\nWrote {out} ({len(cohort)} rows)")

    AUDIT.mkdir(exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    n_her2 = int((with_rna["group"] == "HER2").sum())
    n_lum = int((with_rna["group"] == "Luminal").sum())
    (AUDIT / "metabric_cohort_v4_summary.md").write_text(
        "# METABRIC Cohort v4 Summary (DMOI v0.14 HER2 external validation)\n\n"
        f"Generated: {ts}\n\n"
        "- Source: Curtis 2012 + Pereira 2016 (`brca_metabric`)\n"
        "- HER2 = CLAUDIN_SUBTYPE == 'Her2' (PAM50-style; differs slightly from "
        "TCGA clinical HER2+ — see eval caveat)\n"
        "- Luminal = CLAUDIN_SUBTYPE in {LumA, LumB}\n"
        "- RNA-only (methylation pole silenced at inference, v0.10 pattern)\n\n"
        f"| Subset | with mRNA |\n|---|---|\n| HER2 | {n_her2} |\n| Luminal | {n_lum} |\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
