#!/usr/bin/env python3
"""v0.10: build METABRIC Luminal-vs-Basal cohort for cross-cohort + cross-task DMOI eval.

Parallel of `scripts/build_metabric_cohort.py` (v0.2, LumA-vs-LumB) and
`scripts/build_cohort_v3.py` (v0.9 TCGA Luminal-vs-Basal). Filters
METABRIC patients to those with CLAUDIN_SUBTYPE in {LumA, LumB, Basal}
AND mRNA microarray data available, then assigns Luminal (LumA + LumB)
or Basal labels.

The v0.9 architectural commitment was that the v0.6 framework
generalizes across classification axes when the pole-defining Hallmark
sets are swapped (Luminal pole = ER + ANDROGEN; Basal pole = EMT +
MYC_V1 + G2M). v0.10 tests whether that same framework generalizes
across cohorts too -- METABRIC HT-12 v3 microarray RNA-only with
methylation silenced -- using the same v0.9 priors.

Reads:
  data/metabric/clinical_patient.txt   (4 header lines + PATIENT_ID + CLAUDIN_SUBTYPE)
  data/metabric/mrna_microarray.txt    (header row only)

Writes:
  data/metabric/cohort_v3.tsv          (gitignored -- sample_id, group, has_rna)
  audit/metabric_cohort_v3_summary.md  (committed -- counts only, no PHI)

Expected sizes (from recon, before mRNA intersection):
  CLAUDIN_SUBTYPE  count
  --------------   -----
  LumA             700
  LumB             475
  Basal            209
  --> Luminal     1175  (LumA + LumB)
  --> Basal        209
  Total           1384
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
    """METABRIC clinical files have 4 commented header lines then a real
    header row. Skip them and read the rest."""
    return pd.read_csv(path, sep="\t", comment="#")


def _read_mrna_samples(path: Path) -> list[str]:
    """Pull sample IDs from the mRNA matrix's first row -- no full load."""
    with path.open() as fh:
        header = fh.readline().rstrip("\n").split("\t")
    return header[2:]  # cols 0,1 are Hugo_Symbol, Entrez_Gene_Id


def main() -> int:
    clin_path = METABRIC / "clinical_patient.txt"
    mrna_path = METABRIC / "mrna_microarray.txt"

    for p in (clin_path, mrna_path):
        if not p.exists():
            sys.stderr.write(
                f"ERROR: missing input {p}\n"
                "Run `python scripts/fetch_metabric.py` first.\n",
            )
            return 1

    print(f"Loading METABRIC clinical from {clin_path.name}...")
    clinical = _read_clinical(clin_path)
    print(f"  {len(clinical)} clinical rows")
    if "CLAUDIN_SUBTYPE" not in clinical.columns:
        sys.stderr.write("ERROR: CLAUDIN_SUBTYPE column missing.\n")
        return 1

    # Filter to Luminal (LumA + LumB) + Basal.
    relevant = clinical[
        clinical["CLAUDIN_SUBTYPE"].isin(["LumA", "LumB", "Basal"])
    ].copy()
    n_lumA = int((relevant["CLAUDIN_SUBTYPE"] == "LumA").sum())
    n_lumB = int((relevant["CLAUDIN_SUBTYPE"] == "LumB").sum())
    n_basal = int((relevant["CLAUDIN_SUBTYPE"] == "Basal").sum())
    print(f"  LumA  : {n_lumA}")
    print(f"  LumB  : {n_lumB}")
    print(f"  Basal : {n_basal}")
    print(f"  Total : {len(relevant)}")

    # Map LumA/LumB -> Luminal, Basal -> Basal.
    relevant["group"] = relevant["CLAUDIN_SUBTYPE"].map(
        {"LumA": "Luminal", "LumB": "Luminal", "Basal": "Basal"},
    )

    print(f"\nReading mRNA sample IDs from {mrna_path.name}...")
    mrna_samples = _read_mrna_samples(mrna_path)
    mrna_set = set(mrna_samples)
    print(f"  {len(mrna_samples)} mRNA samples in matrix")

    # Mark has_rna, count per group.
    relevant["has_rna"] = relevant["PATIENT_ID"].isin(mrna_set)
    n_with_rna = int(relevant["has_rna"].sum())
    n_luminal_rna = int(
        ((relevant["group"] == "Luminal") & relevant["has_rna"]).sum(),
    )
    n_basal_rna = int(
        ((relevant["group"] == "Basal") & relevant["has_rna"]).sum(),
    )
    print(f"  Patients with mRNA AND CLAUDIN_SUBTYPE in {{LumA, LumB, Basal}}: "
          f"{n_with_rna} (Luminal={n_luminal_rna}, Basal={n_basal_rna})")

    cohort = pd.DataFrame({
        "sample_id": relevant["PATIENT_ID"],
        "group": relevant["group"],
        "has_rna": relevant["has_rna"],
        "has_meth": False,  # METABRIC has no HM450
    })
    out = METABRIC / "cohort_v3.tsv"
    cohort.to_csv(out, sep="\t", index=False)
    print(f"\nWrote {out} ({len(cohort)} rows)")

    AUDIT.mkdir(exist_ok=True)
    summary_md = AUDIT / "metabric_cohort_v3_summary.md"
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    summary_md.write_text(
        "# METABRIC Cohort v3 Summary (DMOI v0.10 cross-cohort + cross-task eval)\n\n"
        f"Generated: {ts}\n\n"
        "## Source\n\n"
        "- Study: Curtis et al. *Nature* 2012 + Pereira et al. *Nat Commun* 2016\n"
        "- cBioPortal study ID: `brca_metabric`\n"
        "- Subset used: patients with CLAUDIN_SUBTYPE in "
        "{LumA, LumB, Basal} AND mRNA microarray (Illumina HT-12 v3) available\n"
        "- Label assignment: LumA + LumB -> Luminal; Basal -> Basal\n\n"
        "## What v0.10 uses METABRIC for\n\n"
        "Cross-cohort + cross-task external validation for the v0.9\n"
        "framework. v0.9 trained the same v0.6 architecture with\n"
        "Luminal/Basal pole priors on TCGA cohort_v3 and reached AUROC\n"
        "1.000 with 8/8 expected priors in per-pole IG top-5. v0.10 asks\n"
        "whether that finding holds when the trained model is scored on\n"
        "METABRIC (different microarray platform, different patient\n"
        "demographics) using the same RNA-only + meth-silenced + QN-to-TCGA\n"
        "protocol established in v0.2 / v0.4.\n\n"
        "## Cohort sizes\n\n"
        "| Subset | LumA | LumB | Basal | Luminal (LumA+LumB) | Total |\n"
        "|---|---|---|---|---|---|\n"
        f"| All PAM50/CLAUDIN_SUBTYPE-called | {n_lumA} | {n_lumB} | "
        f"{n_basal} | {n_lumA + n_lumB} | {n_lumA + n_lumB + n_basal} |\n"
        f"| With mRNA microarray (cohort_v3.tsv) | — | — | "
        f"{n_basal_rna} | {n_luminal_rna} | {n_with_rna} |\n\n"
        "## Note on excluded subtypes\n\n"
        "METABRIC distribution of CLAUDIN_SUBTYPE (all 1,980 called):\n"
        "- LumA 700 / LumB 475 / Basal 209 (this cohort)\n"
        "- claudin-low 218 / Her2 224 / Normal 148 / NC 6\n"
        "  (excluded -- out of scope for the v0.9 Luminal-vs-Basal target)\n\n"
        "## Reproduce\n\n"
        "```bash\n"
        "python scripts/fetch_metabric.py             # one-time, ~690 MB\n"
        "python scripts/build_metabric_cohort_v3.py\n"
        "```\n",
    )
    print(f"Wrote {summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
