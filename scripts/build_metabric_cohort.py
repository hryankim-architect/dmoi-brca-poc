#!/usr/bin/env python3
"""Build METABRIC LumA/LumB cohort for DMOI v0.2 external validation.

Filters METABRIC patients (n=1,980) to those with PAM50/CLAUDIN_SUBTYPE in
{LumA, LumB} AND mRNA microarray data available.

Reads:
  data/metabric/clinical_patient.txt   (4 header lines + PATIENT_ID + CLAUDIN_SUBTYPE)
  data/metabric/mrna_microarray.txt    (header row only — to confirm sample presence)

Writes:
  data/metabric/cohort.tsv             (gitignored — sample_id, group, has_rna)
  audit/metabric_cohort_summary.md     (committed — n counts only, no PHI)
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
    header row. Skip them and pull PATIENT_ID + CLAUDIN_SUBTYPE."""
    return pd.read_csv(path, sep="\t", comment="#")


def _read_mrna_samples(path: Path) -> list[str]:
    """Pull sample IDs from the mRNA matrix's first row — no full load."""
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
    print(f"  columns: {len(clinical.columns)}")
    if "CLAUDIN_SUBTYPE" not in clinical.columns:
        sys.stderr.write(
            "ERROR: CLAUDIN_SUBTYPE column missing from METABRIC clinical. "
            f"Found: {list(clinical.columns)[:10]}...\n",
        )
        return 1

    # Filter to LumA / LumB.
    luminal = clinical[clinical["CLAUDIN_SUBTYPE"].isin(["LumA", "LumB"])].copy()
    n_lumA = int((luminal["CLAUDIN_SUBTYPE"] == "LumA").sum())
    n_lumB = int((luminal["CLAUDIN_SUBTYPE"] == "LumB").sum())
    print(f"  LumA: {n_lumA}")
    print(f"  LumB: {n_lumB}")
    print(f"  LumA + LumB total: {len(luminal)}")

    print(f"\nReading mRNA sample IDs from {mrna_path.name}...")
    mrna_samples = _read_mrna_samples(mrna_path)
    mrna_set = set(mrna_samples)
    print(f"  {len(mrna_samples)} mRNA samples in matrix")

    # Mark has_rna and filter to those with both LumA/LumB call AND mRNA.
    luminal["has_rna"] = luminal["PATIENT_ID"].isin(mrna_set)
    n_with_rna = int(luminal["has_rna"].sum())
    n_lumA_rna = int(
        ((luminal["CLAUDIN_SUBTYPE"] == "LumA") & luminal["has_rna"]).sum(),
    )
    n_lumB_rna = int(
        ((luminal["CLAUDIN_SUBTYPE"] == "LumB") & luminal["has_rna"]).sum(),
    )
    print(f"  Patients with mRNA AND LumA/LumB call: {n_with_rna} "
          f"(LumA={n_lumA_rna}, LumB={n_lumB_rna})")

    # Cohort table for DMOI external eval.
    cohort = pd.DataFrame({
        "sample_id": luminal["PATIENT_ID"],
        "group": luminal["CLAUDIN_SUBTYPE"],
        "has_rna": luminal["has_rna"],
        "has_meth": False,  # METABRIC has no HM450
    })
    out = METABRIC / "cohort.tsv"
    cohort.to_csv(out, sep="\t", index=False)
    print(f"\nWrote {out} ({len(cohort)} rows)")

    AUDIT.mkdir(exist_ok=True)
    summary_md = AUDIT / "metabric_cohort_summary.md"
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    summary_md.write_text(
        "# METABRIC Cohort Summary (DMOI v0.2 external validation)\n\n"
        f"Generated: {ts}\n\n"
        "## Source\n\n"
        "- Study: Curtis et al. *Nature* 2012 + Pereira et al. *Nat Commun* 2016\n"
        "- cBioPortal study ID: `brca_metabric`\n"
        "- Subset used: patients with PAM50 / CLAUDIN_SUBTYPE in "
        "{LumA, LumB} AND mRNA microarray (Illumina HT-12 v3) available\n\n"
        "## What v0.2 uses METABRIC for\n\n"
        "External validation for DMOI's RNA branch (Path A'). METABRIC has\n"
        "no HM450 methylation, so the methylation pole encoder is silenced\n"
        "(`meth = zeros`) at inference time. This tests whether the\n"
        "hypothesis-conditioned RNA encoder generalizes across cohorts, but\n"
        "does NOT validate the dual-modality story.\n\n"
        "## Cohort sizes\n\n"
        "| Subset | LumA | LumB | Total |\n"
        "|---|---|---|---|\n"
        f"| All PAM50-called METABRIC patients | {n_lumA} | {n_lumB} | "
        f"{n_lumA + n_lumB} |\n"
        f"| With mRNA microarray (cohort.tsv) | {n_lumA_rna} | {n_lumB_rna} | "
        f"{n_with_rna} |\n\n"
        f"## Note on excluded subtypes\n\n"
        "METABRIC distribution of CLAUDIN_SUBTYPE (all 1,980 PAM50-called):\n"
        "- LumA 700 / LumB 475 (this cohort)\n"
        "- claudin-low 218 / Her2 224 / Basal 209 / Normal 148 / NC 6\n"
        "  (excluded — out of scope for DMOI's LumA-vs-LumB target)\n\n"
        "## Reproduce\n\n"
        "```bash\n"
        "python scripts/fetch_metabric.py        # ~690 MB one-time download\n"
        "python scripts/build_metabric_cohort.py\n"
        "```\n",
    )
    print(f"Wrote {summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
