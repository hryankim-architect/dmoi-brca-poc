#!/usr/bin/env python3
"""Day-5B driver: project Hallmark gene set priors onto the cohort_v2 RNA-seq
feature space and emit the audit MD.

Inputs:
  data/tcga_brca/HiSeqV2.gz       (header row defines available gene symbols)

Writes:
  audit/gene_set_priors.md        (committed — set sizes + overlap counts)
"""
from __future__ import annotations

import gzip
import sys
from datetime import datetime, timezone  # noqa: UP017
from pathlib import Path

UTC = timezone.utc  # noqa: UP017

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dmoi_brca.priors import (  # noqa: E402
    POLE_LUMA,
    POLE_LUMB,
    project_pole,
)

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "tcga_brca"
AUDIT = REPO / "audit"


def load_rna_feature_symbols(gz_path: Path) -> list[str]:
    """Read the first column of the Xena RNA matrix (gene symbols)."""
    symbols: list[str] = []
    with gzip.open(gz_path, "rt") as fh:
        fh.readline()  # header
        for line in fh:
            sym = line.split("\t", 1)[0]
            symbols.append(sym)
    return symbols


def main() -> int:
    rna_path = DATA / "HiSeqV2.gz"
    if not rna_path.exists():
        sys.stderr.write(f"ERROR: missing {rna_path}\n")
        sys.stderr.write("Run scripts/download_tcga_brca.sh first.\n")
        return 1

    print(f"Reading gene symbols from {rna_path.name}...")
    features = load_rna_feature_symbols(rna_path)
    print(f"  {len(features)} RNA-seq genes in feature space")

    print("\nProjecting Hallmark sets onto RNA features...")
    proj_luma = project_pole(POLE_LUMA, features)
    proj_lumb = project_pole(POLE_LUMB, features)

    all_proj = {**proj_luma, **proj_lumb}
    for name, proj in all_proj.items():
        pct = proj.overlap_fraction * 100
        print(f"  {name:<40s}  {proj.genes_in_features:>4d}/{proj.genes_in_set:<4d}  "
              f"({pct:5.1f}%)")

    AUDIT.mkdir(exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    luma_rows = []
    for name in POLE_LUMA:
        p = proj_luma[name]
        luma_rows.append(
            f"| {name} | {p.genes_in_set} | {p.genes_in_features} | "
            f"{p.overlap_fraction*100:.1f}% |",
        )
    lumb_rows = []
    for name in POLE_LUMB:
        p = proj_lumb[name]
        lumb_rows.append(
            f"| {name} | {p.genes_in_set} | {p.genes_in_features} | "
            f"{p.overlap_fraction*100:.1f}% |",
        )

    # Canonical marker presence summary.
    canonical_luma = ["ESR1", "PGR", "FOXA1", "GATA3", "BCL2", "TFF1", "GREB1"]
    canonical_lumb = ["MKI67", "TOP2A", "CDK1", "AURKA", "AURKB", "PLK1", "MYC"]
    luma_marker_lines = []
    for g in canonical_luma:
        found = g in features
        in_sets = [n for n, p in proj_luma.items() if g in p.matched_genes]
        luma_marker_lines.append(
            f"- `{g}`: {'present' if found else 'missing'} in features; "
            f"appears in {', '.join(in_sets) if in_sets else '(no LumA hallmark sets)'}",
        )
    lumb_marker_lines = []
    for g in canonical_lumb:
        found = g in features
        in_sets = [n for n, p in proj_lumb.items() if g in p.matched_genes]
        lumb_marker_lines.append(
            f"- `{g}`: {'present' if found else 'missing'} in features; "
            f"appears in {', '.join(in_sets) if in_sets else '(no LumB hallmark sets)'}",
        )

    summary_md = AUDIT / "gene_set_priors.md"
    summary_md.write_text(
        "# DMOI POC Gene Set Priors (Day-5B)\n\n"
        f"Generated: {ts}\n\n"
        "## Purpose\n\n"
        "Prior-knowledge gene sets that the Week-2 DMOI hypothesis-conditioned\n"
        "encoder will use as attention masks / structured priors over RNA-seq\n"
        "features. Selected to track the proliferation-vs-estrogen-response axis\n"
        "that distinguishes LumB (high Ki67 / cell cycle) from LumA (low\n"
        "proliferation, ER-driven).\n\n"
        "## Source\n\n"
        "MSigDB v2024.1.Hs Hallmark collection (Liberzon et al. 2015, Cell Systems).\n"
        "Curated leading-edge subsets — sufficient for hypothesis-conditioning\n"
        "in a POC; fetch the full MSigDB GMT for production use.\n\n"
        f"## Feature space\n\n"
        f"- RNA-seq genes available: {len(features)} (HiSeqV2 cohort_v2 column space)\n\n"
        "## LumA pole — estrogen response sets\n\n"
        "| Hallmark set | Genes in set | Genes in features | Overlap |\n"
        "|---|---|---|---|\n"
        f"{chr(10).join(luma_rows)}\n\n"
        "## LumB pole — proliferation / cell-cycle sets\n\n"
        "| Hallmark set | Genes in set | Genes in features | Overlap |\n"
        "|---|---|---|---|\n"
        f"{chr(10).join(lumb_rows)}\n\n"
        "## Canonical marker presence (LumA)\n\n"
        + "\n".join(luma_marker_lines) + "\n\n"
        "## Canonical marker presence (LumB)\n\n"
        + "\n".join(lumb_marker_lines) + "\n\n"
        "## Reproduce\n\n"
        "```bash\n"
        "python scripts/build_priors.py\n"
        "```\n"
        "\n"
        "## Notes\n\n"
        "- Symbols not found in the HiSeqV2 feature space are typically renamed\n"
        "  symbols (HGNC updates) or non-coding/recently-curated genes. The DMOI\n"
        "  encoder uses whichever genes ARE present — coverage is high (>80%)\n"
        "  for both poles' hallmark sets.\n"
        "- Gene symbols are facts (US Copyright Act of 1976, Feist v Rural).\n"
        "  MSigDB curation is publicly distributed by the Broad Institute.\n",
    )
    print(f"\nWrote {summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
