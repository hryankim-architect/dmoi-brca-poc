#!/usr/bin/env python3
"""Ablation: which of the 5 curated Hallmark sets drives the DMOI-prior advantage?

v0.15 found the 5 curated proliferation/ER sets beat both the full 50-set catalog and
the top-variance baseline. This isolates *which* sets carry that signal: each set alone,
and leave-one-out (the 5-set prior minus one), as label-free RNA feature selectors
through the same downstream protocol (LR + linear SVC, stratified 5-fold weighted-F1,
Calinski-Harabasz / Davies-Bouldin). RNA-only, so the variable is purely the gene set.

Run:  python scripts/ablate_hallmark_sets.py   (~10s; RNA only)
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dmoi_brca.compare_integration import (  # noqa: E402
    eval_selector,
    prior_rna_indices,
    topvar_indices,
    topvar_within,
)
from dmoi_brca.priors import HALLMARK_SETS  # noqa: E402

DATA = REPO / "data" / "tcga_brca"
PAM50 = ("LumA", "LumB", "Basal", "Her2", "Normal")


def _load() -> tuple[np.ndarray, list[str], np.ndarray]:
    clin = pd.read_csv(DATA / "BRCA_clinicalMatrix.tsv", sep="\t",
                       usecols=["sampleID", "PAM50Call_RNAseq"], low_memory=False).dropna()
    clin = clin[clin["PAM50Call_RNAseq"].isin(PAM50)]
    lab = dict(zip(clin["sampleID"].astype(str), clin["PAM50Call_RNAseq"], strict=False))
    with gzip.open(DATA / "HiSeqV2.gz", "rt") as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
    keep = [hdr[0]] + [s for s in hdr[1:] if s in lab]
    rna = pd.read_csv(DATA / "HiSeqV2.gz", sep="\t", usecols=keep, low_memory=False)
    rna = rna.set_index(hdr[0]).T
    genes = rna.columns.tolist()
    y = np.array([lab[s] for s in rna.index])
    return rna.to_numpy(np.float32), genes, y


def main() -> int:
    x, genes, y = _load()
    rows: dict[str, dict] = {}

    # Each curated set alone (all matched genes, label-free).
    for name in HALLMARK_SETS:
        idx = prior_rna_indices(genes, {name: HALLMARK_SETS[name]})
        rows[f"only {name}"] = eval_selector(x[:, idx], y)

    # Leave-one-out: 5-set prior minus one set, capped to 100 by variance.
    for drop in HALLMARK_SETS:
        rest = {k: v for k, v in HALLMARK_SETS.items() if k != drop}
        idx = topvar_within(x, prior_rna_indices(genes, rest), 100)
        rows[f"5-set minus {drop}"] = eval_selector(x[:, idx], y)

    # References: full 5-set (cap 100) and top-variance(100).
    rows["all 5 sets (cap100)"] = eval_selector(
        x[:, topvar_within(x, prior_rna_indices(genes, None), 100)], y)
    rows["top-variance(100)"] = eval_selector(x[:, topvar_indices(x, 100)], y)

    classes = {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True), strict=False)}
    out = {"n": int(len(y)), "classes": classes, "rows": rows}
    (REPO / "audit" / "dmoi_prior_ablation.json").write_text(json.dumps(out, indent=2))
    (REPO / "audit" / "dmoi_prior_ablation.md").write_text(_render(out))
    for name, r in rows.items():
        print(f"{name:34s} n={r['n_features']:4d} LR-wF1={r['lr_weighted_f1']:.3f} "
              f"SVC-wF1={r['svc_weighted_f1']:.3f} CHI={r['calinski_harabasz']:.1f}")
    print("wrote audit/dmoi_prior_ablation.md + .json")
    return 0


def _render(out: dict) -> str:
    rows = "\n".join(
        f"| {name} | {r['n_features']} | {r['lr_weighted_f1']:.3f} | {r['svc_weighted_f1']:.3f} "
        f"| {r['calinski_harabasz']:.1f} | {r['davies_bouldin']:.2f} |"
        for name, r in out["rows"].items()
    )
    return f"""# DMOI prior ablation — which Hallmark sets drive the advantage (RNA-only, PAM50)

n = {out['n']} TCGA-BRCA samples with a PAM50 call ({out['classes']}). RNA-only,
label-free selection; 5-class weighted-F1, stratified 5-fold. The variable is purely
*which Hallmark gene set(s)* define the prior — the downstream classifier is unchanged.

| selector (label-free, RNA) | n_feat | LR wF1 | SVC wF1 | CHI ↑ | DBI ↓ |
|---|---|---|---|---|---|
{rows}

## Reading

- **"only X" rows** show each curated set's standalone discriminative power; a high row
  means that set alone carries much of the PAM50 signal.
- **"5-set minus X" rows** show the cost of dropping one set from the curated prior; a
  large drop vs *all 5 sets* means X is load-bearing, a negligible drop means it is
  redundant with the others.
- Compared against the *top-variance(100)* baseline, this localizes the v0.15 prior
  advantage to specific proliferation/ER biology rather than the prior as a whole.
"""


if __name__ == "__main__":
    raise SystemExit(main())
