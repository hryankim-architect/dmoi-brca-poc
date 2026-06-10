#!/usr/bin/env python3
"""Task generalization: does the prior advantage hold on the binary LumA-vs-LumB pole?

The v0.15 comparison was 5-class PAM50. DMOI's native task is the binary luminal pole
(LumA vs LumB). This re-runs the same label-free selectors (5-set / 50-set prior vs
top-variance, RNA-only) on that binary task and reports AUROC (DMOI's headline metric)
alongside weighted-F1 — to check the prior edge is not an artifact of the 5-class setup.

Run:  python scripts/compare_binary_task.py   (~10s; RNA only)
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
from dmoi_brca.hallmark import load_hallmark_gmt  # noqa: E402

DATA = REPO / "data" / "tcga_brca"


def _load_binary():
    clin = pd.read_csv(DATA / "BRCA_clinicalMatrix.tsv", sep="\t",
                       usecols=["sampleID", "PAM50Call_RNAseq"], low_memory=False).dropna()
    clin = clin[clin["PAM50Call_RNAseq"].isin(["LumA", "LumB"])]
    lab = dict(zip(clin["sampleID"].astype(str), clin["PAM50Call_RNAseq"], strict=False))
    with gzip.open(DATA / "HiSeqV2.gz", "rt") as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
    keep = [hdr[0]] + [s for s in hdr[1:] if s in lab]
    rna = pd.read_csv(DATA / "HiSeqV2.gz", sep="\t", usecols=keep, low_memory=False)
    rna = rna.set_index(hdr[0]).T
    genes = rna.columns.tolist()
    y = np.array([1 if lab[s] == "LumB" else 0 for s in rna.index])  # LumB = positive
    return rna.to_numpy(np.float32), genes, y


def _auroc(x: np.ndarray, y: np.ndarray) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    lr = make_pipeline(StandardScaler(),
                       LogisticRegression(max_iter=2000, class_weight="balanced"))
    return float(cross_val_score(lr, x, y, cv=cv, scoring="roc_auc").mean())


def main() -> int:
    x, genes, y = _load_binary()
    full_sets = load_hallmark_gmt(str(REPO / "data" / "msigdb" / "h.all.v2024.1.Hs.symbols.gmt"))
    selectors = {
        "DMOI-prior(5-set)": topvar_within(x, prior_rna_indices(genes, None), 100),
        "DMOI-prior(50-set)": topvar_within(x, prior_rna_indices(genes, full_sets), 100),
        "top-variance": topvar_indices(x, 100),
    }
    results = {}
    for name, idx in selectors.items():
        r = eval_selector(x[:, idx], y)
        r["auroc"] = _auroc(x[:, idx], y)
        results[name] = r
    out = {"n": int(len(y)), "n_lumB": int(y.sum()), "n_lumA": int((y == 0).sum()),
           "results": results}
    (REPO / "audit" / "dmoi_binary_lumA_lumB.json").write_text(json.dumps(out, indent=2))
    (REPO / "audit" / "dmoi_binary_lumA_lumB.md").write_text(_render(out))
    for name, r in results.items():
        print(f"{name:20s} AUROC={r['auroc']:.3f} wF1={r['lr_weighted_f1']:.3f} "
              f"CHI={r['calinski_harabasz']:.1f}")
    print("wrote audit/dmoi_binary_lumA_lumB.md + .json")
    return 0


def _render(out: dict) -> str:
    rows = "\n".join(
        f"| {name} | {r['auroc']:.3f} | {r['lr_weighted_f1']:.3f} | {r['calinski_harabasz']:.1f} |"
        for name, r in out["results"].items()
    )
    return f"""# Task generalization — binary LumA vs LumB (RNA-only)

n = {out['n']} luminal samples (LumA {out['n_lumA']}, LumB {out['n_lumB']}). Same
label-free RNA selectors as the 5-class comparison; LogisticRegression, stratified
5-fold. AUROC is DMOI's headline metric for this pole task.

| selector (label-free, RNA) | AUROC | LR wF1 | CHI ↑ |
|---|---|---|---|
{rows}

## Reading

- If the biological prior keeps its edge over top-variance here, the v0.15 finding
  generalizes from the 5-class task to DMOI's native binary pole task — it is not a
  5-class artifact.
- Note: these are *label-free selectors + a plain LR*, not DMOI's full supervised,
  prior-conditioned model (which reports AUROC ~0.97 on LumA-vs-LumB). This isolates the
  feature-selection contribution, consistent with the rest of this comparison.
"""


if __name__ == "__main__":
    raise SystemExit(main())
