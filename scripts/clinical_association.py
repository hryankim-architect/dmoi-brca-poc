#!/usr/bin/env python3
"""Interpretability (OncoDB-style): are prior-selected genes clinically meaningful?

Extends the v0.15/(b) interpretability story. For each label-free RNA selector we test
whether the selected genes' expression is associated with clinical variables that are
INDEPENDENT of the PAM50 target — AJCC tumor stage, lymph-node status, and patient age
— then report the fraction of genes significant after BH-FDR (q < 0.05), the criterion
Omran et al. 2025 used via OncoDB. (Hallmark over-representation would be circular here,
since the prior is *defined* from Hallmark sets; clinical association is not.)

Run:  python scripts/clinical_association.py   (~10s; RNA only)
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dmoi_brca.compare_integration import (  # noqa: E402
    bh_fdr,
    prior_rna_indices,
    topvar_indices,
    topvar_within,
)
from dmoi_brca.hallmark import load_hallmark_gmt  # noqa: E402

DATA = REPO / "data" / "tcga_brca"
PAM50 = ("LumA", "LumB", "Basal", "Her2", "Normal")
STAGE = "AJCC_Stage_nature2012"
NODE = "Node_Coded_nature2012"
AGE = "Age_at_Initial_Pathologic_Diagnosis_nature2012"
ALPHA = 0.05


def _load():
    cols = ["sampleID", "PAM50Call_RNAseq", STAGE, NODE, AGE]
    clin = pd.read_csv(DATA / "BRCA_clinicalMatrix.tsv", sep="\t", usecols=cols, low_memory=False)
    clin = clin[clin["PAM50Call_RNAseq"].isin(PAM50)].set_index("sampleID")
    with gzip.open(DATA / "HiSeqV2.gz", "rt") as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
    keep = [hdr[0]] + [s for s in hdr[1:] if s in clin.index]
    rna = pd.read_csv(DATA / "HiSeqV2.gz", sep="\t", usecols=keep, low_memory=False)
    rna = rna.set_index(hdr[0]).T
    clin = clin.loc[rna.index]
    return rna.to_numpy(np.float32), rna.columns.tolist(), clin


def _pvals(x_genes: np.ndarray, clin: pd.DataFrame) -> dict[str, np.ndarray]:
    """Per-gene p-value vs each clinical variable (NaN where a test is not computable)."""
    stage = clin[STAGE].astype("string")
    node = clin[NODE].astype("string")
    age = pd.to_numeric(clin[AGE], errors="coerce").to_numpy()
    out = {"stage": [], "node": [], "age": []}
    for j in range(x_genes.shape[1]):
        col = x_genes[:, j]
        out["stage"].append(_kruskal_p(col, stage))
        out["node"].append(_kruskal_p(col, node))
        m = ~np.isnan(age)
        out["age"].append(spearmanr(col[m], age[m]).pvalue if m.sum() > 10 else np.nan)
    return {k: np.array(v, dtype=float) for k, v in out.items()}


def _kruskal_p(values: np.ndarray, groups: pd.Series) -> float:
    g = groups.to_numpy()
    keep = pd.notna(g)
    vals, lab = values[keep], g[keep]
    samples = [vals[lab == lv] for lv in np.unique(lab)]
    samples = [s for s in samples if len(s) >= 5]
    if len(samples) < 2:
        return np.nan
    try:
        return float(kruskal(*samples).pvalue)
    except ValueError:
        return np.nan


def _assoc_fraction(pvals: dict[str, np.ndarray]) -> dict:
    """Fraction of genes significant (BH q<0.05) per variable and in ANY variable."""
    n = len(next(iter(pvals.values())))
    sig_any = np.zeros(n, dtype=bool)
    per_var = {}
    for var, p in pvals.items():
        valid = ~np.isnan(p)
        q = np.full(n, np.nan)
        if valid.any():
            q[valid] = bh_fdr(p[valid])
        s = (q < ALPHA) & valid
        per_var[var] = round(float(s.mean()), 3)
        sig_any |= s
    return {"per_variable": per_var, "any_variable": round(float(sig_any.mean()), 3)}


def main() -> int:
    x, genes, clin = _load()
    full_sets = load_hallmark_gmt(str(REPO / "data" / "msigdb" / "h.all.v2024.1.Hs.symbols.gmt"))
    selectors = {
        "DMOI-prior(5-set)": topvar_within(x, prior_rna_indices(genes, None), 100),
        "DMOI-prior(50-set)": topvar_within(x, prior_rna_indices(genes, full_sets), 100),
        "top-variance": topvar_indices(x, 100),
    }
    results = {name: _assoc_fraction(_pvals(x[:, idx], clin)) for name, idx in selectors.items()}
    out = {"n": int(x.shape[0]), "alpha": ALPHA, "results": results}
    (REPO / "audit" / "dmoi_clinical_association.json").write_text(json.dumps(out, indent=2))
    (REPO / "audit" / "dmoi_clinical_association.md").write_text(_render(out))
    for name, r in results.items():
        print(f"{name:20s} any={r['any_variable']:.3f}  {r['per_variable']}")
    print("wrote audit/dmoi_clinical_association.md + .json")
    return 0


def _render(out: dict) -> str:
    rows = "\n".join(
        f"| {name} | {r['any_variable']:.3f} | {r['per_variable']['stage']:.3f} "
        f"| {r['per_variable']['node']:.3f} | {r['per_variable']['age']:.3f} |"
        for name, r in out["results"].items()
    )
    return f"""# Clinical association of selected genes (interpretability, OncoDB-style)

n = {out['n']} TCGA-BRCA samples. For each label-free RNA selector (100 genes), the
fraction whose expression is significantly associated (Kruskal-Wallis for categorical
stage / node, Spearman for age; BH-FDR q < {out['alpha']}) with clinical variables that
are **independent of the PAM50 target**. Higher = the selected feature set is more
clinically meaningful, not just predictive of subtype.

| selector (label-free, RNA) | any variable | stage | node | age |
|---|---|---|---|---|
{rows}

## Reading

- A higher *any-variable* fraction for the biological prior than for top-variance means
  the prior selects genes that carry clinical signal beyond subtype separation — the
  same kind of evidence Omran et al. 2025 reported via OncoDB (MOFA+ 59% vs MoGCN 47%).
- Stage/node/age are deliberately not the classification target, so this is a
  non-circular biological-coherence check (unlike Hallmark over-representation, which
  the prior would satisfy by construction).
"""


if __name__ == "__main__":
    raise SystemExit(main())
