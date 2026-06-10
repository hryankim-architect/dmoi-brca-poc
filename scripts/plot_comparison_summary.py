#!/usr/bin/env python3
"""Synthesis figure + note for the DMOI-prior vs unsupervised-integration comparison.

Reads the two audit JSONs produced by `compare_mofa_mogcn.py` and
`ablate_hallmark_sets.py` and renders a two-panel summary bar chart plus a short
synthesis markdown. No new computation — pure presentation of already-audited results.

Run:  python scripts/plot_comparison_summary.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
AUDIT = REPO / "audit"
PNG = AUDIT / "dmoi_comparison_summary.png"


def _short(name: str) -> str:
    return (name.replace(" RNA+meth (100+100)", "")
            .replace("HALLMARK_", "").replace("only ", ""))


def main() -> int:
    cmp_path = AUDIT / "dmoi_vs_mofa_mogcn.json"
    abl_path = AUDIT / "dmoi_prior_ablation.json"
    if not (cmp_path.exists() and abl_path.exists()):
        print("Missing audit JSONs — run compare_mofa_mogcn.py and "
              "ablate_hallmark_sets.py first.", file=sys.stderr)
        return 1
    cmp = json.loads(cmp_path.read_text())
    abl = json.loads(abl_path.read_text())
    mofa = cmp["mofa_plus_reference_f1"]

    main_names = list(cmp["results"])
    main_f1 = [cmp["results"][n]["lr_weighted_f1"] for n in main_names]
    only = {k: v for k, v in abl["rows"].items() if k.startswith("only ")}
    abl_names = list(only)
    abl_f1 = [only[n]["lr_weighted_f1"] for n in abl_names]
    tv100 = abl["rows"]["top-variance(100)"]["lr_weighted_f1"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors1 = ["#2c7fb8" if "DMOI" in n else "#969696" for n in main_names]
    ax1.bar([_short(n) for n in main_names], main_f1, color=colors1)
    ax1.axhline(mofa, ls="--", color="#d95f02",
                label=f"MOFA+ literature ref {mofa:.2f}\n(3-omics, diff. cohort)")
    ax1.set_title("(a) Prior vs unsupervised baselines\nRNA+meth, 100/omics, 5-class PAM50")
    ax1.set_ylabel("LR weighted-F1 (5-fold)")
    ax1.set_ylim(0.70, 0.92)
    ax1.tick_params(axis="x", rotation=20)
    ax1.legend(fontsize=8, loc="lower right")

    ax2.bar([_short(n) for n in abl_names], abl_f1, color="#2c7fb8")
    ax2.axhline(tv100, ls="--", color="#969696", label=f"top-variance(100) {tv100:.2f}")
    ax2.set_title("(d) Single Hallmark set alone\nRNA-only, 5-class PAM50")
    ax2.set_ylabel("LR weighted-F1 (5-fold)")
    ax2.set_ylim(0.70, 0.92)
    ax2.tick_params(axis="x", rotation=30)
    ax2.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(PNG, dpi=130)
    extra = {}
    for key, fn in (("clinical", "dmoi_clinical_association.json"),
                    ("binary", "dmoi_binary_lumA_lumB.json")):
        p = AUDIT / fn
        if p.exists():
            extra[key] = json.loads(p.read_text())
    AUDIT.joinpath("dmoi_comparison_summary.md").write_text(_render(cmp, abl, extra))
    print(f"wrote {PNG.relative_to(REPO)} + audit/dmoi_comparison_summary.md")
    return 0


def _extra_section(extra: dict) -> str:
    if not extra:
        return ""
    lines = ["\n## Generalization & interpretability\n"]
    if "binary" in extra:
        b = extra["binary"]["results"]
        lines.append(
            "6. **Task generalization (binary LumA-vs-LumB, RNA-only):** the prior edge "
            f"holds and widens — 5-set AUROC **{b['DMOI-prior(5-set)']['auroc']:.3f}** vs "
            f"top-variance {b['top-variance']['auroc']:.3f} (50-set "
            f"{b['DMOI-prior(50-set)']['auroc']:.3f}); not a 5-class artifact.")
    if "clinical" in extra:
        c = extra["clinical"]["results"]
        lines.append(
            "7. **(1) Clinical coherence (OncoDB-style, non-circular):** fraction of "
            "selected genes associated with stage/node/age (independent of subtype): "
            f"5-set prior **{c['DMOI-prior(5-set)']['any_variable']:.2f}** vs top-variance "
            f"{c['top-variance']['any_variable']:.2f} vs 50-set "
            f"{c['DMOI-prior(50-set)']['any_variable']:.2f} — mirrors MOFA+>MoGCN (0.59>0.47).")
    return "\n".join(lines) + "\n"


def _render(cmp: dict, abl: dict, extra: dict | None = None) -> str:
    r = cmp["results"]
    j = cmp["rna_jaccard"]
    npg = cmp["n_prior_genes"]
    def f1(name: str) -> float:
        return r[name]["lr_weighted_f1"]
    five = "DMOI-prior(5-set) RNA+meth (100+100)"
    fifty = "DMOI-prior(50-set) RNA+meth (100+100)"
    tv = "top-variance RNA+meth (100+100)"
    return f"""# DMOI prior vs unsupervised integration — synthesis

![summary](dmoi_comparison_summary.png)

Public-data, label-free comparison on TCGA-BRCA PAM50 (n={cmp['n_common']}, RNA+meth).
Every selector is unsupervised (knowledge or variance); the downstream LR/SVC is the
only supervised step, so DMOI's biological prior is compared to MOFA+/MoGCN-style
selection on equal footing. See `dmoi_vs_mofa_mogcn.md` and `dmoi_prior_ablation.md`
for the full tables and caveats.

## What the experiments show

1. **Prior beats statistical selection at equal budget.** 5-set DMOI-prior LR
   weighted-F1 **{f1(five):.3f}** vs top-variance **{f1(tv):.3f}** (100 features/omics).
2. **(a) Specificity, not breadth.** The 5 curated proliferation/ER sets
   ({npg['5-set']} genes) beat the full 50-set catalog ({npg['50-set']} genes):
   {f1(five):.3f} vs **{f1(fifty):.3f}** — widening the prior dilutes the signal back
   toward the variance baseline.
3. **(b) The edge is biological, not variance re-discovery.** Selected RNA genes barely
   overlap top-variance (Jaccard {j['5-set_prior_vs_top-variance']:.3f}).
4. **(c) Microbiome 3rd omic deferred** — absent from the standard cBioPortal BRCA
   study and prior-free; out of scope (documented).
5. **(d) Proliferation axis is load-bearing.** Every single curated set alone beats
   top-variance(100); `G2M_CHECKPOINT` is the strongest single set and costs the most
   when dropped.
{_extra_section(extra or {})}
## Honest scope

The MOFA+ 0.75 dashed line is a *literature reference* (Omran et al. 2025,
doi:10.1186/s12967-025-06662-5; 3-omics incl. microbiome, non-identical cohort), not a
controlled head-to-head. The controlled comparison throughout is DMOI-prior vs
top-variance on identical data, same downstream model.
"""


if __name__ == "__main__":
    raise SystemExit(main())
