#!/usr/bin/env python3
"""Render v0.13 calibration-transfer figures from the audit TSVs.

Reads the two artifacts written by ``scripts/calibrate_transfer.py``:

- ``audit/dmoi_calibration_transfer_v0.13_reliability.tsv``  (per-condition bins)
- ``audit/dmoi_calibration_transfer_v0.13_learning_curve.tsv`` (D1 n x seed)

and writes two PNGs in the repo's ``audit/`` figure style (dpi=110):

- ``..._reliability.png``     reliability curves (confidence vs accuracy) for
  every condition, with per-condition ECE in the legend.
- ``..._learning_curve.png``  D1 METABRIC-mini ECE vs labelled-n (mean +/- std
  over seeds), with the raw (A), naive-TCGA-T (B), and oracle (C) ECE drawn as
  reference lines.

Per-condition ECE is recomputed from the reliability bins
(ECE = sum_b count_b/N * |conf_b - acc_b|), so this script needs only the two
TSVs — no model, no torch.

Reproduce:  python scripts/plot_calibration_transfer.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render to file, never to a display
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
AUDIT = REPO / "audit"
REL_TSV = AUDIT / "dmoi_calibration_transfer_v0.13_reliability.tsv"
LC_TSV = AUDIT / "dmoi_calibration_transfer_v0.13_learning_curve.tsv"

# Stable plotting order + human labels for the conditions.
COND_LABELS = {
    "A_uncalibrated": "A: uncalibrated (raw)",
    "B_TCGA_T": "B: TCGA T (naive)",
    "C_METABRIC_oracle_T": "C: METABRIC oracle T",
    "D2_labelfree_align_TCGA_T": "D2: label-free align + TCGA T",
    "D3_prior_odds": "D3: prior-odds",
}


def _ece_from_bins(df_cond: pd.DataFrame) -> float:
    """ECE = sum_b (count_b / N) * |confidence_b - accuracy_b| over non-empty bins."""
    nz = df_cond[df_cond["bin_count"] > 0]
    total = nz["bin_count"].sum()
    if total == 0:
        return float("nan")
    return float((nz["bin_count"] / total * (nz["bin_confidence"] - nz["bin_accuracy"]).abs()).sum())


def plot_reliability(rel: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect calibration")
    for cond in COND_LABELS:
        sub = rel[rel["condition"] == cond]
        if sub.empty:
            continue
        pts = sub[sub["bin_count"] > 0].sort_values("bin_confidence")
        ece = _ece_from_bins(sub)
        ax.plot(
            pts["bin_confidence"], pts["bin_accuracy"],
            marker="o", markersize=4, linewidth=1.5,
            label=f"{COND_LABELS[cond]}  (ECE={ece:.3f})",
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("mean predicted probability (confidence)")
    ax.set_ylabel("observed positive rate (accuracy)")
    ax.set_title("v0.13 cross-cohort reliability (METABRIC eval slice)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_learning_curve(lc: pd.DataFrame, rel: pd.DataFrame, out_path: Path) -> None:
    grp = lc.groupby("n")["ece"].agg(["mean", "std"]).reset_index().sort_values("n")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(
        grp["n"], grp["mean"], yerr=grp["std"].fillna(0.0),
        marker="o", capsize=4, linewidth=1.8, label="D1 METABRIC-mini T (mean +/- std over seeds)",
    )
    refs = [
        ("A_uncalibrated", "raw / no calibration (A)", "tab:gray", "-"),
        ("B_TCGA_T", "naive TCGA T (B)", "tab:red", "--"),
        ("C_METABRIC_oracle_T", "labelled oracle T (C)", "tab:green", ":"),
    ]
    for cond, label, color, style in refs:
        sub = rel[rel["condition"] == cond]
        if sub.empty:
            continue
        ax.axhline(_ece_from_bins(sub), color=color, linestyle=style, linewidth=1.5, label=label)
    ax.set_xlabel("labelled METABRIC samples used to fit T (n)")
    ax.set_ylabel("ECE on fixed eval slice")
    ax.set_title("v0.13 D1 learning curve: how little target signal recovers calibration")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> int:
    for p in (REL_TSV, LC_TSV):
        if not p.exists():
            raise SystemExit(f"ERROR: missing {p}; run scripts/calibrate_transfer.py first.")
    rel = pd.read_csv(REL_TSV, sep="\t")
    lc = pd.read_csv(LC_TSV, sep="\t")
    plot_reliability(rel, AUDIT / "dmoi_calibration_transfer_v0.13_reliability.png")
    plot_learning_curve(lc, rel, AUDIT / "dmoi_calibration_transfer_v0.13_learning_curve.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
