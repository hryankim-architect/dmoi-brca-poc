"""Regression guard: the committed calibration artifacts still back the documented ECE.

A full recompute needs the TCGA/METABRIC data + `scripts/eval_dmoi.py` (see the Makefile);
this guard is the cheap, always-runnable counterpart. It pins the two calibration figures
the README and `audit/dmoi_eval_v0.md` quote, by reading the committed eval artifacts:

  - the 5-fold ECE cluster from `audit/dmoi_eval_per_fold.tsv`
    (uncalibrated ~0.157 -> optimistic ~0.111 / nested-honest ~0.121), and
  - the held-out test headline from `audit/dmoi_eval_v0.md` (0.143 -> 0.079).

If a re-run of the eval shifts the calibration, those artifacts change and this test
fails — forcing the documented numbers to be updated rather than silently drifting. That
is exactly the class of bug fixed when the stale 0.138->0.077 headline was reconciled to
the reproducible held-out 0.143->0.079.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PER_FOLD_TSV = REPO / "audit" / "dmoi_eval_per_fold.tsv"
EVAL_MD = REPO / "audit" / "dmoi_eval_v0.md"


def _col_mean(rows: list[dict[str, str]], key: str) -> float:
    return sum(float(r[key]) for r in rows) / len(rows)


def test_per_fold_tsv_backs_the_5fold_ece() -> None:
    rows = list(csv.DictReader(PER_FOLD_TSV.open(encoding="utf-8"), delimiter="\t"))
    assert len(rows) == 5, f"expected 5 folds, got {len(rows)}"

    uncalibrated = _col_mean(rows, "ece_optA")        # before temperature scaling
    optimistic = _col_mean(rows, "ece_cal_optA")      # T fit on the eval fold (upper bound)
    nested = _col_mean(rows, "ece_cal_nested")        # honest: T fit on a held-out cal split

    assert round(uncalibrated, 3) == 0.157, f"uncalibrated 5-fold ECE drifted to {uncalibrated:.3f}"
    assert round(optimistic, 3) == 0.111, f"optimistic 5-fold ECE drifted to {optimistic:.3f}"
    assert round(nested, 3) == 0.121, f"nested 5-fold ECE drifted to {nested:.3f}"

    # The relationships the README narrates: scaling helps, and the honest (nested)
    # estimate is not better than the optimistic upper bound.
    assert nested < uncalibrated
    assert optimistic <= nested


def test_eval_md_backs_the_heldout_headline() -> None:
    text = EVAL_MD.read_text(encoding="utf-8")
    before = re.search(r"Test ECE before T-scaling[^\n0-9]*([0-9.]+)", text)
    after = re.search(r"Test ECE after T-scaling[^\n0-9]*([0-9.]+)", text)
    assert before and after, "held-out test ECE lines missing from dmoi_eval_v0.md"

    assert round(float(before.group(1)), 3) == 0.143, f"held-out ECE-before drifted to {before.group(1)}"
    assert round(float(after.group(1)), 3) == 0.079, f"held-out ECE-after drifted to {after.group(1)}"
    # The headline claim: temperature scaling roughly halves the held-out ECE.
    assert float(after.group(1)) < float(before.group(1)) * 0.7
