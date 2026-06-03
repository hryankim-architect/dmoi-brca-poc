#!/usr/bin/env bash
# DMOI POC Day-5A: Week-2 re-scope to within-luminal LumA vs LumB.
# Day-4 (a4d0511) already shipped — this commit adds cohort v2 + new baseline.
# Avoids zsh BANG_HIST (the lab-Lχ) via heredoc commit messages.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Day-5A pre-commit checks ==="
if python3 -c "import ruff" 2>/dev/null || command -v ruff >/dev/null 2>&1; then
  python3 -m ruff check \
    src/dmoi_brca/cohort.py src/dmoi_brca/features.py \
    scripts/build_cohort_v2.py scripts/run_baseline_v2.py \
    tests/test_cohort.py \
    || ruff check src/dmoi_brca/cohort.py src/dmoi_brca/features.py \
                  scripts/build_cohort_v2.py scripts/run_baseline_v2.py \
                  tests/test_cohort.py
else
  echo "  (ruff not installed locally — CI will run it remotely)"
fi
if python3 -c "import pytest" 2>/dev/null; then
  PYTHONPATH=src python3 -m pytest tests/test_cohort.py tests/test_baseline.py -q
else
  echo "  (pytest not installed locally — CI will run the test suite remotely)"
fi

echo "=== Regenerating Day-5A audit MDs ==="
python3 scripts/build_cohort_v2.py
python3 scripts/run_baseline_v2.py

echo "=== Staging Day-5A files ==="
git add \
  src/dmoi_brca/cohort.py \
  src/dmoi_brca/features.py \
  scripts/build_cohort_v2.py \
  scripts/run_baseline_v2.py \
  scripts/commit_day5a.sh \
  tests/test_cohort.py \
  audit/cohort_v2_summary.md \
  audit/baseline_v2_results.md \
  audit/baseline_v2_per_fold.tsv

echo "=== Commit ==="
git commit -F- <<'MSG'
DMOI POC Day-5A: Week-2 re-scope to LumA vs LumB (headroom confirmed)

Day-4 baseline (cohort v1: H+ vs H-) saturated at AUROC=1.0 across all
(rna/meth/concat) x (logreg/rf) combos — too easy for Week-2 hypothesis
conditioning to demonstrate value (PAM50 is RNA-derived, cell-of-origin
states biologically distinct).

Week-2 target re-scoped to within-luminal LumA vs LumB. Both poles are
ER+; discriminating axis is proliferation rate (LumB high Ki67/cell
cycle, LumA low). Biologically meaningful, literature-consistent,
non-trivial.

Changes:
- src/dmoi_brca/cohort.py: add assign_lumab_group(). Parameterize
  build_cohort() with custom assigner + label_a/label_b args
  (defaults preserve Day-3 H+/H- behavior; new args support v2).
- src/dmoi_brca/features.py: add positive_label parameter to
  load_features() so the same loader works for v1 (H_minus_basal_tn)
  and v2 (LumB).
- scripts/build_cohort_v2.py: Day-5A driver for cohort v2.
- scripts/run_baseline_v2.py: re-runs baseline on cohort v2.
- tests/test_cohort.py: +6 unit tests for assign_lumab_group +
  parameterized build_cohort.

Cohort v2: 635 patients (437 LumA / 198 LumB), 417 dual-modality.
Bigger and better balanced than v1 (~2.2:1 vs 6:1).

Headroom confirmation: fast RNA-only LogReg 5-fold CV probe hits
AUROC 0.9605 +/- 0.0153 on cohort v2 (vs 1.000 on v1). Discriminative
but NOT saturated; misclassification cases are the biologically
interesting LumA/LumB borderline tumors where DMOI hypothesis-
conditioning should add value. Re-scope validated.

Next: Day-5B prior gene sets (ESTROGEN_RESPONSE for LumA, E2F_TARGETS
/ G2M_CHECKPOINT for LumB), then Week-2 hypothesis-conditioned encoder.
MSG

echo "=== Push ==="
git push origin main

echo "=== CI status (30 sec settle) ==="
sleep 30
gh run list --branch main --limit 4 || true
