#!/usr/bin/env bash
# DMOI POC Day-4 — commit + push helper.
# Avoids zsh BANG_HIST (the lab-Lχ) by using heredoc commit messages.
# Tolerant of missing local ruff/pytest (CI is the source of truth).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Day-4 pre-commit checks ==="
if python3 -c "import ruff" 2>/dev/null || command -v ruff >/dev/null 2>&1; then
  python3 -m ruff check \
    src/dmoi_brca/features.py src/dmoi_brca/baseline.py \
    scripts/run_baseline.py tests/test_baseline.py \
    || ruff check src/dmoi_brca/features.py src/dmoi_brca/baseline.py \
                  scripts/run_baseline.py tests/test_baseline.py
else
  echo "  (ruff not installed locally — CI will run it remotely)"
fi
if python3 -c "import pytest" 2>/dev/null; then
  PYTHONPATH=src python3 -m pytest tests/test_baseline.py tests/test_cohort.py -q
else
  echo "  (pytest not installed locally — CI will run the test suite remotely)"
fi

echo "=== Staging Day-4 files ==="
git add \
  src/dmoi_brca/features.py \
  src/dmoi_brca/baseline.py \
  scripts/run_baseline.py \
  scripts/commit_day4.sh \
  tests/test_baseline.py \
  audit/baseline_results.md \
  audit/baseline_per_fold.tsv

echo "=== Commit ==="
git commit -F- <<'MSG'
DMOI POC Day-4: baseline saturation + Week-2 re-scope decision

- src/dmoi_brca/features.py: TCGA-BRCA feature loader.
  - RNA-seq (HiSeqV2): full pandas load, 20,530 genes.
  - Methylation (HM450): chunked streaming + top-K variance heap.
    Memory bounded to O(K * n_samples) regardless of total probes.
    Scans all 485,577 probes in ~40s on a Mac, retains top-10K most variable.
- src/dmoi_brca/baseline.py: sklearn LogisticRegression + RandomForest
  with StandardScaler + class_weight='balanced'. StratifiedKFold n=5.
- scripts/run_baseline.py: Day-4 driver. Detects saturation case
  (all AUC >= 0.99) and emits a different honest-scope section in
  the audit MD.
- tests/test_baseline.py: 4 unit tests on synthetic data (signal recovery).
- audit/baseline_results.md: saturation finding documented honestly.
- audit/baseline_per_fold.tsv: per-fold AUC + balanced accuracy.

Result: ALL 6 (feature_set, model) combos hit AUROC = 1.0 in 5-fold CV.
This is a NEGATIVE finding: H+ luminal vs H- basal is too easy a task
(PAM50 labels are RNA-derived; cell-of-origin states are biologically
distinct). Without baseline headroom, Week-2 hypothesis-conditioning
cannot demonstrate value on this target.

Honest next step: re-scope Week-2 to a harder discrimination target
(within-luminal LumA vs LumB, 5yr OS/PFS, neoadjuvant response, or
cross-modal methylation -> RNA-signature regression). See task #151.

The substrate, cohort selection, and feature pipeline are validated.
Only the discrimination target needs adjustment.

Cohort: 395 dual-modality patients (338 H+, 57 H-)
Features: rna=(395, 20530), meth=(395, 10000)
Runtime: ~41s end-to-end on sandbox (streaming + 5-fold CV x 6 combos)
MSG

echo "=== Push ==="
git push origin main

echo "=== CI status (30 sec settle) ==="
sleep 30
gh run list --branch main --limit 4 || true
