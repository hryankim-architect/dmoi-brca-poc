#!/usr/bin/env bash
# DMOI POC Day-3 — commit + push helper.
# Avoids zsh BANG_HIST (Polish-Phase5-Lχ) by using heredoc commit messages.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Day-3 pre-commit checks ==="
if python3 -c "import ruff" 2>/dev/null || command -v ruff >/dev/null 2>&1; then
  python3 -m ruff check src/dmoi_brca/cohort.py scripts/build_cohort.py tests/test_cohort.py \
    || ruff check src/dmoi_brca/cohort.py scripts/build_cohort.py tests/test_cohort.py
else
  echo "  (ruff not installed locally — CI will run it remotely)"
fi
if python3 -c "import pytest" 2>/dev/null; then
  PYTHONPATH=src python3 -m pytest tests/test_cohort.py -q
else
  echo "  (pytest not installed locally — CI will run the test suite remotely)"
fi

echo "=== Staging Day-3 files ==="
git add \
  src/dmoi_brca/cohort.py \
  scripts/build_cohort.py \
  scripts/commit_day3.sh \
  tests/test_cohort.py \
  audit/cohort_summary.md \
  data/manifest.yaml

echo "=== Commit ==="
git commit -F- <<'MSG'
DMOI POC Day-3: TCGA-BRCA cohort selection (H+ luminal vs H- basal/TN)

- src/dmoi_brca/cohort.py: PAM50 + ER/PR/HER2 -> {H_plus_luminal, H_minus_basal_tn}
  Primary PAM50 source: PAM50Call_RNAseq (short labels, ~956/1247 coverage).
  Fallback: PAM50_mRNA_nature2012 (long-form labels normalized to short form).
  Defensive empty-DataFrame check with informative error.
- scripts/build_cohort.py: reproducible Day-3 driver, writes cohort.tsv + audit MD.
- tests/test_cohort.py: 10 unit tests covering normalize, assign, build, empty case.
- audit/cohort_summary.md: Day-3 result snapshot (no PHI, n counts only).
- data/manifest.yaml: actual SHA256 checksums + observed sample counts +
  Day-3 cohort subset actuals (650 total / 547 H+ / 103 H- / 395 dual-modality).

Cohort split actuals:
  H+ (luminal):    547 patients (PAM50 LumA/LumB + ER positive)
  H- (basal/TN):   103 patients (PAM50 Basal + ER/PR/HER2 all negative)
  Both modalities: 395 patients (DMOI dual-modality training set)
MSG

echo "=== Push ==="
git push origin main

echo "=== CI status (3 sec settle) ==="
sleep 3
gh run list --branch main --limit 4 || true
