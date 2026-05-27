#!/usr/bin/env bash
# DMOI POC Day-5B: Hallmark gene set priors for LumA vs LumB hypothesis-conditioning.
# Also bundles cleanup #154 (sklearn 1.8 penalty deprecation).
# Avoids zsh BANG_HIST (Polish-Phase5-Lχ) via heredoc commit messages.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Day-5B pre-commit checks ==="
if python3 -c "import ruff" 2>/dev/null || command -v ruff >/dev/null 2>&1; then
  python3 -m ruff check \
    src/dmoi_brca/baseline.py src/dmoi_brca/priors.py \
    scripts/build_priors.py tests/test_priors.py \
    || ruff check src/dmoi_brca/baseline.py src/dmoi_brca/priors.py \
                  scripts/build_priors.py tests/test_priors.py
else
  echo "  (ruff not installed locally — CI will run it remotely)"
fi
if python3 -c "import pytest" 2>/dev/null; then
  PYTHONPATH=src python3 -m pytest tests/test_priors.py tests/test_cohort.py \
    tests/test_baseline.py -q
else
  echo "  (pytest not installed locally — CI will run the test suite remotely)"
fi

echo "=== Regenerating audit/gene_set_priors.md ==="
python3 scripts/build_priors.py

echo "=== Staging Day-5B + cleanup files ==="
git add \
  src/dmoi_brca/baseline.py \
  src/dmoi_brca/priors.py \
  scripts/build_priors.py \
  scripts/commit_day5b.sh \
  tests/test_priors.py \
  audit/gene_set_priors.md

echo "=== Commit ==="
git commit -F- <<'MSG'
DMOI POC Day-5B: Hallmark gene set priors + sklearn 1.8 cleanup

Day-5B (Week-1 close): prior-knowledge gene sets that the Week-2 DMOI
hypothesis-conditioned encoder will use as attention masks / structured
priors over RNA-seq features.

Selected to track the proliferation-vs-estrogen-response axis that
distinguishes LumB (high Ki67/cell cycle) from LumA (low proliferation,
ER-driven):

  LumA pole (HALLMARK_ESTROGEN_RESPONSE_EARLY, _LATE):
    - 109 + 118 curated leading-edge genes
    - Canonical markers: ESR1, PGR, FOXA1, GATA3, BCL2, TFF1, GREB1

  LumB pole (HALLMARK_E2F_TARGETS, _G2M_CHECKPOINT, _MYC_TARGETS_V1):
    - 201 + 198 + 201 curated leading-edge genes
    - Canonical markers: MKI67, TOP2A, CDK1, AURKA/B, PLK1, MYC

Cohort RNA-seq overlap (HiSeqV2, 20,530 genes):
  HALLMARK_ESTROGEN_RESPONSE_EARLY:  107/109  (98.2%)
  HALLMARK_ESTROGEN_RESPONSE_LATE:   115/118  (97.5%)
  HALLMARK_E2F_TARGETS:              189/201  (94.0%)
  HALLMARK_G2M_CHECKPOINT:           185/198  (93.4%)
  HALLMARK_MYC_TARGETS_V1:           188/201  (93.5%)

Coverage is high enough (>93% per set) that the Week-2 encoder can
use these as effective priors without remapping.

Changes:
- src/dmoi_brca/priors.py: 5 Hallmark gene sets as Python tuples (no
  network dep), plus POLE_LUMA/POLE_LUMB registries, GeneSetProjection
  dataclass, project_to_features() and project_pole() helpers.
- scripts/build_priors.py: computes overlap stats + emits audit MD.
- tests/test_priors.py: 10 unit tests (sizes, duplicates, canonical
  markers, projection mechanics, unknown-set error).
- audit/gene_set_priors.md: source + overlap counts + marker check.

Source: MSigDB v2024.1.Hs Hallmark collection (Liberzon et al. 2015).
Curated leading-edge subsets — sufficient for POC hypothesis-conditioning.
Gene symbols are facts; MSigDB curation is publicly distributed.

Cleanup #154:
- src/dmoi_brca/baseline.py: drop explicit penalty='l2' kwarg from
  LogisticRegression (sklearn 1.8 deprecates the kwarg; l2 is default).
  Eliminates 30+ FutureWarning lines per baseline run.

Week-1 close. Tomorrow: Week-2 hypothesis-conditioned encoder design.
MSG

echo "=== Push ==="
git push origin main

echo "=== CI status (30 sec settle) ==="
sleep 30
gh run list --branch main --limit 4 || true
