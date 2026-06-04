"""Tests for v0.13 cross-cohort calibration transfer.

Covers the deterministic, label-free pieces that the transfer story rests on:

- ``eval.brier_score`` matches sklearn and has the right extremes.
- ``eval.reliability_table`` bins are well-formed and exhaustive.
- ``transfer.affine_align`` / ``transfer.prior_odds_correct`` are deterministic
  and **monotonic** (so AUROC is invariant — the core claim of the audit doc).
- The v0.2 finding reproduces on a fixture: importing TCGA's T (< 1) onto an
  already-calibrated cohort *worsens* ECE.

Temperature *fitting* needs torch, so that assertion is guarded with
``importorskip`` — it runs in CI (torch present) and skips in a torch-free
sandbox. Everything else runs everywhere.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import brier_score_loss, roc_auc_score

from dmoi_brca.eval import brier_score, compute_calibration, reliability_table
from dmoi_brca.transfer import affine_align, prior_odds_correct


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    z = np.exp(-np.abs(x))
    return np.where(x >= 0, 1.0 / (1.0 + z), z / (1.0 + z))


def _calibrated_fixture(n: int = 4000, seed: int = 0):
    """Logits whose sigmoid is the true positive probability => calibrated at T=1."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(0.0, 2.0, size=n)
    p_true = _sigmoid(logits)
    labels = (rng.random(n) < p_true).astype(np.int64)
    return logits, labels


# --------------------------------------------------------------------------- #
# eval.brier_score
# --------------------------------------------------------------------------- #
def test_brier_perfect_and_worst():
    labels = np.array([0, 1, 0, 1])
    assert brier_score(labels, labels.astype(float)) == 0.0
    assert brier_score(labels, 1.0 - labels) == 1.0


def test_brier_matches_sklearn():
    logits, labels = _calibrated_fixture()
    proba = _sigmoid(logits)
    assert np.isclose(brier_score(labels, proba), brier_score_loss(labels, proba))


def test_brier_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape"):
        brier_score(np.array([0, 1]), np.array([0.5]))


# --------------------------------------------------------------------------- #
# eval.reliability_table
# --------------------------------------------------------------------------- #
def test_reliability_table_is_exhaustive_and_bounded():
    logits, labels = _calibrated_fixture()
    proba = _sigmoid(logits)
    table = reliability_table(labels, proba, n_bins=10)
    assert len(table) == 10
    assert sum(b.count for b in table) == len(labels)
    for b in table:
        assert 0.0 <= b.confidence <= 1.0
        assert 0.0 <= b.accuracy <= 1.0
        assert b.count >= 0


# --------------------------------------------------------------------------- #
# transfer.affine_align
# --------------------------------------------------------------------------- #
def test_affine_align_deterministic():
    logits, _ = _calibrated_fixture()
    a = affine_align(logits, src_mean=0.1, src_std=2.0, dst_mean=-0.3, dst_std=1.5)
    b = affine_align(logits, src_mean=0.1, src_std=2.0, dst_mean=-0.3, dst_std=1.5)
    assert np.array_equal(a, b)


def test_affine_align_matches_target_moments():
    logits, _ = _calibrated_fixture()
    aligned = affine_align(
        logits,
        src_mean=float(logits.mean()),
        src_std=float(logits.std()),
        dst_mean=3.0,
        dst_std=0.5,
    )
    assert np.isclose(aligned.mean(), 3.0, atol=1e-6)
    assert np.isclose(aligned.std(), 0.5, atol=1e-6)


def test_affine_align_is_monotonic_auroc_invariant():
    logits, labels = _calibrated_fixture()
    aligned = affine_align(logits, src_mean=0.0, src_std=2.0, dst_mean=1.0, dst_std=0.7)
    assert np.isclose(
        roc_auc_score(labels, _sigmoid(logits)),
        roc_auc_score(labels, _sigmoid(aligned)),
    )


def test_affine_align_rejects_nonpositive_src_std():
    with pytest.raises(ValueError, match="src_std"):
        affine_align(np.array([1.0]), src_mean=0.0, src_std=0.0, dst_mean=0.0, dst_std=1.0)


# --------------------------------------------------------------------------- #
# transfer.prior_odds_correct
# --------------------------------------------------------------------------- #
def test_prior_odds_identity_when_priors_equal():
    proba = np.array([0.1, 0.5, 0.9])
    out = prior_odds_correct(proba, pi_train=0.4, pi_target=0.4)
    assert np.allclose(out, proba)


def test_prior_odds_deterministic_and_monotonic():
    logits, labels = _calibrated_fixture()
    proba = _sigmoid(logits)
    a = prior_odds_correct(proba, pi_train=0.3, pi_target=0.5)
    b = prior_odds_correct(proba, pi_train=0.3, pi_target=0.5)
    assert np.array_equal(a, b)
    # monotonic => ranking preserved => AUROC unchanged
    assert np.isclose(roc_auc_score(labels, proba), roc_auc_score(labels, a))


def test_prior_odds_raises_on_degenerate_prior():
    with pytest.raises(ValueError, match="pi_target"):
        prior_odds_correct(np.array([0.5]), pi_train=0.4, pi_target=1.0)


# --------------------------------------------------------------------------- #
# The v0.2 finding: TCGA T (< 1) over-sharpens an already-calibrated cohort
# --------------------------------------------------------------------------- #
def test_importing_low_T_worsens_ece_on_calibrated_cohort():
    logits, labels = _calibrated_fixture()
    ece_raw = compute_calibration(labels, _sigmoid(logits), n_bins=10).ece
    ece_tcga_T = compute_calibration(labels, _sigmoid(logits / 0.634), n_bins=10).ece
    # raw is calibrated by construction; importing T=0.634 sharpens => worse ECE
    assert ece_tcga_T > ece_raw


def test_temperature_is_auroc_invariant():
    logits, labels = _calibrated_fixture()
    base = roc_auc_score(labels, _sigmoid(logits))
    for T in (0.5, 0.634, 0.934, 2.0):
        assert np.isclose(base, roc_auc_score(labels, _sigmoid(logits / T)))


# --------------------------------------------------------------------------- #
# torch-only: the fitter recovers T ~ 1 on a calibrated fixture
# --------------------------------------------------------------------------- #
def test_fit_temperature_recovers_unity_on_calibrated_fixture():
    pytest.importorskip("torch")
    from dmoi_brca.calibration import fit_temperature

    logits, labels = _calibrated_fixture()
    fit = fit_temperature(logits, labels)
    assert 0.8 < fit.temperature < 1.25
