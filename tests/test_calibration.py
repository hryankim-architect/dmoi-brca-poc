"""Unit tests for dmoi_brca.calibration."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dmoi_brca.calibration import (  # noqa: E402
    CalibrationFit,
    apply_temperature,
    calibrate_fold,
    fit_temperature,
)


def test_apply_temperature_identity_when_T_is_one():
    logits = np.array([-2.0, 0.0, 2.0, 5.0])
    proba = apply_temperature(logits, temperature=1.0)
    # sigmoid(logits / 1) = sigmoid(logits)
    expected = 1.0 / (1.0 + np.exp(-logits))
    assert np.allclose(proba, expected)


def test_apply_temperature_rejects_nonpositive():
    with pytest.raises(ValueError, match="temperature"):
        apply_temperature(np.array([1.0]), temperature=0.0)
    with pytest.raises(ValueError, match="temperature"):
        apply_temperature(np.array([1.0]), temperature=-1.0)


def test_apply_temperature_softens_with_T_gt_1():
    """T > 1 should pull probabilities toward 0.5 (soften)."""
    logits = np.array([5.0, -5.0])  # very confident
    proba_T1 = apply_temperature(logits, temperature=1.0)
    proba_T3 = apply_temperature(logits, temperature=3.0)
    # T=3 outputs should be closer to 0.5 than T=1 outputs
    assert abs(proba_T3[0] - 0.5) < abs(proba_T1[0] - 0.5)
    assert abs(proba_T3[1] - 0.5) < abs(proba_T1[1] - 0.5)


def test_fit_temperature_well_calibrated_input_finds_T_near_one():
    """If logits are already well-calibrated, fitted T ≈ 1."""
    rng = np.random.default_rng(0)
    n = 500
    logits = rng.normal(0, 2, n).astype(np.float32)
    proba = 1.0 / (1.0 + np.exp(-logits))
    labels = (rng.random(n) < proba).astype(np.int64)
    fit = fit_temperature(logits, labels)
    assert isinstance(fit, CalibrationFit)
    assert 0.5 < fit.temperature < 2.0  # close to 1


def test_fit_temperature_overconfident_finds_T_gt_one():
    """If logits are overconfident (too extreme), fitted T should be > 1."""
    rng = np.random.default_rng(0)
    n = 500
    # Generate well-calibrated probabilities, then INFLATE the logits.
    true_logits = rng.normal(0, 1, n).astype(np.float32)
    proba = 1.0 / (1.0 + np.exp(-true_logits))
    labels = (rng.random(n) < proba).astype(np.int64)
    # The model "saw" the same labels but outputs scaled-up logits (overconfident).
    overconfident_logits = (true_logits * 3.0).astype(np.float32)
    fit = fit_temperature(overconfident_logits, labels)
    # Optimal T should recover the scale factor ~3
    assert fit.temperature > 1.5


def test_fit_temperature_reduces_nll():
    rng = np.random.default_rng(0)
    n = 200
    logits = (rng.normal(0, 1, n) * 4.0).astype(np.float32)
    labels = (rng.random(n) < 1.0 / (1.0 + np.exp(-logits / 4.0))).astype(np.int64)
    fit = fit_temperature(logits, labels)
    assert fit.nll_after <= fit.nll_before + 1e-6


def test_fit_temperature_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape"):
        fit_temperature(np.array([1.0, 2.0]), np.array([0, 1, 1]))


def test_fit_temperature_2d_raises():
    with pytest.raises(ValueError, match="1-D"):
        fit_temperature(np.zeros((2, 3)), np.zeros((2, 3)))


def test_calibrate_fold_returns_proba_in_range():
    rng = np.random.default_rng(0)
    n = 100
    logits = rng.normal(0, 2, n).astype(np.float32)
    labels = rng.integers(0, 2, n).astype(np.int64)
    calibrated, fit = calibrate_fold(logits, labels)
    assert calibrated.shape == (n,)
    assert ((calibrated >= 0.0) & (calibrated <= 1.0)).all()
    assert fit.temperature > 0
