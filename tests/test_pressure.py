"""Tests for the sycophancy-style pressure probe (DMOI x sycophancy bridge)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dmoi_brca.pressure import decisions, pressure_probe, push_back  # noqa: E402


def _preds(n=600, seed=0):
    rng = np.random.default_rng(seed)
    comp = rng.integers(0, 3, size=n)
    true_p = np.where(comp == 0, rng.beta(2, 8, n),
             np.where(comp == 1, rng.beta(5, 5, n), rng.beta(8, 2, n)))
    labels = (rng.random(n) < true_p).astype(int)
    proba = np.clip(true_p + rng.normal(0, 0.06, n), 0.0, 1.0)
    return labels, proba


def test_pushback_clipped_and_moves_toward_opposite():
    p = np.array([0.9, 0.1, 0.5])
    out = push_back(p, behavior="blind", strength=0.4)
    assert np.all((out >= 0.0) & (out <= 1.0))
    # a confident-positive call (0.9) is pushed downward (toward the asserted negative)
    assert out[0] < 0.9
    # a confident-negative call (0.1) is pushed upward (toward the asserted positive)
    assert out[1] > 0.1


def test_coupled_barely_moves_confident_calls():
    p = np.array([0.95, 0.5])
    out = push_back(p, behavior="coupled", strength=0.5)
    # confident call (0.95) moves far less than the uncertain one (0.5)
    assert abs(out[0] - 0.95) < 0.05
    assert abs(out[1] - 0.5) > abs(out[0] - 0.95)


def test_same_pre_ece_different_robustness():
    labels, proba = _preds()
    coupled = pressure_probe(labels, proba, behavior="coupled", strength=0.4)
    blind = pressure_probe(labels, proba, behavior="blind", strength=0.4)
    # identical pre-pressure ECE (same probabilities)
    assert coupled.ece_pre == blind.ece_pre
    # the trustworthy model is more robust and degrades calibration less
    assert coupled.robustness_rate > blind.robustness_rate
    assert coupled.ece_post < blind.ece_post


def test_coupled_flips_concentrate_at_low_confidence():
    labels, proba = _preds()
    r = pressure_probe(labels, proba, behavior="coupled", strength=0.4)
    bins = r.flip_by_confidence
    low = bins[0].flip_rate    # |2p-1| in [0,0.2): least confident
    high = bins[-1].flip_rate  # [0.8,1.0]: most confident
    # a trustworthy model flips its uncertain calls, not its confident ones
    assert (high != high) or (low > high)  # high may be nan if empty; else low > high


def test_decisions_threshold():
    assert list(decisions(np.array([0.4, 0.5, 0.6]))) == [0, 1, 1]
