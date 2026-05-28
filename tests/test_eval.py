"""Unit tests for dmoi_brca.eval (Day-4 analytical primitives)."""
from __future__ import annotations

import numpy as np
import pytest

from dmoi_brca.eval import (
    CalibrationReport,
    DisagreementReport,
    PerClassMetrics,
    aggregate_cross_fold,
    build_fold_eval_bundle,
    compute_calibration,
    compute_disagreement_report,
    compute_per_class_metrics,
    concat_fold_predictions,
    confusion_matrix_table,
)

# ---------------------------------------------------------------------------
# Per-class metrics
# ---------------------------------------------------------------------------

def test_per_class_metrics_perfect_classifier():
    labels = np.array([0, 0, 1, 1])
    pred = np.array([0, 0, 1, 1])
    m = compute_per_class_metrics(labels, pred)
    assert m["LumA"].precision == 1.0 and m["LumA"].recall == 1.0 and m["LumA"].f1 == 1.0
    assert m["LumB"].precision == 1.0 and m["LumB"].recall == 1.0 and m["LumB"].f1 == 1.0


def test_per_class_metrics_returns_dataclass():
    labels = np.array([0, 1, 0, 1])
    pred = np.array([0, 0, 0, 1])  # one LumB misclassified
    m = compute_per_class_metrics(labels, pred)
    assert isinstance(m["LumA"], PerClassMetrics)
    assert m["LumB"].n_in_fold == 2
    assert m["LumB"].recall == 0.5  # 1 of 2 LumB caught


def test_per_class_metrics_zero_division_safe():
    labels = np.array([0, 0, 0, 0])  # no positives at all
    pred = np.array([0, 0, 0, 0])
    m = compute_per_class_metrics(labels, pred)
    assert m["LumB"].n_in_fold == 0
    assert m["LumB"].f1 == 0.0  # no positives -> 0, not NaN


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def test_calibration_perfect_calibration_low_ece():
    # All probabilities = exact base rate; ECE should be near 0
    labels = np.array([0, 0, 1, 1, 0, 1, 0, 1])  # 50% positive
    proba = np.full(8, 0.5)
    rep = compute_calibration(labels, proba, n_bins=10)
    assert isinstance(rep, CalibrationReport)
    assert rep.ece == 0.0  # exactly calibrated at 0.5


def test_calibration_extreme_miscalibration():
    # Predict 100% positive for all but labels are 50/50 → ECE near 0.5
    labels = np.array([0, 0, 1, 1])
    proba = np.full(4, 1.0)
    rep = compute_calibration(labels, proba, n_bins=10)
    assert 0.4 < rep.ece < 0.6


def test_calibration_bins_sum_to_n_samples():
    labels = np.random.default_rng(0).integers(0, 2, size=50)
    proba = np.random.default_rng(0).random(50)
    rep = compute_calibration(labels, proba, n_bins=5)
    assert sum(rep.bin_counts) == 50
    assert len(rep.bin_centers) == 5
    assert len(rep.bin_confidence) == 5


def test_calibration_n_bins_too_small_raises():
    with pytest.raises(ValueError, match="n_bins"):
        compute_calibration(np.array([0, 1]), np.array([0.3, 0.7]), n_bins=1)


def test_calibration_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape"):
        compute_calibration(np.array([0, 1]), np.array([0.5]), n_bins=5)


# ---------------------------------------------------------------------------
# Disagreement report
# ---------------------------------------------------------------------------

def test_disagreement_zero_misclass_yields_safe_report():
    labels = np.array([0, 0, 1, 1])
    pred = np.array([0, 0, 1, 1])
    dis = np.array([0.1, 0.2, 0.3, 0.4])
    rep = compute_disagreement_report(labels, pred, dis)
    assert rep.n_misclassified == 0
    assert rep.is_informative is False  # cannot establish information from 0 misclass


def test_disagreement_correlates_with_misclass():
    # Misclassified samples have high disagreement; correctly classified have low.
    rng = np.random.default_rng(42)
    n = 200
    misclass_indicator = rng.choice([0, 1], size=n, p=[0.7, 0.3])
    # Disagreement correlates strongly with misclass
    disagreement = 0.1 + misclass_indicator * 0.6 + rng.normal(0, 0.05, n)
    disagreement = np.clip(disagreement, 0, 1)
    # Synthetic labels + pred consistent with misclass indicator
    labels = rng.choice([0, 1], size=n, p=[0.7, 0.3])
    pred = labels.copy()
    pred[misclass_indicator == 1] = 1 - pred[misclass_indicator == 1]

    rep = compute_disagreement_report(labels, pred, disagreement)
    assert isinstance(rep, DisagreementReport)
    assert rep.mean_dis_incorrect > rep.mean_dis_correct
    assert rep.point_biserial_r > 0.5
    assert rep.point_biserial_p < 0.05
    assert rep.auc_dis_predicts_misclass > 0.85
    assert rep.is_informative is True


def test_disagreement_no_correlation():
    rng = np.random.default_rng(0)
    n = 200
    labels = rng.choice([0, 1], size=n)
    pred = rng.choice([0, 1], size=n)
    dis = rng.random(n)  # random, independent of misclass
    rep = compute_disagreement_report(labels, pred, dis)
    assert abs(rep.point_biserial_r) < 0.3
    # Random disagreement: AUC near 0.5
    assert 0.35 < rep.auc_dis_predicts_misclass < 0.65


def test_disagreement_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        compute_disagreement_report(
            np.array([0, 1]), np.array([0, 1, 1]), np.array([0.1, 0.2, 0.3]),
        )


# ---------------------------------------------------------------------------
# Bundle builder + aggregate
# ---------------------------------------------------------------------------

def test_build_fold_eval_bundle_has_all_reports():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=50)
    proba = rng.random(50)
    dis = rng.random(50)
    b = build_fold_eval_bundle(fold=1, labels=labels, proba=proba, disagreement=dis)
    assert b.fold == 1
    assert b.n_test == 50
    assert set(b.per_class) == {"LumA", "LumB"}
    assert b.calibration is not None
    assert b.disagreement_report is not None


def test_aggregate_cross_fold_means_match_per_fold_values():
    rng = np.random.default_rng(0)
    bundles = []
    for fold in range(1, 6):
        labels = rng.integers(0, 2, size=40)
        proba = rng.random(40)
        dis = rng.random(40)
        bundles.append(build_fold_eval_bundle(fold=fold, labels=labels, proba=proba, disagreement=dis))
    agg = aggregate_cross_fold(bundles)
    assert agg["n_folds"] == 5
    assert "f1_LumA_mean" in agg
    assert "f1_LumB_mean" in agg
    assert "ece_mean" in agg
    assert 0.0 <= agg["f1_LumB_mean"] <= 1.0


def test_concat_fold_predictions_lengths():
    bundles = [
        build_fold_eval_bundle(
            fold=f, labels=np.array([0, 1, 0]),
            proba=np.array([0.1, 0.9, 0.2]),
            disagreement=np.array([0.1, 0.2, 0.3]),
        ) for f in range(3)
    ]
    L, P, D = concat_fold_predictions(bundles)
    assert L.shape == (9,) and P.shape == (9,) and D.shape == (9,)


def test_confusion_matrix_table_keys():
    labels = np.array([0, 0, 1, 1])
    pred = np.array([0, 1, 1, 0])
    cm = confusion_matrix_table(labels, pred)
    assert cm == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
