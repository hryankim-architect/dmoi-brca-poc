"""Unit tests for dmoi_brca.saturation (Lσ helper)."""
from __future__ import annotations

import pytest

from dmoi_brca.saturation import (
    DEFAULT_NEAR_SATURATION_THRESHOLD,
    DEFAULT_SATURATION_THRESHOLD,
    SaturationReport,
    check_saturation,
)


def test_all_saturated_returns_saturated_report():
    auc = [1.0, 1.0, 0.998, 1.0, 0.995]
    report = check_saturation(auc, metric="AUROC")
    assert report.is_saturated
    assert report.is_near_saturated  # implied
    assert report.n_values == 5
    assert report.max_value == 1.0
    assert report.metric_name == "AUROC"


def test_one_below_threshold_not_saturated():
    auc = [1.0, 1.0, 0.98, 1.0]  # one < 0.99
    report = check_saturation(auc)
    assert not report.is_saturated
    # 0.98 still >= 0.95, so near-saturated
    assert report.is_near_saturated


def test_clearly_below_threshold():
    auc = [0.7, 0.8, 0.85, 0.75]
    report = check_saturation(auc)
    assert not report.is_saturated
    assert not report.is_near_saturated


def test_audit_section_saturated_emits_honest_section():
    auc = [1.0] * 5
    report = check_saturation(auc, metric="AUROC")
    md = report.audit_section()
    assert "Saturation finding (honest)" in md
    assert "not** a successful baseline" in md
    assert "AUROC" in md


def test_audit_section_near_saturated_emits_warning():
    auc = [0.96, 0.97, 0.98, 0.97]  # all >= 0.95 but at least one < 0.99
    report = check_saturation(auc)
    md = report.audit_section()
    assert "Near-saturation warning" in md
    assert "Saturation finding" not in md


def test_audit_section_normal_emits_default_scope():
    auc = [0.7, 0.8, 0.75]
    report = check_saturation(auc)
    md = report.audit_section()
    assert "Limitations" in md
    assert "Saturation" not in md


def test_custom_candidate_causes_render():
    auc = [1.0, 1.0]
    causes = (
        "Label was derived from RNA-seq, then RNA used as features",
        "Cell-of-origin states are biologically too distinct",
    )
    report = check_saturation(auc, candidate_causes=causes)
    md = report.audit_section()
    assert "Label was derived from RNA-seq" in md
    assert "biologically too distinct" in md


def test_empty_values_raises():
    with pytest.raises(ValueError, match="empty"):
        check_saturation([])


def test_direction_min_for_loss_metrics():
    # MAE near zero = saturated (predictions perfectly match)
    mae = [0.001, 0.002, 0.0001]
    report = check_saturation(mae, metric="MAE", direction="min")
    assert report.is_saturated


def test_direction_min_high_loss_not_saturated():
    mae = [0.5, 0.4, 0.6]
    report = check_saturation(mae, metric="MAE", direction="min")
    assert not report.is_saturated


def test_invalid_direction_raises():
    with pytest.raises(ValueError, match="direction"):
        check_saturation([0.5], direction="invalid")


def test_report_is_frozen():
    from dataclasses import FrozenInstanceError

    report = check_saturation([0.5])
    with pytest.raises(FrozenInstanceError):
        report.is_saturated = True  # type: ignore[misc]


def test_default_thresholds_have_expected_values():
    assert DEFAULT_SATURATION_THRESHOLD == 0.99
    assert DEFAULT_NEAR_SATURATION_THRESHOLD == 0.95


def test_report_descriptive_stats():
    auc = [0.6, 0.8, 0.9]
    report = check_saturation(auc)
    assert report.min_value == 0.6
    assert report.max_value == 0.9
    assert abs(report.mean_value - 0.7667) < 0.001
    assert isinstance(report, SaturationReport)
