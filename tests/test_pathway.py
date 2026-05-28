"""Unit tests for dmoi_brca.pathway (v0.5 pathway-level IG aggregation)."""
from __future__ import annotations

import numpy as np
import pytest

from dmoi_brca.pathway import PathwayScore, pathway_aggregate, rank_pathways


def test_pathway_aggregate_shapes_basic():
    attr = np.array([
        [0.1, 0.2, -0.3, 0.0],
        [0.5, -0.1, 0.2, 0.3],
    ], dtype=np.float32)
    feature_names = ["g1", "g2", "g3", "g4"]
    pathways = {"P_AB": ["g1", "g2"], "P_CD": ["g3", "g4"]}
    out = pathway_aggregate(attr, feature_names, pathways)
    assert len(out) == 2
    by_name = {s.pathway_name: s for s in out}
    # P_AB: |attr|[:, 0:2] mean = (|0.1|+|0.2|+|0.5|+|0.1|)/4 = 0.225
    np.testing.assert_allclose(by_name["P_AB"].mean_abs_ig, 0.225, atol=1e-6)
    # P_CD: |attr|[:, 2:4] mean = (|-0.3|+|0|+|0.2|+|0.3|)/4 = 0.2
    np.testing.assert_allclose(by_name["P_CD"].mean_abs_ig, 0.2, atol=1e-6)
    assert by_name["P_AB"].n_pathway_genes_in_inputs == 2
    assert by_name["P_CD"].n_pathway_genes_in_inputs == 2


def test_pathway_aggregate_signed_sum_direction():
    """sum_signed encodes direction; positive attributions push positive."""
    attr = np.array([[1.0, 2.0, -0.5]], dtype=np.float32)
    out = pathway_aggregate(
        attr, ["g1", "g2", "g3"],
        {"all_positive": ["g1", "g2"], "mixed": ["g1", "g3"]},
    )
    by_name = {s.pathway_name: s for s in out}
    # all_positive sum = 1.0 + 2.0 = 3.0; signed_mean = 3.0/2 = 1.5
    np.testing.assert_allclose(by_name["all_positive"].sum_signed, 3.0, atol=1e-6)
    np.testing.assert_allclose(by_name["all_positive"].signed_mean, 1.5, atol=1e-6)
    # mixed sum = 1.0 + (-0.5) = 0.5; signed_mean = 0.25
    np.testing.assert_allclose(by_name["mixed"].sum_signed, 0.5, atol=1e-6)
    np.testing.assert_allclose(by_name["mixed"].signed_mean, 0.25, atol=1e-6)


def test_pathway_aggregate_ignores_genes_absent_from_inputs():
    """Pathway can reference genes not in feature_names — they're just dropped."""
    attr = np.array([[1.0, 2.0]], dtype=np.float32)
    out = pathway_aggregate(
        attr, ["g1", "g2"],
        {"P": ["g1", "g_missing", "g_also_missing"]},
    )
    s = out[0]
    # 3 declared, only 1 present in inputs.
    assert s.n_pathway_genes_total == 3
    assert s.n_pathway_genes_in_inputs == 1
    np.testing.assert_allclose(s.mean_abs_ig, 1.0, atol=1e-6)


def test_pathway_aggregate_empty_pathway_returns_zeros():
    """Pathway with no genes in feature_names returns all-zero score."""
    attr = np.array([[1.0, 2.0]], dtype=np.float32)
    out = pathway_aggregate(
        attr, ["g1", "g2"],
        {"orphan": ["x", "y"]},
    )
    s = out[0]
    assert s.n_pathway_genes_in_inputs == 0
    assert s.mean_abs_ig == 0.0
    assert s.sum_signed == 0.0
    assert s.signed_mean == 0.0


def test_pathway_aggregate_averages_across_patients():
    """Per-patient sum then mean across patients (sum_signed semantics)."""
    attr = np.array([
        [1.0, 2.0],   # patient 0: sum = 3.0
        [3.0, 4.0],   # patient 1: sum = 7.0
    ], dtype=np.float32)
    out = pathway_aggregate(attr, ["g1", "g2"], {"P": ["g1", "g2"]})
    # sum_signed = mean(3, 7) = 5.0
    np.testing.assert_allclose(out[0].sum_signed, 5.0, atol=1e-6)


def test_pathway_aggregate_rejects_non_2d():
    attr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    with pytest.raises(ValueError, match="2-D"):
        pathway_aggregate(attr, ["g1", "g2", "g3"], {"P": ["g1"]})


def test_pathway_aggregate_rejects_shape_mismatch():
    attr = np.zeros((2, 3))
    with pytest.raises(ValueError, match="cols"):
        pathway_aggregate(attr, ["g1", "g2"], {"P": ["g1"]})


def test_rank_pathways_by_mean_abs_descending():
    scores = [
        PathwayScore("P1", 10, 5, mean_abs_ig=0.1, sum_signed=0.0, signed_mean=0.0),
        PathwayScore("P2", 10, 5, mean_abs_ig=0.5, sum_signed=0.0, signed_mean=0.0),
        PathwayScore("P3", 10, 5, mean_abs_ig=0.3, sum_signed=0.0, signed_mean=0.0),
    ]
    ranked = rank_pathways(scores, by="mean_abs_ig")
    assert [s.pathway_name for s in ranked] == ["P2", "P3", "P1"]


def test_rank_pathways_ascending():
    scores = [
        PathwayScore("P1", 10, 5, mean_abs_ig=0.1, sum_signed=0.0, signed_mean=0.0),
        PathwayScore("P2", 10, 5, mean_abs_ig=0.5, sum_signed=0.0, signed_mean=0.0),
    ]
    ranked = rank_pathways(scores, by="mean_abs_ig", descending=False)
    assert [s.pathway_name for s in ranked] == ["P1", "P2"]


def test_rank_pathways_rejects_bad_field():
    scores = [PathwayScore("P", 10, 5, 0.1, 0.0, 0.0)]
    with pytest.raises(ValueError, match="by must be"):
        rank_pathways(scores, by="nonexistent_field")
