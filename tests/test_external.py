"""Unit tests for dmoi_brca.external (v0.2 cross-cohort prediction helpers)."""
from __future__ import annotations

import numpy as np
import pytest

from dmoi_brca.external import (
    align_to_train_genes,
    collapse_duplicate_genes,
    gene_overlap_stats,
    make_silenced_meth,
    quantile_normalize_to_train,
)

# ---------------------------------------------------------------------------
# collapse_duplicate_genes
# ---------------------------------------------------------------------------


def test_collapse_duplicate_genes_no_duplicates_returns_unchanged():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    out, names = collapse_duplicate_genes(X, ["A", "B", "C"])
    assert names == ["A", "B", "C"]
    np.testing.assert_array_equal(out, X)


def test_collapse_duplicate_genes_averages_duplicates():
    X = np.array(
        [[2.0, 4.0], [4.0, 8.0], [10.0, 20.0]],  # gene A has two rows
        dtype=np.float32,
    )
    out, names = collapse_duplicate_genes(X, ["A", "A", "B"])
    assert names == ["A", "B"]
    # A row should be (2+4)/2=3, (4+8)/2=6
    np.testing.assert_allclose(out[0], [3.0, 6.0])
    np.testing.assert_allclose(out[1], [10.0, 20.0])


def test_collapse_duplicate_genes_median_aggregator():
    X = np.array([[2.0, 4.0], [4.0, 8.0], [12.0, 20.0]], dtype=np.float32)
    out, names = collapse_duplicate_genes(X, ["A", "A", "A"], aggregator="median")
    assert names == ["A"]
    np.testing.assert_allclose(out[0], [4.0, 8.0])  # median of [2,4,12], [4,8,20]


def test_collapse_duplicate_genes_shape_mismatch_raises():
    X = np.zeros((3, 5))
    with pytest.raises(ValueError, match="expression rows"):
        collapse_duplicate_genes(X, ["A", "B"])  # 2 != 3


def test_collapse_duplicate_genes_bad_aggregator_raises():
    X = np.zeros((2, 5))
    with pytest.raises(ValueError, match="aggregator"):
        collapse_duplicate_genes(X, ["A", "B"], aggregator="sum")


# ---------------------------------------------------------------------------
# align_to_train_genes
# ---------------------------------------------------------------------------


def test_align_to_train_genes_reorders_columns():
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    aligned = align_to_train_genes(
        X, external_genes=["A", "B", "C"], train_genes=["C", "A", "B"],
    )
    np.testing.assert_array_equal(aligned[:, 0], [3.0, 6.0])  # C
    np.testing.assert_array_equal(aligned[:, 1], [1.0, 4.0])  # A
    np.testing.assert_array_equal(aligned[:, 2], [2.0, 5.0])  # B


def test_align_to_train_genes_missing_genes_get_fill_value():
    X = np.array([[1.0, 2.0]], dtype=np.float32)
    aligned = align_to_train_genes(
        X, external_genes=["A", "B"], train_genes=["A", "B", "C", "D"],
        fill_value=0.0,
    )
    assert aligned.shape == (1, 4)
    np.testing.assert_array_equal(aligned[0, 2:], [0.0, 0.0])


def test_align_to_train_genes_extra_external_genes_dropped():
    X = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    aligned = align_to_train_genes(
        X, external_genes=["A", "B", "C", "D"], train_genes=["B", "A"],
    )
    assert aligned.shape == (1, 2)
    assert aligned[0, 0] == 2.0  # B
    assert aligned[0, 1] == 1.0  # A


def test_align_to_train_genes_no_overlap_raises():
    X = np.zeros((1, 2))
    with pytest.raises(ValueError, match="No genes overlap"):
        align_to_train_genes(X, external_genes=["X", "Y"], train_genes=["A", "B"])


def test_align_to_train_genes_shape_mismatch_raises():
    X = np.zeros((1, 3))
    with pytest.raises(ValueError, match="external_X cols"):
        align_to_train_genes(X, external_genes=["A", "B"], train_genes=["A"])


# ---------------------------------------------------------------------------
# gene_overlap_stats
# ---------------------------------------------------------------------------


def test_gene_overlap_stats_basic():
    stats = gene_overlap_stats(["A", "B", "C"], ["B", "C", "D"])
    assert stats["n_external"] == 3
    assert stats["n_train"] == 3
    assert stats["n_shared"] == 2  # B, C
    assert stats["n_external_only"] == 1  # A
    assert stats["n_train_only_mean_imputed"] == 1  # D


# ---------------------------------------------------------------------------
# quantile_normalize_to_train
# ---------------------------------------------------------------------------


def test_quantile_normalize_to_train_matches_distribution():
    """After quantile normalization, external per-gene distribution should
    equal a quantile of train's per-gene distribution."""
    rng = np.random.default_rng(0)
    train = rng.normal(loc=5.0, scale=2.0, size=(200, 4)).astype(np.float32)
    external = rng.normal(loc=10.0, scale=1.0, size=(150, 4)).astype(np.float32)

    normalized = quantile_normalize_to_train(external, train)
    assert normalized.shape == external.shape

    # After QN, normalized per-gene min/max/median should be close to
    # train's per-gene min/max/median (the values are drawn from train).
    for j in range(4):
        # All values must come from train_sorted, so set-membership check:
        train_set = set(np.round(train[:, j], 6))
        norm_unique = set(np.round(normalized[:, j], 6))
        # Every normalized value should appear in the train column.
        assert norm_unique <= train_set


def test_quantile_normalize_to_train_preserves_rank():
    """Quantile normalization preserves the rank ordering of external samples."""
    rng = np.random.default_rng(1)
    train = rng.normal(0, 1, size=(100, 3)).astype(np.float32)
    external = rng.normal(0, 1, size=(50, 3)).astype(np.float32)

    normalized = quantile_normalize_to_train(external, train)
    for j in range(3):
        ranks_before = external[:, j].argsort().argsort()
        ranks_after = normalized[:, j].argsort().argsort()
        assert np.array_equal(ranks_before, ranks_after)


def test_quantile_normalize_to_train_shape_mismatch_raises():
    train = np.zeros((10, 5))
    external = np.zeros((4, 3))
    with pytest.raises(ValueError, match="align gene order"):
        quantile_normalize_to_train(external, train)


# ---------------------------------------------------------------------------
# make_silenced_meth
# ---------------------------------------------------------------------------


def test_make_silenced_meth_zeros_by_default():
    meth = make_silenced_meth(5, 1000)
    assert meth.shape == (5, 1000)
    assert meth.dtype == np.float32
    assert (meth == 0).all()


def test_make_silenced_meth_custom_fill():
    meth = make_silenced_meth(3, 50, fill_value=0.5)
    assert (meth == 0.5).all()


def test_make_silenced_meth_rejects_non_positive():
    with pytest.raises(ValueError, match="shape"):
        make_silenced_meth(0, 5)
    with pytest.raises(ValueError, match="shape"):
        make_silenced_meth(5, -1)
