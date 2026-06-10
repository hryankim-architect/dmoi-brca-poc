from __future__ import annotations

import numpy as np

from dmoi_brca.compare_integration import (
    eval_selector,
    hallmark_gene_universe,
    prior_meth_indices,
    prior_rna_indices,
    topvar_indices,
)

FAKE_SETS = {"SET_A": ("GENE1", "GENE2"), "SET_B": ("GENE2", "GENE3")}


def test_hallmark_universe_unions_and_dedupes():
    assert hallmark_gene_universe(FAKE_SETS) == {"GENE1", "GENE2", "GENE3"}


def test_prior_rna_indices_label_free_selection():
    genes = ["GENE0", "GENE1", "GENEX", "GENE3"]
    # GENE1 (idx1) and GENE3 (idx3) are in the universe; order preserved
    assert prior_rna_indices(genes, FAKE_SETS) == [1, 3]


def test_prior_meth_indices_uses_cis_mapping():
    probes = ["cg01", "cg02", "cg03", "cg04"]
    cis = {"cg01": {"GENE1"}, "cg02": {"INTERGENIC"}, "cg03": set(), "cg04": {"GENE3", "Z"}}
    # cg01 -> GENE1 (in), cg04 -> GENE3 (in); cg02/cg03 excluded
    assert prior_meth_indices(probes, cis, FAKE_SETS) == [0, 3]


def test_topvar_indices_picks_highest_variance():
    X = np.array([[0.0, 0.0, 0.0], [0.0, 5.0, 1.0], [0.0, -5.0, -1.0]])
    # col1 has the largest variance, col2 next, col0 zero
    assert topvar_indices(X, 1) == [1]
    assert topvar_indices(X, 2) == [1, 2]
    assert topvar_indices(X, 99) == [1, 2, 0]  # k clamped to n_cols, var-sorted
    assert topvar_indices(X, 0) == []


def test_eval_selector_shape_and_separable_signal():
    # Two well-separated classes -> downstream classifiers should score high F1.
    rng = np.random.default_rng(0)
    n = 60
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    signal = np.where(y == 1, 3.0, -3.0)[:, None] + rng.normal(0, 0.3, (n, 4))
    out = eval_selector(signal, y, n_splits=5, seed=0)
    assert out["n_features"] == 4
    assert set(out) >= {"lr_weighted_f1", "svc_weighted_f1", "calinski_harabasz", "davies_bouldin"}
    assert 0.0 <= out["lr_weighted_f1"] <= 1.0
    assert out["lr_weighted_f1"] > 0.9  # clearly separable
    assert out["calinski_harabasz"] > 0.0
