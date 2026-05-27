"""Unit tests for dmoi_brca.priors (Day-5B Hallmark gene set priors)."""
from __future__ import annotations

import pytest

from dmoi_brca.priors import (
    HALLMARK_E2F_TARGETS,
    HALLMARK_ESTROGEN_RESPONSE_EARLY,
    HALLMARK_G2M_CHECKPOINT,
    HALLMARK_MYC_TARGETS_V1,
    HALLMARK_SETS,
    POLE_LUMA,
    POLE_LUMB,
    project_pole,
    project_to_features,
)


def test_all_sets_registered():
    assert "HALLMARK_ESTROGEN_RESPONSE_EARLY" in HALLMARK_SETS
    assert "HALLMARK_E2F_TARGETS" in HALLMARK_SETS
    assert len(HALLMARK_SETS) == 5


def test_set_sizes_in_expected_range():
    # MSigDB Hallmark sets are ~150-220 genes each. Our curated subsets should be
    # at least 100 genes and at most 250 to stay representative.
    for name, genes in HALLMARK_SETS.items():
        assert 100 <= len(genes) <= 250, f"{name} has {len(genes)} genes"


def test_no_duplicate_genes_within_set():
    for name, genes in HALLMARK_SETS.items():
        assert len(genes) == len(set(genes)), f"{name} has duplicates"


def test_known_canonical_genes_present():
    # LumA canonical markers should be in estrogen response sets.
    assert "ESR1" in HALLMARK_ESTROGEN_RESPONSE_EARLY
    assert "PGR" in HALLMARK_ESTROGEN_RESPONSE_EARLY
    assert "FOXA1" in HALLMARK_ESTROGEN_RESPONSE_EARLY
    assert "GATA3" in HALLMARK_ESTROGEN_RESPONSE_EARLY

    # LumB canonical proliferation markers should be in E2F / G2M sets.
    assert "MKI67" in HALLMARK_E2F_TARGETS
    assert "CDK1" in HALLMARK_E2F_TARGETS
    assert "TOP2A" in HALLMARK_E2F_TARGETS
    assert "PLK1" in HALLMARK_G2M_CHECKPOINT
    assert "AURKA" in HALLMARK_G2M_CHECKPOINT
    assert "AURKB" in HALLMARK_G2M_CHECKPOINT
    assert "MYC" in HALLMARK_MYC_TARGETS_V1


def test_pole_definitions():
    assert "HALLMARK_ESTROGEN_RESPONSE_EARLY" in POLE_LUMA
    assert "HALLMARK_ESTROGEN_RESPONSE_LATE" in POLE_LUMA
    assert "HALLMARK_E2F_TARGETS" in POLE_LUMB
    assert "HALLMARK_G2M_CHECKPOINT" in POLE_LUMB
    assert "HALLMARK_MYC_TARGETS_V1" in POLE_LUMB
    # Poles should be disjoint by name.
    assert not (set(POLE_LUMA) & set(POLE_LUMB))


def test_project_to_features_basic():
    features = ["ESR1", "FOXA1", "TP53", "GAPDH", "ACTB"]
    proj = project_to_features("HALLMARK_ESTROGEN_RESPONSE_EARLY", features)
    assert proj.name == "HALLMARK_ESTROGEN_RESPONSE_EARLY"
    assert proj.genes_in_features >= 2  # at least ESR1 + FOXA1
    assert all(0 <= i < len(features) for i in proj.feature_indices)
    assert "ESR1" in proj.matched_genes
    assert "FOXA1" in proj.matched_genes
    # Feature index for ESR1 should be 0.
    esr1_pos = proj.matched_genes.index("ESR1")
    assert proj.feature_indices[esr1_pos] == 0


def test_project_to_features_empty_match():
    features = ["NONEXISTENT1", "NONEXISTENT2"]
    proj = project_to_features("HALLMARK_E2F_TARGETS", features)
    assert proj.genes_in_features == 0
    assert proj.feature_indices == ()
    assert proj.overlap_fraction == 0.0
    assert len(proj.missing_genes) == proj.genes_in_set


def test_project_to_features_overlap_fraction():
    features = list(HALLMARK_E2F_TARGETS[:50])
    proj = project_to_features("HALLMARK_E2F_TARGETS", features)
    assert proj.genes_in_features == 50
    assert 0.20 <= proj.overlap_fraction <= 0.45  # 50 / ~200


def test_project_to_features_unknown_set_raises():
    with pytest.raises(KeyError, match="Unknown gene set"):
        project_to_features("HALLMARK_FAKE", ["ESR1"])


def test_project_pole_returns_dict():
    features = ["ESR1", "PGR", "MKI67", "TOP2A", "AURKA", "MYC"]
    proj_luma = project_pole(POLE_LUMA, features)
    proj_lumb = project_pole(POLE_LUMB, features)
    # LumA pole projections should cover the 2 ER sets.
    assert set(proj_luma) == set(POLE_LUMA)
    # LumB pole projections should cover the 3 proliferation sets.
    assert set(proj_lumb) == set(POLE_LUMB)
    # ESR1+PGR should land in LumA; MKI67+TOP2A+AURKA+MYC in LumB.
    assert "ESR1" in proj_luma["HALLMARK_ESTROGEN_RESPONSE_EARLY"].matched_genes
    assert "MKI67" in proj_lumb["HALLMARK_E2F_TARGETS"].matched_genes
    assert "AURKA" in proj_lumb["HALLMARK_G2M_CHECKPOINT"].matched_genes
