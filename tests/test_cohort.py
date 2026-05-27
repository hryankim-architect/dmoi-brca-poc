"""Unit tests for dmoi_brca.cohort (Day-3 cohort selection)."""
from __future__ import annotations

import pandas as pd
import pytest

from dmoi_brca.cohort import (
    PAM50_BASAL,
    PAM50_LUMINAL,
    assign_group,
    assign_lumab_group,
    build_cohort,
    normalize_pam50,
)


def _row(**kw) -> pd.Series:
    base = {
        "PAM50Call_RNAseq": "",
        "PAM50_mRNA_nature2012": "",
        "ER_Status_nature2012": "",
        "PR_Status_nature2012": "",
        "HER2_Final_Status_nature2012": "",
    }
    base.update(kw)
    return pd.Series(base)


def test_normalize_pam50_prefers_rnaseq_call():
    row = _row(PAM50Call_RNAseq="LumA", PAM50_mRNA_nature2012="Luminal B")
    assert normalize_pam50(row) == "LumA"


def test_normalize_pam50_falls_back_to_long_form():
    row = _row(PAM50Call_RNAseq="", PAM50_mRNA_nature2012="Basal-like")
    assert normalize_pam50(row) == "Basal"


def test_normalize_pam50_nan_string_is_empty():
    row = _row(PAM50Call_RNAseq="nan", PAM50_mRNA_nature2012="Luminal A")
    assert normalize_pam50(row) == "LumA"


def test_assign_group_h_plus_luminal():
    row = _row(PAM50Call_RNAseq="LumA", ER_Status_nature2012="Positive")
    assert assign_group(row) == "H_plus_luminal"


def test_assign_group_h_minus_basal_tn():
    row = _row(
        PAM50Call_RNAseq="Basal",
        ER_Status_nature2012="Negative",
        PR_Status_nature2012="Negative",
        HER2_Final_Status_nature2012="Negative",
    )
    assert assign_group(row) == "H_minus_basal_tn"


def test_assign_group_excludes_her2_enriched():
    row = _row(PAM50Call_RNAseq="Her2", ER_Status_nature2012="Negative")
    assert assign_group(row) is None


def test_assign_group_excludes_basal_with_er_positive():
    # Edge case: PAM50=Basal but ER+ — not in either pole, excluded.
    row = _row(
        PAM50Call_RNAseq="Basal",
        ER_Status_nature2012="Positive",
        PR_Status_nature2012="Negative",
        HER2_Final_Status_nature2012="Negative",
    )
    assert assign_group(row) is None


def test_build_cohort_basic():
    clinical = pd.DataFrame([
        {"sampleID": "S1", "PAM50Call_RNAseq": "LumA",
         "PAM50_mRNA_nature2012": "", "ER_Status_nature2012": "Positive",
         "PR_Status_nature2012": "", "HER2_Final_Status_nature2012": ""},
        {"sampleID": "S2", "PAM50Call_RNAseq": "Basal",
         "PAM50_mRNA_nature2012": "", "ER_Status_nature2012": "Negative",
         "PR_Status_nature2012": "Negative", "HER2_Final_Status_nature2012": "Negative"},
        {"sampleID": "S3", "PAM50Call_RNAseq": "Her2",
         "PAM50_mRNA_nature2012": "", "ER_Status_nature2012": "Negative",
         "PR_Status_nature2012": "", "HER2_Final_Status_nature2012": ""},
    ])
    cohort, summary = build_cohort(clinical, {"S1", "S2"}, {"S2"})
    assert len(cohort) == 2
    assert summary.n_luminal_h_plus == 1
    assert summary.n_basal_h_minus == 1
    assert summary.n_both_modalities == 1  # S2 has both
    assert summary.n_rna_only == 1         # S1 RNA-only


def test_build_cohort_empty_raises():
    clinical = pd.DataFrame([
        {"sampleID": "S1", "PAM50Call_RNAseq": "Her2",
         "PAM50_mRNA_nature2012": "", "ER_Status_nature2012": "Negative",
         "PR_Status_nature2012": "", "HER2_Final_Status_nature2012": ""},
    ])
    with pytest.raises(ValueError, match="No patients matched"):
        build_cohort(clinical, set(), set())


def test_constants_unchanged():
    assert frozenset({"LumA", "LumB"}) == PAM50_LUMINAL
    assert frozenset({"Basal"}) == PAM50_BASAL


# ------------------ cohort v2 (LumA vs LumB) tests --------------------------

def test_assign_lumab_group_luma():
    row = _row(PAM50Call_RNAseq="LumA")
    assert assign_lumab_group(row) == "LumA"


def test_assign_lumab_group_lumb():
    row = _row(PAM50Call_RNAseq="LumB")
    assert assign_lumab_group(row) == "LumB"


def test_assign_lumab_group_excludes_basal():
    row = _row(PAM50Call_RNAseq="Basal")
    assert assign_lumab_group(row) is None


def test_assign_lumab_group_excludes_her2():
    row = _row(PAM50Call_RNAseq="Her2")
    assert assign_lumab_group(row) is None


def test_assign_lumab_group_fallback_to_long_form():
    row = _row(PAM50Call_RNAseq="", PAM50_mRNA_nature2012="Luminal A")
    assert assign_lumab_group(row) == "LumA"


def test_build_cohort_with_lumab_assigner():
    clinical = pd.DataFrame([
        {"sampleID": "P1", "PAM50Call_RNAseq": "LumA",
         "PAM50_mRNA_nature2012": "", "ER_Status_nature2012": "",
         "PR_Status_nature2012": "", "HER2_Final_Status_nature2012": ""},
        {"sampleID": "P2", "PAM50Call_RNAseq": "LumB",
         "PAM50_mRNA_nature2012": "", "ER_Status_nature2012": "",
         "PR_Status_nature2012": "", "HER2_Final_Status_nature2012": ""},
        {"sampleID": "P3", "PAM50Call_RNAseq": "Basal",
         "PAM50_mRNA_nature2012": "", "ER_Status_nature2012": "",
         "PR_Status_nature2012": "", "HER2_Final_Status_nature2012": ""},
    ])
    cohort, summary = build_cohort(
        clinical, {"P1", "P2"}, {"P2"},
        assigner=assign_lumab_group,
        label_a="LumA", label_b="LumB",
    )
    assert len(cohort) == 2
    assert summary.n_luminal_h_plus == 1  # label_a count
    assert summary.n_basal_h_minus == 1   # label_b count
    assert summary.n_both_modalities == 1  # P2
