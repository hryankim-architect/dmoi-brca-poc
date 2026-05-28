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
    train_test_split_cohort,
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


# ---------------------------------------------------------------------------
# train_test_split_cohort
# ---------------------------------------------------------------------------


def _make_dual_modality_cohort(n_lumA: int = 40, n_lumB: int = 20) -> pd.DataFrame:
    rows = []
    for i in range(n_lumA):
        rows.append({
            "sample_id": f"A{i:03d}", "group": "LumA",
            "has_rna": True, "has_meth": True,
        })
    for i in range(n_lumB):
        rows.append({
            "sample_id": f"B{i:03d}", "group": "LumB",
            "has_rna": True, "has_meth": True,
        })
    return pd.DataFrame(rows)


def test_train_test_split_adds_split_column():
    cohort = _make_dual_modality_cohort()
    result = train_test_split_cohort(cohort, test_frac=0.2, random_state=2024)
    assert "split" in result.columns
    assert set(result["split"].unique()) == {"train", "test"}


def test_train_test_split_respects_test_frac():
    cohort = _make_dual_modality_cohort(n_lumA=40, n_lumB=20)
    result = train_test_split_cohort(cohort, test_frac=0.2, random_state=2024)
    n_test = (result["split"] == "test").sum()
    n_train = (result["split"] == "train").sum()
    # 20% of 40 = 8, 20% of 20 = 4 → total 12 test, 48 train
    assert n_test == 12
    assert n_train == 48


def test_train_test_split_stratifies():
    cohort = _make_dual_modality_cohort(n_lumA=40, n_lumB=20)
    result = train_test_split_cohort(cohort, test_frac=0.2, random_state=2024)
    test_lumA = ((result["split"] == "test") & (result["group"] == "LumA")).sum()
    test_lumB = ((result["split"] == "test") & (result["group"] == "LumB")).sum()
    # Both classes represented in test
    assert test_lumA == 8
    assert test_lumB == 4


def test_train_test_split_deterministic():
    cohort = _make_dual_modality_cohort()
    r1 = train_test_split_cohort(cohort, test_frac=0.2, random_state=2024)
    r2 = train_test_split_cohort(cohort, test_frac=0.2, random_state=2024)
    assert (r1["split"] == r2["split"]).all()


def test_train_test_split_different_seed_produces_different_assignment():
    cohort = _make_dual_modality_cohort()
    r1 = train_test_split_cohort(cohort, test_frac=0.2, random_state=2024)
    r2 = train_test_split_cohort(cohort, test_frac=0.2, random_state=42)
    # At least one assignment differs
    assert not (r1["split"] == r2["split"]).all()


def test_train_test_split_single_modality_patients_excluded():
    cohort = pd.DataFrame([
        {"sample_id": "A1", "group": "LumA", "has_rna": True, "has_meth": True},
        {"sample_id": "A2", "group": "LumA", "has_rna": True, "has_meth": True},
        {"sample_id": "A3", "group": "LumA", "has_rna": True, "has_meth": False},
        {"sample_id": "B1", "group": "LumB", "has_rna": True, "has_meth": True},
        {"sample_id": "B2", "group": "LumB", "has_rna": True, "has_meth": True},
        {"sample_id": "B3", "group": "LumB", "has_rna": False, "has_meth": True},
    ])
    result = train_test_split_cohort(cohort, test_frac=0.25, random_state=2024)
    excluded = result[result["split"] == ""]
    assert set(excluded["sample_id"]) == {"A3", "B3"}


def test_train_test_split_rejects_bad_frac():
    cohort = _make_dual_modality_cohort()
    with pytest.raises(ValueError, match="test_frac"):
        train_test_split_cohort(cohort, test_frac=0.0)
    with pytest.raises(ValueError, match="test_frac"):
        train_test_split_cohort(cohort, test_frac=1.0)


def test_train_test_split_rejects_bad_stratify_col():
    cohort = _make_dual_modality_cohort()
    with pytest.raises(ValueError, match="stratify_col"):
        train_test_split_cohort(cohort, stratify_col="nonexistent")
