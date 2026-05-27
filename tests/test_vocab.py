"""Unit tests for dmoi_brca.vocab (Lω helper)."""
from __future__ import annotations

import pandas as pd
import pytest

from dmoi_brca.vocab import (
    DefensiveFilterResult,
    VocabMismatchError,
    defensive_filter,
)


def test_primary_column_only_match():
    df = pd.DataFrame({"subtype": ["LumA", "LumB", "Basal", "Her2"]})
    result = defensive_filter(
        df,
        primary_col="subtype",
        allowed_values={"LumA", "LumB"},
    )
    assert isinstance(result, DefensiveFilterResult)
    assert len(result.matched) == 2
    assert result.primary_hit_count == 4  # all rows had values in primary
    assert result.fallback_hit_count == 0
    assert result.unmatched_count == 2  # Basal + Her2 didn't match


def test_fallback_column_used_when_primary_empty():
    df = pd.DataFrame({
        "short": ["LumA", None, ""],
        "long": ["Luminal A", "Luminal B", "Basal-like"],
    })
    result = defensive_filter(
        df,
        primary_col="short",
        fallback_col="long",
        fallback_map={
            "Luminal A": "LumA",
            "Luminal B": "LumB",
            "Basal-like": "Basal",
        },
        allowed_values={"LumA", "LumB", "Basal"},
    )
    # All 3 rows should match (1 via primary, 2 via fallback)
    assert len(result.matched) == 3
    assert result.primary_hit_count == 1
    assert result.fallback_hit_count == 2


def test_no_match_raises_vocab_mismatch():
    df = pd.DataFrame({"subtype": ["Luminal A", "Luminal B"]})  # long form, no fallback
    with pytest.raises(VocabMismatchError, match="No rows matched"):
        defensive_filter(
            df,
            primary_col="subtype",
            allowed_values={"LumA", "LumB"},
        )


def test_no_match_message_includes_seen_values():
    df = pd.DataFrame({"subtype": ["Luminal A", "Basal-like"]})
    with pytest.raises(VocabMismatchError) as excinfo:
        defensive_filter(
            df,
            primary_col="subtype",
            allowed_values={"LumA", "LumB"},
        )
    # The error message should point at the upstream vocab
    msg = str(excinfo.value)
    assert "Luminal A" in msg or "Basal-like" in msg
    assert "Did the upstream dataset change" in msg


def test_no_match_raise_on_empty_false_returns_empty_result():
    df = pd.DataFrame({"subtype": ["Luminal A"]})
    result = defensive_filter(
        df,
        primary_col="subtype",
        allowed_values={"LumA"},
        raise_on_empty=False,
    )
    assert result.matched.empty
    assert result.unmatched_count == 1


def test_fallback_col_without_map_raises():
    df = pd.DataFrame({"short": ["LumA"], "long": ["Luminal A"]})
    with pytest.raises(ValueError, match="fallback_col requires fallback_map"):
        defensive_filter(
            df,
            primary_col="short",
            fallback_col="long",  # but no fallback_map
            allowed_values={"LumA"},
        )


def test_nan_string_in_primary_falls_through():
    # The string "nan" should be treated as missing, not as a valid value
    df = pd.DataFrame({
        "short": ["nan", "LumA"],
        "long": ["Luminal A", "Luminal A"],
    })
    result = defensive_filter(
        df,
        primary_col="short",
        fallback_col="long",
        fallback_map={"Luminal A": "LumA"},
        allowed_values={"LumA"},
    )
    assert len(result.matched) == 2
    assert result.primary_hit_count == 1
    assert result.fallback_hit_count == 1


def test_unique_values_seen_captures_distinct():
    df = pd.DataFrame({"subtype": ["LumA", "LumA", "LumB", "Basal"]})
    result = defensive_filter(
        df,
        primary_col="subtype",
        allowed_values={"LumA", "LumB", "Basal"},
    )
    assert result.unique_values_seen == frozenset({"LumA", "LumB", "Basal"})


def test_label_appears_in_error_message():
    df = pd.DataFrame({"subtype": ["Luminal A"]})
    with pytest.raises(VocabMismatchError, match="patients"):
        defensive_filter(
            df,
            primary_col="subtype",
            allowed_values={"LumA"},
            label="patients",
        )


def test_result_is_frozen():
    from dataclasses import FrozenInstanceError

    df = pd.DataFrame({"subtype": ["LumA"]})
    result = defensive_filter(df, primary_col="subtype", allowed_values={"LumA"})
    with pytest.raises(FrozenInstanceError):
        result.unmatched_count = 999  # type: ignore[misc]
