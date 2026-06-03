"""Defensive vocabulary handling against upstream public datasets (Lω).

When a demo pipeline filters against an external vocabulary
(TCGA / GDC / Xena / ENCODE / Ensembl / any upstream public source), the
vocabulary may differ across mirrors or drift across releases. This module
provides reusable helpers for filtering with fallback columns + failing
loudly when no rows match.

Example:
    >>> import pandas as pd
    >>> from dmoi_brca.vocab import defensive_filter, VocabMismatchError
    >>> df = pd.DataFrame({
    ...     "primary_col": ["a", None, "c"],
    ...     "fallback_col": ["alpha", "beta", "charlie"],
    ... })
    >>> # Filter to known values, falling back to a mapped column.
    >>> result = defensive_filter(
    ...     df,
    ...     primary_col="primary_col",
    ...     fallback_col="fallback_col",
    ...     fallback_map={"alpha": "a", "beta": "b", "charlie": "c"},
    ...     allowed_values={"a", "b", "c"},
    ... )
    >>> result.matched.shape[0]
    3
    >>> result.unmatched_count
    0

If no rows match, a VocabMismatchError is raised with a diagnostic message.
"""
from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from typing import Any

# pandas is optional at import time; only imported lazily inside functions
# so the scaffold can be imported in environments without pandas. Capability
# portraits that use defensive_filter() are expected to have pandas installed.


class VocabMismatchError(ValueError):
    """Raised when a defensive_filter produces zero matched rows.

    The message includes which columns were checked and a summary of the
    upstream values seen vs. the allowed values, so the diagnostic points
    at the likely cause (vocabulary drift) rather than surfacing as a
    cryptic downstream KeyError.
    """


@dataclass(frozen=True)
class DefensiveFilterResult:
    """Structured result of a defensive_filter call.

    Attributes:
        matched:              DataFrame restricted to rows whose normalized
                              value is in allowed_values.
        normalized_column:    Series of the same length as the input frame
                              containing the normalized value per row
                              (empty string if neither column matched).
        unmatched_count:      Rows that had no usable value in either column.
        primary_hit_count:    Rows whose primary column produced the value.
        fallback_hit_count:   Rows that fell back to the fallback column.
        unique_values_seen:   Set of all distinct normalized values present
                              in the input (helpful when allowed_values is
                              wrong and you want to know what the upstream
                              actually contains).
    """

    matched: Any  # pandas DataFrame
    normalized_column: Any  # pandas Series
    unmatched_count: int
    primary_hit_count: int
    fallback_hit_count: int
    unique_values_seen: frozenset[str] = field(default_factory=frozenset)


def _normalize_row(
    row: Mapping[str, Any],
    primary_col: str,
    fallback_col: str | None,
    fallback_map: Mapping[str, str] | None,
) -> tuple[str, str]:
    """Return (normalized_value, source) for one row.

    source is 'primary', 'fallback', or 'none'.
    """
    primary = row.get(primary_col)
    if primary is not None and str(primary).strip() and str(primary).strip().lower() != "nan":
        return str(primary).strip(), "primary"
    if fallback_col is not None and fallback_map is not None:
        fb = row.get(fallback_col)
        if fb is not None and str(fb).strip() and str(fb).strip().lower() != "nan":
            mapped = fallback_map.get(str(fb).strip(), "")
            if mapped:
                return mapped, "fallback"
    return "", "none"


def defensive_filter(
    df: Any,
    *,
    primary_col: str,
    allowed_values: Collection[str],
    fallback_col: str | None = None,
    fallback_map: Mapping[str, str] | None = None,
    raise_on_empty: bool = True,
    label: str = "rows",
) -> DefensiveFilterResult:
    """Filter a DataFrame against an external vocabulary with fallback handling.

    Args:
        df:               Input DataFrame.
        primary_col:      Column whose values should already be in the
                          short/normalized form (allowed_values vocabulary).
        allowed_values:   Set of acceptable normalized values to keep.
        fallback_col:     Optional column whose values are in an alternative
                          form (e.g. long-form labels). Used when primary_col
                          is empty/NaN for a given row.
        fallback_map:     Required if fallback_col is set. Maps fallback
                          values to normalized values.
        raise_on_empty:   If True (default), raise VocabMismatchError when
                          no rows match. Set False to receive an empty
                          matched frame instead.
        label:            Display label for error messages (e.g. "patients",
                          "samples", "rows").

    Returns:
        DefensiveFilterResult with the matched frame plus diagnostic counts.

    Raises:
        VocabMismatchError: If no rows match and raise_on_empty=True.
        ImportError: If pandas is not installed.
    """
    try:
        import pandas as pd  # noqa: F401  (only need the side-effect import)
    except ImportError as e:
        raise ImportError("dmoi_brca.vocab requires pandas") from e

    if fallback_col is not None and fallback_map is None:
        raise ValueError(
            "defensive_filter: fallback_col requires fallback_map to be set",
        )

    allowed_set = set(allowed_values)

    normalized: list[str] = []
    sources: list[str] = []
    for _, row in df.iterrows():
        value, source = _normalize_row(row, primary_col, fallback_col, fallback_map)
        normalized.append(value)
        sources.append(source)

    import pandas as pd

    norm_series = pd.Series(normalized, index=df.index)
    mask = norm_series.isin(allowed_set)
    matched = df[mask].copy()

    primary_hits = sum(1 for s in sources if s == "primary")
    fallback_hits = sum(1 for s in sources if s == "fallback")
    unmatched = len(df) - mask.sum()
    unique_seen = frozenset(v for v in normalized if v)

    if matched.empty and raise_on_empty:
        cols_checked = [primary_col]
        if fallback_col is not None:
            cols_checked.append(fallback_col)
        seen_summary = ", ".join(sorted(unique_seen)[:10])
        if len(unique_seen) > 10:
            seen_summary += f", ... ({len(unique_seen) - 10} more)"
        raise VocabMismatchError(
            f"No {label} matched allowed_values={sorted(allowed_set)} "
            f"using columns={cols_checked}. "
            f"Upstream column values actually seen: [{seen_summary}]. "
            f"Did the upstream dataset change its value vocabulary?",
        )

    return DefensiveFilterResult(
        matched=matched,
        normalized_column=norm_series,
        unmatched_count=int(unmatched),
        primary_hit_count=primary_hits,
        fallback_hit_count=fallback_hits,
        unique_values_seen=unique_seen,
    )
