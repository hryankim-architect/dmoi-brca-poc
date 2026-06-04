"""Saturation detection as a substrate primitive (Lσ).

When a baseline classifier returns near-perfect metrics (AUROC >= 0.99,
F1 = 1.0, etc.), that result is structurally ambiguous: it means either
*perfect classifier* OR *task is too easy / has data leakage / label is
derivable from features by construction*. The latter is far more common
in discrimination tasks where the label was originally derived from one
of the input modalities.

This module provides a reusable substrate primitive for detecting that
saturation case and emitting a structurally-different audit section.

Example:
    >>> from dmoi_brca.saturation import check_saturation
    >>> auc_means = [1.0, 1.0, 0.998, 1.0]   # all >= 0.99
    >>> report = check_saturation(auc_means, metric="AUROC")
    >>> report.is_saturated
    True
    >>> print(report.audit_section())
    ## Saturation finding (honest)
    ...

If the result is saturated, the recommended next step is to re-scope the
discrimination target to something harder, not to ship the result as a
successful baseline.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

# Default thresholds tuned for binary classification metrics in [0, 1].
# Override per-metric if needed.
DEFAULT_SATURATION_THRESHOLD = 0.99
DEFAULT_NEAR_SATURATION_THRESHOLD = 0.95


@dataclass(frozen=True)
class SaturationReport:
    """Structured result of a saturation check on a list of metric values."""

    is_saturated: bool
    is_near_saturated: bool
    max_value: float
    min_value: float
    mean_value: float
    n_values: int
    threshold: float
    metric_name: str
    candidate_causes: tuple[str, ...] = field(default_factory=tuple)

    def audit_section(self) -> str:
        """Return the Markdown audit section appropriate to the result.

        - If is_saturated: the "Saturation finding (honest)" section that
          forces an honest re-scope conversation.
        - If is_near_saturated but not fully saturated: a milder warning.
        - Otherwise: the default "Scope" framing.
        """
        if self.is_saturated:
            causes = (
                "\n".join(f"- {c}" for c in self.candidate_causes)
                if self.candidate_causes
                else (
                    "- Was the label derived from one of the input modalities?\n"
                    "- Are the two classes biologically very distinct cell types?\n"
                    "- Could a single ID-like feature trivially separate them?"
                )
            )
            return (
                "## Saturation finding (honest)\n\n"
                f"All {self.n_values} {self.metric_name} values are >= "
                f"{self.threshold:.2f} (max={self.max_value:.4f}, "
                f"mean={self.mean_value:.4f}).\n\n"
                "This is **not** a successful baseline — it indicates the\n"
                "task is too easy for a meaningful comparison anchor:\n\n"
                f"{causes}\n\n"
                "**Honest next step**: re-scope to a harder discrimination\n"
                "target on the same cohort, rather than shipping this as a\n"
                "baseline number to beat.\n"
            )
        if self.is_near_saturated:
            return (
                "## Near-saturation warning\n\n"
                f"All {self.n_values} {self.metric_name} values are >= "
                f"{DEFAULT_NEAR_SATURATION_THRESHOLD:.2f} (max={self.max_value:.4f}, "
                f"mean={self.mean_value:.4f}) but at least one is below the\n"
                f"strict saturation threshold of {self.threshold:.2f}.\n\n"
                "The task is *probably* too easy, but the next-step model still\n"
                "has small headroom. Consider whether to push harder on a\n"
                "narrower discrimination axis before accepting this as the\n"
                "comparison anchor.\n"
            )
        return (
            "## Limitations\n\n"
            f"Baseline {self.metric_name} values range "
            f"{self.min_value:.4f} - {self.max_value:.4f} "
            f"(mean={self.mean_value:.4f}, n={self.n_values}).\n"
            "This is a non-trivial comparison anchor for the next-step model.\n"
        )


def check_saturation(
    values: Sequence[float],
    *,
    metric: str = "AUROC",
    threshold: float = DEFAULT_SATURATION_THRESHOLD,
    near_threshold: float = DEFAULT_NEAR_SATURATION_THRESHOLD,
    candidate_causes: Sequence[str] | None = None,
    direction: Literal["max", "min"] = "max",
) -> SaturationReport:
    """Detect saturation in a list of metric values.

    Args:
        values:           List of metric values (e.g. AUROC means across CV folds
                          for each (feature_set, model) combination).
        metric:           Display name of the metric (e.g. "AUROC", "F1").
        threshold:        A value is "saturated" if it equals or exceeds this.
                          Default 0.99 for [0,1]-bounded metrics.
        near_threshold:   A value is "near saturated" at this threshold.
                          Default 0.95.
        candidate_causes: Optional list of likely causes to embed in the audit
                          section. Falls back to generic prompts if omitted.
        direction:        "max" if higher = better (AUROC, accuracy, F1).
                          "min" if lower = better (MAE, MSE, loss).
                          For "min", saturation means values <= (1 - threshold)
                          if the metric is in [0, 1]; otherwise compare to 0.

    Returns:
        SaturationReport with is_saturated / is_near_saturated booleans plus
        descriptive statistics. Call .audit_section() for the rendered MD.
    """
    if not values:
        raise ValueError("check_saturation: empty values sequence")
    n = len(values)
    floats = [float(v) for v in values]
    vmax = max(floats)
    vmin = min(floats)
    vmean = sum(floats) / n

    if direction == "max":
        is_saturated = all(v >= threshold for v in floats)
        is_near_saturated = all(v >= near_threshold for v in floats)
    elif direction == "min":
        # For loss-like metrics, saturation means everything near zero.
        # Use (1 - threshold) so threshold=0.99 -> all values <= 0.01.
        # For unbounded loss metrics, callers should set custom thresholds.
        sat_cutoff = 1.0 - threshold
        near_cutoff = 1.0 - near_threshold
        is_saturated = all(v <= sat_cutoff for v in floats)
        is_near_saturated = all(v <= near_cutoff for v in floats)
    else:
        raise ValueError(f"check_saturation: direction must be 'max' or 'min', got {direction!r}")

    return SaturationReport(
        is_saturated=is_saturated,
        is_near_saturated=is_near_saturated,
        max_value=vmax,
        min_value=vmin,
        mean_value=vmean,
        n_values=n,
        threshold=threshold,
        metric_name=metric,
        candidate_causes=tuple(candidate_causes) if candidate_causes else (),
    )
