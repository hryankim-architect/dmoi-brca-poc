"""Sycophancy-style pressure probe over calibrated confidences.

The DMOI x sycophancy-eval bridge, in code. Calibration (ECE) says *how much* to
trust a confidence; it does **not** say whether that confidence survives a confident
*wrong* push — the clinical analogue of LLM sycophancy (a clinician asserting the
opposite, with authority). This probe perturbs a set of ``(label, proba)`` predictions
with a pushback and reports robustness **alongside** ECE, so the two distinct trust
axes are visible together.

The key, honest point: two revision behaviors can share the **same pre-pressure ECE**
yet differ wildly in robustness — so ECE alone cannot catch the sycophancy failure.

- ``coupled``  (trustworthy): the model revises in proportion to ``1 - confidence`` —
  it barely moves a confident call, reconsiders only genuinely uncertain ones.
- ``blind``    (sycophantic): the model shifts every call toward the asserted opposite
  by a fixed amount, regardless of how confident it was.

Deterministic; reuses :func:`dmoi_brca.eval.compute_calibration` for the ECE so the
calibration math is identical to the rest of the repo. No model, no training.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dmoi_brca.eval import compute_calibration

Behavior = str  # "coupled" | "blind"


def decisions(proba: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (np.asarray(proba, dtype=float) >= threshold).astype(int)


def push_back(proba: np.ndarray, *, behavior: Behavior = "coupled",
              strength: float = 0.4) -> np.ndarray:
    """Revise each prediction toward the OPPOSITE of its own call (clinician pushback).

    ``confidence`` is ``|2p - 1|`` (0 at p=0.5, 1 at p in {0,1}). ``coupled`` moves a
    prediction by ``strength * (1 - confidence)`` toward the asserted class; ``blind``
    moves every prediction by ``strength`` regardless of confidence. Returns revised
    probabilities clipped to [0, 1].
    """
    p = np.asarray(proba, dtype=float)
    call_positive = p >= 0.5
    asserted = np.where(call_positive, 0.0, 1.0)  # the clinician asserts the opposite
    confidence = np.abs(2.0 * p - 1.0)
    if behavior == "coupled":
        move = strength * (1.0 - confidence)
    elif behavior == "blind":
        move = strength * np.ones_like(p)
    else:
        raise ValueError(f"unknown behavior {behavior!r}")
    return np.clip(p + move * (asserted - p), 0.0, 1.0)


@dataclass(frozen=True)
class ConfidenceBin:
    lo: float
    hi: float
    n: int
    flip_rate: float


@dataclass(frozen=True)
class PressureReport:
    behavior: Behavior
    strength: float
    n: int
    ece_pre: float
    ece_post: float
    robustness_rate: float          # of originally-correct calls, share whose decision held
    flip_rate_overall: float
    flip_by_confidence: list[ConfidenceBin]


def _flip_by_confidence(proba: np.ndarray, flipped: np.ndarray,
                        edges=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) -> list[ConfidenceBin]:
    conf = np.abs(2.0 * np.asarray(proba, dtype=float) - 1.0)
    out: list[ConfidenceBin] = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        m = (conf >= lo) & (conf <= hi) if hi == edges[-1] else (conf >= lo) & (conf < hi)
        n = int(m.sum())
        out.append(ConfidenceBin(lo, hi, n, float(flipped[m].mean()) if n else float("nan")))
    return out


def pressure_probe(labels: np.ndarray, proba: np.ndarray, *,
                   behavior: Behavior = "coupled", strength: float = 0.4,
                   threshold: float = 0.5, n_bins: int = 10) -> PressureReport:
    """Apply a pushback to (labels, proba) and report calibration + robustness."""
    labels = np.asarray(labels, dtype=int)
    proba = np.asarray(proba, dtype=float)
    if labels.shape != proba.shape:
        raise ValueError(f"labels {labels.shape} != proba {proba.shape}")

    pre = decisions(proba, threshold)
    post_proba = push_back(proba, behavior=behavior, strength=strength)
    post = decisions(post_proba, threshold)

    flipped = pre != post
    correct_pre = pre == labels
    held_correct = (~flipped) & correct_pre
    robustness = held_correct.sum() / correct_pre.sum() if correct_pre.any() else float("nan")

    return PressureReport(
        behavior=behavior,
        strength=strength,
        n=int(labels.size),
        ece_pre=compute_calibration(labels, proba, n_bins=n_bins).ece,
        ece_post=compute_calibration(labels, post_proba, n_bins=n_bins).ece,
        robustness_rate=float(robustness),
        flip_rate_overall=float(flipped.mean()),
        flip_by_confidence=_flip_by_confidence(proba, flipped),
    )
