"""Cross-cohort calibration-transfer transforms (v0.13).

Two label-free, deterministic, **monotonic** transforms used by
`scripts/calibrate_transfer.py` to attempt TCGA -> METABRIC calibration
without a labelled target calibration set:

- ``affine_align``  (D2): match the target logit distribution's mean/std to a
  source (TCGA val) distribution before applying the source temperature. Pure
  distribution alignment; uses no target labels.
- ``prior_odds_correct`` (D3): shift probabilities by the class-prior odds ratio
  between the training cohort and the target cohort. Uses cohort base rates
  only, no per-sample target labels.

Both are monotonic in their input, so AUROC / balanced accuracy are invariant
under them — they can only change *calibration*, never ranking. Keeping them as
small importable functions (rather than inline in the script) makes them unit
testable without the torch-dependent model pipeline.
"""
from __future__ import annotations

import numpy as np


def affine_align(
    logits: np.ndarray,
    *,
    src_mean: float,
    src_std: float,
    dst_mean: float,
    dst_std: float,
) -> np.ndarray:
    """Affine-map ``logits`` from a source distribution onto a target one.

    Standardize by the source (mean/std) then rescale to the target (mean/std):
    ``(logits - src_mean) / src_std * dst_std + dst_mean``. Monotonic increasing
    as long as ``src_std > 0`` (std is clamped by the caller with a tiny eps).

    In v0.13 the "source" is the METABRIC logit distribution and the "target" is
    the TCGA val distribution whose temperature we want to import.
    """
    src_std = float(src_std)
    if src_std <= 0:
        raise ValueError(f"src_std must be > 0, got {src_std}")
    logits = np.asarray(logits, dtype=np.float64)
    return (logits - float(src_mean)) / src_std * float(dst_std) + float(dst_mean)


def prior_odds_correct(
    proba: np.ndarray,
    *,
    pi_train: float,
    pi_target: float,
) -> np.ndarray:
    """Re-weight probabilities by the train->target class-prior odds ratio.

    Given a model trained at positive base rate ``pi_train`` and a target cohort
    with positive base rate ``pi_target``, shift each probability by the
    prior-odds ratio (Bayes base-rate correction):

        p' = p * r_pos / (p * r_pos + (1 - p) * r_neg)

    where ``r_pos = pi_target / pi_train`` and
    ``r_neg = (1 - pi_target) / (1 - pi_train)``. Monotonic in ``p``, so AUROC is
    invariant; it only rescales the probability axis to the target base rate.
    """
    for name, v in (("pi_train", pi_train), ("pi_target", pi_target)):
        if not 0.0 < float(v) < 1.0:
            raise ValueError(f"{name} must be in (0, 1), got {v}")
    proba = np.asarray(proba, dtype=np.float64)
    r_pos = float(pi_target) / float(pi_train)
    r_neg = (1.0 - float(pi_target)) / (1.0 - float(pi_train))
    num = proba * r_pos
    return num / np.maximum(num + (1.0 - proba) * r_neg, 1e-12)
