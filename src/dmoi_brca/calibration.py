"""Temperature scaling for binary classifier calibration (Guo et al. 2017).

Single-parameter post-hoc calibration: learn one scalar T > 0 such that
calibrated_proba = sigmoid(logits / T). T = 1 preserves the model; T > 1
softens overconfident predictions; T < 1 sharpens under-confident ones.

Standard practice fits T on a held-out calibration split. For the DMOI POC
v0.1 we fit T directly on each val fold's logits and report ECE before
and after. This is optimistic (T was tuned to the very fold we measure
on), but it provides an upper bound on what calibration can buy on this
cohort with this architecture. v0.2+ should fit T on a nested calibration
split carved out of the train fold.

Typical interpretation:
    T = 1.0 ± 0.1    well-calibrated already, scaling does nothing useful
    T = 1.5 to 3.0   moderately overconfident model, T > 1 softens
    T > 5.0          severely overconfident; check for label noise or
                     data leakage before trusting the calibrated output
    T < 1.0          under-confident; T < 1 sharpens logits. Less common
                     than over-confidence, but is the regime DMOI POC v0.1
                     actually falls into (cohort_v2 5-fold mean T ~= 0.56)
                     -- likely because the dual-pole sub-classifiers and
                     class-balanced BCE pos_weight together pull logits
                     toward zero, leaving the model's ranking strong
                     (AUROC ~= 0.97) but its probability scale compressed
                     toward 0.5.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

# Temperature is clamped to this range after fitting so a degenerate optimizer
# step (e.g. on a near-separable / extreme input) cannot return T -> 0 or a
# non-finite value and then blow up apply_temperature downstream. The bounds
# are wide enough never to bind on real DMOI logits (observed T ~ 0.4 - 1.0).
T_MIN = 0.05
T_MAX = 100.0


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid (no exp overflow for large |x|).

    Uses exp(-|x|) (argument always <= 0, so it never overflows) and the
    algebraically equivalent branch per sign of x.
    """
    x = np.asarray(x, dtype=np.float64)
    z = np.exp(-np.abs(x))
    return np.where(x >= 0, 1.0 / (1.0 + z), z / (1.0 + z))


@dataclass(frozen=True)
class CalibrationFit:
    temperature: float
    nll_before: float          # negative log-likelihood at T = 1 (uncalibrated)
    nll_after: float           # NLL after fitting T
    converged: bool
    n_iter: int


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    lr: float = 0.01,
    max_iter: int = 200,
    tol: float = 1e-6,
    init_temperature: float = 1.0,
) -> CalibrationFit:
    """Fit a single scalar temperature on (logits, labels) by minimizing BCE NLL.

    Args:
        logits:           Shape (n,) — model's pre-sigmoid logits.
        labels:           Shape (n,) — binary labels in {0, 1}.
        lr:               LBFGS learning rate.
        max_iter:         Max LBFGS iterations.
        tol:              NLL improvement tolerance for early stop.
        init_temperature: Initial T value.

    Returns:
        CalibrationFit with fitted T + before/after NLL.
    """
    if logits.shape != labels.shape:
        raise ValueError(
            f"logits shape {logits.shape} != labels shape {labels.shape}",
        )
    if logits.ndim != 1:
        raise ValueError(f"logits must be 1-D, got shape {logits.shape}")

    logits_t = torch.from_numpy(logits.astype(np.float32))
    labels_t = torch.from_numpy(labels.astype(np.float32))

    # T is parameterized as exp(log_T) to keep T > 0 without constraints.
    log_T = torch.nn.Parameter(
        torch.tensor(float(np.log(init_temperature)), dtype=torch.float32),
    )

    loss_fn = nn.BCEWithLogitsLoss()

    # NLL at T = 1.
    with torch.no_grad():
        nll_before = float(loss_fn(logits_t, labels_t).item())

    optimizer = torch.optim.LBFGS([log_T], lr=lr, max_iter=max_iter, tolerance_grad=tol)
    iter_count = [0]
    last_loss = [float("inf")]

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        T = log_T.exp()
        scaled = logits_t / T
        loss = loss_fn(scaled, labels_t)
        loss.backward()
        iter_count[0] += 1
        last_loss[0] = float(loss.item())
        return loss

    optimizer.step(closure)
    T_raw = float(log_T.exp().item())
    # Guard a degenerate fit: non-finite, or T collapsing toward 0 on a
    # near-separable input. Clamp to [T_MIN, T_MAX] and recompute the NLL so
    # the reported value matches the temperature actually returned.
    if not np.isfinite(T_raw):
        T_raw = 1.0
    T_final = float(np.clip(T_raw, T_MIN, T_MAX))
    with torch.no_grad():
        nll_after = float(loss_fn(logits_t / T_final, labels_t).item())

    converged = abs(nll_before - nll_after) > tol  # at least some movement
    return CalibrationFit(
        temperature=T_final,
        nll_before=nll_before,
        nll_after=nll_after,
        converged=converged,
        n_iter=iter_count[0],
    )


def apply_temperature(
    logits: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Apply a fitted temperature to logits → calibrated probabilities.

    calibrated_proba = sigmoid(logits / T).
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    return _sigmoid(scaled)


def calibrate_fold(
    logits: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, CalibrationFit]:
    """Convenience: fit T on (logits, labels) and return (calibrated_proba, fit)."""
    fit = fit_temperature(logits, labels)
    calibrated = apply_temperature(logits, fit.temperature)
    return calibrated, fit
