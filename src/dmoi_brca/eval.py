"""Day-4 analytical evaluation primitives for DMOI POC.

These helpers consume per-fold prediction arrays (val_labels, val_proba,
val_disagreement) captured in `dmoi_brca.train.FoldResult` and produce:

- Per-class metrics (precision / recall / F1 on the LumB minority).
- Expected Calibration Error (ECE) with reliability bins.
- Disagreement-vs-misclassification analysis — point-biserial correlation
  + AUC of the disagreement score as a misclassification predictor.

The strongest test of DMOI's Option-B thesis (disagreement is INFORMATIVE,
not a regularization target) is whether disagreement is statistically
elevated on misclassified cases. If yes, the dual-pole encoder is doing
its job. If no, the disagreement signal is noise and v0.2 should consider
Option A (auxiliary BCE on sub-classifier scores).
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class PerClassMetrics:
    label: str                  # "LumA" or "LumB"
    label_value: int            # 0 or 1
    n_in_fold: int              # support count
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class CalibrationReport:
    n_bins: int
    ece: float                  # Expected Calibration Error
    bin_centers: tuple[float, ...]   # midpoints of each bin
    bin_confidence: tuple[float, ...]   # mean predicted probability per bin
    bin_accuracy: tuple[float, ...]     # observed positive rate per bin
    bin_counts: tuple[int, ...]         # n samples in each bin


@dataclass(frozen=True)
class DisagreementReport:
    n_samples: int
    n_misclassified: int
    misclass_rate: float
    mean_dis_correct: float          # mean disagreement on correctly classified samples
    mean_dis_incorrect: float        # mean disagreement on misclassified samples
    point_biserial_r: float          # correlation between disagreement and misclass indicator
    point_biserial_p: float          # two-sided p-value
    auc_dis_predicts_misclass: float # how well does disagreement rank misclass samples?
    is_informative: bool             # True if mean_dis_incorrect > mean_dis_correct AND p < 0.05


@dataclass
class FoldEvalBundle:
    """Per-fold concatenation of val arrays + computed reports."""
    fold: int
    n_test: int
    labels: np.ndarray
    proba: np.ndarray
    disagreement: np.ndarray
    pred: np.ndarray = field(default=None)  # type: ignore[assignment]
    per_class: dict[str, PerClassMetrics] = field(default_factory=dict)
    calibration: CalibrationReport | None = None
    disagreement_report: DisagreementReport | None = None


# ---------------------------------------------------------------------------
# Per-class metrics
# ---------------------------------------------------------------------------

def compute_per_class_metrics(
    labels: np.ndarray,
    pred: np.ndarray,
    label_names: Sequence[str] = ("LumA", "LumB"),
) -> dict[str, PerClassMetrics]:
    """Per-class precision/recall/F1 for a binary classification.

    label_names[0] corresponds to label=0, label_names[1] to label=1.
    Returns a dict keyed by label name.
    """
    out: dict[str, PerClassMetrics] = {}
    for value, name in enumerate(label_names):
        n = int((labels == value).sum())
        prec = float(precision_score(labels, pred, pos_label=value, zero_division=0))
        rec = float(recall_score(labels, pred, pos_label=value, zero_division=0))
        f1 = float(f1_score(labels, pred, pos_label=value, zero_division=0))
        out[name] = PerClassMetrics(
            label=name, label_value=value, n_in_fold=n,
            precision=prec, recall=rec, f1=f1,
        )
    return out


# ---------------------------------------------------------------------------
# Expected Calibration Error (ECE)
# ---------------------------------------------------------------------------

def compute_calibration(
    labels: np.ndarray,
    proba: np.ndarray,
    n_bins: int = 10,
) -> CalibrationReport:
    """Expected Calibration Error with uniform-width bins on [0, 1].

    For each bin, computes mean predicted probability vs observed positive
    rate. ECE = sum_b (n_b / N) * |conf_b - acc_b|.
    """
    if proba.shape != labels.shape:
        raise ValueError(f"proba shape {proba.shape} != labels shape {labels.shape}")
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    bin_conf = np.zeros(n_bins)
    bin_acc = np.zeros(n_bins)
    bin_n = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        # Include right edge only in the last bin so probabilities of exactly
        # 1.0 don't fall off the histogram.
        mask = (
            (proba >= lo) & (proba <= hi)
            if b == n_bins - 1
            else (proba >= lo) & (proba < hi)
        )
        bin_n[b] = int(mask.sum())
        if bin_n[b] > 0:
            bin_conf[b] = float(proba[mask].mean())
            bin_acc[b] = float(labels[mask].mean())

    total = len(labels)
    ece = float(
        sum(
            (bin_n[b] / total) * abs(bin_conf[b] - bin_acc[b])
            for b in range(n_bins)
            if bin_n[b] > 0
        ),
    )
    return CalibrationReport(
        n_bins=n_bins,
        ece=ece,
        bin_centers=tuple(float(c) for c in centers),
        bin_confidence=tuple(float(c) for c in bin_conf),
        bin_accuracy=tuple(float(a) for a in bin_acc),
        bin_counts=tuple(int(n) for n in bin_n),
    )


# ---------------------------------------------------------------------------
# Disagreement-vs-misclassification analysis
# ---------------------------------------------------------------------------

def _point_biserial_correlation(
    binary: np.ndarray,
    continuous: np.ndarray,
) -> tuple[float, float]:
    """Compute point-biserial correlation (numpy-only, no scipy dependency).

    For binary in {0, 1} and continuous values, returns (r, two-sided p).
    Implemented via the equivalence to Pearson correlation between the
    binary indicator and the continuous variable.
    """
    n = len(binary)
    if n < 3:
        return 0.0, 1.0
    b = binary.astype(np.float64)
    c = continuous.astype(np.float64)
    bm, cm = b.mean(), c.mean()
    cov = ((b - bm) * (c - cm)).sum()
    sb = np.sqrt(((b - bm) ** 2).sum())
    sc = np.sqrt(((c - cm) ** 2).sum())
    if sb == 0 or sc == 0:
        return 0.0, 1.0
    r = float(cov / (sb * sc))
    # Two-sided p-value via t-distribution with n-2 df (asymptotic).
    # t = r * sqrt(n-2) / sqrt(1 - r^2)
    if abs(r) >= 1.0:
        return r, 0.0
    t = r * np.sqrt(n - 2) / np.sqrt(1.0 - r * r)
    # Use a small-N-safe normal approximation for the p-value. Avoids scipy
    # dependency. Acceptable for n > 30 (the cohort_v2 fold size).
    z = abs(t)
    # 2 * P(Z > z) using complementary error function.
    p = float(math.erfc(z / math.sqrt(2.0)))
    return r, max(min(p, 1.0), 0.0)


def compute_disagreement_report(
    labels: np.ndarray,
    pred: np.ndarray,
    disagreement: np.ndarray,
) -> DisagreementReport:
    """Test whether the disagreement score is informative about misclassification.

    Returns a structured report. The two principal numbers:
        mean_dis_incorrect - mean_dis_correct   (positive = signal)
        point_biserial_r                         (positive + significant = signal)
    Plus auc_dis_predicts_misclass which measures whether disagreement
    ranks misclassified samples higher than correctly classified ones
    (independent of any threshold).
    """
    if not (labels.shape == pred.shape == disagreement.shape):
        raise ValueError(
            f"shape mismatch: labels {labels.shape}, pred {pred.shape}, "
            f"disagreement {disagreement.shape}",
        )
    misclass = (labels != pred).astype(np.int64)
    n_misclass = int(misclass.sum())
    if n_misclass == 0:
        return DisagreementReport(
            n_samples=len(labels), n_misclassified=0, misclass_rate=0.0,
            mean_dis_correct=float(disagreement.mean()),
            mean_dis_incorrect=0.0,
            point_biserial_r=0.0, point_biserial_p=1.0,
            auc_dis_predicts_misclass=0.5,
            is_informative=False,
        )

    mean_correct = float(disagreement[misclass == 0].mean()) if (misclass == 0).any() else 0.0
    mean_incorrect = float(disagreement[misclass == 1].mean())
    r, p = _point_biserial_correlation(misclass, disagreement)

    # AUC of disagreement predicting misclassification (rank-based, threshold-free).
    auc_dis = (
        float(roc_auc_score(misclass, disagreement))
        if 0 < n_misclass < len(labels)
        else 0.5
    )

    is_informative = (mean_incorrect > mean_correct) and (p < 0.05)
    return DisagreementReport(
        n_samples=len(labels), n_misclassified=n_misclass,
        misclass_rate=float(n_misclass / len(labels)),
        mean_dis_correct=mean_correct, mean_dis_incorrect=mean_incorrect,
        point_biserial_r=r, point_biserial_p=p,
        auc_dis_predicts_misclass=auc_dis,
        is_informative=is_informative,
    )


# ---------------------------------------------------------------------------
# Per-fold bundle + aggregate helpers
# ---------------------------------------------------------------------------

def build_fold_eval_bundle(
    fold: int,
    labels: np.ndarray,
    proba: np.ndarray,
    disagreement: np.ndarray,
    *,
    threshold: float = 0.5,
    n_calibration_bins: int = 10,
    label_names: Sequence[str] = ("LumA", "LumB"),
) -> FoldEvalBundle:
    """Bundle per-class metrics + calibration + disagreement report for one fold."""
    pred = (proba >= threshold).astype(np.int64)
    bundle = FoldEvalBundle(
        fold=fold, n_test=len(labels),
        labels=labels, proba=proba, disagreement=disagreement, pred=pred,
    )
    bundle.per_class = compute_per_class_metrics(labels, pred, label_names=label_names)
    bundle.calibration = compute_calibration(labels, proba, n_bins=n_calibration_bins)
    bundle.disagreement_report = compute_disagreement_report(labels, pred, disagreement)
    return bundle


def aggregate_cross_fold(
    bundles: Sequence[FoldEvalBundle],
    label_names: Sequence[str] = ("LumA", "LumB"),
) -> dict[str, float]:
    """Mean + std across folds for headline metrics."""
    if not bundles:
        raise ValueError("aggregate_cross_fold: empty bundle list")

    out: dict[str, float] = {"n_folds": len(bundles)}

    for name in label_names:
        f1s = np.array([b.per_class[name].f1 for b in bundles])
        precs = np.array([b.per_class[name].precision for b in bundles])
        recs = np.array([b.per_class[name].recall for b in bundles])
        out[f"f1_{name}_mean"] = float(f1s.mean())
        out[f"f1_{name}_std"] = float(f1s.std(ddof=1)) if len(f1s) > 1 else 0.0
        out[f"precision_{name}_mean"] = float(precs.mean())
        out[f"recall_{name}_mean"] = float(recs.mean())

    eces = np.array([b.calibration.ece for b in bundles if b.calibration is not None])
    out["ece_mean"] = float(eces.mean())
    out["ece_std"] = float(eces.std(ddof=1)) if len(eces) > 1 else 0.0

    dis_aucs = np.array(
        [b.disagreement_report.auc_dis_predicts_misclass
         for b in bundles if b.disagreement_report is not None],
    )
    out["auc_dis_predicts_misclass_mean"] = float(dis_aucs.mean())
    out["auc_dis_predicts_misclass_std"] = float(dis_aucs.std(ddof=1)) if len(dis_aucs) > 1 else 0.0

    return out


def concat_fold_predictions(
    bundles: Sequence[FoldEvalBundle],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate per-fold val predictions into pooled arrays for OOF analysis."""
    if not bundles:
        raise ValueError("concat_fold_predictions: empty bundles")
    labels = np.concatenate([b.labels for b in bundles])
    proba = np.concatenate([b.proba for b in bundles])
    dis = np.concatenate([b.disagreement for b in bundles])
    return labels, proba, dis


def confusion_matrix_table(labels: np.ndarray, pred: np.ndarray) -> dict[str, int]:
    """Binary confusion matrix as a flat dict (tn / fp / fn / tp)."""
    cm = confusion_matrix(labels, pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


# ---------------------------------------------------------------------------
# Calibration extras (v0.13): Brier score + reliability-table export
# ---------------------------------------------------------------------------

def brier_score(labels: np.ndarray, proba: np.ndarray) -> float:
    """Brier score = mean squared error of probabilistic predictions.

    For binary labels in {0, 1}, lower is better (0 is perfect). Reported
    alongside ECE because ECE is bin-sensitive (a binned summary) while the
    Brier score is a strictly proper scoring rule evaluated per sample. Equal
    to ``sklearn.metrics.brier_score_loss`` for binary labels.
    """
    labels = np.asarray(labels, dtype=np.float64)
    proba = np.asarray(proba, dtype=np.float64)
    if proba.shape != labels.shape:
        raise ValueError(f"proba shape {proba.shape} != labels shape {labels.shape}")
    return float(np.mean((proba - labels) ** 2))


@dataclass(frozen=True)
class ReliabilityBin:
    """One reliability-curve bin: x=confidence, y=accuracy, with its count."""

    center: float       # bin midpoint on [0, 1]
    confidence: float   # mean predicted probability in the bin
    accuracy: float     # observed positive rate in the bin
    count: int          # n samples in the bin


def reliability_table(
    labels: np.ndarray,
    proba: np.ndarray,
    n_bins: int = 10,
) -> list[ReliabilityBin]:
    """Per-bin reliability rows for curve / TSV export.

    Thin wrapper over :func:`compute_calibration` so callers (e.g.
    ``scripts/calibrate_transfer.py``) get ready-to-write rows with the exact
    same binning ECE uses, instead of re-zipping the report tuples by hand.
    """
    rep = compute_calibration(labels, proba, n_bins=n_bins)
    return [
        ReliabilityBin(center=c, confidence=conf, accuracy=acc, count=cnt)
        for c, conf, acc, cnt in zip(
            rep.bin_centers,
            rep.bin_confidence,
            rep.bin_accuracy,
            rep.bin_counts,
            strict=False,
        )
    ]
