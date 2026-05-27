"""Unit tests for dmoi_brca.baseline (Day-4 baseline)."""
from __future__ import annotations

import numpy as np

from dmoi_brca.baseline import aggregate, run_cv


def _make_synthetic(n: int = 60, p: int = 20, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    # Two clusters separable on a single dimension.
    y = rng.integers(0, 2, size=n)
    X = rng.standard_normal((n, p)).astype(np.float32)
    X[:, 0] += y * 2.0  # plant signal
    return X, y


def test_run_cv_returns_per_fold_result():
    X, y = _make_synthetic(n=60)
    results = run_cv({"rna": X}, y, n_splits=5, random_state=0, models=("logreg",))
    assert len(results) == 5
    assert all(r.feature_set == "rna" and r.model == "logreg" for r in results)
    assert all(0.0 <= r.auc <= 1.0 for r in results)
    assert all(0.0 <= r.bacc <= 1.0 for r in results)


def test_run_cv_multiple_feature_sets():
    X1, y = _make_synthetic(n=80, p=15, seed=1)
    X2, _ = _make_synthetic(n=80, p=10, seed=2)
    results = run_cv(
        {"a": X1, "b": X2}, y,
        n_splits=3, random_state=0, models=("logreg", "rf"),
    )
    # 2 feature sets * 2 models * 3 folds = 12
    assert len(results) == 12
    keys = {(r.feature_set, r.model) for r in results}
    assert keys == {("a", "logreg"), ("a", "rf"), ("b", "logreg"), ("b", "rf")}


def test_aggregate_shape():
    X, y = _make_synthetic(n=50)
    results = run_cv({"rna": X}, y, n_splits=5, random_state=0, models=("logreg", "rf"))
    agg = aggregate(results)
    assert ("rna", "logreg") in agg
    assert ("rna", "rf") in agg
    for _key, stats in agg.items():
        assert {"auc_mean", "auc_std", "bacc_mean", "bacc_std", "n_folds"} <= set(stats)
        assert stats["n_folds"] == 5
        assert 0.0 <= stats["auc_mean"] <= 1.0


def test_signal_recovery_logreg_beats_chance():
    # With planted signal, logreg should clearly beat 0.5 AUC.
    X, y = _make_synthetic(n=120, p=30, seed=42)
    results = run_cv({"rna": X}, y, n_splits=5, random_state=0, models=("logreg",))
    agg = aggregate(results)
    assert agg[("rna", "logreg")]["auc_mean"] > 0.7
