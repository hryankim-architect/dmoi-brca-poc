"""Unit tests for dmoi_brca.train (Day-3 training loop)."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dmoi_brca.hypothesis_attention import PoleMaskSet  # noqa: E402
from dmoi_brca.train import (  # noqa: E402
    FoldResult,
    aggregate_fold_results,
    run_dmoi_cv,
    train_one_fold,
)


def _synthetic_dataset(
    n: int = 60,
    n_rna: int = 12,
    n_meth: int = 10,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, PoleMaskSet]]:
    """Small linearly-separable dataset so 5-fold CV converges fast."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    rna = rng.standard_normal((n, n_rna)).astype(np.float32)
    meth = rng.standard_normal((n, n_meth)).astype(np.float32)
    # Plant a per-class signal in both modalities so the model can learn.
    rna[:, 0] += y * 2.0
    meth[:, 0] += y * 2.0

    # Synthetic pole masks (disjoint first/second halves).
    half_rna = n_rna // 2
    half_meth = n_meth // 2
    luma_rna = torch.cat([torch.ones(half_rna), torch.zeros(n_rna - half_rna)])
    lumb_rna = torch.cat([torch.zeros(half_rna), torch.ones(n_rna - half_rna)])
    luma_meth = torch.cat([torch.ones(half_meth), torch.zeros(n_meth - half_meth)])
    lumb_meth = torch.cat([torch.zeros(half_meth), torch.ones(n_meth - half_meth)])
    pole_masks = {
        "LumA": PoleMaskSet("LumA", luma_rna, luma_meth),
        "LumB": PoleMaskSet("LumB", lumb_rna, lumb_meth),
    }
    return rna, meth, y, pole_masks


def test_train_one_fold_returns_fold_result():
    rna, meth, y, pole_masks = _synthetic_dataset(n=60)
    half = len(y) // 2
    result = train_one_fold(
        rna_train=rna[:half], meth_train=meth[:half], y_train=y[:half],
        rna_val=rna[half:], meth_val=meth[half:], y_val=y[half:],
        pole_masks=pole_masks,
        fold=1,
        rna_dim=rna.shape[1], meth_dim=meth.shape[1],
        latent_dim=8, rna_hidden=(16,), meth_hidden=(16,),
        fuse_hidden=(8,), fuse_out=4, head_hidden=4, dropout=0.0,
        n_epochs=5, batch_size=8, lr=1e-3, weight_decay=0.0,
        patience=10, seed=0, device="cpu", verbose=False,
    )
    assert isinstance(result, FoldResult)
    assert result.fold == 1
    assert 0.0 <= result.best_val_auc <= 1.0
    assert 0.0 <= result.best_val_bacc <= 1.0
    assert result.best_epoch >= 1
    assert len(result.train_loss_curve) >= 1
    assert len(result.val_auc_curve) == len(result.train_loss_curve)


def test_train_one_fold_loss_finite():
    rna, meth, y, pole_masks = _synthetic_dataset(n=40)
    half = len(y) // 2
    result = train_one_fold(
        rna_train=rna[:half], meth_train=meth[:half], y_train=y[:half],
        rna_val=rna[half:], meth_val=meth[half:], y_val=y[half:],
        pole_masks=pole_masks,
        fold=1,
        rna_dim=rna.shape[1], meth_dim=meth.shape[1],
        latent_dim=8, rna_hidden=(16,), meth_hidden=(16,),
        fuse_hidden=(8,), fuse_out=4, head_hidden=4, dropout=0.0,
        n_epochs=3, batch_size=8, lr=1e-3, weight_decay=0.0,
        patience=10, seed=0, device="cpu", verbose=False,
    )
    assert all(np.isfinite(result.train_loss_curve))
    assert all(np.isfinite(result.val_auc_curve))


def test_run_dmoi_cv_yields_five_folds():
    rna, meth, y, pole_masks = _synthetic_dataset(n=80)
    results = run_dmoi_cv(
        rna=rna, meth=meth, y=y, pole_masks=pole_masks,
        n_splits=5, random_state=42,
        latent_dim=8, rna_hidden=(16,), meth_hidden=(16,),
        fuse_hidden=(8,), fuse_out=4, head_hidden=4, dropout=0.0,
        n_epochs=3, batch_size=16, lr=1e-3, weight_decay=0.0,
        patience=10, seed=0, device="cpu", verbose=False,
    )
    assert len(results) == 5
    assert all(isinstance(r, FoldResult) for r in results)
    assert {r.fold for r in results} == {1, 2, 3, 4, 5}


def test_aggregate_fold_results_shape():
    rna, meth, y, pole_masks = _synthetic_dataset(n=50)
    results = run_dmoi_cv(
        rna=rna, meth=meth, y=y, pole_masks=pole_masks,
        n_splits=5, random_state=42,
        latent_dim=8, rna_hidden=(16,), meth_hidden=(16,),
        fuse_hidden=(8,), fuse_out=4, head_hidden=4, dropout=0.0,
        n_epochs=2, batch_size=8, lr=1e-3, weight_decay=0.0,
        patience=10, seed=0, device="cpu", verbose=False,
    )
    agg = aggregate_fold_results(results)
    assert set(agg) >= {
        "auc_mean", "auc_std", "bacc_mean", "bacc_std",
        "epoch_mean", "epoch_max", "runtime_sec_total", "n_folds",
    }
    assert agg["n_folds"] == 5
    assert 0.0 <= agg["auc_mean"] <= 1.0


def test_run_dmoi_cv_mismatched_shapes_raises():
    rna, meth, y, pole_masks = _synthetic_dataset(n=40)
    bad_y = y[:30]
    with pytest.raises(ValueError, match="mismatched on axis 0"):
        run_dmoi_cv(
            rna=rna, meth=meth, y=bad_y, pole_masks=pole_masks,
            n_splits=5, random_state=42,
            verbose=False,
        )


def test_calibration_split_carves_stratified_holdout():
    """With calibration_frac=0.2, ~20% of train is held out and exposed via cal_*."""
    rna, meth, y, pole_masks = _synthetic_dataset(n=80, seed=11)
    half = len(y) // 2
    result = train_one_fold(
        rna_train=rna[:half], meth_train=meth[:half], y_train=y[:half],
        rna_val=rna[half:], meth_val=meth[half:], y_val=y[half:],
        pole_masks=pole_masks,
        fold=1,
        rna_dim=rna.shape[1], meth_dim=meth.shape[1],
        latent_dim=8, rna_hidden=(16,), meth_hidden=(16,),
        fuse_hidden=(8,), fuse_out=4, head_hidden=4, dropout=0.0,
        n_epochs=3, batch_size=8, lr=1e-3, weight_decay=0.0,
        patience=10, seed=0, device="cpu", verbose=False,
        calibration_frac=0.2,
    )
    n_train = half
    expected_cal = max(1, int(round(0.2 * (n_train // 2)))) * 2  # both classes
    assert result.n_cal > 0
    # Within +/- 2 of the analytic expectation (rounding per class).
    assert abs(result.n_cal - expected_cal) <= 2
    assert result.cal_labels is not None
    assert result.cal_logits is not None
    assert result.cal_logits.shape == (result.n_cal,)
    assert result.cal_labels.shape == (result.n_cal,)
    # Stratification: both classes present in cal split.
    assert (result.cal_labels == 0).any()
    assert (result.cal_labels == 1).any()
    # Training size shrank by the cal holdout.
    assert result.n_train == n_train - result.n_cal


def test_calibration_split_default_is_zero():
    """Without calibration_frac, cal arrays stay None / n_cal == 0."""
    rna, meth, y, pole_masks = _synthetic_dataset(n=40)
    half = len(y) // 2
    result = train_one_fold(
        rna_train=rna[:half], meth_train=meth[:half], y_train=y[:half],
        rna_val=rna[half:], meth_val=meth[half:], y_val=y[half:],
        pole_masks=pole_masks,
        fold=1,
        rna_dim=rna.shape[1], meth_dim=meth.shape[1],
        latent_dim=8, rna_hidden=(16,), meth_hidden=(16,),
        fuse_hidden=(8,), fuse_out=4, head_hidden=4, dropout=0.0,
        n_epochs=2, batch_size=8, lr=1e-3, weight_decay=0.0,
        patience=10, seed=0, device="cpu", verbose=False,
    )
    assert result.n_cal == 0
    assert result.cal_labels is None
    assert result.cal_logits is None


def test_calibration_split_rejects_invalid_frac():
    rna, meth, y, pole_masks = _synthetic_dataset(n=40)
    half = len(y) // 2
    with pytest.raises(ValueError, match="calibration_frac"):
        train_one_fold(
            rna_train=rna[:half], meth_train=meth[:half], y_train=y[:half],
            rna_val=rna[half:], meth_val=meth[half:], y_val=y[half:],
            pole_masks=pole_masks,
            fold=1,
            rna_dim=rna.shape[1], meth_dim=meth.shape[1],
            latent_dim=8, rna_hidden=(16,), meth_hidden=(16,),
            fuse_hidden=(8,), fuse_out=4, head_hidden=4, dropout=0.0,
            n_epochs=2, batch_size=8, lr=1e-3, weight_decay=0.0,
            patience=10, seed=0, device="cpu", verbose=False,
            calibration_frac=0.6,  # too large
        )


def test_signal_recovery_smoke():
    """With strong planted signal, 5-fold CV mean AUROC must beat 0.5."""
    rna, meth, y, pole_masks = _synthetic_dataset(n=80, seed=7)
    results = run_dmoi_cv(
        rna=rna, meth=meth, y=y, pole_masks=pole_masks,
        n_splits=5, random_state=0,
        latent_dim=8, rna_hidden=(16,), meth_hidden=(16,),
        fuse_hidden=(8,), fuse_out=4, head_hidden=4, dropout=0.0,
        n_epochs=10, batch_size=16, lr=5e-3, weight_decay=0.0,
        patience=10, seed=0, device="cpu", verbose=False,
    )
    agg = aggregate_fold_results(results)
    # Loose: anything materially above 0.5 confirms training is working.
    assert agg["auc_mean"] > 0.6
