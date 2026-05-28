"""Unit tests for dmoi_brca.attribution (v0.3 Integrated Gradients)."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
captum = pytest.importorskip("captum")  # noqa: F841

from dmoi_brca.attribution import (  # noqa: E402
    AttributionResult,
    completeness_residual,
    global_aggregate,
    integrated_gradients_dmoi,
    top_k_per_patient,
)
from dmoi_brca.dmoi_model import DMOIModel  # noqa: E402
from dmoi_brca.hypothesis_attention import PoleMaskSet  # noqa: E402


def _build_tiny_model(seed: int = 0) -> tuple[DMOIModel, dict, int, int]:
    """Tiny but real DMOIModel for attribution smoke tests."""
    torch.manual_seed(seed)
    n_rna, n_meth = 12, 10
    half_rna = n_rna // 2
    half_meth = n_meth // 2
    pole_masks = {
        "LumA": PoleMaskSet(
            "LumA",
            torch.cat([torch.ones(half_rna), torch.zeros(n_rna - half_rna)]),
            torch.cat([torch.ones(half_meth), torch.zeros(n_meth - half_meth)]),
        ),
        "LumB": PoleMaskSet(
            "LumB",
            torch.cat([torch.zeros(half_rna), torch.ones(n_rna - half_rna)]),
            torch.cat([torch.zeros(half_meth), torch.ones(n_meth - half_meth)]),
        ),
    }
    model = DMOIModel(
        rna_dim=n_rna, meth_dim=n_meth, pole_masks=pole_masks,
        latent_dim=8, rna_hidden=(16,), meth_hidden=(16,),
        fuse_hidden=(8,), fuse_out=4, head_hidden=4, dropout=0.0,
        use_disagreement=True,
    )
    model.eval()
    return model, pole_masks, n_rna, n_meth


# ---------------------------------------------------------------------------
# Smoke / shape / determinism
# ---------------------------------------------------------------------------


def test_integrated_gradients_runs_and_returns_correct_shapes():
    model, _, n_rna, n_meth = _build_tiny_model()
    rng = np.random.default_rng(0)
    n = 4
    rna = rng.normal(0, 1, (n, n_rna)).astype(np.float32)
    meth = rng.normal(0, 1, (n, n_meth)).astype(np.float32)
    attr = integrated_gradients_dmoi(model, rna, meth, target="final_logit",
                                     n_steps=20)
    assert isinstance(attr, AttributionResult)
    assert attr.rna_attribution.shape == (n, n_rna)
    assert attr.meth_attribution.shape == (n, n_meth)
    assert attr.target_score.shape == (n,)
    assert attr.f_x.shape == (n,)
    assert attr.f_baseline.shape == (n,)
    assert attr.n_steps == 20
    assert attr.target_name == "final_logit"


def test_integrated_gradients_supports_all_three_targets():
    model, _, n_rna, n_meth = _build_tiny_model()
    rng = np.random.default_rng(1)
    rna = rng.normal(0, 1, (3, n_rna)).astype(np.float32)
    meth = rng.normal(0, 1, (3, n_meth)).astype(np.float32)
    for target in ("final_logit", "lumA_pole", "lumB_pole"):
        attr = integrated_gradients_dmoi(model, rna, meth, target=target, n_steps=10)
        assert attr.target_name == target
        assert attr.rna_attribution.shape == (3, n_rna)


def test_integrated_gradients_rejects_unknown_target():
    model, _, n_rna, n_meth = _build_tiny_model()
    rng = np.random.default_rng(2)
    rna = rng.normal(0, 1, (2, n_rna)).astype(np.float32)
    meth = rng.normal(0, 1, (2, n_meth)).astype(np.float32)
    with pytest.raises(ValueError, match="unknown target_name"):
        integrated_gradients_dmoi(model, rna, meth, target="bogus", n_steps=5)


def test_integrated_gradients_rejects_shape_mismatch():
    model, _, n_rna, n_meth = _build_tiny_model()
    rng = np.random.default_rng(3)
    rna = rng.normal(0, 1, (3, n_rna)).astype(np.float32)
    meth = rng.normal(0, 1, (4, n_meth)).astype(np.float32)
    with pytest.raises(ValueError, match="rna rows"):
        integrated_gradients_dmoi(model, rna, meth, n_steps=5)


def test_integrated_gradients_rejects_bad_n_steps():
    model, _, n_rna, n_meth = _build_tiny_model()
    rng = np.random.default_rng(4)
    rna = rng.normal(0, 1, (2, n_rna)).astype(np.float32)
    meth = rng.normal(0, 1, (2, n_meth)).astype(np.float32)
    with pytest.raises(ValueError, match="n_steps"):
        integrated_gradients_dmoi(model, rna, meth, n_steps=1)


def test_integrated_gradients_deterministic():
    model, _, n_rna, n_meth = _build_tiny_model()
    rng = np.random.default_rng(5)
    rna = rng.normal(0, 1, (2, n_rna)).astype(np.float32)
    meth = rng.normal(0, 1, (2, n_meth)).astype(np.float32)
    a1 = integrated_gradients_dmoi(model, rna, meth, n_steps=30)
    a2 = integrated_gradients_dmoi(model, rna, meth, n_steps=30)
    np.testing.assert_allclose(a1.rna_attribution, a2.rna_attribution, atol=1e-6)
    np.testing.assert_allclose(a1.meth_attribution, a2.meth_attribution, atol=1e-6)


# ---------------------------------------------------------------------------
# Completeness axiom
# ---------------------------------------------------------------------------


def test_completeness_axiom_within_tolerance():
    """IG completeness: sum_i IG_i(x) ≈ f(x) - f(baseline)."""
    model, _, n_rna, n_meth = _build_tiny_model()
    rng = np.random.default_rng(6)
    rna = rng.normal(0, 1, (3, n_rna)).astype(np.float32)
    meth = rng.normal(0, 1, (3, n_meth)).astype(np.float32)
    # Use many steps for a tighter Riemann approximation.
    attr = integrated_gradients_dmoi(model, rna, meth, n_steps=200)
    residuals = completeness_residual(attr)
    # 1e-2 tolerance is the standard IG completeness bar with 200 steps on
    # a small ReLU-bearing model. Captum's own tests use 5e-2.
    assert (residuals < 1e-2).all(), f"completeness residuals too large: {residuals}"


# ---------------------------------------------------------------------------
# top_k_per_patient + global_aggregate
# ---------------------------------------------------------------------------


def test_top_k_per_patient_shapes():
    model, _, n_rna, n_meth = _build_tiny_model()
    rng = np.random.default_rng(7)
    rna = rng.normal(0, 1, (3, n_rna)).astype(np.float32)
    meth = rng.normal(0, 1, (3, n_meth)).astype(np.float32)
    attr = integrated_gradients_dmoi(model, rna, meth, n_steps=10)
    rna_names = [f"RNA{i:02d}" for i in range(n_rna)]
    meth_names = [f"cg{i:06d}" for i in range(n_meth)]
    out = top_k_per_patient(attr, rna_names, meth_names, k=5)
    assert len(out) == 3
    for row in out:
        assert len(row["topk_rna"]) == 5
        assert len(row["topk_meth"]) == 5
        # First entry's |attribution| >= second entry's |attribution|
        for series in (row["topk_rna"], row["topk_meth"]):
            for a, b in zip(series, series[1:], strict=False):
                assert abs(a[1]) >= abs(b[1])


def test_top_k_per_patient_rejects_name_length_mismatch():
    model, _, n_rna, n_meth = _build_tiny_model()
    rng = np.random.default_rng(8)
    rna = rng.normal(0, 1, (2, n_rna)).astype(np.float32)
    meth = rng.normal(0, 1, (2, n_meth)).astype(np.float32)
    attr = integrated_gradients_dmoi(model, rna, meth, n_steps=5)
    with pytest.raises(ValueError, match="rna_feature_names"):
        top_k_per_patient(attr, ["onlyone"], ["cg" * n_meth], k=2)


def test_global_aggregate_returns_ranked():
    model, _, n_rna, n_meth = _build_tiny_model()
    rng = np.random.default_rng(9)
    rna = rng.normal(0, 1, (5, n_rna)).astype(np.float32)
    meth = rng.normal(0, 1, (5, n_meth)).astype(np.float32)
    attr = integrated_gradients_dmoi(model, rna, meth, n_steps=10)
    rna_names = [f"RNA{i:02d}" for i in range(n_rna)]
    meth_names = [f"cg{i:06d}" for i in range(n_meth)]
    out = global_aggregate(attr, rna_names, meth_names, top_k=4)
    assert len(out["rna"]) == 4
    assert len(out["meth"]) == 4
    # Ranked descending by mean |attribution|.
    for series in (out["rna"], out["meth"]):
        for a, b in zip(series, series[1:], strict=False):
            assert a[1] >= b[1]


# ---------------------------------------------------------------------------
# Direction sanity — single-feature dominance
# ---------------------------------------------------------------------------


def test_attribution_direction_sanity():
    """In a tiny model where one RNA feature dominates after training,
    its IG attribution magnitude should exceed the others on average."""
    rng = np.random.default_rng(123)
    # Train a tiny linear-ish model to associate y with rna[:, 0].
    n = 80
    n_rna, n_meth = 6, 4
    y = rng.integers(0, 2, n).astype(np.float32)
    rna = rng.normal(0, 0.5, (n, n_rna)).astype(np.float32)
    rna[:, 0] += y * 3.0  # strong signal on feature 0
    meth = rng.normal(0, 0.5, (n, n_meth)).astype(np.float32)

    half_rna, half_meth = n_rna // 2, n_meth // 2
    pole_masks = {
        "LumA": PoleMaskSet(
            "LumA",
            torch.cat([torch.ones(half_rna), torch.zeros(n_rna - half_rna)]),
            torch.cat([torch.ones(half_meth), torch.zeros(n_meth - half_meth)]),
        ),
        "LumB": PoleMaskSet(
            "LumB",
            torch.cat([torch.zeros(half_rna), torch.ones(n_rna - half_rna)]),
            torch.cat([torch.zeros(half_meth), torch.ones(n_meth - half_meth)]),
        ),
    }
    model = DMOIModel(
        rna_dim=n_rna, meth_dim=n_meth, pole_masks=pole_masks,
        latent_dim=4, rna_hidden=(8,), meth_hidden=(8,),
        fuse_hidden=(4,), fuse_out=2, head_hidden=2, dropout=0.0,
    )
    # Quick supervised loop.
    opt = torch.optim.AdamW(model.parameters(), lr=5e-2)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    rna_t = torch.from_numpy(rna)
    meth_t = torch.from_numpy(meth)
    y_t = torch.from_numpy(y)
    for _ in range(80):
        opt.zero_grad()
        out = model(rna_t, meth_t)
        loss = loss_fn(out["logits"], y_t)
        loss.backward()
        opt.step()

    # Now attribute on a held-out batch.
    rna_eval = rng.normal(0, 0.5, (20, n_rna)).astype(np.float32)
    rna_eval[:, 0] += np.random.default_rng(7).integers(0, 2, 20) * 3.0
    meth_eval = rng.normal(0, 0.5, (20, n_meth)).astype(np.float32)
    attr = integrated_gradients_dmoi(
        model, rna_eval, meth_eval, target="final_logit", n_steps=50,
    )
    mean_abs = np.abs(attr.rna_attribution).mean(axis=0)
    # Feature 0 (the signal) should have the largest mean |attribution|
    # among the LumA-side RNA features (indices 0..2 after masking).
    luma_features = mean_abs[:half_rna]
    assert luma_features.argmax() == 0, (
        f"feature 0 should dominate LumA-side attribution, got argmax="
        f"{luma_features.argmax()}, magnitudes={luma_features}"
    )
