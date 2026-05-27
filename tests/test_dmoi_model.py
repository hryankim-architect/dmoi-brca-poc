"""Unit tests for dmoi_brca.dmoi_model (Day-2 end-to-end model)."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from dmoi_brca.dmoi_model import (  # noqa: E402
    ClassifierHead,
    DMOIModel,
    count_dmoi_parameters,
)
from dmoi_brca.hypothesis_attention import PoleMaskSet  # noqa: E402

# -----------------------------------------------------------------
# Small synthetic pole-mask fixtures (avoid real HM450/Hallmark deps)
# -----------------------------------------------------------------

def _make_synthetic_pole_masks(n_rna: int = 8, n_meth: int = 6) -> dict[str, PoleMaskSet]:
    """Build deterministic synthetic pole masks for fast tests.

    LumA mask: first half of features on; LumB mask: second half on. The
    two masks are intentionally disjoint to verify pole differentiation.
    """
    half_rna = n_rna // 2
    half_meth = n_meth // 2
    luma_rna = torch.cat([torch.ones(half_rna), torch.zeros(n_rna - half_rna)])
    lumb_rna = torch.cat([torch.zeros(half_rna), torch.ones(n_rna - half_rna)])
    luma_meth = torch.cat([torch.ones(half_meth), torch.zeros(n_meth - half_meth)])
    lumb_meth = torch.cat([torch.zeros(half_meth), torch.ones(n_meth - half_meth)])
    return {
        "LumA": PoleMaskSet("LumA", luma_rna, luma_meth),
        "LumB": PoleMaskSet("LumB", lumb_rna, lumb_meth),
    }


# -----------------------------------------------------------------
# ClassifierHead unit tests
# -----------------------------------------------------------------

def test_classifier_head_forward_shape():
    head = ClassifierHead(fuse_dim=8, hidden=16, dropout=0.0)
    batch = 4
    z_luma = torch.randn(batch, 8)
    z_lumb = torch.randn(batch, 8)
    d = torch.rand(batch)
    out = head(z_luma, z_lumb, d)
    assert out.shape == (batch,)


def test_classifier_head_gradient_flow():
    head = ClassifierHead(fuse_dim=8, hidden=16, dropout=0.0)
    z_luma = torch.randn(2, 8, requires_grad=True)
    z_lumb = torch.randn(2, 8, requires_grad=True)
    d = torch.rand(2, requires_grad=True)
    out = head(z_luma, z_lumb, d)
    out.sum().backward()
    for t in (z_luma, z_lumb, d):
        assert t.grad is not None and t.grad.abs().sum() > 0


# -----------------------------------------------------------------
# DMOIModel end-to-end tests
# -----------------------------------------------------------------

def test_dmoi_model_forward_returns_expected_dict():
    pm = _make_synthetic_pole_masks(n_rna=8, n_meth=6)
    model = DMOIModel(
        rna_dim=8, meth_dim=6, pole_masks=pm,
        latent_dim=16, rna_hidden=(32,), meth_hidden=(32,),
        fuse_hidden=(16,), fuse_out=8, head_hidden=8, dropout=0.0,
    )
    model.eval()
    batch = 4
    x_rna = torch.randn(batch, 8)
    x_meth = torch.randn(batch, 6)
    out = model(x_rna, x_meth)
    assert set(out) == {"logits", "pole_scores", "disagreement", "z_fused"}
    assert out["logits"].shape == (batch,)
    assert out["disagreement"].shape == (batch,)
    assert set(out["pole_scores"]) == {"LumA", "LumB"}
    for s in out["pole_scores"].values():
        assert s.shape == (batch,)
        assert ((s >= 0.0) & (s <= 1.0)).all()


def test_dmoi_model_logits_are_finite():
    pm = _make_synthetic_pole_masks()
    model = DMOIModel(
        rna_dim=8, meth_dim=6, pole_masks=pm,
        latent_dim=16, rna_hidden=(32,), meth_hidden=(32,),
        fuse_hidden=(16,), fuse_out=8, head_hidden=8, dropout=0.0,
    )
    x_rna = torch.randn(8, 8) * 3.0
    x_meth = torch.randn(8, 6) * 3.0
    out = model(x_rna, x_meth)
    assert torch.isfinite(out["logits"]).all()


def test_dmoi_model_disagreement_range():
    pm = _make_synthetic_pole_masks()
    model = DMOIModel(
        rna_dim=8, meth_dim=6, pole_masks=pm,
        latent_dim=16, rna_hidden=(32,), meth_hidden=(32,),
        fuse_hidden=(16,), fuse_out=8, head_hidden=8, dropout=0.0,
    )
    x_rna = torch.randn(8, 8)
    x_meth = torch.randn(8, 6)
    out = model(x_rna, x_meth)
    d = out["disagreement"]
    assert ((d >= 0.0) & (d <= 1.0)).all()


def test_dmoi_model_eval_mode_deterministic():
    pm = _make_synthetic_pole_masks()
    model = DMOIModel(
        rna_dim=8, meth_dim=6, pole_masks=pm,
        latent_dim=16, rna_hidden=(32,), meth_hidden=(32,),
        fuse_hidden=(16,), fuse_out=8, head_hidden=8, dropout=0.5,
    )
    model.eval()
    x_rna = torch.randn(4, 8)
    x_meth = torch.randn(4, 6)
    out1 = model(x_rna, x_meth)
    out2 = model(x_rna, x_meth)
    assert torch.allclose(out1["logits"], out2["logits"])


def test_dmoi_model_gradient_flows_end_to_end():
    pm = _make_synthetic_pole_masks()
    model = DMOIModel(
        rna_dim=8, meth_dim=6, pole_masks=pm,
        latent_dim=16, rna_hidden=(32,), meth_hidden=(32,),
        fuse_hidden=(16,), fuse_out=8, head_hidden=8, dropout=0.0,
    )
    model.train()
    x_rna = torch.randn(4, 8)
    x_meth = torch.randn(4, 6)
    labels = torch.randint(0, 2, (4,)).float()
    out = model(x_rna, x_meth)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(out["logits"], labels)
    loss.backward()
    # Encoders + fusers + head should all see gradient.
    components = [
        ("rna_encoder", model.rna_encoder),
        ("meth_encoder", model.meth_encoder),
        ("fusers.LumA", model.fusers["LumA"]),
        ("fusers.LumB", model.fusers["LumB"]),
        ("head", model.head),
    ]
    for name, comp in components:
        total_grad_abs = sum(
            p.grad.abs().sum().item() for p in comp.parameters() if p.grad is not None
        )
        assert total_grad_abs > 0, f"no gradient reached {name}"


def test_dmoi_model_param_count_breakdown():
    pm = _make_synthetic_pole_masks()
    model = DMOIModel(
        rna_dim=8, meth_dim=6, pole_masks=pm,
        latent_dim=16, rna_hidden=(32,), meth_hidden=(32,),
        fuse_hidden=(16,), fuse_out=8, head_hidden=8, dropout=0.0,
    )
    counts = count_dmoi_parameters(model)
    assert set(counts) == {
        "rna_encoder", "meth_encoder", "attention_layers",
        "fusers", "head", "total",
    }
    # PoleAttention is parameter-free.
    assert counts["attention_layers"] == 0
    # Total should be sum of components.
    assert counts["total"] == (
        counts["rna_encoder"] + counts["meth_encoder"]
        + counts["attention_layers"] + counts["fusers"] + counts["head"]
    )


def test_dmoi_model_rejects_mismatched_pole_keys():
    pm = _make_synthetic_pole_masks()
    with pytest.raises(ValueError, match="pole_order"):
        DMOIModel(
            rna_dim=8, meth_dim=6, pole_masks=pm,
            pole_order=("LumA", "NotInMasks"),
        )


def test_dmoi_model_lumA_lumB_pole_scores_differ():
    """Even before training, the two pole sub-classifier scores should differ
    because they consume differently-masked inputs through the same encoder."""
    pm = _make_synthetic_pole_masks()
    model = DMOIModel(
        rna_dim=8, meth_dim=6, pole_masks=pm,
        latent_dim=16, rna_hidden=(32,), meth_hidden=(32,),
        fuse_hidden=(16,), fuse_out=8, head_hidden=8, dropout=0.0,
    )
    model.eval()
    x_rna = torch.randn(8, 8)
    x_meth = torch.randn(8, 6)
    out = model(x_rna, x_meth)
    s_luma = out["pole_scores"]["LumA"]
    s_lumb = out["pole_scores"]["LumB"]
    assert not torch.allclose(s_luma, s_lumb)
