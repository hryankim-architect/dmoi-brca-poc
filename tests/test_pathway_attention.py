"""Unit tests for dmoi_brca.pathway_attention (v0.7)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from dmoi_brca.dmoi_model import DMOIModel
from dmoi_brca.hypothesis_attention import PoleMaskSet
from dmoi_brca.pathway_attention import (
    PathwayPoleAttention,
    compute_pathway_expression_scores,
)

# ---------------------------------------------------------------------------
# compute_pathway_expression_scores
# ---------------------------------------------------------------------------

def test_compute_pathway_scores_shapes_and_values():
    rna = np.array([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]], dtype=np.float32)
    names = ["g1", "g2", "g3", "g4"]
    pathways = {"P_LO": ["g1", "g2"], "P_HI": ["g3", "g4"]}
    scores, pn = compute_pathway_expression_scores(rna, names, pathways)
    assert scores.shape == (2, 2)
    assert pn == ["P_LO", "P_HI"]
    # Patient 0: P_LO = (1+2)/2 = 1.5; P_HI = (3+4)/2 = 3.5
    np.testing.assert_allclose(scores[0], [1.5, 3.5], atol=1e-6)
    # Patient 1: P_LO = (10+20)/2 = 15; P_HI = (30+40)/2 = 35
    np.testing.assert_allclose(scores[1], [15.0, 35.0], atol=1e-6)


def test_compute_pathway_scores_drops_missing_genes():
    rna = np.array([[1.0, 2.0]], dtype=np.float32)
    names = ["g1", "g2"]
    pathways = {"P": ["g1", "g_MISSING", "g_ALSO_MISSING"]}
    scores, _ = compute_pathway_expression_scores(rna, names, pathways)
    # Only g1 is in feature names; mean of one gene = that gene's value.
    np.testing.assert_allclose(scores[0], [1.0], atol=1e-6)


def test_compute_pathway_scores_empty_pathway_returns_zero_column():
    rna = np.array([[1.0, 2.0]], dtype=np.float32)
    names = ["g1", "g2"]
    pathways = {"orphan": ["g_X", "g_Y"]}
    scores, _ = compute_pathway_expression_scores(rna, names, pathways)
    np.testing.assert_allclose(scores[0], [0.0], atol=1e-6)


def test_compute_pathway_scores_rejects_non_2d():
    rna = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    with pytest.raises(ValueError, match="2-D"):
        compute_pathway_expression_scores(rna, ["g1", "g2", "g3"], {"P": ["g1"]})


def test_compute_pathway_scores_rejects_shape_mismatch():
    rna = np.zeros((2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="cols"):
        compute_pathway_expression_scores(rna, ["g1", "g2"], {"P": ["g1"]})


def test_compute_pathway_scores_preserves_pathway_iteration_order():
    """Pathway column order must match the input dict iteration order."""
    rna = np.zeros((1, 2), dtype=np.float32)
    pathways = {"Z_PATH": ["g1"], "A_PATH": ["g2"]}  # alpha-out-of-order
    _, pn = compute_pathway_expression_scores(rna, ["g1", "g2"], pathways)
    assert pn == ["Z_PATH", "A_PATH"]


# ---------------------------------------------------------------------------
# PathwayPoleAttention
# ---------------------------------------------------------------------------

def test_pathway_pole_attention_forward_shape():
    attn = PathwayPoleAttention(n_pathways=5, pole_order=("LumA", "LumB"))
    x = torch.randn(3, 5)
    out = attn(x)
    assert tuple(out.shape) == (3, 2)


def test_pathway_pole_attention_weights_are_softmax_normalized():
    attn = PathwayPoleAttention(n_pathways=7, pole_order=("LumA", "LumB"))
    w = attn.attn_weights
    assert tuple(w.shape) == (2, 7)
    sums = w.sum(dim=-1)
    np.testing.assert_allclose(sums.detach().numpy(), np.array([1.0, 1.0]), atol=1e-5)


def test_pathway_pole_attention_weights_init_near_uniform():
    """Small init_std means initial softmax is close to uniform."""
    attn = PathwayPoleAttention(
        n_pathways=10, pole_order=("LumA", "LumB"), init_std=0.001,
    )
    w = attn.attn_weights.detach().numpy()
    # Uniform = 0.1; with tiny init the spread should be small.
    assert np.allclose(w, 0.1, atol=0.001)


def test_pathway_pole_attention_gradient_flows():
    attn = PathwayPoleAttention(n_pathways=4, pole_order=("LumA", "LumB"))
    x = torch.randn(2, 4, requires_grad=False)
    out = attn(x)
    loss = out.sum()
    loss.backward()
    assert attn.attn_logits.grad is not None
    assert torch.isfinite(attn.attn_logits.grad).all()
    # Non-zero gradient (some learning signal).
    assert attn.attn_logits.grad.abs().sum() > 0


def test_pathway_pole_attention_top_k_pathways():
    attn = PathwayPoleAttention(n_pathways=5, pole_order=("LumA", "LumB"))
    # Hand-set logits so LumA peaks at index 2, LumB peaks at index 4.
    with torch.no_grad():
        attn.attn_logits.zero_()
        attn.attn_logits[0, 2] = 10.0
        attn.attn_logits[1, 4] = 10.0
    names = ["p0", "p1", "p2", "p3", "p4"]
    top = attn.top_k_pathways(names, k=2)
    assert top["LumA"][0][0] == "p2"
    assert top["LumB"][0][0] == "p4"
    # Softmax weight at the peak should dominate (>0.9 with logit gap of 10).
    assert top["LumA"][0][1] > 0.9
    assert top["LumB"][0][1] > 0.9


def test_pathway_pole_attention_rejects_bad_input_dim():
    attn = PathwayPoleAttention(n_pathways=5, pole_order=("LumA", "LumB"))
    bad = torch.randn(3, 7)  # wrong feature dim
    with pytest.raises(ValueError, match="n_pathways"):
        attn(bad)


def test_pathway_pole_attention_rejects_non_2d_input():
    attn = PathwayPoleAttention(n_pathways=5, pole_order=("LumA", "LumB"))
    bad = torch.randn(5)
    with pytest.raises(ValueError, match="batch, n_pathways"):
        attn(bad)


def test_pathway_pole_attention_top_k_rejects_wrong_names_length():
    attn = PathwayPoleAttention(n_pathways=5, pole_order=("LumA", "LumB"))
    with pytest.raises(ValueError, match="n_pathways"):
        attn.top_k_pathways(["p0", "p1"], k=2)


def test_pathway_pole_attention_empty_pole_order_raises():
    with pytest.raises(ValueError, match="pole_order"):
        PathwayPoleAttention(n_pathways=5, pole_order=())


def test_pathway_pole_attention_zero_pathways_raises():
    with pytest.raises(ValueError, match="n_pathways"):
        PathwayPoleAttention(n_pathways=0, pole_order=("LumA",))


# ---------------------------------------------------------------------------
# PathwayPoleAttention -- v0.8 Variant C (proj_dim kwarg)
# ---------------------------------------------------------------------------

def test_pathway_pole_attention_scalar_mode_out_dim_equals_n_poles():
    attn = PathwayPoleAttention(n_pathways=5, pole_order=("LumA", "LumB"))
    assert attn.proj_dim is None
    assert attn.projections is None
    assert attn.out_dim == 2


def test_pathway_pole_attention_vector_mode_out_dim_is_npoles_times_projdim():
    attn = PathwayPoleAttention(
        n_pathways=5, pole_order=("LumA", "LumB"), proj_dim=8,
    )
    assert attn.proj_dim == 8
    assert attn.projections is not None
    assert attn.out_dim == 2 * 8


def test_pathway_pole_attention_vector_mode_forward_shape():
    attn = PathwayPoleAttention(
        n_pathways=5, pole_order=("LumA", "LumB"), proj_dim=4,
    )
    x = torch.randn(3, 5)
    out = attn(x)
    assert tuple(out.shape) == (3, 8)


def test_pathway_pole_attention_vector_mode_gradient_flows_to_projections():
    attn = PathwayPoleAttention(
        n_pathways=4, pole_order=("LumA", "LumB"), proj_dim=3,
    )
    out = attn(torch.randn(2, 4))
    out.sum().backward()
    assert attn.attn_logits.grad is not None
    for pole_proj in attn.projections.values():
        assert pole_proj.weight.grad is not None
        assert torch.isfinite(pole_proj.weight.grad).all()
        assert pole_proj.weight.grad.abs().sum() > 0


def test_pathway_pole_attention_vector_mode_param_delta_matches_expected():
    """Vector mode adds n_poles * n_pathways * proj_dim params + bookkeeping."""
    n_paths, n_poles, proj_dim = 7, 2, 5
    scalar = PathwayPoleAttention(n_pathways=n_paths, pole_order=("LumA", "LumB"))
    vector = PathwayPoleAttention(
        n_pathways=n_paths, pole_order=("LumA", "LumB"), proj_dim=proj_dim,
    )
    n_scalar = sum(p.numel() for p in scalar.parameters())
    n_vector = sum(p.numel() for p in vector.parameters())
    # Each pole gets nn.Linear(n_paths, proj_dim, bias=False)
    expected_delta = n_poles * n_paths * proj_dim
    assert n_vector - n_scalar == expected_delta


def test_pathway_pole_attention_rejects_nonpositive_proj_dim():
    with pytest.raises(ValueError, match="proj_dim"):
        PathwayPoleAttention(
            n_pathways=5, pole_order=("LumA",), proj_dim=0,
        )


def test_dmoi_model_pathway_proj_dim_wires_vector_branch():
    """DMOIModel.pathway_proj_dim should propagate to PathwayPoleAttention."""
    m = DMOIModel(
        rna_dim=8, meth_dim=4, pole_masks=_tiny_pole_masks(8, 4),
        latent_dim=4, rna_hidden=(8,), meth_hidden=(4,),
        fuse_hidden=(8,), fuse_out=4, head_hidden=4,
        n_pathways=6, pathway_proj_dim=3,
    )
    assert m.pathway_attention.proj_dim == 3
    assert m.pathway_attention.out_dim == 2 * 3
    out = m(torch.randn(2, 8), torch.randn(2, 4), torch.randn(2, 6))
    assert tuple(out["pole_pathway_feat"].shape) == (2, 6)
    assert tuple(out["logits"].shape) == (2,)


# ---------------------------------------------------------------------------
# DMOIModel n_pathways switching
# ---------------------------------------------------------------------------

def _tiny_pole_masks(rna_dim: int = 8, meth_dim: int = 4):
    return {
        p: PoleMaskSet(
            pole_name=p,
            rna_mask=torch.ones(rna_dim),
            meth_mask=torch.ones(meth_dim),
        )
        for p in ("LumA", "LumB")
    }


def _tiny_model(*, n_pathways: int = 0):
    return DMOIModel(
        rna_dim=8, meth_dim=4, pole_masks=_tiny_pole_masks(8, 4),
        latent_dim=4, rna_hidden=(8,), meth_hidden=(4,),
        fuse_hidden=(8,), fuse_out=4, head_hidden=4,
        n_pathways=n_pathways,
    )


def test_dmoi_model_n_pathways_zero_matches_v06_signature():
    m = _tiny_model(n_pathways=0)
    assert m.pathway_attention is None
    x_rna = torch.randn(2, 8)
    x_meth = torch.randn(2, 4)
    out = m(x_rna, x_meth)
    assert tuple(out["logits"].shape) == (2,)
    assert "pole_pathway_feat" not in out


def test_dmoi_model_n_pathways_positive_wires_branch():
    m = _tiny_model(n_pathways=6)
    assert m.pathway_attention is not None
    x_rna = torch.randn(2, 8)
    x_meth = torch.randn(2, 4)
    x_path = torch.randn(2, 6)
    out = m(x_rna, x_meth, x_path)
    assert tuple(out["logits"].shape) == (2,)
    assert "pole_pathway_feat" in out
    assert tuple(out["pole_pathway_feat"].shape) == (2, 2)  # 2 poles


def test_dmoi_model_v07_requires_x_pathway_when_wired():
    m = _tiny_model(n_pathways=6)
    with pytest.raises(ValueError, match="n_pathways>0.*x_pathway=None"):
        m(torch.randn(2, 8), torch.randn(2, 4))


def test_dmoi_model_v06_rejects_unexpected_x_pathway():
    m = _tiny_model(n_pathways=0)
    with pytest.raises(ValueError, match="n_pathways=0.*non-None x_pathway"):
        m(torch.randn(2, 8), torch.randn(2, 4), torch.randn(2, 5))


def test_dmoi_model_v07_param_delta_is_small():
    """v0.7 adds n_pathways*n_poles attn logits + 2 extra head input dims."""
    m6 = _tiny_model(n_pathways=0)
    m7 = _tiny_model(n_pathways=10)
    n6 = sum(p.numel() for p in m6.parameters())
    n7 = sum(p.numel() for p in m7.parameters())
    delta = n7 - n6
    # 10 pathways * 2 poles = 20 attn logits + 2 extra head input feats * head_hidden=4 = 8
    # Total expected: 28 (give or take bias / LayerNorm reshuffling around the head boundary)
    assert 20 < delta < 40, f"expected ~28, got {delta}"


def test_dmoi_model_v07_backprop_reaches_pathway_attention():
    m = _tiny_model(n_pathways=5)
    out = m(torch.randn(3, 8), torch.randn(3, 4), torch.randn(3, 5))
    out["logits"].sum().backward()
    assert m.pathway_attention.attn_logits.grad is not None
    assert torch.isfinite(m.pathway_attention.attn_logits.grad).all()
