"""Unit tests for dmoi_brca.fusion (Day-2)."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from dmoi_brca.fusion import PoleFuser, disagreement_score  # noqa: E402


def test_pole_fuser_forward_shapes():
    fuser = PoleFuser(latent_dim=16, fuse_hidden=(32,), fuse_out=8, dropout=0.0)
    batch = 4
    z_rna = torch.randn(batch, 16)
    z_meth = torch.randn(batch, 16)
    z_pole, s_pole = fuser(z_rna, z_meth)
    assert z_pole.shape == (batch, 8)
    assert s_pole.shape == (batch,)


def test_pole_fuser_subclf_range_is_probability():
    fuser = PoleFuser(latent_dim=16, fuse_hidden=(32,), fuse_out=8, dropout=0.0)
    fuser.eval()
    z_rna = torch.randn(8, 16) * 5  # larger inputs
    z_meth = torch.randn(8, 16) * 5
    _, s_pole = fuser(z_rna, z_meth)
    assert ((s_pole >= 0.0) & (s_pole <= 1.0)).all()


def test_pole_fuser_shape_mismatch_raises():
    fuser = PoleFuser(latent_dim=16)
    z_rna = torch.randn(4, 16)
    z_meth = torch.randn(4, 12)  # wrong shape
    with pytest.raises(ValueError, match="z_rna shape"):
        fuser(z_rna, z_meth)


def test_pole_fuser_wrong_latent_dim_raises():
    fuser = PoleFuser(latent_dim=16)
    z_rna = torch.randn(4, 12)  # latent_dim mismatch
    z_meth = torch.randn(4, 12)
    with pytest.raises(ValueError, match="input latent dim"):
        fuser(z_rna, z_meth)


def test_pole_fuser_gradient_flow():
    fuser = PoleFuser(latent_dim=16, fuse_hidden=(32,), fuse_out=8, dropout=0.0)
    z_rna = torch.randn(4, 16, requires_grad=True)
    z_meth = torch.randn(4, 16, requires_grad=True)
    z_pole, s_pole = fuser(z_rna, z_meth)
    (z_pole.sum() + s_pole.sum()).backward()
    assert z_rna.grad is not None and z_rna.grad.abs().sum() > 0
    assert z_meth.grad is not None and z_meth.grad.abs().sum() > 0


def test_pole_fuser_eval_mode_deterministic():
    fuser = PoleFuser(latent_dim=16, fuse_hidden=(32,), fuse_out=8, dropout=0.5)
    fuser.eval()
    z_rna = torch.randn(4, 16)
    z_meth = torch.randn(4, 16)
    z1, s1 = fuser(z_rna, z_meth)
    z2, s2 = fuser(z_rna, z_meth)
    assert torch.allclose(z1, z2)
    assert torch.allclose(s1, s2)


def test_disagreement_score_perfect_agreement_is_zero():
    # Both perspectives say "definitely LumA": s_LumA ~ 1.0, s_LumB ~ 0.0
    s_luma = torch.tensor([1.0, 1.0])
    s_lumb = torch.tensor([0.0, 0.0])
    d = disagreement_score(s_luma, s_lumb)
    assert torch.allclose(d, torch.tensor([0.0, 0.0]))


def test_disagreement_score_perfect_disagreement_is_one():
    # LumA branch says LumA; LumB branch ALSO says LumA (i.e. s_LumB ~ 0).
    # Both AGREE this is LumA, so disagreement should be near 0.
    # Real disagreement: LumA branch says LumA (s_LumA=1), LumB branch says LumB (s_LumB=1).
    # disagreement = |1 - (1 - 1)| = |1 - 0| = 1.
    s_luma = torch.tensor([1.0])
    s_lumb = torch.tensor([1.0])
    d = disagreement_score(s_luma, s_lumb)
    assert torch.allclose(d, torch.tensor([1.0]))


def test_disagreement_score_range():
    s_luma = torch.rand(100)
    s_lumb = torch.rand(100)
    d = disagreement_score(s_luma, s_lumb)
    assert ((d >= 0.0) & (d <= 1.0)).all()


def test_disagreement_score_intermediate():
    # Slight disagreement: s_LumA = 0.7, s_LumB = 0.4 → expected 1-0.4=0.6, so |0.7-0.6|=0.1
    s_luma = torch.tensor([0.7])
    s_lumb = torch.tensor([0.4])
    d = disagreement_score(s_luma, s_lumb)
    assert torch.allclose(d, torch.tensor([0.1]), atol=1e-6)


def test_pole_fuser_param_count_sane():
    """Default config: 2*128 -> 128 -> 64 + sub_clf 64->1.
    Rough: 256*128 + 128 + 128*64 + 64 + 64*1 + 1 = ~41,300.
    """
    fuser = PoleFuser(latent_dim=128, fuse_hidden=(128,), fuse_out=64)
    n = sum(p.numel() for p in fuser.parameters())
    assert 35_000 < n < 50_000, f"unexpected fuser param count: {n}"
