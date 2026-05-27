"""Unit tests for dmoi_brca.encoder (Day-1 Step 3)."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from dmoi_brca.encoder import (  # noqa: E402
    MethEncoder,
    RNAEncoder,
    count_parameters,
)


def test_rna_encoder_forward_shape():
    enc = RNAEncoder(in_dim=20530, out_dim=128)
    x = torch.randn(8, 20530)
    z = enc(x)
    assert z.shape == (8, 128)


def test_meth_encoder_forward_shape():
    enc = MethEncoder(in_dim=10000, out_dim=128)
    x = torch.randn(8, 10000)
    z = enc(x)
    assert z.shape == (8, 128)


def test_rna_encoder_different_batch_sizes():
    enc = RNAEncoder(in_dim=20530, out_dim=128)
    for bs in (1, 4, 32, 128):
        x = torch.randn(bs, 20530)
        z = enc(x)
        assert z.shape == (bs, 128)


def test_encoder_smaller_bottleneck():
    enc = RNAEncoder(in_dim=20530, hidden_dims=(512, 128), out_dim=64)
    x = torch.randn(4, 20530)
    z = enc(x)
    assert z.shape == (4, 64)


def test_encoder_gradient_flows():
    enc = RNAEncoder(in_dim=100, hidden_dims=(64,), out_dim=32)
    x = torch.randn(4, 100, requires_grad=False)
    z = enc(x)
    loss = z.sum()
    loss.backward()
    # Every parameter should have a gradient
    for name, p in enc.named_parameters():
        assert p.grad is not None, f"no grad on {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad on {name}"


def test_encoder_deterministic_with_seed():
    """Same seed + eval mode -> same output."""
    torch.manual_seed(42)
    enc1 = RNAEncoder(in_dim=100, hidden_dims=(64,), out_dim=32)
    enc1.eval()

    torch.manual_seed(42)
    enc2 = RNAEncoder(in_dim=100, hidden_dims=(64,), out_dim=32)
    enc2.eval()

    x = torch.randn(4, 100)
    z1 = enc1(x)
    z2 = enc2(x)
    assert torch.allclose(z1, z2)


def test_dropout_differs_between_train_and_eval():
    """In train mode dropout perturbs; in eval mode it's deterministic."""
    enc = RNAEncoder(in_dim=100, hidden_dims=(64,), out_dim=32, dropout=0.5)
    x = torch.randn(8, 100)

    enc.train()
    torch.manual_seed(0)
    z_train_a = enc(x)
    torch.manual_seed(1)
    z_train_b = enc(x)
    # Two train passes with different RNG should differ due to dropout
    assert not torch.allclose(z_train_a, z_train_b)

    enc.eval()
    z_eval_a = enc(x)
    z_eval_b = enc(x)
    # Eval mode disables dropout - two passes should be identical
    assert torch.allclose(z_eval_a, z_eval_b)


def test_default_rna_encoder_param_count():
    """Sanity-check the default architecture's parameter count.

    Default: 20530 -> 1024 -> 256 -> 128
    Linear params (with bias): 20530*1024 + 1024 + 1024*256 + 256 + 256*128 + 128
                             = 21,022,720 + 263,424 + 33,024  (rough)
    Plus LayerNorm: 2 * (1024 + 256) = 2,560
    """
    enc = RNAEncoder()
    n = count_parameters(enc)
    # Should be roughly 21M; check loose bounds.
    assert 20_000_000 < n < 23_000_000, f"unexpected param count: {n}"


def test_default_meth_encoder_param_count():
    """Default meth: 10000 -> 512 -> 128, ~5.2M params."""
    enc = MethEncoder()
    n = count_parameters(enc)
    assert 5_000_000 < n < 6_000_000, f"unexpected param count: {n}"


def test_encoder_output_is_finite():
    """No NaN/Inf in the encoder output for reasonable inputs."""
    enc = RNAEncoder(in_dim=100, hidden_dims=(64,), out_dim=32)
    x = torch.randn(16, 100) * 5  # somewhat larger scale
    z = enc(x)
    assert torch.isfinite(z).all()


def test_encoder_runs_on_cpu_explicit():
    """Explicitly request CPU device — guards against accidental CUDA hardcoding."""
    enc = RNAEncoder(in_dim=100, hidden_dims=(64,), out_dim=32)
    enc = enc.to("cpu")
    x = torch.randn(4, 100, device="cpu")
    z = enc(x)
    assert z.device.type == "cpu"
    assert z.shape == (4, 32)
