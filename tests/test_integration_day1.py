"""Day-1 integration smoke test — encoder + PoleAttention forward pass.

Verifies that the Day-1 building blocks compose end-to-end without errors,
returning the four pole-conditioned latents the DMOI model will fuse in Day-2:

    z_rna_LumA  = RNAEncoder( PoleAttention(LumA_rna_mask)(x_rna)  )
    z_rna_LumB  = RNAEncoder( PoleAttention(LumB_rna_mask)(x_rna)  )
    z_meth_LumA = MethEncoder(PoleAttention(LumA_meth_mask)(x_meth))
    z_meth_LumB = MethEncoder(PoleAttention(LumB_meth_mask)(x_meth))

The same encoder is shared across both poles — only the input is gated
differently. This is intentional: pole specificity is a property of the
input projection (the mask), not of the encoder weights.
"""
from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from dmoi_brca.encoder import MethEncoder, RNAEncoder  # noqa: E402
from dmoi_brca.hypothesis_attention import (  # noqa: E402
    PoleAttention,
    load_hm450_cis_mapping,
    make_pole_masks,
)
from dmoi_brca.priors import POLE_LUMA, POLE_LUMB  # noqa: E402

PROBEMAP_FIXTURE = """#id\tgene\tchrom\tchromStart\tchromEnd\tstrand
cg00000001\tESR1\tchr6\t100\t102\t.
cg00000002\tFOXA1,GATA3\tchr14\t200\t202\t.
cg00000003\tMKI67\tchr10\t300\t302\t.
cg00000004\tCDK1,TOP2A\tchr10\t400\t402\t.
cg00000005\tAURKA\tchr20\t500\t502\t.
cg00000006\tBOGUS_GENE\tchr1\t600\t602\t.
"""


@pytest.fixture
def fixture_setup(tmp_path: Path):
    probemap_file = tmp_path / "probemap.tsv"
    probemap_file.write_text(PROBEMAP_FIXTURE)
    cis = load_hm450_cis_mapping(probemap_file)

    # Small feature dims so the test runs quickly.
    n_rna = 8
    n_meth = 6
    feature_genes = ["ESR1", "PGR", "FOXA1", "MKI67", "CDK1", "AURKA", "TP53", "GAPDH"]
    feature_probes = [
        "cg00000001", "cg00000002", "cg00000003",
        "cg00000004", "cg00000005", "cg00000006",
    ]

    poles = {"LumA": POLE_LUMA, "LumB": POLE_LUMB}
    pole_masks = make_pole_masks(feature_genes, feature_probes, cis, poles)

    return {
        "n_rna": n_rna,
        "n_meth": n_meth,
        "feature_genes": feature_genes,
        "feature_probes": feature_probes,
        "pole_masks": pole_masks,
    }


def test_pole_masks_have_expected_dimensions(fixture_setup):
    pm = fixture_setup["pole_masks"]
    n_rna = fixture_setup["n_rna"]
    n_meth = fixture_setup["n_meth"]
    for pole_name in ("LumA", "LumB"):
        assert pm[pole_name].rna_mask.shape == (n_rna,)
        assert pm[pole_name].meth_mask.shape == (n_meth,)


def test_pole_masks_have_expected_canonical_coverage(fixture_setup):
    """LumA mask should hit ER-axis markers; LumB should hit proliferation markers."""
    pm = fixture_setup["pole_masks"]
    # LumA: ESR1, PGR (not in fixture genes -> 0), FOXA1
    # PGR is in POLE_LUMA hallmark sets, included in mask if present in feature_genes.
    assert pm["LumA"].n_rna_on >= 3   # ESR1 + PGR + FOXA1 at minimum
    # LumB: MKI67, CDK1, AURKA
    assert pm["LumB"].n_rna_on >= 3
    # No gene should be in both LumA and LumB masks for these canonical markers.
    overlap = (pm["LumA"].rna_mask * pm["LumB"].rna_mask).sum().item()
    assert overlap == 0


def test_full_forward_pass_yields_four_latents(fixture_setup):
    """The Day-2 fusion layer expects four (batch, latent_dim) tensors."""
    n_rna = fixture_setup["n_rna"]
    n_meth = fixture_setup["n_meth"]
    pm = fixture_setup["pole_masks"]
    latent_dim = 16
    batch = 4

    # Small encoders so the test is fast (matching the tiny feature dims).
    rna_enc = RNAEncoder(in_dim=n_rna, hidden_dims=(32,), out_dim=latent_dim, dropout=0.0)
    meth_enc = MethEncoder(in_dim=n_meth, hidden_dims=(32,), out_dim=latent_dim, dropout=0.0)

    # Per-modality + per-pole attention layers (parameter-free).
    attn_rna_luma = PoleAttention(pm["LumA"].rna_mask)
    attn_rna_lumb = PoleAttention(pm["LumB"].rna_mask)
    attn_meth_luma = PoleAttention(pm["LumA"].meth_mask)
    attn_meth_lumb = PoleAttention(pm["LumB"].meth_mask)

    x_rna = torch.randn(batch, n_rna)
    x_meth = torch.randn(batch, n_meth)

    rna_enc.eval()  # disable dropout for deterministic shape check
    meth_enc.eval()

    z_rna_luma = rna_enc(attn_rna_luma(x_rna))
    z_rna_lumb = rna_enc(attn_rna_lumb(x_rna))
    z_meth_luma = meth_enc(attn_meth_luma(x_meth))
    z_meth_lumb = meth_enc(attn_meth_lumb(x_meth))

    for z in (z_rna_luma, z_rna_lumb, z_meth_luma, z_meth_lumb):
        assert z.shape == (batch, latent_dim)
        assert torch.isfinite(z).all()


def test_pole_conditioned_latents_differ_from_unmasked(fixture_setup):
    """Pole-masked input should yield a different latent than the unmasked input.

    Asserts the attention layer is actually changing the signal — not a no-op.
    """
    n_rna = fixture_setup["n_rna"]
    pm = fixture_setup["pole_masks"]
    rna_enc = RNAEncoder(in_dim=n_rna, hidden_dims=(32,), out_dim=16, dropout=0.0)
    rna_enc.eval()
    attn_luma = PoleAttention(pm["LumA"].rna_mask)

    x_rna = torch.randn(2, n_rna)
    z_unmasked = rna_enc(x_rna)
    z_masked = rna_enc(attn_luma(x_rna))

    # Should differ because the mask zeros out some columns of x.
    assert not torch.allclose(z_unmasked, z_masked)


def test_lumA_vs_lumB_latents_differ(fixture_setup):
    """The two pole-conditioned inputs should produce different latents from
    the same encoder, even before any pole-specific training."""
    n_rna = fixture_setup["n_rna"]
    pm = fixture_setup["pole_masks"]
    rna_enc = RNAEncoder(in_dim=n_rna, hidden_dims=(32,), out_dim=16, dropout=0.0)
    rna_enc.eval()
    attn_luma = PoleAttention(pm["LumA"].rna_mask)
    attn_lumb = PoleAttention(pm["LumB"].rna_mask)

    x_rna = torch.randn(2, n_rna)
    z_luma = rna_enc(attn_luma(x_rna))
    z_lumb = rna_enc(attn_lumb(x_rna))

    # Different masks -> different inputs -> different latents.
    assert not torch.allclose(z_luma, z_lumb)


def test_gradient_flows_through_full_pipeline(fixture_setup):
    """End-to-end gradient check: loss on z_LumA must flow back to encoder weights."""
    n_rna = fixture_setup["n_rna"]
    pm = fixture_setup["pole_masks"]
    rna_enc = RNAEncoder(in_dim=n_rna, hidden_dims=(32,), out_dim=16, dropout=0.0)
    rna_enc.train()
    attn_luma = PoleAttention(pm["LumA"].rna_mask)

    x_rna = torch.randn(2, n_rna)
    z = rna_enc(attn_luma(x_rna))
    z.sum().backward()
    for name, p in rna_enc.named_parameters():
        # Only the first Linear layer's params attached to masked-out feature columns
        # may have zero grad; the rest must have non-trivial gradient.
        if "0.weight" in name:
            # First linear weight: some columns may be zero due to the input mask.
            assert p.grad is not None
            continue
        assert p.grad is not None, f"no grad for {name}"
        # Most params should have non-zero grad (allow occasional zero rows).
        assert p.grad.abs().sum() > 0, f"all-zero grad for {name}"
