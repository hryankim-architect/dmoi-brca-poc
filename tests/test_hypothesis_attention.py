"""Unit tests for dmoi_brca.hypothesis_attention (Day-1 Step 4)."""
from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from dmoi_brca.hypothesis_attention import (  # noqa: E402
    PoleAttention,
    PoleMaskSet,
    load_hm450_cis_mapping,
    make_meth_mask,
    make_pole_masks,
    make_rna_mask,
    summarize_mask_coverage,
)
from dmoi_brca.priors import POLE_LUMA, POLE_LUMB  # noqa: E402, I001

# ---------------------------------------------------------------------------
# Tiny in-memory HM450 probemap fixture (no real file fetch required)
# ---------------------------------------------------------------------------

PROBEMAP_FIXTURE = """#id\tgene\tchrom\tchromStart\tchromEnd\tstrand
cg00000001\tESR1\tchr6\t100\t102\t.
cg00000002\t.\tchr7\t200\t202\t.
cg00000003\tFOXA1,GATA3\tchr14\t300\t302\t.
cg00000004\tMKI67\tchr10\t400\t402\t.
cg00000005\tCDK1,TOP2A\tchr10\t500\t502\t.
cg00000006\tAURKA\tchr20\t600\t602\t.
cg00000007\tRANDOM_NONHALLMARK_GENE\tchr1\t700\t702\t.
"""


@pytest.fixture
def cis_mapping_file(tmp_path: Path) -> Path:
    p = tmp_path / "probemap_fixture.tsv"
    p.write_text(PROBEMAP_FIXTURE)
    return p


def test_load_hm450_cis_mapping_parses_rows(cis_mapping_file: Path):
    m = load_hm450_cis_mapping(cis_mapping_file)
    assert m["cg00000001"] == {"ESR1"}
    assert m["cg00000002"] == set()                       # intergenic
    assert m["cg00000003"] == {"FOXA1", "GATA3"}          # multi-gene
    assert m["cg00000004"] == {"MKI67"}
    assert m["cg00000005"] == {"CDK1", "TOP2A"}
    assert m["cg00000007"] == {"RANDOM_NONHALLMARK_GENE"}


def test_load_hm450_cis_mapping_rejects_bad_header(tmp_path: Path):
    bad = tmp_path / "bad.tsv"
    bad.write_text("not-a-real-header\nfoo\tbar\n")
    with pytest.raises(ValueError, match="Unexpected HM450 probemap header"):
        load_hm450_cis_mapping(bad)


def test_make_rna_mask_basic():
    feature_genes = ["ESR1", "FOXA1", "TP53", "GAPDH", "MKI67"]
    mask = make_rna_mask(feature_genes, POLE_LUMA)
    # ESR1 and FOXA1 are LumA hallmark genes; others are not.
    assert mask[0].item() == 1.0  # ESR1
    assert mask[1].item() == 1.0  # FOXA1
    assert mask[2].item() == 0.0  # TP53 not in estrogen hallmarks
    assert mask[3].item() == 0.0  # GAPDH
    assert mask[4].item() == 0.0  # MKI67 is LumB, not LumA


def test_make_rna_mask_lumb_canonical():
    feature_genes = ["MKI67", "CDK1", "TOP2A", "AURKA", "ESR1"]
    mask = make_rna_mask(feature_genes, POLE_LUMB)
    assert mask[0].item() == 1.0  # MKI67 in E2F_TARGETS
    assert mask[1].item() == 1.0  # CDK1 in E2F_TARGETS / G2M
    assert mask[2].item() == 1.0  # TOP2A in E2F_TARGETS / G2M
    assert mask[3].item() == 1.0  # AURKA in G2M
    assert mask[4].item() == 0.0  # ESR1 is LumA only


def test_make_rna_mask_unknown_pole_set_raises():
    with pytest.raises(KeyError, match="Unknown Hallmark set"):
        make_rna_mask(["ESR1"], ("HALLMARK_NONEXISTENT",))


def test_make_meth_mask_basic(cis_mapping_file: Path):
    cis = load_hm450_cis_mapping(cis_mapping_file)
    probes = ["cg00000001", "cg00000002", "cg00000003", "cg00000004", "cg00000007"]

    luma_mask = make_meth_mask(probes, cis, POLE_LUMA)
    # cg00000001 -> ESR1 (LumA), cg00000003 -> FOXA1,GATA3 (LumA)
    assert luma_mask[0].item() == 1.0
    assert luma_mask[1].item() == 0.0  # intergenic
    assert luma_mask[2].item() == 1.0
    assert luma_mask[3].item() == 0.0  # MKI67 is LumB
    assert luma_mask[4].item() == 0.0  # nonhallmark gene

    lumb_mask = make_meth_mask(probes, cis, POLE_LUMB)
    # cg00000004 -> MKI67 (LumB)
    assert lumb_mask[0].item() == 0.0  # ESR1 is LumA
    assert lumb_mask[3].item() == 1.0  # MKI67


def test_make_meth_mask_unknown_probe_is_zero(cis_mapping_file: Path):
    cis = load_hm450_cis_mapping(cis_mapping_file)
    probes = ["cg99999999_NOT_IN_MANIFEST"]
    mask = make_meth_mask(probes, cis, POLE_LUMA)
    assert mask[0].item() == 0.0


def test_make_pole_masks_returns_both_poles(cis_mapping_file: Path):
    cis = load_hm450_cis_mapping(cis_mapping_file)
    # Use GAPDH (not in any Hallmark set) as the third gene rather than TP53.
    # TP53 is actually in HALLMARK_E2F_TARGETS (DNA damage response axis),
    # so picking it as a "nonhallmark" filler gene would inadvertently raise
    # LumB's mask count by 1.
    genes = ["ESR1", "MKI67", "GAPDH"]
    probes = ["cg00000001", "cg00000004", "cg00000007"]
    poles = {"LumA": POLE_LUMA, "LumB": POLE_LUMB}
    result = make_pole_masks(genes, probes, cis, poles)

    assert set(result) == {"LumA", "LumB"}
    for ms in result.values():
        assert isinstance(ms, PoleMaskSet)
        assert ms.rna_mask.shape == (3,)
        assert ms.meth_mask.shape == (3,)

    # LumA mask: ESR1 + cg00000001
    assert result["LumA"].n_rna_on == 1
    assert result["LumA"].n_meth_on == 1
    # LumB mask: MKI67 + cg00000004
    assert result["LumB"].n_rna_on == 1
    assert result["LumB"].n_meth_on == 1


def test_pole_attention_forward_shape_and_values():
    mask = torch.tensor([1.0, 0.0, 1.0, 0.0])
    attn = PoleAttention(mask)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    z = attn(x)
    expected = torch.tensor([[1.0, 0.0, 3.0, 0.0], [5.0, 0.0, 7.0, 0.0]])
    assert torch.allclose(z, expected)
    assert z.shape == x.shape


def test_pole_attention_2d_mask_raises():
    with pytest.raises(ValueError, match="mask must be 1-D"):
        PoleAttention(torch.zeros(2, 3))


def test_pole_attention_dim_mismatch_raises():
    attn = PoleAttention(torch.tensor([1.0, 0.0]))
    x = torch.randn(4, 5)
    with pytest.raises(ValueError, match="does not match mask length"):
        attn(x)


def test_pole_attention_is_parameter_free():
    """The mask is a buffer, not a parameter — optimizer should see 0 params."""
    attn = PoleAttention(torch.tensor([1.0, 0.0, 1.0]))
    n_params = sum(p.numel() for p in attn.parameters())
    assert n_params == 0


def test_pole_attention_device_movement():
    """Mask buffer should move with .to(device)."""
    attn = PoleAttention(torch.tensor([1.0, 0.0, 1.0]))
    attn = attn.to("cpu")
    assert attn.mask.device.type == "cpu"
    x = torch.randn(2, 3, device="cpu")
    z = attn(x)
    assert z.device.type == "cpu"


def test_pole_mask_set_summary_includes_counts():
    ms = PoleMaskSet(
        pole_name="LumA",
        rna_mask=torch.tensor([1.0, 0.0, 1.0]),
        meth_mask=torch.tensor([1.0, 1.0, 0.0, 0.0]),
    )
    s = ms.summary()
    assert "LumA" in s
    assert "rna 2/3" in s
    assert "meth 2/4" in s


def test_summarize_mask_coverage_multiple_poles():
    masks = [
        PoleMaskSet("LumA", torch.tensor([1.0, 1.0]), torch.tensor([1.0, 0.0])),
        PoleMaskSet("LumB", torch.tensor([0.0, 1.0]), torch.tensor([1.0, 1.0])),
    ]
    s = summarize_mask_coverage(masks)
    assert "LumA" in s
    assert "LumB" in s
    assert s.count("\n") == 1


def test_pole_attention_gradient_passes_through():
    """Gradient should flow through unmasked positions."""
    mask = torch.tensor([1.0, 0.0, 1.0])
    attn = PoleAttention(mask)
    x = torch.randn(2, 3, requires_grad=True)
    z = attn(x)
    z.sum().backward()
    # Gradient at masked positions should be 0, at unmasked should be 1
    assert torch.allclose(x.grad, mask.expand(2, 3))
