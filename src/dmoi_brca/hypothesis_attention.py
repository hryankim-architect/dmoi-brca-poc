"""Hypothesis-conditioned attention masks for DMOI POC.

Each biological pole (LumA, LumB) is defined by a set of MSigDB Hallmark gene
sets (see `dmoi_brca.priors`). This module builds two binary masks per pole:

- **RNA mask**: 1 at gene_i iff gene_i is in any of the pole's hallmark sets.
- **Methylation mask**: 1 at probe_j iff probe_j's HM450 cis-annotated gene
  list intersects any of the pole's hallmark sets.

The masks are applied **at the input layer** (before encoding) via
`PoleAttention`, which gates each pole-conditioned input through the
shared encoder. The same encoder is used twice per modality, once per pole:

    x_rna_LumA  = x_rna  ⊙ mask_LumA_rna   → encoder_rna → z_rna_LumA
    x_rna_LumB  = x_rna  ⊙ mask_LumB_rna   → encoder_rna → z_rna_LumB
    x_meth_LumA = x_meth ⊙ mask_LumA_meth  → encoder_meth → z_meth_LumA
    x_meth_LumB = x_meth ⊙ mask_LumB_meth  → encoder_meth → z_meth_LumB

This is the "gated multiplication" path in the design doc — simplest first.
Promote to dot-product attention if Day-2-3 metrics show it's the bottleneck.

The cis-mapping TSV is the UCSC Xena HM450 probemap (~395k probes), fetched
on Day-1 Step 2. See `scripts/fetch_hm450_manifest.sh` and
`audit/hm450_probemap_summary.md`.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from dmoi_brca.priors import HALLMARK_SETS


def load_hm450_cis_mapping(path: Path) -> dict[str, set[str]]:
    """Parse the Xena HM450 probemap into a probe-id -> {gene symbols} mapping.

    Probemap format (TSV with header line starting `#id`):

        #id           gene           chrom   chromStart  chromEnd  strand
        cg13332474    .              chr7    25935146    25935148  .
        cg00651829    RSPH14,GNAZ    chr22   23413065    23413067  .
        cg17027195    AUTS2          chr7    69064092    69064094  .

    Intergenic rows (gene = ".") are stored with an empty gene set so callers
    can distinguish "probe known but intergenic" from "probe not in manifest".
    """
    mapping: dict[str, set[str]] = {}
    with Path(path).open() as fh:
        header = fh.readline()
        if not header.startswith("#id"):
            raise ValueError(
                f"Unexpected HM450 probemap header: {header!r}. "
                "Expected first line to start with '#id'.",
            )
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            probe_id, gene_field = parts[0], parts[1]
            if gene_field == "." or gene_field == "":
                mapping[probe_id] = set()
            else:
                mapping[probe_id] = {g.strip() for g in gene_field.split(",") if g.strip()}
    return mapping


def _pole_gene_universe(pole_set_names: Sequence[str]) -> set[str]:
    """Union the Hallmark gene sets named in `pole_set_names`."""
    universe: set[str] = set()
    for name in pole_set_names:
        if name not in HALLMARK_SETS:
            raise KeyError(
                f"Unknown Hallmark set: {name}. "
                f"Available: {sorted(HALLMARK_SETS)}",
            )
        universe.update(HALLMARK_SETS[name])
    return universe


def make_rna_mask(
    feature_genes: Sequence[str],
    pole_set_names: Sequence[str],
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a binary mask of length len(feature_genes) for one pole.

    mask[i] = 1.0 if feature_genes[i] is in any Hallmark set named in
    pole_set_names; else 0.0. Returned as a 1-D torch tensor.
    """
    universe = _pole_gene_universe(pole_set_names)
    mask_list = [1.0 if g in universe else 0.0 for g in feature_genes]
    return torch.tensor(mask_list, dtype=dtype)


def make_meth_mask(
    feature_probes: Sequence[str],
    cis_mapping: dict[str, set[str]],
    pole_set_names: Sequence[str],
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a binary mask of length len(feature_probes) for one pole.

    mask[j] = 1.0 if feature_probes[j]'s cis gene set intersects the pole's
    hallmark gene universe; else 0.0. Probes not present in the cis_mapping
    are treated as intergenic (mask 0).
    """
    universe = _pole_gene_universe(pole_set_names)
    mask_list: list[float] = []
    for probe in feature_probes:
        cis_genes = cis_mapping.get(probe, set())
        mask_list.append(1.0 if cis_genes & universe else 0.0)
    return torch.tensor(mask_list, dtype=dtype)


@dataclass(frozen=True)
class PoleMaskSet:
    """A pair of binary masks for one pole — one over RNA, one over meth.

    Attributes:
        pole_name:        Human-readable pole identifier (e.g. "LumA", "LumB").
        rna_mask:         Binary tensor of shape (n_rna_features,).
        meth_mask:        Binary tensor of shape (n_meth_features,).
        n_rna_on:         Count of nonzero entries in rna_mask.
        n_meth_on:        Count of nonzero entries in meth_mask.
    """

    pole_name: str
    rna_mask: torch.Tensor
    meth_mask: torch.Tensor

    @property
    def n_rna_on(self) -> int:
        return int(self.rna_mask.sum().item())

    @property
    def n_meth_on(self) -> int:
        return int(self.meth_mask.sum().item())

    def summary(self) -> str:
        return (
            f"PoleMaskSet({self.pole_name}: "
            f"rna {self.n_rna_on}/{self.rna_mask.numel()}, "
            f"meth {self.n_meth_on}/{self.meth_mask.numel()})"
        )


def make_pole_masks(
    feature_genes: Sequence[str],
    feature_probes: Sequence[str],
    cis_mapping: dict[str, set[str]],
    poles: dict[str, Sequence[str]],
    *,
    dtype: torch.dtype = torch.float32,
) -> dict[str, PoleMaskSet]:
    """Convenience builder for both LumA / LumB pole masks at once.

    Args:
        feature_genes:    Gene symbols matching the RNA encoder's input dim.
        feature_probes:   HM450 probe IDs matching the meth encoder's input dim.
        cis_mapping:      Output of `load_hm450_cis_mapping(...)`.
        poles:            Dict pole_name -> tuple of Hallmark set names defining the pole.
                          Typically: {"LumA": POLE_LUMA, "LumB": POLE_LUMB}.

    Returns:
        Dict pole_name -> PoleMaskSet with RNA and meth masks.
    """
    return {
        pole_name: PoleMaskSet(
            pole_name=pole_name,
            rna_mask=make_rna_mask(feature_genes, set_names, dtype=dtype),
            meth_mask=make_meth_mask(feature_probes, cis_mapping, set_names, dtype=dtype),
        )
        for pole_name, set_names in poles.items()
    }


class PoleAttention(nn.Module):
    """Gated-multiplication pole attention.

    Given an input tensor x of shape (batch, n_features) and a 1-D mask
    of shape (n_features,), returns x ⊙ mask broadcast across the batch.

    Intentionally parameter-free for the v0.1 implementation — the mask is
    a fixed prior, not a learned parameter. v0.2 may promote this to a
    learnable scalar-per-feature gain on top of the binary mask, or to
    dot-product attention over latent dims.
    """

    def __init__(self, mask: torch.Tensor) -> None:
        super().__init__()
        if mask.dim() != 1:
            raise ValueError(f"PoleAttention mask must be 1-D, got shape {tuple(mask.shape)}")
        # Register as buffer (not parameter) so it moves with .to(device) but
        # doesn't get included in the optimizer's parameter set.
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.mask.numel():
            raise ValueError(
                f"PoleAttention: last dim of x ({x.shape[-1]}) "
                f"does not match mask length ({self.mask.numel()})",
            )
        return x * self.mask


def summarize_mask_coverage(
    pole_masks: Iterable[PoleMaskSet],
) -> str:
    """Render a one-line-per-pole summary for audit MDs / log lines."""
    return "\n".join(m.summary() for m in pole_masks)
