"""DMOI per-modality encoders.

Two parallel MLP encoders that project the high-dimensional input modalities
(RNA-seq, methylation) into a shared latent dimension. The downstream
hypothesis-conditioned attention layer (`dmoi_brca.hypothesis_attention`)
operates on these latents.

Design choices (see DMOI/Week-2-Day-1-Design.md):

- **Small bottleneck (128 latent dim)** to mitigate overfitting on the
  n=417 dual-modality cohort. Larger latents are configurable via the
  constructor but the default is conservative.
- **LayerNorm + ReLU + Dropout** instead of BatchNorm: with batch size 64
  on a fold of ~334 train samples, batch statistics would be too noisy.
- **No final activation on the latent output**: downstream attention layer
  can normalize or activate as needed.
- **CPU + MPS both supported**: pure nn.Linear, no kernel-specific ops.

Example:
    >>> import torch
    >>> from dmoi_brca.encoder import RNAEncoder, MethEncoder
    >>> rna_enc = RNAEncoder(in_dim=20530)
    >>> meth_enc = MethEncoder(in_dim=10000)
    >>> x_rna = torch.randn(64, 20530)
    >>> x_meth = torch.randn(64, 10000)
    >>> z_rna = rna_enc(x_rna)    # (64, 128)
    >>> z_meth = meth_enc(x_meth) # (64, 128)
    >>> assert z_rna.shape == (64, 128)
    >>> assert z_meth.shape == (64, 128)
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def _build_mlp(
    in_dim: int,
    hidden_dims: Sequence[int],
    out_dim: int,
    dropout: float,
) -> nn.Sequential:
    """Construct an MLP with LayerNorm + ReLU + Dropout between hidden layers.

    The final layer (hidden -> out_dim) is a bare Linear with no
    norm/activation/dropout — downstream layers handle that.
    """
    dims = [in_dim, *list(hidden_dims), out_dim]
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:  # not the final layer
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class RNAEncoder(nn.Module):
    """RNA-seq feature encoder.

    Default architecture for the cohort_v2 baseline (20,530 HiSeqV2 genes):
        20530 -> 1024 -> 256 -> 128

    Total parameters ~ 21.3M, dominated by the first linear layer
    (20530 * 1024). For tighter overfitting control, pass smaller
    `hidden_dims` and/or `out_dim`.
    """

    def __init__(
        self,
        in_dim: int = 20_530,
        hidden_dims: Sequence[int] = (1024, 256),
        out_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.net = _build_mlp(in_dim, hidden_dims, out_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MethEncoder(nn.Module):
    """Methylation (HM450) feature encoder.

    Default architecture for the cohort_v2 baseline (10,000 top-variance probes):
        10000 -> 512 -> 128

    Smaller than the RNA encoder because the meth feature space is already
    pre-filtered to top-variance probes. Total parameters ~ 5.2M.
    """

    def __init__(
        self,
        in_dim: int = 10_000,
        hidden_dims: Sequence[int] = (512,),
        out_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.net = _build_mlp(in_dim, hidden_dims, out_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def count_parameters(module: nn.Module) -> int:
    """Return the total number of trainable parameters in a module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
