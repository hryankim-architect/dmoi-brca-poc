"""Per-pole fusion: combine RNA + methylation latents into one pole latent
plus a sub-classifier score.

Given the pole-conditioned latents from the encoders:

    z_rna_pole   in R^{batch x latent_dim}
    z_meth_pole  in R^{batch x latent_dim}

PoleFuser produces:

    z_pole       in R^{batch x fuse_dim}     # the fused pole representation
    s_pole       in R^{batch}                # sub-classifier probability for this pole

Day-2 v0.1 implementation uses concat + 2-layer MLP. v0.2 may promote
to cross-attention between RNA and meth latents if Day-3 metrics show
this is the bottleneck (see DMOI/Week-2-Day-1-Design.md §3, §12).
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class PoleFuser(nn.Module):
    """Concat-then-MLP fusion of RNA + meth latents for one pole.

    Outputs:
        z_pole: (batch, fuse_dim) — fused latent, fed to the final classifier.
        s_pole: (batch,)          — sub-classifier probability for this pole.
                                    Used (a) to build the disagreement signal
                                    and (b) as auxiliary supervision if a
                                    pole-wise BCE term is added (Option A
                                    from the design doc; not used in v0.1).
    """

    def __init__(
        self,
        latent_dim: int = 128,
        fuse_hidden: Sequence[int] = (128,),
        fuse_out: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.fuse_out = fuse_out

        # MLP fuses the concat of RNA + meth latents (dim = 2 * latent_dim).
        in_dim = 2 * latent_dim
        dims = [in_dim, *list(fuse_hidden), fuse_out]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(dropout))
        self.fuse_mlp = nn.Sequential(*layers)

        # Sub-classifier: one scalar logit per sample.
        self.sub_clf = nn.Linear(fuse_out, 1)

    def forward(
        self,
        z_rna: torch.Tensor,
        z_meth: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if z_rna.shape != z_meth.shape:
            raise ValueError(
                f"PoleFuser: z_rna shape {tuple(z_rna.shape)} "
                f"!= z_meth shape {tuple(z_meth.shape)}",
            )
        if z_rna.shape[-1] != self.latent_dim:
            raise ValueError(
                f"PoleFuser: input latent dim {z_rna.shape[-1]} != "
                f"expected {self.latent_dim}",
            )
        x = torch.cat([z_rna, z_meth], dim=-1)  # (batch, 2*latent_dim)
        z_pole = self.fuse_mlp(x)               # (batch, fuse_out)
        s_pole_logit = self.sub_clf(z_pole).squeeze(-1)  # (batch,)
        s_pole = torch.sigmoid(s_pole_logit)              # (batch,)
        return z_pole, s_pole


def disagreement_score(s_luma: torch.Tensor, s_lumb: torch.Tensor) -> torch.Tensor:
    """Compute the per-sample disagreement signal between the two pole sub-classifiers.

    Convention: s_LumA ≈ P(patient is LumA | LumA-conditioned view),
                s_LumB ≈ P(patient is LumB | LumB-conditioned view).
                If both perspectives agree, s_LumA + s_LumB ≈ 1, so
                disagreement = |s_LumA - (1 - s_LumB)|. Range [0, 1].

    High disagreement = ambiguous borderline tumor case. Per the design
    doc (Option B, §5), v0.1 does NOT penalize disagreement; instead it
    is passed forward as an extra input feature to the final classifier.
    """
    return (s_luma - (1.0 - s_lumb)).abs()
