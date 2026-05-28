"""DMOIModel — ties encoder + PoleAttention + PoleFuser + ClassifierHead.

Per-batch forward returns:
    logits        : (batch,) - final classification logit (LumA=1 / LumB=0).
    pole_scores   : dict {pole_name: (batch,)} - sub-classifier probs per pole.
    disagreement  : (batch,) - |s_LumA - (1 - s_LumB)| signal.

Per design doc §3, the same encoder is used for both poles — pole specificity
is encoded in the input attention mask, not in encoder weights. The
classifier head consumes [z_LumA_fused, z_LumB_fused, disagreement_feature]
to produce the final logit.

Loss design (v0.1, Option B from design doc §5):
    L = BCEWithLogitsLoss(logits, labels, pos_weight=class_balanced)
    (No disagreement penalty; disagreement is *input* to the classifier,
     not a regularized output. v0.2 may add an Option-A auxiliary term.)
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from dmoi_brca.encoder import MethEncoder, RNAEncoder
from dmoi_brca.fusion import PoleFuser, disagreement_score
from dmoi_brca.hypothesis_attention import PoleAttention, PoleMaskSet


class ClassifierHead(nn.Module):
    """Final binary classifier head.

    Input is [z_LumA_fused, z_LumB_fused, (disagreement_scalar)] concatenated
    along the feature dim. Returns a single logit per sample.

    Args:
        fuse_dim:          Per-pole fused latent dimension.
        hidden:            Hidden layer width.
        dropout:           Dropout rate.
        use_disagreement:  If True (default), include the disagreement scalar
                           as an extra input feature. Day-4 ablation toggles
                           this to False to measure the disagreement signal's
                           empirical contribution.
    """

    def __init__(
        self,
        fuse_dim: int,
        hidden: int = 32,
        dropout: float = 0.3,
        use_disagreement: bool = True,
    ) -> None:
        super().__init__()
        self.use_disagreement = use_disagreement
        in_dim = 2 * fuse_dim + (1 if use_disagreement else 0)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        z_luma: torch.Tensor,
        z_lumb: torch.Tensor,
        disagreement: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_disagreement:
            d = disagreement.unsqueeze(-1)
            x = torch.cat([z_luma, z_lumb, d], dim=-1)
        else:
            x = torch.cat([z_luma, z_lumb], dim=-1)
        return self.net(x).squeeze(-1)


class DMOIModel(nn.Module):
    """End-to-end DMOI model.

    Composition:
        x_rna, x_meth
          → PoleAttention(pole, modality) gates the input per pole
          → shared RNA / Meth encoder produces pole-conditioned latents
          → PoleFuser per pole combines RNA + meth latents -> (z_pole, s_pole)
          → ClassifierHead consumes [z_LumA, z_LumB, disagreement] -> logit
    """

    def __init__(
        self,
        rna_dim: int,
        meth_dim: int,
        pole_masks: dict[str, PoleMaskSet],
        *,
        latent_dim: int = 128,
        rna_hidden: Sequence[int] = (1024, 256),
        meth_hidden: Sequence[int] = (512,),
        fuse_hidden: Sequence[int] = (128,),
        fuse_out: int = 64,
        head_hidden: int = 32,
        dropout: float = 0.3,
        pole_order: tuple[str, str] = ("LumA", "LumB"),
        use_disagreement: bool = True,
    ) -> None:
        super().__init__()
        if set(pole_order) != set(pole_masks):
            raise ValueError(
                f"pole_order {pole_order} keys != pole_masks keys "
                f"{sorted(pole_masks)}",
            )
        self.pole_order = pole_order
        self.rna_dim = rna_dim
        self.meth_dim = meth_dim
        self.latent_dim = latent_dim
        self.fuse_out = fuse_out
        self.use_disagreement = use_disagreement

        # Shared encoders — used twice per modality (once per pole).
        self.rna_encoder = RNAEncoder(
            in_dim=rna_dim, hidden_dims=rna_hidden,
            out_dim=latent_dim, dropout=dropout,
        )
        self.meth_encoder = MethEncoder(
            in_dim=meth_dim, hidden_dims=meth_hidden,
            out_dim=latent_dim, dropout=dropout,
        )

        # Per-pole attention layers (parameter-free, fixed binary masks).
        self.attn_rna = nn.ModuleDict({
            pole: PoleAttention(pole_masks[pole].rna_mask) for pole in pole_order
        })
        self.attn_meth = nn.ModuleDict({
            pole: PoleAttention(pole_masks[pole].meth_mask) for pole in pole_order
        })

        # Per-pole fusers.
        self.fusers = nn.ModuleDict({
            pole: PoleFuser(
                latent_dim=latent_dim,
                fuse_hidden=fuse_hidden,
                fuse_out=fuse_out,
                dropout=dropout,
            )
            for pole in pole_order
        })

        # Final classifier head. The ablation flag controls whether the
        # disagreement scalar is included as an input feature.
        self.head = ClassifierHead(
            fuse_dim=fuse_out,
            hidden=head_hidden,
            dropout=dropout,
            use_disagreement=use_disagreement,
        )

    def forward(
        self,
        x_rna: torch.Tensor,
        x_meth: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass. Returns a dict containing:

        - logits        : (batch,) — final binary logit (positive = LumA).
        - pole_scores   : dict{"LumA": s_luma, "LumB": s_lumb}, each (batch,).
        - disagreement  : (batch,) — |s_LumA - (1 - s_LumB)|.
        """
        # Per-pole forward through encoder + fuser.
        z_fused: dict[str, torch.Tensor] = {}
        s_pole: dict[str, torch.Tensor] = {}
        for pole in self.pole_order:
            x_rna_pole = self.attn_rna[pole](x_rna)
            x_meth_pole = self.attn_meth[pole](x_meth)
            z_rna = self.rna_encoder(x_rna_pole)
            z_meth = self.meth_encoder(x_meth_pole)
            z, s = self.fusers[pole](z_rna, z_meth)
            z_fused[pole] = z
            s_pole[pole] = s

        # Disagreement signal between the two pole sub-classifiers.
        disagreement = disagreement_score(s_pole["LumA"], s_pole["LumB"])

        # Final head.
        logits = self.head(z_fused["LumA"], z_fused["LumB"], disagreement)

        return {
            "logits": logits,
            "pole_scores": s_pole,
            "disagreement": disagreement,
            "z_fused": z_fused,
        }


def count_dmoi_parameters(model: DMOIModel) -> dict[str, int]:
    """Break parameter count down by major component for sanity reporting."""
    def n(m: nn.Module) -> int:
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    return {
        "rna_encoder": n(model.rna_encoder),
        "meth_encoder": n(model.meth_encoder),
        "attention_layers": n(model.attn_rna) + n(model.attn_meth),  # should be 0
        "fusers": n(model.fusers),
        "head": n(model.head),
        "total": n(model),
    }
