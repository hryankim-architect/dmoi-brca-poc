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

v0.7 — optional pathway-pole attention (Variant D from v0.7 design doc).
If `n_pathways > 0`, a `PathwayPoleAttention` module learns a per-pole
softmax distribution over a Hallmark catalog of pathway-level scores.
The per-pole pathway feature is concatenated into the classifier head
input. v0.6-compatible default: `n_pathways=0` skips the branch entirely.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from dmoi_brca.encoder import MethEncoder, RNAEncoder
from dmoi_brca.fusion import PoleFuser, disagreement_score
from dmoi_brca.hypothesis_attention import PoleAttention, PoleMaskSet
from dmoi_brca.pathway_attention import PathwayPoleAttention


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
        n_pole_pathway_feats: int = 0,
    ) -> None:
        super().__init__()
        self.use_disagreement = use_disagreement
        self.n_pole_pathway_feats = n_pole_pathway_feats
        in_dim = (
            2 * fuse_dim
            + (1 if use_disagreement else 0)
            + n_pole_pathway_feats
        )
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
        pole_pathway_feat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts: list[torch.Tensor] = [z_luma, z_lumb]
        if self.use_disagreement:
            parts.append(disagreement.unsqueeze(-1))
        if self.n_pole_pathway_feats > 0:
            if pole_pathway_feat is None:
                raise ValueError(
                    "ClassifierHead expects pole_pathway_feat when "
                    "n_pole_pathway_feats > 0",
                )
            if pole_pathway_feat.shape[-1] != self.n_pole_pathway_feats:
                raise ValueError(
                    f"pole_pathway_feat last dim "
                    f"{pole_pathway_feat.shape[-1]} != expected "
                    f"{self.n_pole_pathway_feats}",
                )
            parts.append(pole_pathway_feat)
        x = torch.cat(parts, dim=-1)
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
        n_pathways: int = 0,
        pathway_proj_dim: int | None = None,
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
        self.n_pathways = n_pathways

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

        # v0.7: optional pathway-pole attention. When n_pathways > 0, a
        # learnable softmax distribution over Hallmark pathways is fit
        # per pole. The per-pole pathway feature is concatenated into
        # the ClassifierHead input. n_pathways=0 (default) restores v0.6.
        # v0.8 Variant C: pathway_proj_dim>0 promotes the scalar-per-pole
        # feature to a vector-per-pole projection so the head can read
        # per-pathway direction signals, not just aggregate magnitude.
        self.pathway_attention: PathwayPoleAttention | None = None
        n_pole_pathway_feats = 0
        if n_pathways > 0:
            self.pathway_attention = PathwayPoleAttention(
                n_pathways=n_pathways,
                pole_order=pole_order,
                proj_dim=pathway_proj_dim,
            )
            n_pole_pathway_feats = self.pathway_attention.out_dim

        # Final classifier head. The ablation flag controls whether the
        # disagreement scalar is included as an input feature.
        self.head = ClassifierHead(
            fuse_dim=fuse_out,
            hidden=head_hidden,
            dropout=dropout,
            use_disagreement=use_disagreement,
            n_pole_pathway_feats=n_pole_pathway_feats,
        )

    def forward(
        self,
        x_rna: torch.Tensor,
        x_meth: torch.Tensor,
        x_pathway: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass. Returns a dict containing:

        - logits        : (batch,) — final binary logit (positive = LumA).
        - pole_scores   : dict{"LumA": s_luma, "LumB": s_lumb}, each (batch,).
        - disagreement  : (batch,) — |s_LumA - (1 - s_LumB)|.
        - pole_pathway_feat : (batch, n_poles) — per-pole pathway feature
          (only present when pathway_attention is wired in; otherwise
          this key is omitted).

        Args:
            x_rna:     (batch, rna_dim).
            x_meth:    (batch, meth_dim).
            x_pathway: (batch, n_pathways) when v0.7 pathway branch is
                       wired (DMOIModel was built with n_pathways>0);
                       must be None when the branch is disabled.
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

        # v0.7: pathway-pole attention branch.
        pole_pathway_feat: torch.Tensor | None = None
        if self.pathway_attention is not None:
            if x_pathway is None:
                raise ValueError(
                    "DMOIModel was constructed with n_pathways>0 but "
                    "forward got x_pathway=None",
                )
            pole_pathway_feat = self.pathway_attention(x_pathway)
        elif x_pathway is not None:
            # Caller passed a pathway tensor but model wasn't built for it.
            raise ValueError(
                "DMOIModel was constructed with n_pathways=0 but "
                "forward got a non-None x_pathway",
            )

        # Final head.
        logits = self.head(
            z_fused["LumA"], z_fused["LumB"], disagreement,
            pole_pathway_feat=pole_pathway_feat,
        )

        out: dict[str, torch.Tensor] = {
            "logits": logits,
            "pole_scores": s_pole,
            "disagreement": disagreement,
            "z_fused": z_fused,
        }
        if pole_pathway_feat is not None:
            out["pole_pathway_feat"] = pole_pathway_feat
        return out


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
