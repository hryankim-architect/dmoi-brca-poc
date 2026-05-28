"""v0.7: learnable pathway-pole attention.

The v0.5 / v0.6 pathway IG rollups showed that the LumA pole loads
estrogen-response pathways and the LumB pole loads cell-cycle pathways
on both TCGA test and METABRIC. That finding was **post-hoc**: the
model itself attends to genes through the hand-picked pole masks in
`priors.HALLMARK_SETS`, and the pathway-level alignment was
reconstructed at interpretation time.

v0.7 makes that alignment **learnable**. The model gets:

1. Per-patient × per-Hallmark-pathway expression scores computed
   deterministically from RNA features (mean of pathway-member-gene
   expressions, mean-imputed when a pathway gene is absent from the
   feature vocabulary).
2. A `PathwayPoleAttention` module with one learnable softmax
   distribution over the Hallmark catalog per pole. After training,
   the softmax-normalized attention matrix IS the interpretation
   artifact -- we can read off "what pathways did the model decide
   define LumA / LumB?" and compare to the v0.6 hand-picked masks.

Total new trainable parameters: `n_poles * n_pathways` (100 for the
v0.6 default of 2 poles x 50 Hallmark sets). The module is designed
as a thin addition to v0.6; if the pathway branch is starved by the
gene-level branch, AUROC stays at v0.6's 0.968 and the learned
attention is the entire v0.7 deliverable.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
from torch import nn


def compute_pathway_expression_scores(
    rna: np.ndarray,
    feature_names: Sequence[str],
    pathways: Mapping[str, Sequence[str]],
) -> tuple[np.ndarray, list[str]]:
    """Per-patient mean expression of each pathway's member genes.

    Distinct from `dmoi_brca.pathway.pathway_aggregate`, which takes
    per-patient × per-gene IG attribution. This helper takes raw
    expression (or standardized expression) and returns per-patient
    × per-pathway mean values for use as model inputs.

    Args:
        rna:           (n_patients, n_genes) expression matrix.
        feature_names: length n_genes; gene names indexing rna columns.
        pathways:      pathway name -> list of gene names. Genes not
                       present in feature_names are silently dropped
                       from the per-pathway mean. Pathways with zero
                       genes in feature_names produce a column of
                       zeros (caller can drop them if desired).

    Returns:
        (scores, pathway_names) where
          scores: (n_patients, n_pathways) float32 array, columns
                  ordered by `pathway_names`.
          pathway_names: list[str] of the pathway columns in the
                  same order as `pathways.keys()` iteration.
    """
    if rna.ndim != 2:
        raise ValueError(
            f"rna must be 2-D (n_patients, n_genes), got {rna.shape}",
        )
    if rna.shape[1] != len(feature_names):
        raise ValueError(
            f"rna cols {rna.shape[1]} != feature_names length "
            f"{len(feature_names)}",
        )

    name_to_idx = {name: j for j, name in enumerate(feature_names)}
    pathway_names = list(pathways.keys())
    n_patients = rna.shape[0]
    out = np.zeros((n_patients, len(pathway_names)), dtype=np.float32)
    for j, pname in enumerate(pathway_names):
        idx = [name_to_idx[g] for g in pathways[pname] if g in name_to_idx]
        if not idx:
            # Pathway has zero genes in feature vocabulary; leave column zeros.
            continue
        out[:, j] = rna[:, idx].mean(axis=1).astype(np.float32)
    return out, pathway_names


class PathwayPoleAttention(nn.Module):
    """Learnable softmax attention over pathway scores, per pole (Variant D).

    For each pole P, learns a probability distribution `pole_attn[P, :]`
    over the n_pathways Hallmark sets via a softmax over a free
    parameter vector. The per-pole pathway feature is

        pole_feature[batch, P] = sum_k pole_attn[P, k] * pathway_scores[batch, k]

    i.e. each pole reads pathway scores through its own learned
    softmax-mixture. After training, `attn_weights` is the
    interpretation artifact.

    Args:
        n_pathways:  number of Hallmark pathways.
        pole_order:  tuple of pole names (e.g. ("LumA", "LumB")). The
                     order of the first dim of the attention matrix.

    Parameters:
        attn_logits: (n_poles, n_pathways) -- pre-softmax logits. Init
                     to small zero-mean noise so each pole starts near
                     a uniform mixture and gradient descent breaks the
                     symmetry.
    """

    def __init__(
        self,
        n_pathways: int,
        pole_order: Sequence[str],
        init_std: float = 0.5,
    ) -> None:
        super().__init__()
        if not pole_order:
            raise ValueError("pole_order must be non-empty")
        if n_pathways <= 0:
            raise ValueError(f"n_pathways must be > 0, got {n_pathways}")
        self.pole_order: tuple[str, ...] = tuple(pole_order)
        self.n_pathways = n_pathways
        # v0.7.1 Phase B: init_std default raised 0.01 -> 0.5 so each pole's
        # attention starts asymmetrically. Phase A's 0.01 left the softmax so
        # close to uniform that the (already-tiny) gradient was overwhelmed
        # by wd=1e-4 pulling logits toward zero before symmetry could break.
        self.attn_logits = nn.Parameter(
            torch.randn(len(self.pole_order), n_pathways) * init_std,
        )

    @property
    def attn_weights(self) -> torch.Tensor:
        """Softmax-normalized attention per pole. Shape (n_poles, n_pathways)."""
        return torch.softmax(self.attn_logits, dim=-1)

    def forward(self, pathway_scores: torch.Tensor) -> torch.Tensor:
        """Compute per-pole pathway feature.

        Args:
            pathway_scores: (batch, n_pathways).

        Returns:
            (batch, n_poles) -- per-patient per-pole pathway feature
            (scalar per pole; expand later if a richer feature is
            needed).
        """
        if pathway_scores.ndim != 2:
            raise ValueError(
                f"pathway_scores must be (batch, n_pathways), got "
                f"{tuple(pathway_scores.shape)}",
            )
        if pathway_scores.shape[1] != self.n_pathways:
            raise ValueError(
                f"pathway_scores cols {pathway_scores.shape[1]} != "
                f"n_pathways {self.n_pathways}",
            )
        # (batch, P) = (batch, K) @ (K, P)
        return pathway_scores @ self.attn_weights.T

    def top_k_pathways(
        self,
        pathway_names: Sequence[str],
        k: int = 3,
    ) -> dict[str, list[tuple[str, float]]]:
        """For each pole, return the top-k pathways by learned weight.

        Useful for interpretation after training (the v0.7 deliverable).

        Args:
            pathway_names: must be the same order used to compute
                pathway_scores at training time (i.e. the order
                returned by `compute_pathway_expression_scores`).
            k:             number of top pathways per pole.

        Returns:
            dict mapping pole name -> [(pathway_name, weight), ...]
            sorted by weight descending.
        """
        if len(pathway_names) != self.n_pathways:
            raise ValueError(
                f"pathway_names length {len(pathway_names)} != "
                f"n_pathways {self.n_pathways}",
            )
        weights = self.attn_weights.detach().cpu().numpy()
        out: dict[str, list[tuple[str, float]]] = {}
        for i, pole in enumerate(self.pole_order):
            order = np.argsort(-weights[i])[:k]
            out[pole] = [(pathway_names[j], float(weights[i, j])) for j in order]
        return out
