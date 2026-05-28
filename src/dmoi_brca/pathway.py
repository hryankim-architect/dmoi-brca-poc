"""Pathway-level aggregation of per-gene IG attributions (v0.5).

Given a per-patient × per-gene attribution matrix and a pathway → gene-set
mapping (typically `priors.HALLMARK_SETS`), roll up to per-pathway scores:

- `mean_abs_ig`  : mean |IG| across genes in the pathway, averaged over
                   patients. The "how loudly does this pathway speak"
                   metric.
- `sum_signed`   : sum of signed IG across genes in the pathway, averaged
                   over patients. The "which direction does this pathway
                   push the prediction" metric.
- `signed_mean`  : sum_signed / n_genes_in_pathway. Direction normalized
                   by pathway size, so larger pathways don't dominate.
- `n_pathway_genes_in_inputs` : how many of the pathway's genes are
                                actually represented in the input feature
                                set (the rest are absent / mean-imputed).
- `n_inputs_in_pathway` : redundant — same number, kept for readability.

The v0.3+v0.4 finding was at the per-gene level: "lumA picked FOXC1/BCL2,
lumB picked RANBP1/NBN/ZW10/POLA2." The v0.5 pathway aggregation expresses
the same finding one level up: "lumA loaded heavily on
ESTROGEN_RESPONSE_EARLY/LATE; lumB loaded heavily on E2F_TARGETS,
G2M_CHECKPOINT, MYC_TARGETS_V1." Both views agree because the per-gene
top features happen to be members of the expected Hallmark sets.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PathwayScore:
    """Per-pathway aggregate scores across the cohort."""
    pathway_name: str
    n_pathway_genes_total: int        # genes in the Hallmark set definition
    n_pathway_genes_in_inputs: int    # of those, how many appear in feature_names
    mean_abs_ig: float                # mean |IG| across patients and pathway genes
    sum_signed: float                 # mean over patients of sum_g IG[patient, g in pathway]
    signed_mean: float                # sum_signed / n_pathway_genes_in_inputs


def pathway_aggregate(
    attribution: np.ndarray,
    feature_names: Sequence[str],
    pathways: Mapping[str, Sequence[str]],
) -> list[PathwayScore]:
    """Aggregate per-patient × per-gene attribution into per-pathway scores.

    Args:
        attribution:   Shape (n_patients, n_genes), signed IG values.
        feature_names: Length n_genes. Names that index attribution[:, j].
        pathways:      Dict mapping pathway_name -> iterable of gene names.

    Returns:
        List[PathwayScore], one per pathway in `pathways`. Pathways with
        zero representation in `feature_names` are returned with all
        scores set to 0 and `n_pathway_genes_in_inputs == 0` so the
        caller can drop them if desired.
    """
    if attribution.ndim != 2:
        raise ValueError(
            f"attribution must be 2-D (n_patients, n_genes), got "
            f"{attribution.shape}",
        )
    if attribution.shape[1] != len(feature_names):
        raise ValueError(
            f"attribution cols {attribution.shape[1]} != feature_names "
            f"length {len(feature_names)}",
        )

    name_to_idx = {name: j for j, name in enumerate(feature_names)}
    out: list[PathwayScore] = []
    for pathway_name, genes in pathways.items():
        gene_set = set(genes)
        indices = [
            name_to_idx[g] for g in gene_set if g in name_to_idx
        ]
        if not indices:
            out.append(PathwayScore(
                pathway_name=pathway_name,
                n_pathway_genes_total=len(gene_set),
                n_pathway_genes_in_inputs=0,
                mean_abs_ig=0.0,
                sum_signed=0.0,
                signed_mean=0.0,
            ))
            continue
        sub = attribution[:, indices]
        # mean |IG| across patients × pathway-member genes.
        mean_abs = float(np.abs(sub).mean())
        # Per patient, sum signed IG over pathway-member genes, then mean over patients.
        sum_signed = float(sub.sum(axis=1).mean())
        signed_mean = sum_signed / len(indices)
        out.append(PathwayScore(
            pathway_name=pathway_name,
            n_pathway_genes_total=len(gene_set),
            n_pathway_genes_in_inputs=len(indices),
            mean_abs_ig=mean_abs,
            sum_signed=sum_signed,
            signed_mean=signed_mean,
        ))
    return out


def rank_pathways(
    scores: Sequence[PathwayScore],
    by: str = "mean_abs_ig",
    descending: bool = True,
) -> list[PathwayScore]:
    """Sort PathwayScores by one of the score fields.

    Args:
        scores:     output of `pathway_aggregate`.
        by:         "mean_abs_ig" / "sum_signed" / "signed_mean".
        descending: largest first if True.
    """
    valid = {"mean_abs_ig", "sum_signed", "signed_mean"}
    if by not in valid:
        raise ValueError(f"by must be one of {sorted(valid)}, got {by!r}")
    keyfn = lambda s: getattr(s, by)  # noqa: E731
    return sorted(scores, key=keyfn, reverse=descending)
