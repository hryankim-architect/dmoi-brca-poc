"""External-cohort prediction helpers for DMOI v0.2 (Path A').

Cross-cohort generalization needs three things to line up between the
training cohort (TCGA-BRCA) and the external cohort (METABRIC):

1. **Gene-symbol alignment.** TCGA HiSeqV2 has 20,530 Hugo symbols;
   METABRIC HT-12 v3 has 20,603. We project the external matrix into
   the train gene order, mean-imputing genes that exist in train but
   not in the external cohort.

2. **Cross-cohort distribution matching.** Even after gene alignment,
   per-gene expression distributions differ between platforms (RNA-seq
   FPKM vs Illumina microarray intensity). `quantile_normalize_to_train`
   forces the external matrix's per-gene rank-distribution to match the
   train's per-gene rank-distribution — a standard approach in
   cross-platform validation studies.

3. **Modality silencing.** METABRIC has no HM450 methylation. The
   methylation branch is silenced at inference time by passing a
   zero-tensor of the correct shape. Because the train StandardScaler
   centers methylation at zero, this is equivalent to passing the train
   mean — the model's methylation pole encoder sees its "neutral" input.

The cost of (3) is that the dual-perspective architecture's methylation
contribution is muted. v0.2 reports this as a degraded-mode external
test — primary claim (RNA encoder generalizes) is tested, but the
dual-modality claim is not. CPTAC and other public cohorts have no
paired RNA+HM450 BRCA data, so multi-modal external validation is
deferred to a hypothetical v0.3 with a non-public cohort.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def collapse_duplicate_genes(
    expression: np.ndarray,
    gene_symbols: Sequence[str],
    *,
    aggregator: str = "mean",
) -> tuple[np.ndarray, list[str]]:
    """Collapse rows sharing the same Hugo symbol.

    METABRIC's mRNA matrix has ~30-40 duplicate Hugo symbols (mostly
    pseudogenes and ambiguous mappings). We average their expression
    rows so each gene gets a single row.

    Args:
        expression:    Shape (n_genes, n_samples).
        gene_symbols:  Length n_genes.
        aggregator:    'mean' (default) or 'median'.

    Returns:
        (collapsed_expression, unique_gene_symbols) — both with length
        equal to the number of unique gene symbols.
    """
    if expression.shape[0] != len(gene_symbols):
        raise ValueError(
            f"expression rows {expression.shape[0]} != gene_symbols length "
            f"{len(gene_symbols)}",
        )
    if aggregator not in ("mean", "median"):
        raise ValueError(f"aggregator must be 'mean' or 'median', got {aggregator}")

    seen: dict[str, list[int]] = {}
    for i, g in enumerate(gene_symbols):
        seen.setdefault(g, []).append(i)

    unique_genes = list(seen)
    out = np.empty((len(unique_genes), expression.shape[1]), dtype=expression.dtype)
    agg = np.nanmean if aggregator == "mean" else np.nanmedian
    for j, g in enumerate(unique_genes):
        idx = seen[g]
        out[j] = expression[idx[0]] if len(idx) == 1 else agg(expression[idx], axis=0)
    return out, unique_genes


def align_to_train_genes(
    external_X: np.ndarray,
    external_genes: Sequence[str],
    train_genes: Sequence[str],
    *,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Project external expression matrix into the train gene order.

    Args:
        external_X:     Shape (n_external_samples, n_external_genes).
        external_genes: Length n_external_genes.
        train_genes:    Length n_train_genes — the model's expected order.
        fill_value:     Value for train_genes that are absent from
                        external_genes. Default 0.0 = "mean after the
                        train-fitted StandardScaler is applied", which
                        is the natural neutral input.

    Returns:
        Shape (n_external_samples, n_train_genes) — column order matches
        train_genes exactly.
    """
    if external_X.shape[1] != len(external_genes):
        raise ValueError(
            f"external_X cols {external_X.shape[1]} != external_genes length "
            f"{len(external_genes)}",
        )

    external_index = {g: j for j, g in enumerate(external_genes)}
    n_samples = external_X.shape[0]
    aligned = np.full(
        (n_samples, len(train_genes)), fill_value, dtype=external_X.dtype,
    )
    n_matched = 0
    for j_train, g in enumerate(train_genes):
        j_ext = external_index.get(g)
        if j_ext is not None:
            aligned[:, j_train] = external_X[:, j_ext]
            n_matched += 1
    if n_matched == 0:
        raise ValueError(
            "No genes overlap between external and train. Check gene-symbol "
            "format consistency (Hugo vs Entrez vs Ensembl).",
        )
    return aligned


def gene_overlap_stats(
    external_genes: Sequence[str],
    train_genes: Sequence[str],
) -> dict[str, int]:
    """How many genes are shared / external-only / train-only?"""
    ext = set(external_genes)
    train = set(train_genes)
    return {
        "n_external": len(ext),
        "n_train": len(train),
        "n_shared": len(ext & train),
        "n_external_only": len(ext - train),
        "n_train_only_mean_imputed": len(train - ext),
    }


def quantile_normalize_to_train(
    external_X: np.ndarray,
    train_X: np.ndarray,
) -> np.ndarray:
    """Map each external sample's per-gene distribution to the train's.

    For each gene independently, sort the train values and the external
    values, then rewrite each external value with the train value at the
    same rank position. This produces an external matrix whose per-gene
    distributions match the train matrix's exactly, eliminating
    platform-specific scale and shape differences.

    Args:
        external_X: Shape (n_external_samples, n_genes).
        train_X:    Shape (n_train_samples,  n_genes) — reference.

    Returns:
        Shape (n_external_samples, n_genes) — values drawn from train's
        per-gene empirical distribution.
    """
    if external_X.shape[1] != train_X.shape[1]:
        raise ValueError(
            f"external_X cols {external_X.shape[1]} != train_X cols "
            f"{train_X.shape[1]} — align gene order first.",
        )

    n_ext = external_X.shape[0]
    n_train = train_X.shape[0]
    n_genes = external_X.shape[1]
    out = np.empty_like(external_X, dtype=np.float32)

    # Per-gene quantile match. Precompute sorted train columns once.
    train_sorted = np.sort(train_X, axis=0)
    for j in range(n_genes):
        # Rank external values in [0, 1).
        ext_col = external_X[:, j]
        ranks = ext_col.argsort().argsort()  # rank positions (0..n_ext-1)
        # Map rank to a train index: rank q out of n_ext -> floor(q * n_train / n_ext)
        train_idx = (ranks.astype(np.float64) * n_train / n_ext).astype(np.int64)
        train_idx = np.clip(train_idx, 0, n_train - 1)
        out[:, j] = train_sorted[train_idx, j]
    return out


def make_silenced_meth(
    n_external_samples: int,
    n_meth_features: int,
    *,
    fill_value: float = 0.0,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Build a zero-tensor for the methylation branch when the external
    cohort has no methylation data.

    The train StandardScaler centers each meth feature at zero, so
    fill_value=0.0 corresponds to the train per-feature mean — the
    methylation encoder's neutral input.
    """
    if n_external_samples <= 0 or n_meth_features <= 0:
        raise ValueError(
            f"shape must be positive, got ({n_external_samples}, {n_meth_features})",
        )
    return np.full((n_external_samples, n_meth_features), fill_value, dtype=dtype)
