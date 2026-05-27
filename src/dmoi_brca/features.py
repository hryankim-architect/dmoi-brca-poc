"""TCGA-BRCA feature loading + top-variance selection for the Day-4 baseline.

Inputs come from UCSC Xena Hub (sample-matched, pre-normalized):
- HiSeqV2.gz             : RNA-seq HiSeqV2 Polya+ FPKM, log2(x+1), 20,530 genes
- HumanMethylation450.gz : HM450 beta values, 485,577 probes
- cohort.tsv             : Day-3 output (sample_id, group, has_rna, has_meth)

Methylation strategy: the full HM450 matrix (~485k probes x ~888 samples) is
~1.7 GB dense in float32. We stream probe-by-probe and keep only the top-K
highest-variance probes in a min-heap — memory bounded to O(K * n_samples)
regardless of total probe count. RNA-seq (20k genes) is small enough to load
whole with pandas.

Missing values: probes with NA in any cohort sample are excluded (permissive
baseline; no imputation in capability-portrait scope).
"""
from __future__ import annotations

import gzip
import heapq
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class FeatureMatrices:
    sample_ids: list[str]               # length n
    y: np.ndarray                       # length n, 0=H+, 1=H-
    rna: np.ndarray                     # n x p_rna
    meth: np.ndarray                    # n x p_meth
    rna_features: list[str]             # length p_rna (gene symbols)
    meth_features: list[str]            # length p_meth (probe ids)


def _load_xena_full(gz_path: Path, sample_ids: set[str]) -> tuple[pd.DataFrame, list[str]]:
    """Load a Xena matrix subsetted to requested sample IDs (small matrices only).

    Xena matrices are TSV: first column = feature id, remaining columns = sample IDs.
    Returns (samples x features) DataFrame and feature_ids in original order.
    """
    with gzip.open(gz_path, "rt") as fh:
        header = fh.readline().rstrip("\n").split("\t")
    feature_col = header[0]
    keep_cols = [feature_col] + [s for s in header[1:] if s in sample_ids]
    df = pd.read_csv(gz_path, sep="\t", usecols=keep_cols, low_memory=False)
    feature_ids = df[feature_col].astype(str).tolist()
    df = df.set_index(feature_col).T  # rows = samples
    return df, feature_ids


def _stream_topk_meth(
    gz_path: Path,
    sample_ids: set[str],
    k: int,
    chunksize: int = 20_000,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Stream HM450 matrix in chunks, retain top-k probes by sample variance.

    Uses pandas chunked TSV reader (~10x faster than per-line Python parsing).
    Memory bounded to O(chunksize * n_samples + k * n_samples) regardless of
    total probe count.

    Returns:
        X            : (n_samples, k) float32 array, rows in sample_order
        probes       : list[str] of length k, sorted high to low variance
        sample_order : list[str] sample IDs corresponding to X rows
    """
    with gzip.open(gz_path, "rt") as fh:
        header = fh.readline().rstrip("\n").split("\t")
    feature_col = header[0]
    keep_cols = [feature_col] + [c for c in header[1:] if c in sample_ids]
    sample_order = keep_cols[1:]
    if not sample_order:
        raise ValueError(f"No requested samples found in {gz_path.name} header")

    # Min-heap of (variance, tie_id, probe_id, values_array). tie_id avoids
    # numpy array comparison when variances tie.
    heap: list[tuple[float, int, str, np.ndarray]] = []
    tie = 0
    skipped_na = 0
    seen = 0

    reader = pd.read_csv(
        gz_path, sep="\t", usecols=keep_cols, chunksize=chunksize,
        low_memory=False, dtype={feature_col: str},
    )
    for chunk in reader:
        # Drop probes with any NA in this cohort.
        chunk_drop = chunk.dropna()
        skipped_na += len(chunk) - len(chunk_drop)
        seen += len(chunk)

        if chunk_drop.empty:
            continue

        probe_ids = chunk_drop[feature_col].astype(str).to_numpy()
        values = chunk_drop[sample_order].to_numpy(dtype=np.float32)
        # Variance across samples (axis=1: row-wise across columns).
        vars_ = values.var(axis=1)

        for i in range(len(probe_ids)):
            v = float(vars_[i])
            if len(heap) < k:
                heapq.heappush(heap, (v, tie, str(probe_ids[i]), values[i].copy()))
            elif v > heap[0][0]:
                heapq.heapreplace(heap, (v, tie, str(probe_ids[i]), values[i].copy()))
            tie += 1

        print(f"    [features.meth-stream] read {seen:,} probes, "
              f"skipped {skipped_na:,} NA-bearing, heap={len(heap)}")

    print(f"    [features.meth-stream] DONE: {seen:,} probes scanned, "
          f"{skipped_na:,} skipped (NA), retained top-{len(heap)}")

    # Sort heap high-to-low variance for stable feature order.
    heap.sort(key=lambda x: -x[0])
    probes = [item[2] for item in heap]
    X = np.stack([item[3] for item in heap], axis=1)  # samples x probes
    return X, probes, sample_order


def load_features(
    cohort_tsv: Path,
    rna_gz: Path,
    meth_gz: Path,
    *,
    meth_topk: int = 10_000,
    rna_topk: int | None = None,
    dual_modality_only: bool = True,
    positive_label: str = "H_minus_basal_tn",
) -> FeatureMatrices:
    """Load aligned (X_rna, X_meth, y) for the cohort.

    Args:
        cohort_tsv:         Cohort file (sample_id/group/has_rna/has_meth).
                            Day-3 cohort (H+/H-) or cohort_v2 (LumA/LumB).
        rna_gz / meth_gz:   Xena gzipped matrices.
        meth_topk:          Keep top-K most-variable methylation probes (default 10k).
        rna_topk:           If set, also keep top-K most-variable RNA genes.
        dual_modality_only: If True, restrict to patients with both modalities.
        positive_label:     Group label that gets y=1. Default 'H_minus_basal_tn'.
                            For cohort_v2 use 'LumB' (the higher-proliferation pole).

    Returns:
        FeatureMatrices with aligned sample order across rna/meth/y.
    """
    cohort = pd.read_csv(cohort_tsv, sep="\t")
    if dual_modality_only:
        cohort = cohort[cohort["has_rna"] & cohort["has_meth"]].copy()
    if cohort.empty:
        raise ValueError(f"Empty cohort after dual-modality filter in {cohort_tsv}")

    wanted_samples = set(cohort["sample_id"].astype(str))

    group_counts = cohort["group"].value_counts().to_dict()
    print(f"  [features] cohort: {len(cohort)} patients (groups: {group_counts})")

    print(f"  [features] loading RNA-seq from {rna_gz.name} (full load)...")
    rna_df, rna_feats = _load_xena_full(rna_gz, wanted_samples)
    print(f"    rna shape: {rna_df.shape}")

    print(f"  [features] streaming methylation from {meth_gz.name} "
          f"(top-{meth_topk} variance probes)...")
    meth_X_full, meth_feats_full, meth_sample_order = _stream_topk_meth(
        meth_gz, wanted_samples, meth_topk,
    )
    meth_df = pd.DataFrame(meth_X_full, index=meth_sample_order, columns=meth_feats_full)

    # Intersect sample sets across all 3 sources.
    common = sorted(set(rna_df.index) & set(meth_df.index) & wanted_samples)
    print(f"  [features] intersecting samples: {len(common)}")
    if not common:
        raise ValueError("No samples common to cohort + RNA + methylation matrices")

    rna_df = rna_df.loc[common]
    meth_df = meth_df.loc[common]

    # Label vector: y=1 if group equals positive_label.
    cohort = cohort.set_index("sample_id").loc[common]
    y = (cohort["group"] == positive_label).astype(int).to_numpy()

    rna_X = rna_df.to_numpy(dtype=np.float32)
    meth_X = meth_df.to_numpy(dtype=np.float32)
    meth_feats = meth_df.columns.tolist()

    # Optional top-variance trim for RNA-seq (the meth side already trimmed in stream).
    if rna_topk is not None and rna_topk < rna_X.shape[1]:
        var = np.nanvar(rna_X, axis=0)
        top_idx = np.argpartition(-var, rna_topk)[:rna_topk]
        top_idx = top_idx[np.argsort(-var[top_idx])]
        rna_X = rna_X[:, top_idx]
        rna_feats = [rna_feats[i] for i in top_idx]
        print(f"    rna trimmed to top-{rna_topk}: {rna_X.shape}")

    print(f"  [features] final shapes: rna={rna_X.shape}, meth={meth_X.shape}")
    return FeatureMatrices(
        sample_ids=common,
        y=y,
        rna=rna_X,
        meth=meth_X,
        rna_features=rna_feats,
        meth_features=meth_feats,
    )
