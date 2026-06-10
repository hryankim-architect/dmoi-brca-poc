"""Label-free feature-selection comparison: DMOI biological prior vs baselines.

Context. Omran et al. 2025 (J Transl Med, doi:10.1186/s12967-025-06662-5) compared
two *unsupervised* multi-omics integrators as feature selectors for TCGA-BRCA PAM50
subtyping — MOFA+ (statistical) and MoGCN (deep learning) — by feeding the selected
features to downstream linear (SVC) and nonlinear (LR) classifiers and reporting
weighted-F1 under cross-validation. MOFA+ was the better selector (best F1 0.75).

This module evaluates DMOI's biological prior on the *same footing*: the Hallmark
gene-set + HM450 cis-mapping feature restriction is itself **label-free** (it depends
only on prior knowledge, not on the subtype labels), so it is a directly comparable
unsupervised feature selector. We benchmark three label-free selectors —

    * DMOI-prior   : RNA genes in the Hallmark pole universe + methylation probes
                     cis-mapped to those genes (knowledge-based).
    * top-variance : the n highest-variance features (statistical, the usual baseline).
    * (the published MOFA+/MoGCN numbers are cited as a literature reference; their
       exact 960-sample / 3-omics dataset is not re-run here — see the script.)

— through the *same* downstream protocol (LR + linear SVC, stratified k-fold,
weighted-F1) plus paradigm-neutral feature-quality metrics (Calinski-Harabasz and
Davies-Bouldin of the selected-feature space against the true subtypes).

Selectors here take only knowledge/variance — never ``y`` — keeping the comparison
to MOFA+/MoGCN fair (unsupervised vs unsupervised). The downstream classifier is the
only supervised step, identical across selectors.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from dmoi_brca.priors import HALLMARK_SETS


def hallmark_gene_universe(
    hallmark_sets: Mapping[str, Sequence[str]] | None = None,
) -> set[str]:
    """Union of all genes across the given Hallmark sets (default: the 5 curated sets)."""
    sets = hallmark_sets if hallmark_sets is not None else HALLMARK_SETS
    universe: set[str] = set()
    for genes in sets.values():
        universe.update(genes)
    return universe


def prior_rna_indices(
    feature_genes: Sequence[str],
    hallmark_sets: Mapping[str, Sequence[str]] | None = None,
) -> list[int]:
    """Column indices of RNA features whose gene is in the Hallmark universe (label-free)."""
    universe = hallmark_gene_universe(hallmark_sets)
    return [i for i, g in enumerate(feature_genes) if g in universe]


def prior_meth_indices(
    feature_probes: Sequence[str],
    cis_mapping: Mapping[str, set[str]],
    hallmark_sets: Mapping[str, Sequence[str]] | None = None,
) -> list[int]:
    """Column indices of methylation probes cis-mapped to a Hallmark gene (label-free)."""
    universe = hallmark_gene_universe(hallmark_sets)
    out: list[int] = []
    for j, probe in enumerate(feature_probes):
        genes = cis_mapping.get(probe)
        if genes and (genes & universe):
            out.append(j)
    return out


def topvar_indices(X: np.ndarray, k: int) -> list[int]:
    """Indices of the k highest-variance columns of X (label-free statistical baseline)."""
    if k <= 0:
        return []
    k = min(k, X.shape[1])
    var = np.nanvar(X, axis=0)
    idx = np.argsort(-var)[:k]
    return idx.tolist()


def topvar_within(X: np.ndarray, candidate_idx: Sequence[int], k: int) -> list[int]:
    """Top-k highest-variance columns among ``candidate_idx``, as ORIGINAL indices.

    Lets a knowledge prior (candidate_idx) be capped to a fixed budget by variance —
    still label-free — while keeping the original column indices so selections from
    different selectors can be compared (e.g. via ``jaccard_index``).
    """
    cand = list(candidate_idx)
    if k <= 0 or not cand:
        return []
    var = np.nanvar(X[:, cand], axis=0)
    order = np.argsort(-var)[: min(k, len(cand))]
    return [cand[i] for i in order]


def jaccard_index(a: Sequence[int], b: Sequence[int]) -> float:
    """Jaccard overlap |A∩B| / |A∪B| of two index/feature sets (0.0 if both empty)."""
    sa, sb = set(a), set(b)
    union = sa | sb
    return len(sa & sb) / len(union) if union else 0.0


def eval_selector(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int = 5,
    seed: int = 42,
) -> dict[str, float]:
    """Downstream evaluation of a selected-feature matrix, matching Omran et al.

    Returns weighted-F1 from LogisticRegression (nonlinear-equivalent in the paper)
    and a linear SVC under stratified k-fold CV, plus Calinski-Harabasz (higher=better)
    and Davies-Bouldin (lower=better) of the standardized feature space vs the true
    labels. ``y`` is used ONLY by the supervised downstream step / cluster metrics —
    never by the selectors. sklearn is imported lazily so the core package import
    stays dependency-light.
    """
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
    from sklearn.metrics import (  # noqa: PLC0415
        calinski_harabasz_score,
        davies_bouldin_score,
    )
    from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: PLC0415
    from sklearn.pipeline import make_pipeline  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415
    from sklearn.svm import SVC  # noqa: PLC0415

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    lr = make_pipeline(StandardScaler(),
                       LogisticRegression(max_iter=2000, class_weight="balanced"))
    sv = make_pipeline(StandardScaler(), SVC(kernel="linear", class_weight="balanced"))
    f1_lr = float(cross_val_score(lr, X, y, cv=cv, scoring="f1_weighted").mean())
    f1_svc = float(cross_val_score(sv, X, y, cv=cv, scoring="f1_weighted").mean())
    Xz = StandardScaler().fit_transform(X)
    return {
        "n_features": int(X.shape[1]),
        "lr_weighted_f1": f1_lr,
        "svc_weighted_f1": f1_svc,
        "calinski_harabasz": float(calinski_harabasz_score(Xz, y)),
        "davies_bouldin": float(davies_bouldin_score(Xz, y)),
    }
