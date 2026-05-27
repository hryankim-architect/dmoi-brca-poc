"""Day-4 baselines: sklearn LogReg + RF on RNA / meth / concatenated features.

These are the comparison anchors before DMOI hypothesis-conditioning is
introduced in Week-2. We use stratified 5-fold CV reporting AUROC + balanced
accuracy because the H+/H- split is imbalanced (~547 vs ~103).

The point is NOT to push these baselines hard — it's to record honest numbers
on the dual-modality cohort so the Week-2 hypothesis-conditioned encoder can
be measured against a non-trivial reference.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class FoldResult:
    fold: int
    feature_set: str          # "rna" | "meth" | "concat"
    model: str                # "logreg" | "rf"
    auc: float
    bacc: float
    n_train: int
    n_test: int
    n_pos_train: int          # H- count in train fold
    n_pos_test: int


def _make_pipeline(model: str, random_state: int) -> Pipeline:
    if model == "logreg":
        # Note: penalty='l2' is the sklearn default; explicit kwarg dropped to
        # silence sklearn 1.8+ FutureWarning on penalty deprecation.
        clf = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            random_state=random_state,
        )
    elif model == "rf":
        clf = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        )
    else:
        raise ValueError(f"unknown model: {model}")
    return Pipeline([("scaler", StandardScaler(with_mean=True)), ("clf", clf)])


def _eval_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model: str,
    random_state: int,
) -> tuple[float, float]:
    pipe = _make_pipeline(model, random_state)
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    pred = (proba >= 0.5).astype(int)
    bacc = balanced_accuracy_score(y_test, pred)
    return float(auc), float(bacc)


def run_cv(
    feature_sets: dict[str, np.ndarray],
    y: np.ndarray,
    *,
    n_splits: int = 5,
    random_state: int = 42,
    models: tuple[str, ...] = ("logreg", "rf"),
    progress: Callable[[str], None] | None = None,
) -> list[FoldResult]:
    """Run stratified K-fold CV for each (feature_set, model) combo."""
    if progress is None:
        def progress(msg: str) -> None:
            print(f"  [baseline] {msg}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    results: list[FoldResult] = []
    folds = list(skf.split(np.zeros(len(y)), y))
    for fset_name, X in feature_sets.items():
        for model in models:
            progress(f"running {fset_name} / {model} ({n_splits}-fold CV)...")
            for fold_idx, (tr_idx, te_idx) in enumerate(folds, start=1):
                X_tr, X_te = X[tr_idx], X[te_idx]
                y_tr, y_te = y[tr_idx], y[te_idx]
                auc, bacc = _eval_fold(X_tr, y_tr, X_te, y_te, model, random_state)
                results.append(FoldResult(
                    fold=fold_idx,
                    feature_set=fset_name,
                    model=model,
                    auc=auc,
                    bacc=bacc,
                    n_train=len(tr_idx),
                    n_test=len(te_idx),
                    n_pos_train=int(y_tr.sum()),
                    n_pos_test=int(y_te.sum()),
                ))
    return results


def aggregate(results: list[FoldResult]) -> dict[tuple[str, str], dict[str, float]]:
    """Mean ± std across folds for each (feature_set, model)."""
    out: dict[tuple[str, str], dict[str, float]] = {}
    keys = sorted({(r.feature_set, r.model) for r in results})
    for key in keys:
        rs = [r for r in results if (r.feature_set, r.model) == key]
        aucs = np.array([r.auc for r in rs])
        baccs = np.array([r.bacc for r in rs])
        out[key] = {
            "auc_mean": float(aucs.mean()),
            "auc_std": float(aucs.std(ddof=1)) if len(aucs) > 1 else 0.0,
            "bacc_mean": float(baccs.mean()),
            "bacc_std": float(baccs.std(ddof=1)) if len(baccs) > 1 else 0.0,
            "n_folds": len(rs),
        }
    return out
