"""DMOI training loop — per-fold trainer + 5-fold CV runner.

Uses the **same** StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
as `dmoi_brca.baseline.run_cv`, so the per-fold metrics drop directly into
a head-to-head comparison MD against baseline_v2_per_fold.tsv.

Label convention (matches `dmoi_brca.baseline` + Day-5A baseline_v2):
    y[i] = 1 if patient is LumB (higher-proliferation, harder-to-call minority)
    y[i] = 0 if patient is LumA (lower-proliferation, ER-driven majority)

Loss: BCEWithLogitsLoss with pos_weight = n_neg / n_pos to handle the
~2.3:1 LumA:LumB imbalance.

Per-epoch flow:
    train pass -> compute mean training loss
    val pass   -> compute AUROC + balanced accuracy
    if val AUROC improved: snapshot best weights, reset patience
    if patience exhausted: stop early

Per-fold output (used by the audit driver in scripts/train_dmoi.py):
    FoldResult(fold, best_val_auc, best_val_bacc, best_epoch, train_loss_curve,
               val_auc_curve, mask_on_counts, runtime_seconds)
"""
from __future__ import annotations

import copy
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dmoi_brca.dmoi_model import DMOIModel
from dmoi_brca.hypothesis_attention import PoleMaskSet


@dataclass
class FoldResult:
    fold: int
    best_val_auc: float
    best_val_bacc: float
    best_epoch: int
    n_train: int
    n_test: int
    n_pos_train: int
    n_pos_test: int
    train_loss_curve: list[float] = field(default_factory=list)
    val_auc_curve: list[float] = field(default_factory=list)
    val_bacc_curve: list[float] = field(default_factory=list)
    runtime_seconds: float = 0.0
    # Day-4 additions — analytical artifacts captured at best epoch.
    # Each array is shape (n_test,); aligned by sample index in the val fold.
    val_labels: np.ndarray | None = None
    val_proba: np.ndarray | None = None
    val_logits: np.ndarray | None = None  # pre-sigmoid; for Day-5 temperature scaling
    val_disagreement: np.ndarray | None = None
    val_sample_ids: list[str] = field(default_factory=list)
    # Day-5 additions — nested calibration split (subset of train fold held out
    # from training and used exclusively for fitting temperature scaling).
    # Lets us report an honest (non-optimistic) calibrated ECE on val.
    cal_labels: np.ndarray | None = None
    cal_logits: np.ndarray | None = None  # logits at best epoch on cal split
    n_cal: int = 0


def _resolve_device(prefer: str = "auto") -> torch.device:
    """Pick MPS (Mac Apple Silicon) if available, else CPU."""
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer in ("mps", "auto") and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_one_fold(
    *,
    rna_train: np.ndarray,
    meth_train: np.ndarray,
    y_train: np.ndarray,
    rna_val: np.ndarray,
    meth_val: np.ndarray,
    y_val: np.ndarray,
    pole_masks: dict[str, PoleMaskSet],
    fold: int,
    rna_dim: int,
    meth_dim: int,
    latent_dim: int = 128,
    rna_hidden: Sequence[int] = (1024, 256),
    meth_hidden: Sequence[int] = (512,),
    fuse_hidden: Sequence[int] = (128,),
    fuse_out: int = 64,
    head_hidden: int = 32,
    dropout: float = 0.3,
    n_epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    patience: int = 10,
    seed: int = 42,
    device: str = "auto",
    verbose: bool = True,
    use_disagreement: bool = True,
    aux_weight: float = 0.0,
    calibration_frac: float = 0.0,
    pick_best_epoch: bool = True,
) -> FoldResult:
    """Train one DMOI model on one fold's data, return per-epoch metrics + best val AUC.

    If `calibration_frac > 0`, that fraction of the train fold is held out
    (stratified on y) and excluded from training. The model is evaluated on
    this calibration split at the best epoch and the logits are returned in
    `cal_logits` / `cal_labels` for downstream temperature scaling.

    If `pick_best_epoch=False`, the model trains for the full `n_epochs` with
    no early stopping and no best-epoch selection — the returned `val_*`
    arrays come from the FINAL epoch, not the val-AUC-maximizing one. Use
    this when val is actually a held-out test split that must not leak into
    model selection (e.g., for the v0.2 TCGA 80/20 held-out scoring).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    dev = _resolve_device(device)
    t_start = time.time()

    # Optional: carve a stratified calibration split out of train.
    # Standardization happens AFTER this split so the scaler is fit only
    # on data the model actually trains on.
    cal_idx_local: np.ndarray | None = None
    if calibration_frac > 0:
        if not 0.0 < calibration_frac < 0.5:
            raise ValueError(
                f"calibration_frac must be in (0, 0.5), got {calibration_frac}",
            )
        rng = np.random.default_rng(seed + 1_000 + fold)
        n = len(y_train)
        # Stratified split: hold out `calibration_frac` of each class.
        cal_mask = np.zeros(n, dtype=bool)
        for class_value in (0, 1):
            class_idx = np.where(y_train == class_value)[0]
            n_hold = max(1, int(round(len(class_idx) * calibration_frac)))
            chosen = rng.choice(class_idx, size=n_hold, replace=False)
            cal_mask[chosen] = True
        cal_idx_local = np.where(cal_mask)[0]
        train_idx_local = np.where(~cal_mask)[0]
        rna_cal_raw = rna_train[cal_idx_local]
        meth_cal_raw = meth_train[cal_idx_local]
        y_cal = y_train[cal_idx_local]
        rna_train = rna_train[train_idx_local]
        meth_train = meth_train[train_idx_local]
        y_train = y_train[train_idx_local]

    # Standardize using train fold stats only.
    rna_scaler = StandardScaler().fit(rna_train)
    meth_scaler = StandardScaler().fit(meth_train)
    rna_tr = rna_scaler.transform(rna_train).astype(np.float32)
    meth_tr = meth_scaler.transform(meth_train).astype(np.float32)
    rna_va = rna_scaler.transform(rna_val).astype(np.float32)
    meth_va = meth_scaler.transform(meth_val).astype(np.float32)

    # To tensors.
    X_rna_tr = torch.from_numpy(rna_tr).to(dev)
    X_meth_tr = torch.from_numpy(meth_tr).to(dev)
    y_tr_t = torch.from_numpy(y_train.astype(np.float32)).to(dev)
    X_rna_va = torch.from_numpy(rna_va).to(dev)
    X_meth_va = torch.from_numpy(meth_va).to(dev)

    # Calibration split tensors (if requested).
    X_rna_cal: torch.Tensor | None = None
    X_meth_cal: torch.Tensor | None = None
    if cal_idx_local is not None:
        rna_ca = rna_scaler.transform(rna_cal_raw).astype(np.float32)
        meth_ca = meth_scaler.transform(meth_cal_raw).astype(np.float32)
        X_rna_cal = torch.from_numpy(rna_ca).to(dev)
        X_meth_cal = torch.from_numpy(meth_ca).to(dev)

    # Class-balanced positive weight (LumB pos = label 1).
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float32, device=dev)

    # Build model + optimizer.
    model = DMOIModel(
        rna_dim=rna_dim, meth_dim=meth_dim, pole_masks=pole_masks,
        latent_dim=latent_dim, rna_hidden=rna_hidden, meth_hidden=meth_hidden,
        fuse_hidden=fuse_hidden, fuse_out=fuse_out, head_hidden=head_hidden,
        dropout=dropout, use_disagreement=use_disagreement,
    ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Training DataLoader.
    ds = TensorDataset(X_rna_tr, X_meth_tr, y_tr_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    best_val_auc = -float("inf")
    best_val_bacc = 0.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_val_proba: np.ndarray | None = None
    best_val_logits: np.ndarray | None = None
    best_val_disagreement: np.ndarray | None = None
    best_cal_logits: np.ndarray | None = None
    patience_left = patience
    train_loss_curve: list[float] = []
    val_auc_curve: list[float] = []
    val_bacc_curve: list[float] = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        for b_rna, b_meth, b_y in loader:
            opt.zero_grad()
            out = model(b_rna, b_meth)
            loss = loss_fn(out["logits"], b_y)
            if aux_weight > 0:
                # Option A: supervised sub-classifiers.
                # Convention: label=1 means LumB, label=0 means LumA.
                # LumA branch should predict P(patient is LumA) = 1 - label.
                # LumB branch should predict P(patient is LumB) = label.
                eps = 1e-7
                s_luma = out["pole_scores"]["LumA"].clamp(eps, 1.0 - eps)
                s_lumb = out["pole_scores"]["LumB"].clamp(eps, 1.0 - eps)
                luma_target = 1.0 - b_y
                lumb_target = b_y
                aux_loss = (
                    nn.functional.binary_cross_entropy(s_luma, luma_target)
                    + nn.functional.binary_cross_entropy(s_lumb, lumb_target)
                )
                loss = loss + aux_weight * aux_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite training loss at fold {fold} epoch {epoch}: {loss.item()}",
                )
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.item()))
        mean_loss = float(np.mean(epoch_losses))
        train_loss_curve.append(mean_loss)

        # Validation.
        model.eval()
        with torch.no_grad():
            val_out = model(X_rna_va, X_meth_va)
            val_logits_np = val_out["logits"].detach().cpu().numpy()
            val_proba = torch.sigmoid(val_out["logits"]).detach().cpu().numpy()
            val_disagreement = val_out["disagreement"].detach().cpu().numpy()
            cal_logits_np: np.ndarray | None = None
            if X_rna_cal is not None and X_meth_cal is not None:
                cal_out = model(X_rna_cal, X_meth_cal)
                cal_logits_np = cal_out["logits"].detach().cpu().numpy()
        val_auc = float(roc_auc_score(y_val, val_proba))
        val_pred = (val_proba >= 0.5).astype(int)
        val_bacc = float(balanced_accuracy_score(y_val, val_pred))
        val_auc_curve.append(val_auc)
        val_bacc_curve.append(val_bacc)

        if verbose:
            print(f"  fold {fold} ep {epoch:02d}  "
                  f"train_loss={mean_loss:.4f}  val_auc={val_auc:.4f}  val_bacc={val_bacc:.4f}",
                  flush=True)

        if not pick_best_epoch:
            # No peeking at val for best-epoch selection — use the latest
            # epoch's predictions unconditionally. This mode is for held-out
            # test scoring where val == test and any val-driven selection
            # would leak.
            best_val_auc = val_auc
            best_val_bacc = val_bacc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_val_proba = val_proba.copy()
            best_val_logits = val_logits_np.copy()
            best_val_disagreement = val_disagreement.copy()
            if cal_logits_np is not None:
                best_cal_logits = cal_logits_np.copy()
        elif val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_bacc = val_bacc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_val_proba = val_proba.copy()
            best_val_logits = val_logits_np.copy()
            best_val_disagreement = val_disagreement.copy()
            if cal_logits_np is not None:
                best_cal_logits = cal_logits_np.copy()
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                if verbose:
                    print(f"  fold {fold} early stop at epoch {epoch} "
                          f"(best epoch {best_epoch}, best val_auc {best_val_auc:.4f})")
                break

    # Restore best weights into model so the caller can introspect (e.g.
    # to extract calibration plots in a later phase).
    if best_state is not None:
        model.load_state_dict(best_state)

    runtime = time.time() - t_start
    cal_labels_out: np.ndarray | None = (
        y_cal.astype(np.int64).copy() if cal_idx_local is not None else None
    )
    n_cal_out = int(len(y_cal)) if cal_idx_local is not None else 0
    return FoldResult(
        fold=fold,
        best_val_auc=best_val_auc,
        best_val_bacc=best_val_bacc,
        best_epoch=best_epoch,
        n_train=len(y_train),
        n_test=len(y_val),
        n_pos_train=n_pos,
        n_pos_test=int(y_val.sum()),
        train_loss_curve=train_loss_curve,
        val_auc_curve=val_auc_curve,
        val_bacc_curve=val_bacc_curve,
        runtime_seconds=runtime,
        val_labels=y_val.astype(np.int64).copy(),
        val_proba=best_val_proba,
        val_logits=best_val_logits,
        val_disagreement=best_val_disagreement,
        cal_labels=cal_labels_out,
        cal_logits=best_cal_logits,
        n_cal=n_cal_out,
    )


def run_dmoi_cv(
    *,
    rna: np.ndarray,
    meth: np.ndarray,
    y: np.ndarray,
    pole_masks: dict[str, PoleMaskSet],
    n_splits: int = 5,
    random_state: int = 42,
    verbose: bool = True,
    **fold_kwargs,
) -> list[FoldResult]:
    """Run K-fold CV, returning per-fold results.

    Same StratifiedKFold contract as `dmoi_brca.baseline.run_cv` so that the
    DMOI fold-by-fold metrics drop directly into a head-to-head comparison
    against baseline_v2_per_fold.tsv.
    """
    if rna.shape[0] != y.shape[0] or meth.shape[0] != y.shape[0]:
        raise ValueError(
            f"rna {rna.shape}, meth {meth.shape}, y {y.shape} mismatched on axis 0",
        )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results: list[FoldResult] = []
    for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
        if verbose:
            print(f"\n=== Fold {fold_idx} / {n_splits} ===", flush=True)
            print(f"  train n={len(tr_idx)} (pos={int(y[tr_idx].sum())}), "
                  f"val n={len(te_idx)} (pos={int(y[te_idx].sum())})", flush=True)
        result = train_one_fold(
            rna_train=rna[tr_idx], meth_train=meth[tr_idx], y_train=y[tr_idx],
            rna_val=rna[te_idx], meth_val=meth[te_idx], y_val=y[te_idx],
            pole_masks=pole_masks,
            fold=fold_idx,
            rna_dim=rna.shape[1], meth_dim=meth.shape[1],
            verbose=verbose,
            **fold_kwargs,
        )
        results.append(result)
        if verbose:
            print(f"  fold {fold_idx} done: best_val_auc={result.best_val_auc:.4f} "
                  f"@ epoch {result.best_epoch} ({result.runtime_seconds:.1f}s)",
                  flush=True)
    return results


def aggregate_fold_results(results: list[FoldResult]) -> dict[str, float]:
    """Mean ± std across folds, ready for the audit MD comparison table."""
    aucs = np.array([r.best_val_auc for r in results])
    baccs = np.array([r.best_val_bacc for r in results])
    epochs = np.array([r.best_epoch for r in results])
    runtimes = np.array([r.runtime_seconds for r in results])
    return {
        "auc_mean": float(aucs.mean()),
        "auc_std": float(aucs.std(ddof=1)) if len(aucs) > 1 else 0.0,
        "bacc_mean": float(baccs.mean()),
        "bacc_std": float(baccs.std(ddof=1)) if len(baccs) > 1 else 0.0,
        "epoch_mean": float(epochs.mean()),
        "epoch_max": float(epochs.max()),
        "runtime_sec_total": float(runtimes.sum()),
        "n_folds": len(results),
    }
