"""Per-patient Integrated Gradients attribution for DMOI (v0.3).

Wraps Captum's `IntegratedGradients` to attribute three different
DMOI output heads — the final logit, the LumA pole sub-classifier
score, and the LumB pole sub-classifier score — back to per-feature
contributions across the RNA and methylation inputs.

Why three targets:
    1. final_logit  — primary output; the clinical headline answer.
    2. lumA_pole    — `s_LumA`; "why did the LumA branch say this score?"
    3. lumB_pole    — `s_LumB`; "why did the LumB branch say this score?"

Baseline: zero tensor in the standardized domain. Because train
StandardScaler centers each feature at zero, a zero input == the
train per-feature mean. This produces the clean interpretation
"compared to an average TCGA train patient."

Completeness axiom (verified by `tests/test_attribution.py`):
    sum_i IG_i(x) == f(x) - f(0)   within numerical tolerance.

The DMOI model's `forward` returns a dict; Captum needs a tensor.
We use closure-based wrappers that select the target tensor.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from captum.attr import IntegratedGradients

# Top-level so users can document or override per-call.
DEFAULT_N_STEPS = 50


@dataclass(frozen=True)
class AttributionResult:
    """Per-patient IG attribution for a single target."""
    target_name: str                      # "final_logit" / "lumA_pole" / "lumB_pole"
    sample_indices: tuple[int, ...]       # positions in the input arrays
    target_score: np.ndarray              # (n_samples,) the value being attributed
    rna_attribution: np.ndarray           # (n_samples, n_rna_features)
    meth_attribution: np.ndarray          # (n_samples, n_meth_features)
    rna_input: np.ndarray                 # (n_samples, n_rna_features) standardized input
    meth_input: np.ndarray                # (n_samples, n_meth_features) standardized input
    f_x: np.ndarray                       # (n_samples,) model output at x
    f_baseline: np.ndarray                # (n_samples,) model output at baseline
    n_steps: int                          # IG steps used


def _select_target_tensor(
    out: dict,
    target_name: str,
    pole_order: tuple[str, str] = ("LumA", "LumB"),
) -> torch.Tensor:
    """Pull the requested scalar tensor out of DMOIModel's forward dict.

    `pole_order[0]` is the "negative-class" pole (default LumA) and
    `pole_order[1]` is the "positive-class" pole (default LumB). Target
    names `lumA_pole` / `lumB_pole` are kept as the public API so that
    cohort-agnostic drivers (v0.6, v0.9) refer to "pole 0" / "pole 1"
    without caring what the actual class labels are.
    """
    if target_name == "final_logit":
        return out["logits"]
    if target_name == "lumA_pole":
        return out["pole_scores"][pole_order[0]]
    if target_name == "lumB_pole":
        return out["pole_scores"][pole_order[1]]
    raise ValueError(
        f"unknown target_name {target_name!r}; "
        "expected final_logit / lumA_pole / lumB_pole",
    )


def integrated_gradients_dmoi(
    model,                                # DMOIModel (avoid circular import)
    rna: np.ndarray,
    meth: np.ndarray,
    *,
    target: str = "final_logit",
    n_steps: int = DEFAULT_N_STEPS,
    device: str = "cpu",
    pole_order: tuple[str, str] = ("LumA", "LumB"),
) -> AttributionResult:
    """Run Integrated Gradients on the chosen DMOI output target.

    Args:
        model:      A trained DMOIModel (forward returns a dict).
        rna:        (n_samples, n_rna_features) standardized RNA input.
        meth:       (n_samples, n_meth_features) standardized methylation input.
        target:     "final_logit" / "lumA_pole" / "lumB_pole". The
                    pole names here are positional ("pole 0", "pole 1");
                    use `pole_order` to map them to the model's actual
                    pole names (e.g. Luminal/Basal for v0.9).
        n_steps:    Riemann-sum steps for the IG integral. Default 50.
        device:     torch device. Default "cpu" (deterministic).
        pole_order: model's pole order. Default ("LumA", "LumB") for
                    v0.6 backward-compat. Pass ("Luminal", "Basal") for
                    v0.9.

    Returns:
        AttributionResult with rna/meth attribution + bookkeeping fields.
    """
    if rna.shape[0] != meth.shape[0]:
        raise ValueError(
            f"rna rows {rna.shape[0]} != meth rows {meth.shape[0]}",
        )
    if n_steps < 2:
        raise ValueError(f"n_steps must be >= 2, got {n_steps}")

    dev = torch.device(device)
    model = model.to(dev).eval()

    rna_t = torch.from_numpy(rna.astype(np.float32)).to(dev)
    meth_t = torch.from_numpy(meth.astype(np.float32)).to(dev)
    rna_baseline = torch.zeros_like(rna_t)
    meth_baseline = torch.zeros_like(meth_t)

    def _fwd(r: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        out = model(r, m)
        return _select_target_tensor(out, target, pole_order)

    # Evaluate f(x) and f(baseline) for the completeness check + reporting.
    with torch.no_grad():
        f_x = _fwd(rna_t, meth_t).detach().cpu().numpy()
        f_baseline = _fwd(rna_baseline, meth_baseline).detach().cpu().numpy()

    ig = IntegratedGradients(_fwd)
    attr_rna, attr_meth = ig.attribute(
        inputs=(rna_t, meth_t),
        baselines=(rna_baseline, meth_baseline),
        n_steps=n_steps,
    )
    return AttributionResult(
        target_name=target,
        sample_indices=tuple(range(rna.shape[0])),
        target_score=f_x.astype(np.float32),
        rna_attribution=attr_rna.detach().cpu().numpy().astype(np.float32),
        meth_attribution=attr_meth.detach().cpu().numpy().astype(np.float32),
        rna_input=rna.astype(np.float32),
        meth_input=meth.astype(np.float32),
        f_x=f_x.astype(np.float64),
        f_baseline=f_baseline.astype(np.float64),
        n_steps=n_steps,
    )


def completeness_residual(attr: AttributionResult) -> np.ndarray:
    """Per-sample IG completeness residual: |sum(IG) - (f(x) - f(0))|."""
    total = (
        attr.rna_attribution.sum(axis=1) + attr.meth_attribution.sum(axis=1)
    )
    expected = attr.f_x - attr.f_baseline
    return np.abs(total - expected).astype(np.float64)


def top_k_per_patient(
    attr: AttributionResult,
    rna_feature_names: Sequence[str],
    meth_feature_names: Sequence[str],
    k: int = 10,
) -> list[dict]:
    """Per-patient top-K contributors by |attribution|, across both modalities.

    Returns a list of dicts (one per patient) of the form:
        {
            "sample_idx": int,
            "target_score": float,
            "topk_rna":  [(feature_name, attribution, input_value), ...] length k,
            "topk_meth": [(feature_name, attribution, input_value), ...] length k,
        }
    """
    if len(rna_feature_names) != attr.rna_attribution.shape[1]:
        raise ValueError(
            f"rna_feature_names length {len(rna_feature_names)} != "
            f"rna attribution cols {attr.rna_attribution.shape[1]}",
        )
    if len(meth_feature_names) != attr.meth_attribution.shape[1]:
        raise ValueError(
            f"meth_feature_names length {len(meth_feature_names)} != "
            f"meth attribution cols {attr.meth_attribution.shape[1]}",
        )
    out: list[dict] = []
    rna_names = list(rna_feature_names)
    meth_names = list(meth_feature_names)
    for i in range(attr.rna_attribution.shape[0]):
        # Take top-K by absolute value, but keep the signed attribution.
        rna_abs_top = np.argpartition(-np.abs(attr.rna_attribution[i]), k - 1)[:k]
        rna_abs_top = rna_abs_top[np.argsort(-np.abs(attr.rna_attribution[i, rna_abs_top]))]
        meth_abs_top = np.argpartition(-np.abs(attr.meth_attribution[i]), k - 1)[:k]
        meth_abs_top = meth_abs_top[
            np.argsort(-np.abs(attr.meth_attribution[i, meth_abs_top]))
        ]
        out.append({
            "sample_idx": int(attr.sample_indices[i]),
            "target_score": float(attr.target_score[i]),
            "topk_rna": [
                (
                    rna_names[j],
                    float(attr.rna_attribution[i, j]),
                    float(attr.rna_input[i, j]),
                )
                for j in rna_abs_top
            ],
            "topk_meth": [
                (
                    meth_names[j],
                    float(attr.meth_attribution[i, j]),
                    float(attr.meth_input[i, j]),
                )
                for j in meth_abs_top
            ],
        })
    return out


def global_aggregate(
    attr: AttributionResult,
    rna_feature_names: Sequence[str],
    meth_feature_names: Sequence[str],
    top_k: int = 50,
) -> dict[str, list[tuple[str, float]]]:
    """Global feature importance = mean |attribution| across patients.

    Returns:
        {"rna": [(name, mean_abs_attr), ...] top_k,
         "meth": [(name, mean_abs_attr), ...] top_k}
    """
    if len(rna_feature_names) != attr.rna_attribution.shape[1]:
        raise ValueError("rna feature_names length mismatch")
    if len(meth_feature_names) != attr.meth_attribution.shape[1]:
        raise ValueError("meth feature_names length mismatch")
    rna_mean = np.abs(attr.rna_attribution).mean(axis=0)
    meth_mean = np.abs(attr.meth_attribution).mean(axis=0)
    rna_top_idx = np.argsort(-rna_mean)[:top_k]
    meth_top_idx = np.argsort(-meth_mean)[:top_k]
    return {
        "rna": [(rna_feature_names[i], float(rna_mean[i])) for i in rna_top_idx],
        "meth": [(meth_feature_names[i], float(meth_mean[i])) for i in meth_top_idx],
    }
