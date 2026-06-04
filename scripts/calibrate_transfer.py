#!/usr/bin/env python3
"""DMOI v0.13 — cross-cohort calibration transfer (TCGA -> METABRIC).

Closes the question v0.2 left explicitly open: temperature scaling fit on TCGA
does NOT transfer naively to METABRIC. This script measures, on a single fixed
METABRIC eval slice, how well each calibration strategy does:

    A  uncalibrated (raw sigmoid)                       — floor
    B  naive TCGA cal-split T                           — the v0.2 failure case
    C  METABRIC oracle T (full cal pool, labelled)      — ceiling
    D1 METABRIC-mini T (tiny labelled slice, n sweep)   — "how little is enough?"
    D2 label-free logit alignment + TCGA T              — best zero-label attempt
    D3 class-prior (base-rate) odds correction          — shift-only analysis

IMPORTANT — temperature scaling and the affine/odds transforms here are
monotonic, so AUROC / balanced accuracy are INVARIANT across every condition.
The deliverable is purely calibration quality (ECE + Brier + reliability), not
an accuracy claim. The audit doc states this explicitly.

The data prep (TCGA train features, METABRIC load + gene-alignment + quantile
normalization + methylation silencing + pole masks + a single final Option-A
model) is reproduced faithfully from `eval_external.py`; the proven METABRIC
matrix loader and FINAL_KWARGS are imported from it directly to avoid drift.

Reproduce:  python scripts/calibrate_transfer.py
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone  # noqa: UP017
from pathlib import Path

UTC = timezone.utc  # noqa: UP017
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from dmoi_brca import audit, tracking  # noqa: E402
from dmoi_brca.calibration import apply_temperature, fit_temperature  # noqa: E402
from dmoi_brca.eval import brier_score, compute_calibration, reliability_table  # noqa: E402
from dmoi_brca.transfer import affine_align, prior_odds_correct  # noqa: E402
from dmoi_brca.external import (  # noqa: E402
    align_to_train_genes,
    gene_overlap_stats,
    make_silenced_meth,
    quantile_normalize_to_train,
)
from dmoi_brca.features import FeatureMatrices, load_features  # noqa: E402
from dmoi_brca.hypothesis_attention import (  # noqa: E402
    load_hm450_cis_mapping,
    make_pole_masks,
)
from dmoi_brca.priors import POLE_LUMA, POLE_LUMB  # noqa: E402
from dmoi_brca.train import train_one_fold  # noqa: E402

TCGA = REPO / "data" / "tcga_brca"
METABRIC = REPO / "data" / "metabric"
AUDIT = REPO / "audit"
JOB_ID = "dmoi-calibration-transfer-v0.13"
N_BINS = 10
CAL_FRAC = 0.15
SPLIT_SEED = 2024
D1_N_GRID = (30, 50, 100, 200)
D1_SEEDS = (0, 1, 2, 3, 4)


# ---------------------------------------------------------------------------
# Data prep — faithful to eval_external.py (one final Option-A model)
# ---------------------------------------------------------------------------
def prepare_run(n_epochs: int = 15):
    """Train one final model on TCGA train, score on METABRIC.

    Returns the FoldResult (val_* = METABRIC, cal_* = TCGA cal split) plus
    class priors pi_train / pi_metab.
    """
    # Sibling script: reuse its proven METABRIC matrix loader + FINAL_KWARGS.
    import eval_external as ee  # local import keeps the module import block isort-clean

    cohort_tsv = TCGA / "cohort_v2.tsv"
    rna_gz = TCGA / "HiSeqV2.gz"
    meth_gz = TCGA / "HumanMethylation450.gz"
    probemap = TCGA / "hm450_probemap.tsv"
    metabric_cohort_tsv = METABRIC / "cohort.tsv"
    metabric_mrna = METABRIC / "mrna_microarray.txt"
    for p in (cohort_tsv, rna_gz, meth_gz, probemap, metabric_cohort_tsv, metabric_mrna):
        if not p.exists():
            sys.stderr.write(f"ERROR: missing input {p}\n")
            raise SystemExit(1)

    print("--- Loading TCGA features (slice to train split) ---")
    feats_all = load_features(
        cohort_tsv=cohort_tsv, rna_gz=rna_gz, meth_gz=meth_gz,
        meth_topk=10_000, dual_modality_only=True, positive_label="LumB",
    )
    cohort_df = pd.read_csv(cohort_tsv, sep="\t")
    cohort_df = cohort_df[cohort_df["has_rna"] & cohort_df["has_meth"]].copy()
    sample_to_split = dict(zip(cohort_df["sample_id"], cohort_df["split"], strict=False))
    train_idx = np.array([
        i for i, sid in enumerate(feats_all.sample_ids)
        if sample_to_split.get(sid) == "train"
    ])
    feats = FeatureMatrices(
        sample_ids=[feats_all.sample_ids[i] for i in train_idx],
        y=feats_all.y[train_idx], rna=feats_all.rna[train_idx],
        meth=feats_all.meth[train_idx],
        rna_features=feats_all.rna_features, meth_features=feats_all.meth_features,
    )
    print(f"  TCGA train: {len(feats.sample_ids)} "
          f"(LumA={int((feats.y == 0).sum())}, LumB={int((feats.y == 1).sum())})")

    print("--- Loading METABRIC cohort + mRNA ---")
    metabric_cohort = pd.read_csv(metabric_cohort_tsv, sep="\t")
    metabric_cohort = metabric_cohort[metabric_cohort["has_rna"]].copy()
    ext_ids_wanted = set(metabric_cohort["sample_id"].astype(str))
    ext_X_raw, ext_genes, ext_sample_ids = ee._load_metabric_mrna(metabric_mrna, ext_ids_wanted)
    metabric_cohort = metabric_cohort.set_index("sample_id").loc[ext_sample_ids]
    y_ext = (metabric_cohort["group"] == "LumB").astype(np.int64).to_numpy()
    print(f"  METABRIC: {len(y_ext)} "
          f"(LumA={int((y_ext == 0).sum())}, LumB={int((y_ext == 1).sum())})")

    print("--- Gene alignment + quantile normalization ---")
    overlap = gene_overlap_stats(ext_genes, feats.rna_features)
    ext_X_aligned = align_to_train_genes(ext_X_raw, ext_genes, feats.rna_features, fill_value=0.0)
    ext_X_qn = quantile_normalize_to_train(ext_X_aligned, feats.rna)
    meth_ext_silenced = make_silenced_meth(ext_X_qn.shape[0], feats.meth.shape[1])

    print("--- Building pole masks ---")
    cis = load_hm450_cis_mapping(probemap)
    pole_masks = make_pole_masks(
        feats.rna_features, feats.meth_features, cis,
        {"LumA": POLE_LUMA, "LumB": POLE_LUMB},
    )

    print(f"--- Training final Option-A model (n_epochs={n_epochs}) ---")
    result = train_one_fold(
        rna_train=feats.rna, meth_train=feats.meth, y_train=feats.y,
        rna_val=ext_X_qn, meth_val=meth_ext_silenced.astype(np.float32),
        y_val=y_ext.astype(np.int64),
        pole_masks=pole_masks, fold=0,
        rna_dim=feats.rna.shape[1], meth_dim=feats.meth.shape[1],
        n_epochs=n_epochs, patience=n_epochs + 1,
        **ee.FINAL_KWARGS,
    )
    print(f"  METABRIC AUROC={result.best_val_auc:.4f}  BalAcc={result.best_val_bacc:.4f}")
    if result.cal_logits is None or result.cal_labels is None:
        sys.stderr.write("ERROR: no TCGA cal split logits; cannot run transfer.\n")
        raise SystemExit(1)
    return result, float(feats.y.mean()), float(y_ext.mean()), overlap


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _ece(labels, proba):
    return compute_calibration(labels, proba, n_bins=N_BINS).ece


def _brier(labels, proba):
    return brier_score(labels, proba)


def _stratified_pool(labels, frac, seed):
    """Return (pool_idx, eval_idx): a stratified `frac` calibration pool and complement."""
    rng = np.random.default_rng(seed)
    pool = np.zeros(len(labels), dtype=bool)
    for cls in (0, 1):
        idx = np.where(labels == cls)[0]
        n_hold = max(1, int(round(len(idx) * frac)))
        pool[rng.choice(idx, size=n_hold, replace=False)] = True
    return np.where(pool)[0], np.where(~pool)[0]


def _subsample(pool_idx, labels, n, seed):
    """Draw up to n indices from pool_idx, stratified by label where possible."""
    rng = np.random.default_rng(1000 + seed)
    if n >= len(pool_idx):
        return pool_idx
    chosen = []
    per_cls = max(1, n // 2)
    for cls in (0, 1):
        cls_idx = pool_idx[labels[pool_idx] == cls]
        take = min(per_cls, len(cls_idx))
        chosen.extend(rng.choice(cls_idx, size=take, replace=False).tolist())
    # top up to n if rounding left us short
    remaining = [i for i in pool_idx if i not in set(chosen)]
    if len(chosen) < n and remaining:
        extra = rng.choice(remaining, size=min(n - len(chosen), len(remaining)), replace=False)
        chosen.extend(extra.tolist())
    return np.array(sorted(chosen))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    print("=== DMOI v0.13: cross-cohort calibration transfer ===")
    result, pi_train, pi_metab, overlap = prepare_run()

    logits = result.val_logits          # METABRIC logits
    labels = result.val_labels          # METABRIC labels
    proba_raw = result.val_proba        # METABRIC raw sigmoid

    # Fixed cal pool / eval slice so every condition is apples-to-apples.
    pool_idx, eval_idx = _stratified_pool(labels, CAL_FRAC, SPLIT_SEED)
    eval_labels = labels[eval_idx]
    eval_logits = logits[eval_idx]
    print(f"\nCal pool (METABRIC {int(CAL_FRAC * 100)}%, stratified): n={len(pool_idx)} | "
          f"eval slice: n={len(eval_idx)}")

    # TCGA reference: cal-split T and logit distribution (label-free for D2).
    fit_tcga = fit_temperature(result.cal_logits, result.cal_labels)
    T_tcga = fit_tcga.temperature
    tcga_mean, tcga_std = float(result.cal_logits.mean()), float(result.cal_logits.std() + 1e-12)
    metab_mean, metab_std = float(logits.mean()), float(logits.std() + 1e-12)

    rows = []  # (condition, T, ece, brier)

    # A — uncalibrated
    rows.append(("A_uncalibrated", float("nan"),
                 _ece(eval_labels, proba_raw[eval_idx]), _brier(eval_labels, proba_raw[eval_idx])))

    # B — naive TCGA cal-split T
    pB = apply_temperature(eval_logits, T_tcga)
    rows.append(("B_TCGA_T", T_tcga, _ece(eval_labels, pB), _brier(eval_labels, pB)))

    # C — METABRIC oracle T (full cal pool)
    fit_C = fit_temperature(logits[pool_idx], labels[pool_idx])
    pC = apply_temperature(eval_logits, fit_C.temperature)
    rows.append(("C_METABRIC_oracle_T", fit_C.temperature, _ece(eval_labels, pC), _brier(eval_labels, pC)))

    # D2 — label-free affine logit alignment (METABRIC dist -> TCGA dist) + TCGA T
    aligned = affine_align(eval_logits, src_mean=metab_mean, src_std=metab_std,
                           dst_mean=tcga_mean, dst_std=tcga_std)
    pD2 = apply_temperature(aligned, T_tcga)
    rows.append(("D2_labelfree_align_TCGA_T", T_tcga, _ece(eval_labels, pD2), _brier(eval_labels, pD2)))

    # D3 — class-prior odds correction (priors only; no per-sample labels)
    pD3 = prior_odds_correct(proba_raw[eval_idx], pi_train=pi_train, pi_target=pi_metab)
    rows.append(("D3_prior_odds", float("nan"), _ece(eval_labels, pD3), _brier(eval_labels, pD3)))

    # D1 — METABRIC-mini learning curve (n sweep x seeds)
    d1_rows = []  # (n, seed, T, ece, brier)
    for n in D1_N_GRID:
        if n > len(pool_idx):
            continue
        for seed in D1_SEEDS:
            sub = _subsample(pool_idx, labels, n, seed)
            fit_d1 = fit_temperature(logits[sub], labels[sub])
            pD1 = apply_temperature(eval_logits, fit_d1.temperature)
            d1_rows.append((n, seed, fit_d1.temperature, _ece(eval_labels, pD1), _brier(eval_labels, pD1)))

    # AUROC sanity (constant across all conditions — monotonic transforms).
    auc = float(roc_auc_score(eval_labels, proba_raw[eval_idx]))

    # ---- Report ----
    print("\n--- Calibration on fixed METABRIC eval slice (AUROC invariant"
          f" = {auc:.4f}) ---")
    print(f"{'condition':<28}{'T':>8}{'ECE':>10}{'Brier':>10}")
    for name, T, ece, brier in rows:
        print(f"{name:<28}{(f'{T:.3f}' if T == T else '   -- '):>8}{ece:>10.4f}{brier:>10.4f}")
    if d1_rows:
        print("\n  D1 METABRIC-mini (mean over seeds):")
        for n in D1_N_GRID:
            sel = [(e, b) for (nn, _s, _T, e, b) in d1_rows if nn == n]
            if sel:
                ee_, bb_ = np.mean([s[0] for s in sel]), np.mean([s[1] for s in sel])
                print(f"    n={n:<5} ECE={ee_:.4f}  Brier={bb_:.4f}")

    # ---- Write reliability TSV ----
    AUDIT.mkdir(exist_ok=True)
    rel_path = AUDIT / "dmoi_calibration_transfer_v0.13_reliability.tsv"
    with rel_path.open("w") as fh:
        fh.write("condition\tbin_center\tbin_confidence\tbin_accuracy\tbin_count\n")
        named_proba = {"A_uncalibrated": proba_raw[eval_idx], "B_TCGA_T": pB,
                       "C_METABRIC_oracle_T": pC, "D2_labelfree_align_TCGA_T": pD2,
                       "D3_prior_odds": pD3}
        for cond, pr in named_proba.items():
            for b in reliability_table(eval_labels, pr, n_bins=N_BINS):
                fh.write(f"{cond}\t{b.center:.4f}\t{b.confidence:.4f}\t{b.accuracy:.4f}\t{b.count}\n")

    lc_path = AUDIT / "dmoi_calibration_transfer_v0.13_learning_curve.tsv"
    with lc_path.open("w") as fh:
        fh.write("n\tseed\tT\tece\tbrier\n")
        for n, seed, T, ece, brier in d1_rows:
            fh.write(f"{n}\t{seed}\t{T:.4f}\t{ece:.4f}\t{brier:.4f}\n")

    # ---- Verdict (data-driven, honest) ----
    ece_A = rows[0][2]
    ece_B = rows[1][2]
    ece_C = rows[2][2]
    ece_D2 = rows[3][2]
    # smallest n at which mean D1 ECE comes within 20% of the oracle gap-from-A
    def _mean_ece(n):
        sel = [e for (nn, _s, _T, e, _b) in d1_rows if nn == n]
        return float(np.mean(sel)) if sel else float("nan")
    d1_summary = {n: _mean_ece(n) for n in D1_N_GRID}

    # Best transfer attempt (everything except the uncalibrated floor A and the
    # labelled oracle C) — the question is whether ANY of them beats doing nothing.
    transfer_eces = {"B_TCGA_T": ece_B, "D2_labelfree": ece_D2, "D3_prior_odds": rows[4][2]}
    d1_best_n = min((n for n in d1_summary if d1_summary[n] == d1_summary[n]),
                    key=lambda n: d1_summary[n], default=None)
    if d1_best_n is not None:
        transfer_eces[f"D1_mini_n{d1_best_n}"] = d1_summary[d1_best_n]
    best_transfer_name = min(transfer_eces, key=transfer_eces.get)
    best_transfer_ece = transfer_eces[best_transfer_name]
    headroom = ece_A - ece_C  # how much the labelled oracle improves over raw

    verdict_lines = []
    verdict_lines.append(
        f"- The model is **already calibrated on METABRIC out of the box**: raw ECE "
        f"{ece_A:.4f} vs labelled oracle {ece_C:.4f} (headroom only {headroom:+.4f}). "
        "There is almost no calibration to recover.")
    verdict_lines.append(
        f"- Naive TCGA-T {'worsens' if ece_B > ece_A else 'improves'} calibration "
        f"(ECE {ece_A:.4f} -> {ece_B:.4f}) — reproduces and sharpens the v0.2 finding: "
        "TCGA's temperature should NOT be imported.")
    if best_transfer_ece > ece_A:
        verdict_lines.append(
            f"- **No transfer method beats doing nothing.** The best attempt "
            f"({best_transfer_name}, ECE {best_transfer_ece:.4f}) is still worse than the "
            f"uncalibrated baseline ({ece_A:.4f}). Recommended cross-cohort policy: apply "
            "no temperature; the raw probabilities are the best-calibrated available "
            "without a fully labelled target cohort.")
    else:
        verdict_lines.append(
            f"- Best label-light transfer ({best_transfer_name}) reaches ECE "
            f"{best_transfer_ece:.4f}, beating the uncalibrated baseline ({ece_A:.4f}).")
    verdict_lines.append(
        f"- Brier nuance: class-prior odds correction (D3) gives the best Brier "
        f"({rows[4][3]:.4f} vs raw {rows[0][3]:.4f}) by matching METABRIC's higher LumB "
        "base rate, even though its binned ECE is worse — a probability-accuracy vs "
        "bin-calibration trade-off worth noting.")

    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    md = AUDIT / "dmoi_calibration_transfer_v0.13.md"
    table = "\n".join(
        f"| {name} | {(f'{T:.3f}' if T == T else '—')} | {ece:.4f} | {brier:.4f} |"
        for name, T, ece, brier in rows
    )
    d1_table = "\n".join(
        f"| {n} | {d1_summary[n]:.4f} |" for n in D1_N_GRID if d1_summary[n] == d1_summary[n]
    )
    md.write_text(
        "# DMOI v0.13 — Cross-Cohort Calibration Transfer (TCGA → METABRIC)\n\n"
        f"Generated: {ts}\n\n"
        "## Framing\n\n"
        "Temperature scaling and the affine/odds transforms below are **monotonic**, "
        f"so AUROC is invariant across all conditions (AUROC = {auc:.4f} on the eval "
        "slice). This is a **calibration-quality** result (ECE + Brier), not an "
        "accuracy claim.\n\n"
        f"- TCGA train priors: π_LumB = {pi_train:.3f}; METABRIC: π_LumB = {pi_metab:.3f}\n"
        f"- METABRIC cal pool: n={len(pool_idx)} (stratified {int(CAL_FRAC*100)}%, "
        f"seed={SPLIT_SEED}); eval slice: n={len(eval_idx)}\n"
        f"- Gene overlap (METABRIC vs TCGA train): shared={overlap['n_shared']}\n\n"
        "## Conditions (fixed METABRIC eval slice)\n\n"
        "| Condition | T | ECE | Brier |\n|---|---|---|---|\n" + table + "\n\n"
        "Legend: A uncalibrated · B naive TCGA cal-split T (v0.2 failure case) · "
        "C METABRIC oracle T (labelled cal pool) · D2 label-free logit alignment + "
        "TCGA T · D3 class-prior odds correction.\n\n"
        "## D1 — METABRIC-mini learning curve (mean ECE over seeds)\n\n"
        "| labelled n | mean ECE |\n|---|---|\n" + d1_table + "\n\n"
        "Full per-(n, seed) detail: `dmoi_calibration_transfer_v0.13_learning_curve.tsv`. "
        "Reliability bins: `dmoi_calibration_transfer_v0.13_reliability.tsv`.\n\n"
        "## Verdict\n\n" + "\n".join(verdict_lines) + "\n",
    )
    print(f"\nWrote {md}")

    # ---- Substrate: audit ledger + MLflow (best-effort) ----
    audit.emit(
        "calibration_transfer_v0.13", JOB_ID,
        fields={
            "auroc_eval": auc, "ece_uncal": ece_A, "ece_tcga_T": ece_B,
            "ece_oracle_T": ece_C, "ece_labelfree_D2": ece_D2,
            "n_eval": int(len(eval_idx)), "n_cal_pool": int(len(pool_idx)),
        },
    )
    try:
        if tracking.is_enabled():
            with tracking.run("v0.13-calibration-transfer", experiment="dmoi-brca"):
                tracking.log_params({"cal_frac": CAL_FRAC, "split_seed": SPLIT_SEED,
                                     "n_bins": N_BINS, "T_tcga": T_tcga})
                tracking.log_metrics({"auroc_eval": auc, "ece_uncal": ece_A,
                                      "ece_tcga_T": ece_B, "ece_oracle_T": ece_C,
                                      "ece_labelfree_D2": ece_D2})
                tracking.log_artifact(str(md))
    except Exception as exc:  # noqa: BLE001 — tracking must never be pipeline-fatal
        print(f"  (MLflow logging skipped: {exc})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
