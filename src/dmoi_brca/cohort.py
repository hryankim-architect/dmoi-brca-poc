"""TCGA-BRCA cohort selection — ER+ luminal vs TN basal poles.

Reads UCSC Xena clinical phenotype matrix + RNA-seq + HM450 sample IDs,
returns cohort.tsv with (sample_id, group, has_rna, has_meth) ready for
DMOI POC training.

Group definitions (DMOI POC v0.1):
    H+ (luminal):    PAM50 in {LumA, LumB} AND ER status positive
    H- (basal/TN):   PAM50 in {Basal} AND ER/PR/HER2 all negative
    (Other PAM50 subtypes: Her2-enriched, Normal-like — excluded from POC)

PAM50 source columns (UCSC Xena BRCA_clinicalMatrix.tsv):
    PAM50Call_RNAseq         — short labels (LumA / LumB / Basal / Her2 / Normal); ~956/1247 coverage
    PAM50_mRNA_nature2012    — long labels (Luminal A / Luminal B / Basal-like / ...); ~522/1247 coverage
    Primary = PAM50Call_RNAseq, fallback = PAM50_mRNA_nature2012 normalized to short form.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PAM50_LUMINAL: frozenset[str] = frozenset({"LumA", "LumB"})
PAM50_BASAL: frozenset[str] = frozenset({"Basal"})

# Map long-form PAM50 labels (PAM50_mRNA_nature2012) to short form (PAM50Call_RNAseq).
PAM50_LONG_TO_SHORT: dict[str, str] = {
    "Luminal A": "LumA",
    "Luminal B": "LumB",
    "Basal-like": "Basal",
    "HER2-enriched": "Her2",
    "Normal-like": "Normal",
}


@dataclass
class CohortSummary:
    n_luminal_h_plus: int
    n_basal_h_minus: int
    n_both_modalities: int
    n_rna_only: int
    n_meth_only: int


def load_clinical(path: Path) -> pd.DataFrame:
    """Load Xena BRCA_clinicalMatrix.tsv with key columns only."""
    df = pd.read_csv(path, sep="\t", low_memory=False)
    keep = [
        "sampleID",
        "PAM50Call_RNAseq",
        "PAM50_mRNA_nature2012",
        "ER_Status_nature2012",
        "PR_Status_nature2012",
        "HER2_Final_Status_nature2012",
    ]
    available = [c for c in keep if c in df.columns]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(
            f"Expected clinical columns missing from {path}: {missing}. "
            f"Found columns include: {list(df.columns)[:20]}..."
        )
    return df[available].copy()


def normalize_pam50(row: pd.Series) -> str:
    """Return short-form PAM50 label, preferring RNAseq call, falling back to long-form."""
    short = str(row.get("PAM50Call_RNAseq", "")).strip()
    if short and short.lower() != "nan":
        return short
    long_form = str(row.get("PAM50_mRNA_nature2012", "")).strip()
    return PAM50_LONG_TO_SHORT.get(long_form, "")


def assign_group(row: pd.Series) -> str | None:
    pam50 = normalize_pam50(row)
    er = str(row.get("ER_Status_nature2012", "")).strip().lower()
    pr = str(row.get("PR_Status_nature2012", "")).strip().lower()
    her2 = str(row.get("HER2_Final_Status_nature2012", "")).strip().lower()

    if pam50 in PAM50_LUMINAL and er == "positive":
        return "H_plus_luminal"
    if pam50 in PAM50_BASAL and er == "negative" and pr == "negative" and her2 == "negative":
        return "H_minus_basal_tn"
    return None


def assign_lumab_group(row: pd.Series) -> str | None:
    """Cohort v2 (Week-2 re-scope): within-luminal LumA vs LumB discrimination.

    Both poles are ER+ — the discriminating axis is proliferation
    (LumB high Ki67 / cell cycle, LumA low). This is a harder, biologically
    meaningful target that gives DMOI hypothesis-conditioning real headroom.

    Returns "LumA" / "LumB" / None. ER-positivity NOT enforced — the PAM50
    LumA/LumB call IS the ER+ filter by definition.
    """
    pam50 = normalize_pam50(row)
    if pam50 == "LumA":
        return "LumA"
    if pam50 == "LumB":
        return "LumB"
    return None


def build_cohort(
    clinical: pd.DataFrame,
    rna_sample_ids: set[str],
    meth_sample_ids: set[str],
    *,
    assigner=assign_group,
    label_a: str = "H_plus_luminal",
    label_b: str = "H_minus_basal_tn",
) -> tuple[pd.DataFrame, CohortSummary]:
    """Build a cohort table with (sample_id, group, has_rna, has_meth).

    Args:
        clinical:         Xena clinical phenotype matrix.
        rna_sample_ids:   Set of sample IDs present in the RNA-seq matrix.
        meth_sample_ids:  Set of sample IDs present in the methylation matrix.
        assigner:         Row -> {label_a | label_b | None} function.
                          Default: assign_group (H+/H- poles).
                          Alternative: assign_lumab_group (LumA/LumB within-luminal).
        label_a/label_b:  Labels returned by assigner. Default matches assign_group.
    """
    rows = []
    for _, row in clinical.iterrows():
        group = assigner(row)
        if group is None:
            continue
        sid = row["sampleID"]
        rows.append({
            "sample_id": sid,
            "group": group,
            "has_rna": sid in rna_sample_ids,
            "has_meth": sid in meth_sample_ids,
        })

    cohort = pd.DataFrame(rows, columns=["sample_id", "group", "has_rna", "has_meth"])

    if cohort.empty:
        raise ValueError(
            f"No patients matched {label_a} or {label_b} criteria. "
            "Check PAM50/ER/PR/HER2 column values in clinical matrix — "
            "did UCSC Xena change the value vocabulary?"
        )

    summary = CohortSummary(
        n_luminal_h_plus=int((cohort["group"] == label_a).sum()),
        n_basal_h_minus=int((cohort["group"] == label_b).sum()),
        n_both_modalities=int((cohort["has_rna"] & cohort["has_meth"]).sum()),
        n_rna_only=int((cohort["has_rna"] & ~cohort["has_meth"]).sum()),
        n_meth_only=int((~cohort["has_rna"] & cohort["has_meth"]).sum()),
    )
    return cohort, summary


def read_sample_ids_from_xena(gz_path: Path) -> set[str]:
    """Xena matrix files are sample_id-as-column-headers TSVs (gzipped)."""
    import gzip
    with gzip.open(gz_path, "rt") as fh:
        header = fh.readline().rstrip("\n").split("\t")
    return set(header[1:])  # first column is gene/probe id


def train_test_split_cohort(
    cohort: pd.DataFrame,
    *,
    test_frac: float = 0.2,
    random_state: int = 2024,
    stratify_col: str = "group",
    dual_modality_only: bool = True,
) -> pd.DataFrame:
    """Add a stratified `split` column to the cohort (train / test).

    Only patients that pass the dual-modality filter are eligible for the
    split — single-modality patients get split="" (excluded from both train
    and test).

    The random_state is intentionally distinct from the CV seed used in
    `dmoi_brca.train.run_dmoi_cv` (random_state=42) so the cohort-level
    train/test split is not entangled with the CV fold assignments.

    Args:
        cohort:             Cohort DataFrame with sample_id, group, has_rna,
                            has_meth columns.
        test_frac:          Fraction of dual-modality patients to assign
                            to test. Must be in (0, 1).
        random_state:       Seed for reproducible split.
        stratify_col:       Column to stratify on (default 'group').
        dual_modality_only: If True, only assign split values to patients
                            with both modalities; otherwise to all rows.

    Returns:
        Copy of `cohort` with an added `split` column in
        {"train", "test", ""} where "" means excluded from both.
    """
    if not 0.0 < test_frac < 1.0:
        raise ValueError(f"test_frac must be in (0, 1), got {test_frac}")
    if stratify_col not in cohort.columns:
        raise ValueError(f"stratify_col '{stratify_col}' not in cohort columns")

    result = cohort.copy()
    result["split"] = ""

    if dual_modality_only:
        eligible_mask = result["has_rna"] & result["has_meth"]
    else:
        eligible_mask = pd.Series([True] * len(result), index=result.index)

    eligible = result[eligible_mask]
    if eligible.empty:
        raise ValueError(
            "No eligible patients for train/test split "
            "(dual_modality_only filter removed everyone)",
        )

    # Per-class stratified hold-out: from each class, take round(n * test_frac)
    # for test and the rest for train. The per-class seed mixes the class value
    # so changing the class label set doesn't shuffle the existing split.
    test_indices: list = []
    train_indices: list = []
    for class_value, group_df in eligible.groupby(stratify_col, sort=True):
        n_test = max(1, int(round(len(group_df) * test_frac)))
        class_seed = random_state + hash(str(class_value)) % 10_000
        shuffled = group_df.sample(frac=1.0, random_state=class_seed)
        test_indices.extend(shuffled.index[:n_test])
        train_indices.extend(shuffled.index[n_test:])

    result.loc[test_indices, "split"] = "test"
    result.loc[train_indices, "split"] = "train"
    return result
