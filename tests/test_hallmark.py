"""Unit tests for dmoi_brca.hallmark (v0.6 gmt-file loader)."""
from __future__ import annotations

from pathlib import Path

import pytest

from dmoi_brca.hallmark import (
    DEFAULT_HALLMARK_GMT,
    load_hallmark_gmt,
    summarize_hallmark,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = REPO_ROOT / DEFAULT_HALLMARK_GMT


def test_default_hallmark_file_is_checked_in():
    """The checked-in gmt file must exist at the documented path."""
    assert DEFAULT_PATH.is_file(), (
        f"expected gmt at {DEFAULT_PATH!s} — check data/msigdb/README.md"
    )


def test_load_hallmark_gmt_has_50_sets():
    """MSigDB Hallmark v2024.1 ships exactly 50 sets."""
    sets = load_hallmark_gmt(DEFAULT_PATH)
    assert len(sets) == 50


def test_load_hallmark_gmt_canonical_set_names_present():
    """Spot-check the five sets v0.5 already uses are in the catalog."""
    sets = load_hallmark_gmt(DEFAULT_PATH)
    for canonical in (
        "HALLMARK_ESTROGEN_RESPONSE_EARLY",
        "HALLMARK_ESTROGEN_RESPONSE_LATE",
        "HALLMARK_E2F_TARGETS",
        "HALLMARK_G2M_CHECKPOINT",
        "HALLMARK_MYC_TARGETS_V1",
    ):
        assert canonical in sets, f"{canonical} missing from catalog"


def test_load_hallmark_gmt_genes_are_unique_within_a_set():
    """No duplicate genes inside a single set."""
    sets = load_hallmark_gmt(DEFAULT_PATH)
    for name, genes in sets.items():
        assert len(genes) == len(set(genes)), f"{name} has duplicate genes"


def test_load_hallmark_gmt_gene_counts_sane():
    """Hallmark sets range from ~30 (NOTCH) to 200 (cap) in v2024.1.Hs."""
    sets = load_hallmark_gmt(DEFAULT_PATH)
    for name, genes in sets.items():
        assert 30 <= len(genes) <= 200, f"{name} has {len(genes)} genes"


def test_summarize_hallmark_matches_load():
    sets = load_hallmark_gmt(DEFAULT_PATH)
    summary = summarize_hallmark(sets)
    assert set(summary.keys()) == set(sets.keys())
    for name, count in summary.items():
        assert count == len(sets[name])


def test_load_hallmark_gmt_missing_file_raises(tmp_path: Path):
    bogus = tmp_path / "does_not_exist.gmt"
    with pytest.raises(FileNotFoundError):
        load_hallmark_gmt(bogus)


def test_load_hallmark_gmt_parses_minimal_fixture(tmp_path: Path):
    """Tiny fixture exercises the parser independent of MSigDB."""
    fixture = tmp_path / "mini.gmt"
    fixture.write_text(
        "SET_A\thttps://example.org/A\tGENE1\tGENE2\tGENE3\n"
        "SET_B\thttps://example.org/B\tGENEA\tGENEB\n",
        encoding="utf-8",
    )
    out = load_hallmark_gmt(fixture)
    assert out == {
        "SET_A": ["GENE1", "GENE2", "GENE3"],
        "SET_B": ["GENEA", "GENEB"],
    }


def test_load_hallmark_gmt_rejects_short_line(tmp_path: Path):
    fixture = tmp_path / "bad.gmt"
    fixture.write_text("ONLY_TWO\tno_genes\n", encoding="utf-8")
    with pytest.raises(ValueError, match="at least 3"):
        load_hallmark_gmt(fixture)


def test_load_hallmark_gmt_rejects_empty_file(tmp_path: Path):
    fixture = tmp_path / "empty.gmt"
    fixture.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="no gene sets"):
        load_hallmark_gmt(fixture)


def test_load_hallmark_gmt_skips_blank_lines(tmp_path: Path):
    fixture = tmp_path / "blanks.gmt"
    fixture.write_text(
        "\n"
        "SET_A\tdesc\tG1\tG2\n"
        "\n"
        "SET_B\tdesc\tG3\n"
        "\n",
        encoding="utf-8",
    )
    out = load_hallmark_gmt(fixture)
    assert list(out.keys()) == ["SET_A", "SET_B"]


def test_load_hallmark_gmt_dedupes_within_set(tmp_path: Path):
    fixture = tmp_path / "dup.gmt"
    fixture.write_text(
        "SET\tdesc\tA\tB\tA\tC\tB\n",
        encoding="utf-8",
    )
    out = load_hallmark_gmt(fixture)
    assert out["SET"] == ["A", "B", "C"]
