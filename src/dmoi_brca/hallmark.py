"""GMT-file loader for MSigDB Hallmark gene sets (v0.6).

v0.5 rolled per-gene IG up to 5 hand-picked Hallmark sets that already
lived in `priors.py` for the pole masks. v0.6 widens that to the full
50-set Hallmark collection so the v0.5 "the top pathways are the
expected ones" finding can't be dismissed as an artifact of which
sets we chose to load.

The parser is intentionally tiny — a single-pass split of the
`set_name<TAB>description_url<TAB>gene1<TAB>gene2<TAB>...` format.
No new dependencies. The data file lives at
`data/msigdb/h.all.v2024.1.Hs.symbols.gmt` with provenance and
CC-BY 4.0 attribution in `data/msigdb/README.md`.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

# Default path relative to repo root. Resolved lazily so the import
# doesn't fail in environments where the file isn't checked out.
DEFAULT_HALLMARK_GMT = "data/msigdb/h.all.v2024.1.Hs.symbols.gmt"


def load_hallmark_gmt(path: str | Path | None = None) -> dict[str, list[str]]:
    """Parse an MSigDB GMT file into a dict[set_name, gene_list].

    Args:
        path: Path to a GMT file. If None, uses `DEFAULT_HALLMARK_GMT`
              resolved relative to the current working directory.

    Returns:
        Ordered dict mapping pathway name (e.g.
        "HALLMARK_ESTROGEN_RESPONSE_EARLY") to a deduplicated list of
        gene symbols in the order they appear in the file.

    Raises:
        FileNotFoundError: if the gmt file isn't where we expect.
        ValueError: if the file is empty or any line has fewer than 3
                    tab-separated columns (name, url/description, then
                    one or more genes).
    """
    gmt_path = Path(path) if path is not None else Path(DEFAULT_HALLMARK_GMT)
    if not gmt_path.is_file():
        raise FileNotFoundError(f"Hallmark GMT not found at {gmt_path!s}")

    sets: dict[str, list[str]] = {}
    with gmt_path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, 1):
            line = raw.rstrip("\n").rstrip("\r")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                raise ValueError(
                    f"{gmt_path}: line {line_no} has {len(parts)} "
                    f"columns, need at least 3 (name, url, gene)",
                )
            name, _url, *genes = parts
            # GMT files occasionally include empty trailing tabs.
            cleaned = [g for g in genes if g]
            # Deduplicate while preserving order.
            seen: set[str] = set()
            unique: list[str] = []
            for g in cleaned:
                if g not in seen:
                    seen.add(g)
                    unique.append(g)
            sets[name] = unique

    if not sets:
        raise ValueError(f"{gmt_path}: no gene sets found")
    return sets


def summarize_hallmark(
    sets: Mapping[str, list[str]],
) -> dict[str, int]:
    """Return {set_name: gene_count} for a parsed Hallmark catalog.

    Handy for audit-MD tables and sanity checks.
    """
    return {name: len(genes) for name, genes in sets.items()}
