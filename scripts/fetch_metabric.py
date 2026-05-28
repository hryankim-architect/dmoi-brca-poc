#!/usr/bin/env python3
"""Fetch METABRIC data from the cBioPortal datahub (GitHub LFS).

Downloads the three files needed for DMOI v0.2 Path A' external validation:
  - data_clinical_patient.txt           (~408 KB)
  - data_clinical_sample.txt            (~299 KB)
  - data_mrna_illumina_microarray.txt   (~690 MB)

Total ~690 MB on first run. Files are streamed to disk with progress
reporting. Idempotent — skips any file already present at the expected
size.

Source: https://github.com/cBioPortal/datahub/tree/master/public/brca_metabric
        served via Git LFS media URLs.

Output: data/metabric/{clinical_patient,clinical_sample,mrna_microarray}.txt
"""
from __future__ import annotations

import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "data" / "metabric"
BASE_URL = (
    "https://media.githubusercontent.com/media/cBioPortal/datahub/"
    "master/public/brca_metabric"
)

FILES = {
    # local_name : (remote_name, expected_min_size_bytes)
    "clinical_patient.txt": ("data_clinical_patient.txt", 100_000),
    "clinical_sample.txt": ("data_clinical_sample.txt", 100_000),
    "mrna_microarray.txt": ("data_mrna_illumina_microarray.txt", 500_000_000),
}


def _download_with_progress(url: str, dest: Path) -> None:
    """Stream a file from url to dest with a stderr progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {url}", file=sys.stderr)
    print(f"           -> {dest}", file=sys.stderr)
    with urllib.request.urlopen(url, timeout=60) as r:
        total = int(r.headers.get("Content-Length", 0))
        chunk = 1 << 20  # 1 MiB
        seen = 0
        with dest.open("wb") as f:
            while True:
                data = r.read(chunk)
                if not data:
                    break
                f.write(data)
                seen += len(data)
                if total > 0:
                    pct = seen / total * 100.0
                    print(
                        f"\r    {seen / 1_000_000:7.1f} / {total / 1_000_000:7.1f} MB"
                        f"  ({pct:5.1f}%)",
                        end="", file=sys.stderr, flush=True,
                    )
        print("", file=sys.stderr)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"METABRIC datahub fetch -> {OUT_DIR}")
    print(f"Source base: {BASE_URL}")

    for local_name, (remote_name, min_size) in FILES.items():
        dest = OUT_DIR / local_name
        if dest.exists() and dest.stat().st_size >= min_size:
            print(
                f"  skip  {local_name} (exists, "
                f"{dest.stat().st_size / 1_000_000:.1f} MB)",
            )
            continue
        url = f"{BASE_URL}/{remote_name}"
        try:
            _download_with_progress(url, dest)
        except urllib.error.HTTPError as e:
            sys.stderr.write(f"ERROR fetching {url}: HTTP {e.code}\n")
            return 1
        if dest.stat().st_size < min_size:
            sys.stderr.write(
                f"ERROR: {dest} size {dest.stat().st_size} < expected min "
                f"{min_size}. Re-run to retry.\n",
            )
            return 1
        print(f"  done  {local_name} ({dest.stat().st_size / 1_000_000:.1f} MB)")

    print("\nAll METABRIC files present.")
    print("Next: python scripts/build_metabric_cohort.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
