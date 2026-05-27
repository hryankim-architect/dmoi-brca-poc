#!/usr/bin/env bash
# Download TCGA-BRCA RNA-seq + DNA methylation + clinical for DMOI POC.
# Source: UCSC Xena Hub (xenabrowser.net) — pre-normalized + sample-matched.
#
# Usage:
#   nohup bash scripts/download_tcga_brca.sh > /tmp/dmoi_brca_download.log 2>&1 &

set -uo pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DEST="$REPO_ROOT/data/tcga_brca"
LOG_PREFIX="[dmoi_brca_download $(date -u +%Y-%m-%dT%H:%M:%SZ)]"

mkdir -p "$DEST"
echo "$LOG_PREFIX starting; dest=$DEST"

XENA_BASE="https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download"

# RNA-seq HiSeqV2 (~25 MB gzipped, ~1100 BRCA samples)
RNA_URL="$XENA_BASE/TCGA.BRCA.sampleMap%2FHiSeqV2.gz"
RNA_OUT="$DEST/HiSeqV2.gz"
if [[ ! -f "$RNA_OUT" ]]; then
  echo "$LOG_PREFIX downloading RNA-seq HiSeqV2 (~25 MB)..."
  curl -fL --continue-at - -o "$RNA_OUT" "$RNA_URL" || echo "$LOG_PREFIX WARN: RNA-seq fetch failed"
fi

# DNA methylation HM450 (~700 MB gzipped)
METH_URL="$XENA_BASE/TCGA.BRCA.sampleMap%2FHumanMethylation450.gz"
METH_OUT="$DEST/HumanMethylation450.gz"
if [[ ! -f "$METH_OUT" ]]; then
  echo "$LOG_PREFIX downloading HM450 methylation (~700 MB, this takes a while)..."
  curl -fL --continue-at - -o "$METH_OUT" "$METH_URL" || echo "$LOG_PREFIX WARN: methylation fetch failed"
fi

# Clinical phenotype matrix (~2 MB)
CLIN_URL="$XENA_BASE/TCGA.BRCA.sampleMap%2FBRCA_clinicalMatrix"
CLIN_OUT="$DEST/BRCA_clinicalMatrix.tsv"
if [[ ! -f "$CLIN_OUT" ]]; then
  echo "$LOG_PREFIX downloading clinical phenotype matrix..."
  curl -fL --continue-at - -o "$CLIN_OUT" "$CLIN_URL" || echo "$LOG_PREFIX WARN: clinical fetch failed"
fi

# sha256 sums
echo "$LOG_PREFIX computing sha256 sums"
cd "$DEST"
shasum -a 256 HiSeqV2.gz HumanMethylation450.gz BRCA_clinicalMatrix.tsv 2>/dev/null > sha256sums.txt
cat sha256sums.txt

echo "$LOG_PREFIX file sizes:"
du -h "$DEST"/*
echo "$LOG_PREFIX done"
