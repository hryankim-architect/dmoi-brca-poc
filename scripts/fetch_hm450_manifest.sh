#!/usr/bin/env bash
# Fetch the UCSC Xena HM450 probeMap (probe -> gene cis-mapping).
#
# Used by DMOI Week-2 to build the methylation-side hypothesis-conditioned
# attention mask: a probe is "in pole P" iff its annotated gene is in
# the pole's Hallmark gene set.
#
# Source: UCSC Xena Hub (sample-matched, pre-normalized).
# Mirror: tcga-xena-hub.s3.us-east-1.amazonaws.com (same bucket as the
#         BRCA RNA-seq + HM450 matrices fetched on Day-2 of Week-1).
#
# Output:
#   data/tcga_brca/hm450_probemap.tsv         (~18 MB; gitignored)
#   data/tcga_brca/sha256sums.txt             (appended)
#
# Format (TSV with header):
#   #id  gene  chrom  chromStart  chromEnd  strand
#   cg13332474  .              chr7  25935146  25935148  .
#   cg00651829  RSPH14,GNAZ    chr22 23413065  23413067  .
#   cg17027195  AUTS2          chr7  69064092  69064094  .
#
# The "gene" column may be:
#   - a single HGNC symbol         e.g. AUTS2
#   - a comma-separated list       e.g. RSPH14,GNAZ
#   - a dot "."                    intergenic (no annotated cis gene)
set -euo pipefail

cd "$(dirname "$0")/.."
DATA_DIR="data/tcga_brca"
OUT="${DATA_DIR}/hm450_probemap.tsv"
URL="https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/probeMap/illuminaMethyl450_hg19_GPL16304_TCGAlegacy"

mkdir -p "$DATA_DIR"

if [[ -f "$OUT" ]]; then
  echo "Already fetched: $OUT ($(wc -l < "$OUT") rows)"
  echo "Delete to re-fetch."
  exit 0
fi

echo "Fetching HM450 probeMap from Xena..."
curl --fail --silent --show-error --location \
  --output "$OUT" \
  "$URL"

ROWS=$(wc -l < "$OUT")
SIZE_MB=$(du -m "$OUT" | cut -f1)
echo "Downloaded: ${ROWS} rows, ${SIZE_MB} MB"

# Append checksum (sha256sums.txt is created by download_tcga_brca.sh in Week-1)
CHECKSUM_FILE="${DATA_DIR}/sha256sums.txt"
SHA=$(shasum -a 256 "$OUT" | awk '{print $1}')
echo "${SHA}  hm450_probemap.tsv" >> "$CHECKSUM_FILE"
echo "SHA-256: $SHA"
echo "Appended to: $CHECKSUM_FILE"
