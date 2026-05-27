#!/usr/bin/env bash
# Re-clone upstream reference repos that are gitignored from the main tree.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$REPO_ROOT/reference"
[[ ! -d "$REPO_ROOT/reference/MGDMCL" ]] && \
  git clone --depth 1 https://github.com/wxchen-uestc/MGDMCL "$REPO_ROOT/reference/MGDMCL"
echo "MGDMCL upstream cloned to reference/MGDMCL"
