#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/ELITR/elitr-testset.git"

if [[ -z "${H2T_DATADIR:-}" ]]; then
  echo "ERROR: H2T_DATADIR is not set"; exit 1
fi

DATA_ROOT="${H2T_DATADIR%/}"
DEST="${DATA_ROOT}/elitr-testset"
SPARSE_PATH="documents/iwslt2020-nonnative-slt/testset"
REVISION="${1:-}"

if [[ ! -d "$DEST/.git" ]]; then
  echo "[fetch] sparse clone to: $DEST"
  git clone --filter=blob:none --sparse "$REPO_URL" "$DEST"
  git -C "$DEST" sparse-checkout set "$SPARSE_PATH"
else
  echo "[fetch] repo exists. updating..."
  git -C "$DEST" fetch origin --prune
  git -C "$DEST" sparse-checkout set "$SPARSE_PATH"
  git -C "$DEST" pull --ff-only || true
fi

if [[ -n "$REVISION" ]]; then
  echo "[fetch] checkout revision: $REVISION"
  git -C "$DEST" checkout --detach "$REVISION"
fi

# sanity check
if [[ ! -d "$DEST/$SPARSE_PATH" ]]; then
  echo "ERROR: expected path missing: $DEST/$SPARSE_PATH"; exit 2
fi

echo "[fetch] OK: $DEST/$SPARSE_PATH"