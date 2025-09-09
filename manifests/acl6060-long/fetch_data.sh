#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="$H2T_DATADIR/acl6060-long/audio/en"
mkdir -p "$OUT_DIR"

OWNER="sarapapi"
REPO="hearing2translate"
TAG="data-share-acl6060"
PATTERN="acl6060_long"

urls=$(curl -sL "https://api.github.com/repos/$OWNER/$REPO/releases/tags/$TAG" \
  | jq -r '.assets[] | .browser_download_url')

found=0
for u in $urls; do
  if [[ "$u" == *"$PATTERN"* ]]; then
    fname="$OUT_DIR/$(basename "$u")"
    echo "[DL] $u → $fname"
    curl -L "$u" -o "$fname"
    unzip -q "$fname" -d "$OUT_DIR"
    rm "$fname"
    found=1
  fi
done

if [[ $found -eq 0 ]]; then
  echo "[ERROR] No assets matched pattern: $PATTERN"
  exit 2
fi

echo "[OK] extracted to: $OUT_DIR"