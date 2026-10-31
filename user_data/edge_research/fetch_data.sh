#!/usr/bin/env bash
# Recreate the research dataset without exchange API access.
# Source: public repo DigiTuccar/HistoricalDataForTradeBacktest (freqtrade feather format).
#
# usage: fetch_data.sh [regex-of-files] [dest]   (default: all 1h futures candles + funding)
set -euo pipefail
PATTERN="${1:--(1h-futures|8h-funding_rate)\.feather$}"
DEST="${2:-$(cd "$(dirname "$0")/.." && pwd)/data}"
REPO=DigiTuccar/HistoricalDataForTradeBacktest
WORK="${WORK:-$(mktemp -d)}"

if [ ! -d "$WORK/hd/.git" ]; then
  git clone -q --filter=blob:none --no-checkout --depth 1 "https://github.com/$REPO" "$WORK/hd"
fi
SHA=$(git -C "$WORK/hd" rev-parse HEAD)
git -C "$WORK/hd" ls-tree -r --name-only HEAD binance | grep -E -- "$PATTERN" > "$WORK/want.txt"
echo "fetching $(wc -l < "$WORK/want.txt") files @ $SHA -> $DEST"

fetch() {
  local f="$1"
  [ -s "$DEST/$f" ] && return 0
  mkdir -p "$DEST/$(dirname "$f")"
  curl -sSf --retry 4 --retry-delay 2 -o "$DEST/$f.tmp" \
    "https://raw.githubusercontent.com/$REPO/$SHA/$f" && mv "$DEST/$f.tmp" "$DEST/$f"
}
export -f fetch
export DEST REPO SHA
xargs -P 16 -I{} bash -c 'fetch "$@"' _ {} < "$WORK/want.txt"
echo done
