#!/usr/bin/env bash
# Build benchmarks/corpus/corpus.txt — a single natural-English corpus for the
# long-context benchmark lane (bench-natural.py).
#
# Source: Project Gutenberg (gutenberg.org), public-domain plain-text books
# (all listed titles are published well before 1900 and are public domain in
# the US; Gutenberg texts are freely distributable).
#
# For each book we:
#   1. download the cache/epub UTF-8 text (stable URL form),
#   2. verify the download is real, reasonably-sized English text,
#   3. strip the Project Gutenberg license header/footer boilerplate
#      (everything before the "*** START OF THE PROJECT GUTENBERG EBOOK"
#      marker and everything from the "*** END OF ..." marker on),
#   4. concatenate into corpus.txt separated by blank lines.
#
# Idempotent: safe to re-run; always rebuilds corpus.txt from scratch.
set -euo pipefail

cd "$(dirname "$0")"
OUT="corpus.txt"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

# "id|short-name|minimum-bytes" — minimum size guards against error pages /
# truncated downloads masquerading as a book.
BOOKS=(
  "2600|war_and_peace|1500000"   # Tolstoy, ~590k words
  "2701|moby_dick|1000000"        # Melville, ~260k words
  "84|frankenstein|250000"        # Mary Shelley, ~75k words
)

# Gutenberg's canonical URL pattern for UTF-8 text of ebook <id>.
url_for() { printf 'https://www.gutenberg.org/cache/epub/%s/pg%s.txt' "$1" "$1"; }

# Strip PG boilerplate: keep text strictly between the START and END markers.
strip_pg() {
  awk '
    /^\*\*\* *START OF (THE|THIS) PROJECT GUTENBERG EBOOK/ { keep=1; next }
    /^\*\*\* *END OF (THE|THIS) PROJECT GUTENBERG EBOOK/   { keep=0 }
    keep { print }
  ' "$1"
}

: > "$STAGE/parts.list"
for entry in "${BOOKS[@]}"; do
  IFS='|' read -r id name minbytes <<<"$entry"
  raw="$STAGE/${name}.raw.txt"
  clean="$STAGE/${name}.txt"
  url="$(url_for "$id")"

  echo "fetching ebook $id ($name): $url"
  curl -fsSL --retry 3 --retry-delay 2 -A 'curl/8' -o "$raw" "$url"

  # --- integrity checks -------------------------------------------------
  actual=$(wc -c < "$raw")
  if (( actual < minbytes )); then
    echo "ERROR: ebook $id downloaded only $actual bytes (expected >= $minbytes)" >&2
    exit 1
  fi
  # Must look like English prose, not an HTML error page or gzip blob.
  if grep -qi '<html\|<!doctype' "$raw"; then
    echo "ERROR: ebook $id looks like HTML, not plain text" >&2
    exit 1
  fi
  if ! grep -qm1 -E '\b(the|and|of|to|was)\b' "$raw"; then
    echo "ERROR: ebook $id lacks common English function words" >&2
    exit 1
  fi
  if ! grep -q 'PROJECT GUTENBERG' "$raw"; then
    echo "ERROR: ebook $id has no Project Gutenberg markers (wrong file?)" >&2
    exit 1
  fi

  strip_pg "$raw" > "$clean"
  words=$(wc -w < "$clean")
  echo "  ebook $id: $actual raw bytes, $words words after boilerplate strip"
  echo "$clean" >> "$STAGE/parts.list"
done

# --- concatenate ---------------------------------------------------------
: > "$OUT"
first=1
while IFS= read -r part; do
  if (( first )); then first=0; else printf '\n\n' >> "$OUT"; fi
  cat "$part" >> "$OUT"
done < "$STAGE/parts.list"

# --- final guard: corpus must stay under 20 MB ---------------------------
size=$(wc -c < "$OUT")
total_words=$(wc -w < "$OUT")
max=$((20 * 1024 * 1024))
if (( size > max )); then
  echo "ERROR: corpus.txt is $size bytes, over the 20 MB cap" >&2
  exit 1
fi

echo "corpus.txt: $size bytes, $total_words words"
