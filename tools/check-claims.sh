#!/usr/bin/env bash
# Claim-hygiene guards (added 2026-09-26 review fixes).
# - GONE_RE claims must not appear in README.md or the launcher (prose facts)
# - the ghost handle must appear nowhere in tracked files
# - committed .json artifacts must parse (JSON or JSONL) and contain no stray stdout
# - HAVE facts must exist in README.md (or the artifact named in HAVE_ART)
# Exit 1 on any failure.
set -u
cd "$(dirname "$0")/.."
fail=0

GONE_RE=( "2026-08-31" "164/164 problems" "(126/146)" "2026-09-25 on the commit above"
          "26.4 GiB of KV" "24.9" "72.5 tok/s" "91.3–91.5" "1,857 accepted"
          "4,023 drafted" "1,341 drafts" "176 tok/s code" "133–176"
          "133.5" "144.1" "2,059,947 prompt tokens held at once"
          "≈ 97.5%" "| Validated | 2026-09-22, end-to-end |" )
for p in "${GONE_RE[@]}"; do
  grep -qF -- "$p" README.md launch/serve-qwen38-flashnext-nightly.sh \
    && { echo "STILL PRESENT (prose): $p"; fail=1; }
done

# the ghost handle must be gone from ALL tracked files (excluding this tool)
git grep -qF "tonyd615" -- ':!tools' && { echo "STILL PRESENT: tonyd615"; fail=1; }

# committed JSON artifacts: valid JSON or JSONL, and no stray stdout lines
for f in benchmarks/*.json; do
  python3 -m json.tool "$f" >/dev/null 2>&1 || {
    python3 -c "import json,sys; [json.loads(l) for l in open(sys.argv[1]) if l.strip()]" "$f" \
      || { echo "bad json/jsonl: $f"; fail=1; }
  }
  grep -q "^written:" "$f" && { echo "stray stdout in $f"; fail=1; }
done

# HAVE facts
HAVE_RE="tonyd2wild 2026-09-02 123.9/143.8 1b88c3af 2,112,392 final ~1 s gddd6fbca1 runs-native-mtp-2026-09-26.json 91.0–93.7 41.8% (JSONL)"
for p in $HAVE_RE; do
  grep -qF -- "$p" README.md || { echo "MISSING in README.md: $p"; fail=1; }
done
grep -qF "pool_correction_note" benchmarks/runs-fullctx-native-2026-09-26.json \
  || { echo "MISSING: pool_correction_note"; fail=1; }
grep -qF "spec_delta" benchmarks/runs-native-mtp-2026-09-26.json \
  || { echo "MISSING: spec_delta in mtp artifact"; fail=1; }

python3 -m py_compile benchmarks/bench.py benchmarks/bench-fullctx-conc.py || fail=1

[ $fail -eq 0 ] && echo "ALL CLAIM GUARDS PASS" || echo "GUARD FAILURES"
exit $fail
