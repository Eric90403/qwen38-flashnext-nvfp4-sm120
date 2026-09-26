#!/usr/bin/env python3
"""bench-median.py — median-of-3 aggregation for the prose/code suite.

Reads runs-prose-{1,2,3}.json (produced by bench.py, fresh RUN_BASE per
pass) and writes bench-median.json.

Failed-request handling (the reason this script exists as a file): a
request that errored or collapsed to a 1-token stop (ttft_s null, or
completion_tokens <= 1 — the known reasoning-parser artifact from the
debugging section) is NOT performance data. Such rows are excluded from
per-field stats and counted in `excluded` with the reason, so a failed
pass can never masquerade as a range floor again (review observation C).
Concurrency rows aggregate output as sum(completion_tokens)/wall_s using
only streams that produced real output.
"""
import json
import statistics

PASSES = ("runs-prose-1.json", "runs-prose-2.json", "runs-prose-3.json")
OUT = "bench-median.json"


def valid(row: dict) -> bool:
    return row.get("ttft_s") is not None and row.get("completion_tokens", 0) > 1


def main() -> None:
    runs = [json.load(open(p)) for p in PASSES]
    names: list[str] = []
    for r in runs:
        for t in r["tests"]:
            if t["test"] not in names:
                names.append(t["test"])

    out = {
        "passes": [r["started"] for r in runs],
        "note": ("median (min-max) of 3 fresh-seed passes, n=3 per cell; "
                 "errored/1-token rows excluded from stats and counted in 'excluded'"),
        "tests": [],
    }
    for name in names:
        rows = [t for r in runs for t in r["tests"] if t["test"] == name]
        is_group = "per_stream" in rows[0]
        if is_group:
            good, bad = rows, 0
        else:
            good = [t for t in rows if valid(t)]
            bad = len(rows) - len(good)
        row = {"test": name, "n": len(good)}
        if bad:
            row["excluded"] = f"{bad} errored/1-token row(s) (ttft null or completion_tokens<=1)"
        if is_group:
            # Exclusion applies per STREAM: a stream that produced no real
            # output is dropped from the aggregate and counted.
            agg = []
            dropped = 0
            for t in good:
                streams = [x for x in t["per_stream"] if x.get("completion_tokens", 0) > 1]
                dropped += len(t["per_stream"]) - len(streams)
                if streams:
                    agg.append(sum(x["completion_tokens"] for x in streams) / t["wall_s"])
            row["n"] = sum(
                sum(1 for x in t["per_stream"] if x.get("completion_tokens", 0) > 1)
                for t in good
            )
            if dropped:
                row["excluded"] = f"{dropped} errored/1-token stream(s) (ttft null or completion_tokens<=1)"
            if agg:
                row["aggregate_output_tok_s"] = {
                    "median": round(statistics.median(agg), 1),
                    "min": round(min(agg), 1),
                    "max": round(max(agg), 1),
                }
        else:
            for f in ("ttft_s", "prefill_tok_s", "decode_tok_s", "wall_s"):
                vals = [t[f] for t in good if t.get(f) is not None]
                if vals:
                    row[f] = {
                        "median": round(statistics.median(vals), 2),
                        "min": min(vals),
                        "max": max(vals),
                    }
        out["tests"].append(row)
        print(json.dumps(row))

    json.dump(out, open(OUT, "w"), indent=2)
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()
