#!/usr/bin/env python3
"""Benchmark matrix for the Qwen3.8-Flash-Next NVFP4 / 2x RTX PRO 5000 recipe.

Measures, against a running server:
  - TTFT and prefill throughput at short / mid / long / full context
  - Decode throughput (prose and code generation) at each context length
  - Concurrent decode at 2 / 3 / 4 parallel streams

Methodology notes (these matter, see README debugging section):
  - /v1/completions endpoint, temperature 0.7, per-request seed.
  - Filler context is regenerated with a DIFFERENT seed per request so
    vLLM's automatic prefix caching cannot reuse KV between tests.
  - Filler is synthetic pseudo-words, so the model cannot "continue"
    it and must follow the trailing instruction.
  - TTFT = time to first streamed chunk; decode rate = completion_tokens
    divided by (total - ttft).

Usage: python3 bench.py [http://host:8007]   (writes results JSON to stdout)
"""
import json
import random
import string
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8007").rstrip("/") + "/v1"
MODEL = "qwen38-flashnext-nvfp4"

# Synthetic word list: defeats prefix caching via per-request seeding and
# gives the model nothing to memorably continue.
_r = random.Random(20260925)
SYNTH = ["".join(_r.choices(string.ascii_lowercase, k=_r.randint(3, 9))) for _ in range(4096)]

PROSE_TASK = ("Write a long, detailed, factual essay about the history and "
              "engineering of suspension bridges. Write continuously; do not "
              "stop early. Begin the essay now.")
CODE_TASK = ("Write complete, working, commented Python source code implementing "
             "Dijkstra's shortest-path algorithm with a heap-based priority "
             "queue, an adjacency-list graph class, and a test suite at the "
             "bottom. Write the full file; do not stop early. Begin now.")


def filler(approx_tokens: int, seed: int) -> str:
    # Calibrated: random lowercase pseudo-words tokenize at ~3.4 tokens/word
    # with this tokenizer (measured against the server: 150,000 words ->
    # 509,987 prompt tokens). Slightly over-generate, then trim to budget.
    rng = random.Random(seed)
    nwords = int(approx_tokens / 3.4)
    words = [rng.choice(SYNTH) for _ in range(nwords)]
    return " ".join(words)


def one_request(ctx_tokens: int, task: str, max_tokens: int, seed: int) -> dict:
    prompt = filler(ctx_tokens, seed) + "\n\n" + task
    body = {
        "model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0.7, "seed": seed, "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        BASE + "/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "curl/8"},
    )
    t0 = time.time()
    ttft = None
    chunks = 0
    usage = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            d = json.loads(payload)
            if d.get("usage"):
                usage = d["usage"]
            txt = d["choices"][0].get("text", "") if d.get("choices") else ""
            if txt:
                chunks += 1
                if ttft is None:
                    ttft = time.time() - t0
    total = time.time() - t0
    ct = (usage or {}).get("completion_tokens", chunks)
    pt = (usage or {}).get("prompt_tokens")
    gen_t = max(total - (ttft or 0), 1e-6)
    return {
        "ctx_requested": ctx_tokens, "prompt_tokens": pt, "completion_tokens": ct,
        "ttft_s": round(ttft, 2) if ttft else None,
        "total_s": round(total, 2),
        "prefill_tok_s": round(pt / ttft, 0) if ttft and pt else None,
        "decode_tok_s": round(ct / gen_t, 1),
    }


def run_matrix():
    # RUN_BASE makes every invocation use fresh seeds: re-running the suite
    # must NOT replay identical prompts into vLLM's automatic prefix cache
    # (that silently zeroes TTFT and fakes prefill throughput — the exact
    # trap the debugging section documents). Original 2026-09-25 single-run
    # results used fixed seeds 1001-1009; keep those numbers as the
    # published single-run reference, but never re-measure with them.
    RUN_BASE = random.randrange(10**9)
    results = {"started": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
               "endpoint": BASE, "run_base": RUN_BASE, "tests": []}

    # 1) Context sweep: TTFT/prefill + prose decode at increasing context.
    seed = RUN_BASE
    for ctx in (1000, 8000, 32000, 128000, 500000):
        seed += 1
        r = one_request(ctx, PROSE_TASK, 400, seed)
        r["test"] = f"prose ctx~{ctx}"
        results["tests"].append(r)
        print(json.dumps(r), flush=True)

    # 2) Code generation at short / mid / long context.
    for ctx in (1000, 32000, 128000):
        seed += 1
        r = one_request(ctx, CODE_TASK, 400, seed)
        r["test"] = f"code  ctx~{ctx}"
        results["tests"].append(r)
        print(json.dumps(r), flush=True)

    # 3) Concurrency sweep: c parallel prose requests at mid context.
    for c in (2, 3, 4):
        seed += 10
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=c) as ex:
            rs = list(ex.map(
                lambda i: one_request(8000, PROSE_TASK, 256, seed + i), range(c)))
        wall = time.time() - t0
        agg = sum(x["completion_tokens"] for x in rs) / wall
        entry = {
            "test": f"concurrency={c} (prose, ctx~8000, 256 tok each)",
            "per_stream": rs,
            "wall_s": round(wall, 2),
            "aggregate_decode_tok_s": round(agg, 1),
        }
        results["tests"].append(entry)
        print(json.dumps(entry), flush=True)

    results["finished"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    return results


if __name__ == "__main__":
    out = run_matrix()
    with open("bench-results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwritten: bench-results.json")
