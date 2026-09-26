#!/usr/bin/env python3
"""Full-context boundary + 4-way full-context concurrency proof.

Two claims this file settles with measurements (2026-09-25 review):

1. BOUNDARY: one ~522K-token input + 1024-token output — a single request
   that exercises essentially the entire advertised 524,288-token window.
   (The published sweeps topped out at ~500K input.)

2. CONC4: four SIMULTANEOUS ~515K-token requests. The KV pool size
   depends on the build (2,118,489 legacy / 2,112,392 native) — read it
   from the server's /metrics (kv_cache_size_tokens), do not hardcode;
   KV_POOL_TOKENS env overrides the default. 4 x (515K + 256 out + template)
   ~= 2.061M = 97.3-97.5% of the pool. If all four complete, "4 concurrent
   full-length requests" is a measured fact, not a capacity extrapolation.
   If they don't, the README gets reworded to the honest number and the
   failure mode gets documented.

Anti-prefix-cache discipline: RUN_BASE is fresh per invocation; every
request (within and across groups) uses a distinct seed, so no request's
filler is a prefix of another's and re-runs never replay cached prefixes.
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
RUN_BASE = random.randrange(10**9)
KV_POOL_TOKENS = int(__import__("os").environ.get("KV_POOL_TOKENS", "2112392"))

_r = random.Random(20260925)
SYNTH = ["".join(_r.choices(string.ascii_lowercase, k=_r.randint(3, 9))) for _ in range(4096)]

TASK = ("Write a detailed essay about the history and engineering of "
        "suspension bridges. Write continuously; do not stop early. "
        "Begin the essay now.")


def filler(approx_tokens: int, seed: int) -> str:
    # ~3.4 tokens per synthetic word (calibrated 2026-09-25 against this
    # server: 150,000 words -> 509,987 prompt tokens).
    rng = random.Random(seed)
    return " ".join(rng.choice(SYNTH) for _ in range(int(approx_tokens / 3.4)))


def one_chat(ctx_tokens: int, max_tokens: int, seed: int) -> dict:
    body = {
        "model": MODEL,
        "messages": [{"role": "user",
                      "content": filler(ctx_tokens, seed) + "\n\n" + TASK}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "seed": seed,
        # Thinking off: same rationale as bench-code-long.py — we are
        # measuring visible-output generation, and thinking would spend
        # the budget invisibly.
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        BASE + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "curl/8"},
    )
    t0 = time.time()
    ttft = None
    usage = None
    finish = None
    n_chunks = 0
    with urllib.request.urlopen(req, timeout=3600) as r:
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
            if d.get("choices"):
                ch = d["choices"][0]
                delta = ch.get("delta") or {}
                if delta.get("content"):
                    n_chunks += 1
                    if ttft is None:
                        ttft = time.time() - t0
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    total = time.time() - t0
    ct = (usage or {}).get("completion_tokens", n_chunks)
    pt = (usage or {}).get("prompt_tokens")
    gen_t = max(total - (ttft or 0), 1e-6)
    return {
        "ctx_requested": ctx_tokens, "prompt_tokens": pt,
        "completion_tokens": ct, "finish_reason": finish,
        "ttft_s": round(ttft, 2) if ttft else None,
        "total_s": round(total, 2),
        "prefill_tok_s": round(pt / ttft, 0) if ttft and pt else None,
        "decode_tok_s": round(ct / gen_t, 1) if ttft else None,
    }


def main():
    results = {"started": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
               "endpoint": "chat_completions", "run_base": RUN_BASE,
               "kv_pool_tokens_expected": KV_POOL_TOKENS, "tests": []}

    # 1) BOUNDARY: ~522K input + 1024 output, single stream.
    seed = RUN_BASE
    r = one_chat(522000, 1024, seed)
    r["test"] = "boundary single-request ~522K in + 1024 out"
    results["tests"].append(r)
    print(json.dumps(r), flush=True)

    # 2) CONC4: four simultaneous ~515K requests, 256 output each.
    # 4 x (~515K + 256 + template ~30) ~= 2.061M vs 2.118M pool.
    seed += 100
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=4) as ex:
        rs = list(ex.map(lambda i: one_chat(515000, 256, seed + i), range(4)))
    wall = time.time() - t0
    agg = sum(x["completion_tokens"] for x in rs) / wall
    total_prompt = sum(x["prompt_tokens"] or 0 for x in rs)
    entry = {
        "test": "4-way full-context concurrency (~515K each, 256 out)",
        "per_stream": rs,
        "wall_s": round(wall, 2),
        "aggregate_output_tok_s": round(agg, 1),
        "sum_prompt_tokens": total_prompt,
        "kv_pool_tokens_expected": KV_POOL_TOKENS,
        "pool_utilization_pct": round(100.0 * (total_prompt + 4 * 256) / KV_POOL_TOKENS, 1),
    }
    results["tests"].append(entry)
    print(json.dumps(entry), flush=True)

    results["finished"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    with open("bench-fullctx-conc.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nwritten: bench-fullctx-conc.json")


if __name__ == "__main__":
    main()
