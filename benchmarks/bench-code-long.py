#!/usr/bin/env python3
"""Code-generation performance, long-output edition — chat endpoint.

Rebuilt on /v1/chat/completions after the completions endpoint showed
model-behavior collapse at long context (empty content, 1-token stops):
the raw completions path bypasses the model's chat template. Every real
consumer of this server (agents, Hermes, OpenAI clients) uses chat, so
chat is the honest thing to benchmark.

Complements bench.py (400-token outputs). This measures the cost of LONG
generations: 2048-token completions (a realistic "whole module + tests"
response) across short/mid/long/full context and concurrency 1-4.

Anti-prefix-cache discipline: the filler seed is derived from a fresh
random RUN_BASE each invocation, so re-runs never replay a previous
run's context. Filler is also unique per stream within a group.
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
RUN_BASE = random.randrange(10**9)  # fresh every invocation

_r = random.Random(20260925)
SYNTH = ["".join(_r.choices(string.ascii_lowercase, k=_r.randint(3, 9))) for _ in range(4096)]

CODE_TASK = (
    "Write a complete, production-quality Python package in one file: a "
    "weighted directed graph library. Include: an Edge dataclass with "
    "validation; a Graph class with add/remove node and edge operations, "
    "edge-weight mutation, and neighbor queries; Dijkstra and Bellman-Ford "
    "shortest paths (the latter detecting negative cycles and raising "
    "NegativeCycleError); topological sort raising CycleError; strongly "
    "connected components (Tarjan); a minimum-spanning-arborescence via "
    "Chu-Liu/Edmonds; full docstrings with complexity notes on every public "
    "method; and a complete pytest test suite at the bottom covering happy "
    "paths, edge cases (empty graph, disconnected nodes, duplicate edges), "
    "and error paths. Write the entire file. Do not stop early."
)


def filler(approx_tokens: int, seed: int) -> str:
    rng = random.Random(seed)
    return " ".join(rng.choice(SYNTH) for _ in range(int(approx_tokens / 3.4)))


def one_chat(ctx_tokens: int, max_tokens: int, seed: int) -> dict:
    body = {
        "model": MODEL,
        "messages": [{"role": "user",
                      "content": filler(ctx_tokens, seed) + "\n\n" + CODE_TASK}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "seed": seed,
        # enable_thinking=False: the Qwen-native switch (chat template kwarg).
        # Without it the model spends the whole 2048-token budget in
        # reasoning_content and no code is produced; with it, the visible
        # stream is the code from token one — which is what we time.
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
               "task": "long-output code (2048 tokens)", "tests": []}

    # Long outputs across short/mid/long/full context.
    seed = RUN_BASE
    for ctx in (1000, 32000, 128000, 500000):
        seed += 1
        r = one_chat(ctx, 2048, seed)
        r["test"] = f"chat code-2048tok ctx~{ctx}"
        results["tests"].append(r)
        print(json.dumps(r), flush=True)

    # Concurrency sweep with long outputs at mid context.
    for c in (1, 2, 3, 4):
        seed += 10
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=c) as ex:
            rs = list(ex.map(lambda i: one_chat(32000, 2048, seed + i), range(c)))
        wall = time.time() - t0
        agg = sum(x["completion_tokens"] for x in rs) / wall
        entry = {
            "test": f"chat code-2048tok concurrency={c} (ctx~32000)",
            "per_stream": rs,
            "wall_s": round(wall, 2),
            "aggregate_decode_tok_s": round(agg, 1),
        }
        results["tests"].append(entry)
        print(json.dumps(entry), flush=True)

    results["finished"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    with open("bench-code-long.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nwritten: bench-code-long.json")


if __name__ == "__main__":
    main()
