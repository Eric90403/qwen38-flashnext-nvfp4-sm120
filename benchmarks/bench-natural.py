#!/usr/bin/env python3
"""Natural-text long-context performance, chat endpoint.

Sibling of bench-code-long.py. Same one_chat() flow, anti-prefix-cache
discipline (fresh random RUN_BASE per invocation, disjoint per-cell windows
within a run), and metrics — but the filler is real English prose from
benchmarks/corpus/corpus.txt (see benchmarks/corpus/README.md and
fetch_corpus.sh) instead of synthetic pseudo-words.

Why: the synthetic lanes defeat vLLM automatic prefix caching, but this
model has a PLE (n-gram embedding) system that may behave differently on
natural text. This lane measures prefill/decode/TTFT on prose at
32k / 128k / 500k context.

Token calibration (measured, not assumed):
  Probe: first 30,000 whitespace words of corpus.txt as one chat message,
  max_tokens=16, against the live server (model qwen38-flashnext-nvfp4),
  2026-09-25. usage.prompt_tokens = 39,553 (includes ~30 tokens of chat
  template overhead), so corpus tokens/word = (39553 - 30) / 30000 = 1.317.
TOKENS_PER_WORD = 1.317

Run (full benchmark — schedules a lot of GPU time):
  python3 bench-natural.py [http://host:port]
"""
import json
import os
import random
import sys
import time
import urllib.request

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8007").rstrip("/") + "/v1"
MODEL = "qwen38-flashnext-nvfp4"
RUN_BASE = random.randrange(10**9)  # fresh every invocation

CORPUS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "corpus", "corpus.txt")

# See module docstring for the measurement behind this constant.
TOKENS_PER_WORD = 1.317

NATURAL_TASK = (
    "Write a detailed essay about the history and engineering of suspension "
    "bridges. Write continuously; do not stop early. Begin the essay now."
)

# Token targets. The words->tokens estimate (1.317 tok/word) under-counts
# real prompts by ~7% because paragraph breaks tokenize too (measured
# 2026-09-25: est 32,000 -> actual 34,126; est 128,065 -> actual 137,142).
# At the top cell that overshoot would cross the 524,288 window, so the
# 500K cell targets 480,000 est (~514K actual) — safely inside.
CTX_TARGETS = (32000, 128000, 480000)
MAX_TOKENS = 512


def load_paragraphs() -> list[str]:
    """corpus.txt split on blank lines (Gutenberg texts use \n\n between
    paragraphs). Runs of 3+ newlines normalize to exactly one blank line."""
    with open(CORPUS, encoding="utf-8") as f:
        text = f.read()
    paras = [p.strip() for p in text.replace("\r\n", "\n").split("\n\n")]
    return [p for p in paras if p]


def shuffled(paras: list[str]) -> list[str]:
    """One deterministic shuffle per invocation, seeded by RUN_BASE:
    different runs see different paragraph orderings (anti-prefix-cache)."""
    out = list(paras)
    random.Random(RUN_BASE).shuffle(out)
    return out


def window(paras: list[str], start: int, target_tokens: int) -> tuple[str, int, int]:
    """Contiguous slice of `paras` from index `start` whose cumulative word
    count * TOKENS_PER_WORD first reaches target_tokens. Returns
    (text, next_free_index, est_tokens)."""
    need = target_tokens / TOKENS_PER_WORD
    words = 0
    i = start
    while i < len(paras) and words < need:
        words += len(paras[i].split())
        i += 1
    if words < need:
        raise SystemExit(
            f"corpus exhausted: only {words:.0f} words available from index "
            f"{start}; need ~{need:.0f} for {target_tokens} tokens. "
            "Re-run benchmarks/corpus/fetch_corpus.sh with more books.")
    return "\n\n".join(paras[start:i]), i, int(words * TOKENS_PER_WORD)


def one_chat(filler: str, ctx_requested: int, seed: int) -> dict:
    body = {
        "model": MODEL,
        "messages": [{"role": "user",
                      "content": filler + "\n\n" + NATURAL_TASK}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.7,
        "seed": seed,
        # enable_thinking=False: the Qwen-native switch (chat template kwarg),
        # as in bench-code-long.py — without it the whole budget goes to
        # reasoning_content and no visible essay is produced.
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
        "ctx_requested": ctx_requested, "prompt_tokens": pt,
        "completion_tokens": ct, "finish_reason": finish,
        "ttft_s": round(ttft, 2) if ttft else None,
        "total_s": round(total, 2),
        "prefill_tok_s": round(pt / ttft, 0) if ttft and pt else None,
        "decode_tok_s": round(ct / gen_t, 1) if ttft else None,
    }


def main():
    if not os.path.exists(CORPUS):
        raise SystemExit(f"missing {CORPUS} — run benchmarks/corpus/fetch_corpus.sh first")

    paras = shuffled(load_paragraphs())

    results = {"started": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
               "endpoint": "chat_completions", "run_base": RUN_BASE,
               "filler": "gutenberg natural English",
               "tokens_per_word": TOKENS_PER_WORD,
               "task": "natural-text essay (512-token outputs)", "tests": []}

    # Cells consume disjoint contiguous windows of the shuffled paragraph
    # list, so no cell's prompt is a prefix (or overlap) of another's.
    cursor = 0
    seed = RUN_BASE
    for ctx in CTX_TARGETS:
        seed += 1
        text, cursor, est = window(paras, cursor, ctx)
        r = one_chat(text, ctx, seed)
        r["test"] = f"natural-text ctx~{ctx}"
        r["est_prompt_tokens"] = est
        results["tests"].append(r)
        print(json.dumps(r), flush=True)

    results["finished"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "bench-natural.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print("\nwritten: bench-natural.json")


if __name__ == "__main__":
    main()
