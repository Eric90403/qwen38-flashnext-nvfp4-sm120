# Qwen3.8-Flash-Next NVFP4 on 2× RTX PRO 5000 72GB (sm_120) — a vLLM recipe that fits 512K context

**What this is.** A working, measured single-node deployment of
[`nvidia/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4)
on **two workstation Blackwell cards (RTX PRO 5000 72GB, compute capability
12.0)** using vLLM with tensor parallelism 2 — serving the full
**524,288-token context** on 144 GB of VRAM. Threadripper PRO host, 125 GB
system RAM, no NVLink, Docker. We couldn't find another public recipe for
this exact combination — 2× RTX PRO 5000 72GB + vLLM TP2 + the NVIDIA
NVFP4 checkpoint + PLE host offload + full 512K context — measured
end-to-end; the value here is the exact flags, the failure modes they
prevent, and the raw results.

**Why it's not obvious.** The official recipes for this model assume
datacenter GPUs (B200, TP4/TP8) or 2× DGX Spark. The community recipes for
consumer/workstation Blackwell stop short of two things we do here:

1. **512K context on 72 GB cards.** The model carries a 51B-parameter n-gram
   embedding table (PLE) that naively eats ~24 GiB *per GPU* at TP2, leaving
   no KV-cache room past ~395K tokens. We park that table in pinned host RAM
   — natively via vLLM's `--engram-config` CPU offload (#54371, in every
   nightly since 2026-09-09; before that, via our patch to the generic UVA
   offloader). This frees ~23.8 GiB/GPU and grows the KV pool to 2,112,392
   tokens — four concurrent full-length requests, verified live at 97.5%
   pool occupancy.
2. **The sm_120 + PCIe TP2 trap field.** vLLM's custom all-reduce deadlocks
   this class of machine; the inductor autotuner OOMs by cloning the PLE
   table; the V2 runner's kernel warmup wedged (fixed upstream since
   2026-09-22); and at high
   gpu-memory-utilization a fresh full-length prefill OOMs one rank and
   wedges the other. Each failure mode looks unrelated to the next.
   We document all four with symptoms and fixes.

**Provenance.** The recipe was developed on **Eric's** ([@Eric90403](https://github.com/Eric90403))
workstation — hardware, development instigation, and the acceptance bar —
and debugged and written up by **Hermes Agent** (Nous Research)
session-by-session against the live machine, including a multi-day py-spy
session to pin the all-reduce deadlock. It stands on real shoulders; see
[Credits](#credits).

---

## The model in one paragraph

Qwen3.8-Flash-Next (`model_type: qwen4_exp`) is Qwen's 125B-parameter MoE
with only ~6B active parameters per token — a hybrid of Gated DeltaNet and
QSA attention layers, plus two unusual satellites: a **51B n-gram embedding
table (PLE)** and a **4B MTP draft head**. It is the architecture preview of
Qwen4. The NVFP4 checkpoint (`nvidia/Qwen3.8-Flash-Next-NVFP4`, on HF since
2026-08-31, quantized with NVIDIA Model Optimizer v0.46.0; the PLE and MTP
tensors are carried byte-for-byte from `Qwen3.8-Flash-Next-FP8`) puts
~63 GiB of weights land on each card during load; at steady state
~38.8 GiB of weights + non-torch stays on-card (the ~23.8 GiB PLE shard
pins to host RAM), leaving **26.4 GiB of KV per card = 2,112,392 tokens**
for the pool.

## Hardware we validated on

| | |
|---|---|
| GPU | 2× NVIDIA RTX PRO 5000 Blackwell 72GB (sm_120), no NVLink (`nvidia-smi topo -m`: `NODE`) |
| CPU / RAM | AMD Threadripper PRO on a TRX50-class platform, 125 GB DDR5 |
| Host driver | 595.91.07 (Ubuntu) |
| Serving | Docker, `vllm/vllm-openai:nightly` — vLLM `0.28.1rc1.dev437+ge962733e0` (upstream commit [`e962733`](https://github.com/vllm-project/vllm/commit/e962733e0), image pulled digest `sha256:89dd8f44…`, image built 2026-09-05), container torch 2.13.0+cu130 |
| Validated | 2026-09-22, end-to-end |

Requirements that are hard: **≥ ~100 GB host RAM** (each rank pins a 23.8 GiB
table copy — see the `uva.py` patch) and **two 72 GB Blackwell cards**. A
single card cannot hold the TP2 shards; this is not a TP1 recipe.

## Quick start

```bash
# 0. Preflight: verify your machine can actually run this (2 GPUs, VRAM,
#    host RAM, driver, model path, image present).
./preflight.sh

# 1. Fetch the NVFP4 checkpoint (~124 GB; NVIDIA Open Model License +
#    Qwen Community License — accept terms, `hf` CLI)
hf download nvidia/Qwen3.8-Flash-Next-NVFP4 --local-dir ./qwen38-flash-next-nvfp4

# 2. Launch (launcher finds patches/ relative to itself; MODEL overrides path)
MODEL=./qwen38-flash-next-nvfp4 ./launch/serve-qwen38-flashnext-nightly.sh

# 3. Verify (~8 min cold start; ~3 min with the named-cache volumes warm)
curl -s http://localhost:8007/v1/models      # -> qwen38-flashnext-nvfp4
curl -s http://localhost:8007/health
curl -s http://localhost:8007/metrics | grep cache_config_info
#    ^ confirm kv_cache_size_tokens≈2112392 (at GPU_UTIL=0.94, MTP=0) —
#      proof the Engram CPU offload is active. With MTP=1 the pool is
#      1739614 tokens (the MTP head claims the difference).
```

If your host runs `nvidia-container-toolkit`, delete the "manual GPU
injection" block from the launcher and pass `--gpus all` instead — that
block exists because our host deliberately doesn't run the toolkit hook.

**The default mode (`NATIVE=1`) runs ZERO patches.** The launcher pins the
2026-09-26 nightly (vLLM `0.30.1rc1.dev193+gddd6fbca1`) and asserts the
image's own Engram/PLE modules import before launching. `NATIVE=0` restores
the September-5 pinned build with the two historical bind-mounted files in
`patches/` — kept for rollback and for anyone who cannot bump images; the
launcher refuses to pair a mode with the wrong image. See `patches/README.md`
for what each patch did and which upstream PR replaced it.

## The flags, and why each one is load-bearing

```bash
vllm serve /model \
  --served-model-name qwen38-flashnext-nvfp4 \
  --quantization modelopt \
  --tensor-parallel-size 2 \
  --distributed-executor-backend mp \
  --gpu-memory-utilization 0.94 \
  --max-model-len 524288 \
  --max-num-seqs 4 \
  --disable-custom-all-reduce \
  --no-enable-flashinfer-autotune \
  --compilation-config '{"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --engram-config '{"cpu_offload": true}' \
  --hf-overrides '{"text_config":{"rope_parameters":{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":262144}}}' \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 --trust-remote-code --port 8007
```

Plus container env: `-e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`,
`-e NCCL_P2P_DISABLE=1`. (The legacy `NATIVE=0` build instead carries
`--cpu-offload-gb 24 --cpu-offload-params ngram_embedding.weight`, the two
bind-mounted patch files, and `-e VLLM_SKIP_WARMUP_KERNELS=1`.)

| Flag | Why | What breaks without it |
|---|---|---|
| `--engram-config '{"cpu_offload": true}'` | The model-declared Engram/PLE path (#54371): the FP8 n-gram table shards (~23.8 GiB/rank) are allocated directly in pinned host RAM at load and read from the Triton kernel over UVA, with prefetch on a side CUDA stream. `true` is the default; passing it explicitly keeps the launch self-documenting | KV pool caps at ~395K tokens — **524K context simply won't fit** |
| *(legacy `NATIVE=0`)* `--cpu-offload-gb 24 --cpu-offload-params ngram_embedding.weight` + `patches/uva.py` | The generic UVAOffloader parks the table behind a zero-copy device view; the patch made its loader allocate pinned directly (upstream keeps pageable + pinned copies alive at once: 102 GB for TP2 on a 125 GB host). Superseded entirely by the native row above — the native path never puts the table on GPU | Kernel OOM during weight load; the vLLM log ends mid-startup with **no error** and released VRAM |
| `--disable-custom-all-reduce` | vLLM's custom P2P all-reduce **deadlocks on this hardware class** — both ranks spin forever at the first embedding all-reduce, even with `NCCL_P2P_DISABLE=1` (that env var only steers NCCL, not vLLM's custom kernels) | Wedge at ~100 W GPU / 0% mem util, first request never completes. Symptom class: stack parked inside `hyperconnection.py mix` (backpressure *behind* the jammed all-reduce) |
| `--compilation-config '{"mode":0, "cudagraph_mode":"FULL_DECODE_ONLY"}'` | Mode 0 (no torch.compile/inductor) makes the autotuner's PLE-table-clone OOM unreachable (same failure tonyd615 documented on 2× Spark; vLLM PR #55272), while CUDA graphs still capture decode | Inductor autotune clones the full 47.7 GiB PLE table as a compile-time constant → OOM. Decode without graphs: 24.9 tok/s vs 72.5 with |
| `--distributed-executor-backend mp` | Multiproc TP on sm_120 workstation cards | — (Ray backend works but adds nothing here) |
| *(legacy `NATIVE=0` only)* `VLLM_SKIP_WARMUP_KERNELS=1` + `patches/gpu_worker.py` | The V2 runner's `warmup_kernels` ran a forward pass **without PLE inputs**, spinning the PLE custom op for 40+ min at ~100 W. Upstream fixed this properly (#55146 + #58197: mode-0/eager skips JIT warmup — in every nightly since 2026-09-22), so `NATIVE=1` needs neither the flag nor the patch | On the legacy build: 40+ minute hang during startup, GPU pegged at ~100 W |
| `--hf-overrides` YaRN **nested under `text_config`** + `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` | 524288 = 2× the native 262144 window; Qwen's own card prescribes YaRN factor 2.0. The nesting is mandatory — **top-level `rope_parameters` is a silent no-op** in vLLM's `_apply_dict_overrides` for qwen4_exp (fix originally noted by MiaAI-Lab) | Silent no-op: you believe you have YaRN, you have the 262K window, and long-context quality is quietly wrong |
| `--gpu-memory-utilization 0.94` `--max-num-seqs 4` | Steady state per card (vLLM startup log): 38.8 GiB weights + non-torch (PLE shard pinned in host RAM), 1.6 GiB activation/graphs, **26.5 GiB KV = 2,112,392 tokens** — **4 concurrent full-length requests verified live**: four simultaneous ~515K-token prompts (2,059,947 prompt tokens at once ≈ 97.5% of the pool) all completed on the native build, see `benchmarks/runs-fullctx-native-2026-09-26.json`. 0.94, not higher: a **fresh ~512K prefill OOMs at 0.97** — the QSA prefill indexer needs ~2 GB of activation headroom beyond steady state; at 0.97 rank 1 died mid-prefill (512 MiB alloc failure, 491 MiB free) and rank 0 spun at 100% waiting for its dead peer | At 0.97: single 500K-token prompt bricks the server; other in-flight requests hang forever behind the dead rank |
| `--no-enable-flashinfer-autotune` | Avoids first-request autotune stalls on SM120 where FlashInfer AOT cubins already cover this checkpoint's shapes | Long, unpredictable first requests |
| `--quantization modelopt` + `--trust-remote-code` | NVFP4 via NVIDIA Model Optimizer's quant config (checkpoint ships `hf_quant_config.json`); qwen4_exp modeling code ships with the checkpoint | Refuses to load / wrong kernels |

**MTP speculative decoding: validated on the native build (2026-09-26).**
`MTP=1` appends `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`.
It boots clean on the pinned Sep-26 nightly: the block-FP8 ModelOpt loading
fix (#55513, merged 2026-09-08) makes the `Qwen4ExpMTP` architecture resolve
and load against this checkpoint. Measured single-stream (one pass each):
decode 133–144 tok/s prose and 176 tok/s code vs 90.9–93.7 with `MTP=0`,
acceptance 46% of draft tokens (1,857 accepted of 4,023 drafted over 1,341
drafts, temp 0.7). The cost: the MTP head plus speculator graph capture
shrink the KV pool 2,112,392 → 1,739,614 tokens, so full-context concurrency
drops 4.03× → 3.32× — **the four-concurrent-full-length claim needs
`MTP=0`.** A 512K-token needle stayed correct with MTP on (recall is not
speculation-dependent, but we checked). Some Qwen4Exp-specific MTP fixes are
still open upstream (e.g. #56742) as of 2026-09-25. (The old
"needs #55313/#55513" note in earlier revisions was wrong — #55313 does not
exist.)

**MTP under multi-stream full-context load (2026-09-26, measured).** Three
simultaneous ~515K-token requests, 256 output each, identical seeds, MTP=1
vs MTP=0 on the same native build:

| | MTP=1 (pool 1,739,614) | MTP=0 (pool 2,112,392) |
|---|---|---|
| All 3 complete | yes | yes |
| Wall clock | 168.7 s | **161.0 s** |
| Pool occupancy | 88.8% | 73.2% |
| Last-stream decode (alone at the tail) | **121.2 tok/s** | 77.3 tok/s |
| Draft acceptance under load | 36% (399/1,101) | — |

Reading: with short outputs this workload is **prefill-bound**, so MTP buys
no wall-clock (it costs ~5% — speculator overhead on 256-token segments
while chunked prefill dominates the batch). MTP's benefit appears the
moment decode dominates — the tail stream finishing alone decoded at 121
vs 77 tok/s (+57%), matching the single-stream 133–176 vs 91–94 numbers.
Choose by workload: long generations → MTP=1; many short-output
full-context lookups → MTP=0. Raw data:
`benchmarks/runs-conc3-mtp-2026-09-26.json`,
`benchmarks/runs-conc3-mtp0-2026-09-26.json`.

## Measured performance

**Native build — nightly `gddd6fbca1` (2026-09-26, zero patches).** Single
pass per cell; treat as ±3% of a 3-pass median. Raw files:
`benchmarks/runs-native-ple-2026-09-26.json`,
`benchmarks/runs-fullctx-native-2026-09-26.json`. Decode at the same
protocol (prose task, `MTP=0` unless noted):

| Prompt size | TTFT | Prefill tok/s | Decode tok/s | Decode tok/s (MTP=1) |
|---|---|---|---|---|
| ~1K | 0.13 s | 8,269 | 93.7 | 133.5 |
| ~8K | 0.78 s | 10,387 | 93.5 | — |
| ~32K | 2.64 s | 12,126 | 92.8 | 144.1 |
| ~128K | 10.8 s | 11,835 | 91.8 | — |
| **~500K** | **48.7 s** | **10,283** | **90.9** | **133.0** |

The native offload path (#54371 + #56926 serialized huge-page host tables)
decodes ~15% faster than the patched build at every context length, and
single-stream MTP roughly doubles it again (133–176 tok/s; see the MTP note
above for the KV-pool cost). Code lane, MTP=0: 93.3 / 92.2 / 91.3–91.5
(1K/32K/128K). Concurrency (MTP=0): c=3 140.3 and c=4 141.5 aggregate —
in line with the legacy medians (126/146); the c=2 pass measured 79.8 due
to TTFT serialization in a single pass and is a timing artifact, not a
regression. Full-context boundary (522K in + 1,024 out, single request):
TTFT 51.8 s, decode 89.2 tok/s; four simultaneous ~515K requests all
completed, wall 215.8 s (~97.5% of the 2,112,392-token pool). 524K needle
recall 3/3 at 25/50/90% depth (MTP=0) and correct at 50% with MTP=1.
(Honest raw-data note: one `code ~128K` row of the native sweep shows
`completion_tokens: 1` — an early-EOS artifact at temp 0.7, reproduced
neither before nor after; clean retests at that cell ran 91.3–91.5 tok/s.)

**Legacy patched build — `e962733e0` (2026-09-25; headline cells are
median of 3 passes, range in `benchmarks/bench-median.json`).** Kept for
comparison; the method notes below apply to both eras.

Method: `/v1/completions`, temperature 0.7, streamed; filler context is
random pseudo-words with a **fresh per-run seed base** (defeats vLLM's
automatic prefix caching — identical filler across tests silently reuses
KV and inflates prefill numbers; this bit us once, see the debugging
section); TTFT = time to first streamed token. Decode rate = completion
tokens / (total − TTFT).

**Prefill + decode, single stream, prose generation (n=3):**

| Prompt size | TTFT | Prefill tok/s | Decode tok/s |
|---|---|---|---|
| ~1K | 0.12 s | 8,854 | 80.5 |
| ~8K | 0.76 s | 10,487 | 79.8 |
| ~32K | 3.05 s | 10,497 | 79.0 |
| ~128K | 12.5 s | 10,188 | 78.8 |
| **~500K** | **55.1 s** | **9,065** | **78.1** |

**Code generation (same protocol, Dijkstra-with-tests task, n=3):**

| Prompt size | TTFT | Prefill tok/s | Decode tok/s |
|---|---|---|---|
| ~1K | 0.12 s | 8,863 | 79.7 |
| ~32K | 3.05 s | 10,479 | 79.6 |
| ~128K | 12.6 s | 10,159 | 78.2 |

**Natural-text lane (real English corpus — Gutenberg prose, not synthetic
pseudo-words — because this model's PLE/n-gram embedding system could in
principle treat real text differently; it doesn't):**

| Prompt size | TTFT | Prefill tok/s | Decode tok/s |
|---|---|---|---|
| ~32K | 3.25 s | 10,536 | 80.1 |
| ~128K | 13.1 s | 10,412 | 79.8 |
| **~511K** | **56.4 s** | **9,062** | **78.2** |

**Long code outputs (fixed 2,048-token sustained-generation workload —
every run ends at `finish_reason: length`, so this measures sustained
generation speed, not "a complete module + tests" — chat
endpoint, thinking disabled, graph-library task):**

| Prompt size | TTFT | Prefill tok/s | Decode tok/s |
|---|---|---|---|
| ~1K | 0.15 s | 7,927 | 80.1 |
| ~32K | 3.0 s | 10,814 | 79.7 |
| ~128K | 12.4 s | 10,373 | 78.6 |
| **~500K** | **55.1 s** | **9,082** | **77.6** |

Long-output **concurrency** — end-to-end output throughput (total generated
tokens / total wall time, **including prefill/TTFT**, which is why these
aggregate figures are lower than pure decode rates), 2,048-token
generations, ~32K context:

| Streams | Aggregate output (incl. prefill) | Per-stream decode | Wall time |
|---|---|---|---|
| 1 | 71.1 | 79.5 | 28.8 s |
| 2 | 115.0 | 64–69 | 35.6 s |
| 3 | 150.9 | 55–65 | 40.7 s |
| 4 | **181.7** | 50–62 | 45.1 s |

(Concurrency decode rates are lower than the 256-token case because each
stream runs for the full 2,048 tokens — the batch never drains to
single-stream speedups.)

**Full-context concurrency (the 4-way proof):** four simultaneous
~515K-token prompts all completed — 2,059,947 prompt tokens held at once,
~97.5% of the KV pool. Behavior at that occupancy: chunked prefill admits
streams staggered (TTFTs 57–236 s), and near-full-pool decode batches
serialize — expect minutes, not seconds, per stream. Raw data:
`benchmarks/bench-fullctx-conc.json`.

Decode speed is essentially flat from 1K to 500K context (≈79–81 tok/s) —
context length costs prefill time, not generation speed. Prefill holds
~10K tok/s to 128K and drops only ~13% at the 500K extreme.

**Concurrency (prose, ~8K context, 256 output tokens each; n=3; aggregate =
end-to-end output throughput **including prefill/TTFT**):**

| Streams | Aggregate output (incl. prefill) | Per-stream decode | Wall time |
|---|---|---|---|
| 1 | 80.5 | 80.5 | — |
| 2 | 98.1 (98.1–99.2) | 58–71 | 5.22 s |
| 3 | 123.9 (123.9–124.3) | 48–66 | 6.2 s |
| 4 | **143.8 (143.2–144.0)** | 41–64 | 7.12 s |

*(Ranges are n=3 fresh-seed passes; one `code ctx~1000` pass errored —
the known 1-token reasoning-parser artifact — and is excluded from stats
and counted in `bench-median.json` rather than shown as a range floor.)*

Throughput scales sub-linearly (batching amortizes the MoE weight reads);
per-stream rate at c=4 is still a comfortable 41–64 tok/s.

| Other | Value |
|---|---|
| **HumanEval+ pass@1 (EvalPlus, greedy, thinking off)** | **0.945 base / 0.921+** — 164/164 problems, sandboxed evaluation |
| Cold start | ~8 min (JIT caches in named Docker volumes make restarts fast) |
| Long-context recall | YaRN 2.0 needle tests 3/3 correct at 25% / 50% / 90% depth (2026-09-22; re-verified 3/3 on the native build 2026-09-26, plus 1/1 at 50% depth with MTP=1) |
| KV pool (GMU 0.94) | 2,112,392 tokens — 4 concurrent full-context requests **verified live** (four ~515K prompts, ~97.5% occupancy, `runs-fullctx-native-2026-09-26.json`) |
| Full-window boundary | single request: 521,681-token prompt + 1,024 output — TTFT 51.8 s, 89.2 tok/s (native build; legacy was 55.7 s / 77.4 tok/s, `bench-fullctx-conc.json`) |

Eval configuration: [EvalPlus](https://github.com/evalplus/evalplus) `--backend openai`
against the live server, temperature 0.0 (`--greedy`),
`chat_template_kwargs: {"enable_thinking": false}` (EvalPlus's OpenAI
backend needed a one-line local patch to pass this through — without it
the model spends the whole sampling budget reasoning and returns empty
completions; same failure mode as the completions-endpoint bench, see
`benchmarks/bench-code-long.py`). Raw samples + eval config in
`benchmarks/evalplus-humaneval/`.

Alternative lane: the GGUF build (Unsloth UD-Q4_K_XL via llama.cpp) hits
~92 tok/s single-stream on the same cards — faster solo, but no tensor-
parallel KV pool, no concurrent streams. We run vLLM.

## Debugging playbook (the lessons)

- **Wedge at ~100 W / 0% memory utilization = something *spins*, it does not compile.**
  `py-spy dump` the worker (not in the image: `docker exec <c> pip install py-spy`).
  Add `CUDA_LAUNCH_BLOCKING=1` to move the error onto the exact blocking launch.
- **Both ranks parked in the same custom all-reduce = symmetric deadlock, not divergence.**
  Fix: `--disable-custom-all-reduce`.
- **Log ends mid-startup, no traceback, VRAM released → host OOM, look at the kernel.**
  `sudo dmesg -T | grep -i oom`. Pinned pages don't show as anonymous in smaps —
  they hide in a shmem-like category, so process RSS alone misleads you.
- **V1 runner cannot run qwen4_exp at all** ("PLE inputs were not prepared") —
  stay on the V2 runner and use the warmup-skip patch.
- **One rank at 100% forever, its peer idle = the peer died, this one waits.**
  With multiproc TP, a CUDA OOM in one worker does not fail the request —
  the surviving rank spins on the dead peer's all-reduce forever. Look for
  `torch.OutOfMemoryError` in `docker logs` and `No available shared memory
  broadcast block found in 60 seconds` repeating in the engine log; that
  pair means a rank is already gone. If the OOM is in
  `qsa_select_paged_prefill` during a very long prompt, lower
  `--gpu-memory-utilization` (0.94 is the validated value here) — the QSA
  prefill indexer needs ~2 GB of activation headroom beyond steady state.
- **Prefix-caching benchmark trap:** vLLM reuses KV for identical filler prompts
  across tests — seed your bench text per stream or your throughput is fiction.
- **The model reasons in `<think>`** — give needle/QA tests a generous
  `max_tokens` (≥700) or you'll "measure" a truncated empty answer.

## Repo layout

```
README.md                                   this file
LICENSE                                     Apache-2.0 (covers the patch files, derived from vLLM)
preflight.sh                                run this first: GPU/VRAM/RAM/driver/model/image checks
.gitignore
launch/serve-qwen38-flashnext-nightly.sh    the launcher (digest-pinned; DRY_RUN=1 inspects it; PUBLISH=1 / DEBUG=1 opt-ins)
patches/uva.py                              pinned-direct UVA offloader -> vllm/model_executor/offloader/uva.py
patches/gpu_worker.py                       warmup_kernels skip -> vllm/v1/worker/gpu_worker.py
patches/README.md                           exact vLLM commit the patches were cut against; re-diff procedure; deletion criteria
benchmarks/bench.py                         prose+code ctx sweep, completions endpoint (fresh RUN_BASE seeds)
benchmarks/bench-code-long.py               fixed 2048-token sustained-generation code bench, chat endpoint
benchmarks/bench-fullctx-conc.py            full-window boundary + 4-way full-context concurrency proof
benchmarks/bench-natural.py                 natural-English (Gutenberg corpus) long-context lane
benchmarks/corpus/                          corpus fetch script + provenance (corpus.txt itself is gitignored)
benchmarks/bench-median.py                 median-of-3 aggregation (excludes errored rows, counts them)
benchmarks/bench-median.json                median (min-max) of 3 fresh-seed passes for the headline cells
benchmarks/runs-prose-{1,2,3}.json          the raw passes behind the medians
benchmarks/bench-results.json               the original 2026-09-25 single-run reference
benchmarks/bench-code-long.json             long-output code run (raw)
benchmarks/bench-fullctx-conc.json          boundary + 4-way proof (raw, legacy build)
benchmarks/runs-native-ple-2026-09-26.json  native build (zero-patch) full sweep, raw
benchmarks/runs-fullctx-native-2026-09-26.json  native build boundary + 4-way proof, raw
benchmarks/bench-natural.json               natural-text lane (raw)
benchmarks/evalplus-humaneval/              HumanEval+ samples + eval configuration
```

## Credits

- **Eric ([@Eric90403](https://github.com/Eric90403)) — Meatbag using Hermes** — kept the
  power on through the multi-day deadlock hunt, set the bar ("it must fit
  512K or it doesn't ship").
- **Hermes Agent (Nous Research)** — authored this recipe: the flag-by-flag
  forensics, py-spy sessions, the `uva.py`/`gpu_worker.py` patches, the
  benchmarks, and this write-up.
- **@blackwellboy** — expert review of this repo (2026-09-25), with our
  sincere thanks. His 12-point feedback materially improved this revision:
  the digest-pinned image default, the CTX↔RoPE linkage, localhost-default
  binding, metric naming, claim wording, and the warmup-patch caveat all
  came from his review — and he independently verified the `gpu_worker.py`
  patch edit against upstream before we did anything else. Errors that
  survived his review are ours.
- **Qwen Team, Alibaba** — Qwen3.8-Flash-Next weights and architecture.
- **NVIDIA** — the NVFP4 checkpoint and Model Optimizer; the
  Dynamo recipe (B200 lane) as prior art.
- **vLLM project** — nightly `qwen4_exp` support, `UVAOffloader`,
  YaRN plumbing. Relevant PRs: #54371 (PLE CPU offload), #55272 (autotune
  PLE-clone OOM), #55513 (block FP8 MTP fix, merged 2026-09-08 — after
  our pinned image), #56742 (Qwen4Exp MTP fixes, open).
- **tonyd2wild** — first documentation of the inductor PLE-table-clone OOM on
  2× DGX Spark; our mode-0 workaround stands on that finding.
- **MiaAI-Lab** — documented that `rope_parameters` must nest under
  `text_config`; their Dual-DGX-Spark repo is the sibling recipe (sm_121,
  multi-node) to this one (sm_120, single node).
- **Unsloth** — GGUF lane and documentation that made the fallback real.

*Disclaimers: performance varies with driver, nightly build, and prompt mix —
our numbers are dated 2026-09-25 on the commit above. Nothing here is
affiliated with or endorsed by Qwen, NVIDIA, or Nous Research.*
