#!/usr/bin/env bash
set -euo pipefail

# nvidia/Qwen3.8-Flash-Next-NVFP4 on vLLM NIGHTLY (upstream qwen4_exp),
# TP=2 on 2x RTX PRO 5000 72GB (sm_120), single node, no NVLink.
# Validated 2026-09-22 on vllm/vllm-openai:nightly
# (vllm 0.28.1rc1.dev437+ge962733e0, image created 2026-09-05).
# See ../patches/README.md for what the two bind-mounted files change and
# which upstream commit they were read against.
#
# NATIVE PLE era (2026-09-26): upstream #54371 (merged 2026-09-09) ships
# model-declared Engram/PLE CPU offload in every nightly after it; the two
# bind-mount patches were superseded upstream (#58197 + #55146 replaced
# gpu_worker.py; uva.py only mattered for the GENERIC offloader, which the
# native path does not use). NATIVE=1 (default) runs ZERO patches against
# the pinned 2026-09-26 nightly (vllm 0.30.1rc1.dev193+gddd6fbca1).
# NATIVE=0 is the legacy pinned Sep-5 build WITH patches, kept for
# rollback — the launcher refuses to pair either mode with the wrong image.
NATIVE=${NATIVE:-1}
LEGACY_IMAGE=vllm/vllm-openai@sha256:89dd8f442a3f4c08c6b3cd634c4f735cd709160651c296596673cf974ea6ee39
NATIVE_IMAGE=vllm/vllm-openai@sha256:1b88c3afc77c73a3858730254d6eaf73fd7ff6e226ced60f069d9277fd8d6b2b
if [[ "$NATIVE" == "1" ]]; then
  IMAGE=${IMAGE:-$NATIVE_IMAGE}
else
  IMAGE=${IMAGE:-$LEGACY_IMAGE}
fi
if [[ "$NATIVE" == "1" && "$IMAGE" == "$LEGACY_IMAGE" ]]; then
  echo "ERROR: NATIVE=1 (no patches) with the Sep-5 legacy image: that build" >&2
  echo "       predates the eager-mode warmup fix (#58197) and would wedge at" >&2
  echo "       startup. Use NATIVE=0 for the legacy image." >&2
  exit 1
fi
if [[ "$NATIVE" != "1" && "$IMAGE" != "$LEGACY_IMAGE" ]]; then
  echo "ERROR: NATIVE=0 bind-mounts whole-file patches; pairing them with" >&2
  echo "       any image other than the pinned Sep-5 build silently replaces" >&2
  echo "       newer code with older files. Re-diff patches/ first" >&2
  echo "       (../patches/README.md), or run NATIVE=1." >&2
  exit 1
fi
#
# What each unusual piece here is for (full rationale in ../README.md):
#   * qwen4_exp is MERGED upstream: the stock nightly image runs this model,
#     no fork overlay. NATIVE=1 has ZERO source deltas; NATIVE=0 carries the
#     two historical bind-mounted files (see ../patches/README.md).
#   * PLE CPU offload — the piece that makes 524288 ctx fit on 72 GB cards:
#     NATIVE=1: --engram-config '{"cpu_offload": true}' (#54371): the model
#       itself allocates the FP8 n-gram (PLE) table shards (~23.8 GiB/rank)
#       in pinned host RAM at load; lookups run from the Triton kernel over
#       UVA, prefetching on a side CUDA stream. #56926 packs the host tables
#       into huge pages and serializes offloaded lookups.
#     NATIVE=0: --cpu-offload-gb 24 --cpu-offload-params
#       ngram_embedding.weight — the GENERIC UVA offloader parks the table
#       behind a zero-copy device view; patches/uva.py fixes a host-OOM in
#       that loader (pageable + pinned copies coexist: ~51 GB/worker,
#       102 GB for TP2 on a 125 GB host). Without offload of any kind the
#       KV pool caps at ~395k tokens and 524288 ctx does not fit.
#   * --disable-custom-all-reduce: vLLM's custom P2P all-reduce deadlocks
#     this box class (both ranks spin at the first embedding all-reduce).
#     NCCL_P2P_DISABLE=1 does NOT cover it — that env var only steers NCCL.
#   * --compilation-config mode 0 + FULL_DECODE_ONLY: inductor's compile-
#     time autotune block clones the full 47.7 GiB PLE table as a constant
#     -> OOM (PR #55272 discussion; first documented on 2x DGX Spark by
#     tonyd2wild). Mode 0 never enters inductor; CUDA graphs still capture
#     decode (eager-vs-graphs A/B measured early in development, ~3x). Do not add --enforce-eager.
#   * Warmup wedge (NATIVE=0 only): the V2 runner's warmup_kernels runs a
#     forward WITHOUT prepared PLE inputs, and the qwen4_exp PLE op spins
#     forever on sm_120 (40+ min startup wedge at ~100 W). patches/gpu_worker.py
#     honors VLLM_SKIP_WARMUP_KERNELS=1 to skip it. Upstream fixed this
#     properly in #55146 + #58197 (eager/mode-0 skips JIT warmup) — both in
#     every nightly since 2026-09-22, so NATIVE=1 needs no patch: startup is
#     clean (observed 2026-09-26). V2 runner stays ON in both modes (V1
#     errors "PLE inputs were not prepared" and cannot run this model at all).
#   * GDN decode: nightly default is the AOT fused CUDA kernel. If the op
#     is ever missing vLLM logs "Falling back to the Triton GDN decode
#     path" — watch for that line after image bumps.
#   * gdn_prefill_backend left auto -> FlashInfer prefill (AOT cubins,
#     head_k_dim=128 satisfied by this checkpoint).
#   * KV is BF16 (nightly QSA allowlist: auto|bfloat16 — no FP8 KV).
#   * MTP: validated on the native Sep-26 nightly (README: 3-pass prose+code
#     sweep 2026-09-26, acceptance 42% of drafted tokens; multi-stream numbers
#     in runs-conc3-mtp-2026-09-26.json). NOT validated on the legacy Sep-5
#     pinned build (image 2026-09-05,
#     commit e962733). Upstream has moved since the pin — #55513 (block
#     FP8 MTP fix for ModelOpt checkpoints) merged 2026-09-08, and the
#     Qwen4Exp-specific MTP fixes are still open (e.g. #56742) as of
#     2026-09-25. The old "#55313/#55513 needed" note was wrong: #55313
#     does not exist. Revalidate on a bumped image before enabling.
#     MTP=1 keeps the switch ready for that day (~2.5 GiB/GPU cost).
#   * VRAM/GPU at steady state (live startup log, gpu_worker.py:879):
#     38.8 GiB weights + non-torch (the ~23.8 GiB PLE shard pins to host
#     RAM), ~1.6 GiB peak activation, 0.06 GiB CUDA graphs, KV 26.5 GiB
#     = 2,118,489 tokens (GPU_UTIL=0.94; vLLM logs "Maximum concurrency
#     for 524,288 tokens per request: 4.04x"). Checkpoint on disk:
#     123.57 GiB across 11 shards.
#   * Triton/inductor caches persisted in named volumes: any JIT is paid
#     once across container recreation.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_DIR="$SCRIPT_DIR/../patches"

MTP=${MTP:-0}
CTX=${CTX:-524288}
NUM_SEQS=${NUM_SEQS:-4}
# 0.94, not higher: a fresh ~512K-token prefill OOMs at 0.97. The QSA
# prefill indexer (qsa_select_paged_prefill -> _prefill_logits) wants
# ~2 GB of activation headroom beyond the steady-state allocation; at
# 0.97 rank 1 dies with a 512 MiB alloc failure and rank 0 spins at 100%
# waiting for its dead peer (observed 2026-09-25). 0.94 costs ~7.5% of
# the KV pool (2.12M vs 2.29M tokens) and survives full-length prefill.
GPU_UTIL=${GPU_UTIL:-0.94}
MAX_BT=${MAX_BT:-}
PORT=${PORT:-8007}
NAME=${NAME:-qwen38-nightly}
# IMAGE is resolved by the mode gate at the top: NATIVE_IMAGE (Sep-26
# nightly, zero patches) or LEGACY_IMAGE (Sep-5 build + patches). Override
# with IMAGE=... only after reading the pairing warnings there.
MODEL=${MODEL:-/mnt/4tb-nvme/4tb-nvme-models/qwen38-flash-next-nvfp4}
# DRY_RUN=1 prints the docker command and exits BEFORE any side effect
# (no container removal, no docker run) — use it to inspect the launch
# config: CTX -> RoPE linkage, PUBLISH binding, DEBUG caps.
DRY_RUN=${DRY_RUN:-0}
# PUBLISH=0 (default) binds 127.0.0.1 only. PUBLISH=1 binds 0.0.0.0 —
# the author serves remote clients over a tailnet this way via a systemd
# unit outside this repo.
PUBLISH=${PUBLISH:-0}
# DEBUG=1 adds SYS_PTRACE + unconfined apparmor/seccomp — needed only
# for py-spy deadlock hunts, not for normal serving.
DEBUG=${DEBUG:-0}

# --- manual GPU injection (this host has no nvidia-container-toolkit) --------
# If your host DOES have the toolkit, delete both blocks below and pass
# `--gpus all` to docker run instead.
mapfile -t LIBS < <(ldconfig -p | grep -oE '/lib/x86_64-linux-gnu/lib(nvidia|cuda)[^ ]*\.so\.[0-9a-z]+' | sort -u)
VOL_ARGS=()
for lib in "${LIBS[@]}"; do
  VOL_ARGS+=(-v "$lib:/usr/local/nvidia/lib64/$(basename "$lib"):ro")
done
DEV_ARGS=(--device /dev/nvidia0 --device /dev/nvidia1 --device /dev/nvidiactl
          --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools)

# --- model args --------------------------------------------------------------
# CTX is linked to the RoPE configuration (2026-09-25 review):
#   262144 = the model's native window -> NO hf-overrides, NO long-len env
#   524288 = 2x native -> YaRN factor 2.0 override (Qwen card guidance)
#   anything else -> refuse: unvalidated RoPE configs silently degrade
#   long-context quality (see README debugging section).
# MUST nest rope_parameters under text_config -- top-level rope_parameters
# is a SILENT no-op in vLLM's _apply_dict_overrides for qwen4_exp (fix
# originally noted by MiaAI-Lab).
case "$CTX" in
  262144)
    ROPE_ARGS=()
    ;;
  524288)
    ROPE_ARGS=(
      --hf-overrides '{"text_config":{"rope_parameters":{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":262144}}}'
    )
    ;;
  *)
    echo "ERROR: CTX=${CTX} has no validated RoPE configuration." >&2
    echo "       Use CTX=262144 (native window, no YaRN) or CTX=524288" >&2
    echo "       (YaRN factor 2.0). For anything else, set the RoPE" >&2
    echo "       parameters yourself and own the long-context risk." >&2
    exit 1
    ;;
esac

args=(
  --model /model
  --served-model-name qwen38-flashnext-nvfp4
  --quantization modelopt
  --tensor-parallel-size 2
  --distributed-executor-backend mp
  --gpu-memory-utilization "$GPU_UTIL"
  --max-model-len "$CTX"
  --max-num-seqs "$NUM_SEQS"
  --no-enable-flashinfer-autotune
  --disable-custom-all-reduce
  --compilation-config '{"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}'
  # PLE offload flag is chosen by mode (see header): appended after the array.
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  --reasoning-parser qwen3
  --trust-remote-code
  --port "$PORT"
)
args+=("${ROPE_ARGS[@]}")
# PLE offload path by mode (header notes explain both):
if [[ "$NATIVE" == "1" ]]; then
  args+=(--engram-config '{"cpu_offload": true}')
else
  args+=(--cpu-offload-gb 24 --cpu-offload-params ngram_embedding.weight)
fi
if [[ "${MTP}" == "1" ]]; then
  args+=(--speculative-config '{"method":"mtp","num_speculative_tokens":3}')
fi
if [[ -n "${MAX_BT}" ]]; then
  args+=(--max-num-batched-tokens "$MAX_BT")
fi

# CTX=524288 needs the long-max-model-len override; 262144 is native and
# must NOT carry it. Built as an array so docker run sees -e + value as
# two properly quoted words (a bare $() expansion here once parsed the
# env var as an image name and aborted the launch). Defined before the
# DRY_RUN block so dry-run output reflects the real launch.
LONG_LEN=()
if [[ "$CTX" == "524288" ]]; then
  LONG_LEN=(-e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1)
fi

# --- network + security gating ------------------------------------------------
if [[ "${PUBLISH}" == "1" ]]; then
  BIND="0.0.0.0"
else
  BIND="127.0.0.1"
fi
EXTRA_OPTS=()
if [[ "${DEBUG}" == "1" ]]; then
  EXTRA_OPTS+=(--cap-add SYS_PTRACE --security-opt apparmor=unconfined --security-opt seccomp=unconfined)
fi

# --- DRY_RUN: print and exit before ANY side effect ---------------------------
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "# DRY_RUN: would execute the following (no side effects performed):"
  echo "BIND=${BIND} CTX=${CTX} (rope args: ${ROPE_ARGS[*]:-none}) IMAGE=${IMAGE}"
  echo "long-max-model-len env: ${LONG_LEN[*]:-none}"
  echo "docker rm -f ${NAME}   (skipped in DRY_RUN)"
  echo "docker run -d --name ${NAME} \\"
  echo "  ${VOL_ARGS[*]} ${DEV_ARGS[*]} \\"
  echo "  ${EXTRA_OPTS[*]:+${EXTRA_OPTS[*]}} \\"
  echo "  --ipc=host \\"
  echo "  -p ${BIND}:${PORT}:${PORT} \\"
  echo "  -v ${MODEL}:/model:ro \\"
  if [[ "$NATIVE" == "1" ]]; then
    echo "  (NATIVE=1: no patch mounts, no VLLM_SKIP_WARMUP_KERNELS)"
  else
    echo "  -v ${PATCH_DIR}/uva.py:.../vllm/model_executor/offloader/uva.py:ro \\"
    echo "  -v ${PATCH_DIR}/gpu_worker.py:.../vllm/v1/worker/gpu_worker.py:ro \\"
    echo "  -e VLLM_SKIP_WARMUP_KERNELS=1"
  fi
  echo "  -e NCCL_P2P_DISABLE=1 \\"
  if [[ "$CTX" == "524288" ]]; then echo "  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \\"; fi
  echo "  -e NCCL_DEBUG=WARN -e VLLM_RPC_TIMEOUT=900000 \\"
  echo "  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200 \\"
  echo "  ${IMAGE} \\"
  echo "  ${args[*]}"
  exit 0
fi

# --- (re)start ---------------------------------------------------------------
# PATCH_MOUNTS is the single source of truth for the two bind mounts —
# used first by the assertion probe below, then by the real launch. The
# target paths hardcode the image's python3.12 dist-packages layout.
# In NATIVE=1 mode both arrays stay empty: no patches, no warmup env.
PATCH_MOUNTS=()
SKIPWARM_ENV=()
if [[ "$NATIVE" != "1" ]]; then
  PATCH_MOUNTS=(
    -v "$PATCH_DIR/uva.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/offloader/uva.py:ro"
    -v "$PATCH_DIR/gpu_worker.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py:ro"
  )
  SKIPWARM_ENV=(-e VLLM_SKIP_WARMUP_KERNELS=1)
fi
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "ERROR: image $IMAGE is not present locally — docker pull it first" >&2
  echo "       (the launcher never pulls; see preflight.sh check 8)." >&2
  exit 1
fi
# Boot-time assertion, mode-aware:
#  NATIVE=1 -> prove the image ITSELF carries the model-side Engram/PLE
#    CPU-offload path (config module + ngram embedding module importable).
#    If a future nightly refactors these paths, the launch fails HERE
#    instead of silently loading the whole 47.7 GiB table onto the GPUs.
#  NATIVE=0 -> the legacy patch assertion: prove the image's OWN interpreter
#    imports both modules WITH the "TRX50 patch" markers, BEFORE touching the
#    running container. A moved dist-packages path (image bump) or a lost
#    mount then fails the launch here instead of silently serving unpatched
#    code — which on this recipe means host OOM at load (uva.py) or the
#    40-minute warmup wedge (gpu_worker.py). This proves the mounts are
#    EFFECTIVE, not that their content is fresh; on every image bump the
#    re-diff in ../patches/README.md remains mandatory.
if [[ "$NATIVE" == "1" ]]; then
  if ! probe_out="$(docker run --rm --entrypoint python3 "$IMAGE" -c '
import vllm.config.engram, vllm.models.qwen4_exp.nvidia.ngram_embedding  # noqa: F401
print("native engram modules import OK")' 2>&1)"; then
    echo "ERROR: image $IMAGE lacks the native Engram/PLE CPU-offload path" >&2
    echo "       (needs vLLM >= #54371, merged 2026-09-09, or a refactor that" >&2
    echo "       moved these modules — check the probe output below)." >&2
    printf '%s\n' "$probe_out" | tail -n 5 >&2
    exit 1
  fi
else
if ! probe_out="$(docker run --rm -i --entrypoint python3 "${PATCH_MOUNTS[@]}" "$IMAGE" - <<'PYEOF' 2>&1
import importlib, inspect, sys
bad = []
for mod in ("vllm.model_executor.offloader.uva", "vllm.v1.worker.gpu_worker"):
    f = inspect.getfile(importlib.import_module(mod))
    if "TRX50 patch" not in open(f).read():
        bad.append(f)
if bad:
    sys.exit("UNPATCHED IMPORT PATHS: " + ", ".join(bad))
PYEOF
)"; then
  echo "ERROR: patch assertion FAILED — the image's interpreter did not" >&2
  echo "       import uva.py/gpu_worker.py with the TRX50 patch markers:" >&2
  printf '%s\n' "$probe_out" | tail -n 3 >&2
  echo "       Most likely the image's python path moved (the mounts hardcode" >&2
  echo "       python3.12) or a patch file lost its marker. See patches/README.md." >&2
  exit 1
fi
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true
for v in "$NAME-triton-cache" "$NAME-inductor-cache" "$NAME-vllm-cache"; do
  docker volume create "$v" >/dev/null 2>&1 || true
done

docker run -d --name "$NAME" \
  "${VOL_ARGS[@]}" "${DEV_ARGS[@]}" \
  "${EXTRA_OPTS[@]}" \
  --ipc=host \
  -p "${BIND}:${PORT}:${PORT}" \
  -v "$MODEL:/model:ro" \
  "${PATCH_MOUNTS[@]}" \
  "${SKIPWARM_ENV[@]}" \
  -v "${NAME}-vllm-cache:/root/.cache/vllm" \
  -v "${NAME}-triton-cache:/root/.triton" \
  -v "${NAME}-inductor-cache:/tmp/torchinductor_root" \
  -e NCCL_P2P_DISABLE=1 \
  "${LONG_LEN[@]}" \
  -e NCCL_DEBUG=WARN \
  -e VLLM_RPC_TIMEOUT=900000 \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200 \
  "$IMAGE" \
  "${args[@]}"

echo "container ${NAME} started (MTP=${MTP}, CTX=${CTX}, NUM_SEQS=${NUM_SEQS}, port ${PORT})."
echo "tail logs:   docker logs -f ${NAME}"
echo "health:      curl -s http://localhost:${PORT}/health"
