#!/usr/bin/env bash
set -euo pipefail

# nvidia/Qwen3.8-Flash-Next-NVFP4 on vLLM NIGHTLY (upstream qwen4_exp),
# TP=2 on 2x RTX PRO 5000 72GB (sm_120), single node, no NVLink.
# Validated 2026-09-22 on vllm/vllm-openai:nightly
# (vllm 0.28.1rc1.dev437+ge962733e0, image created 2026-09-05).
# See ../patches/README.md for what the two bind-mounted files change and
# which upstream commit they were read against.
#
# What each unusual piece here is for (full rationale in ../README.md):
#   * qwen4_exp is MERGED upstream: the stock nightly image runs this model,
#     no fork overlay. Two bind-mounted files are the only source deltas.
#   * --cpu-offload-gb 24 --cpu-offload-params ngram_embedding.weight:
#     native UVA offload parks the FP8 n-gram (PLE) table shard (~23.8 GiB
#     per rank) in pinned host RAM behind a zero-copy device view. Without
#     it the KV pool caps at ~395k tokens and 524288 ctx does not fit.
#     patches/uva.py fixes a host-OOM in that loader (pageable + pinned
#     copies coexist: ~51 GB/worker, 102 GB for TP2 on a 125 GB host).
#   * --disable-custom-all-reduce: vLLM's custom P2P all-reduce deadlocks
#     this box class (both ranks spin at the first embedding all-reduce).
#     NCCL_P2P_DISABLE=1 does NOT cover it — that env var only steers NCCL.
#   * --compilation-config mode 0 + FULL_DECODE_ONLY: inductor's compile-
#     time autotune block clones the full 47.7 GiB PLE table as a constant
#     -> OOM (PR #55272 discussion; first documented on 2x DGX Spark by
#     tonyd615). Mode 0 never enters inductor; CUDA graphs still capture
#     decode (24.9 -> 72.5 tok/s). Do not add --enforce-eager.
#   * VLLM_SKIP_WARMUP_KERNELS=1 + patches/gpu_worker.py: the V2 runner's
#     warmup_kernels runs a forward WITHOUT prepared PLE inputs, and the
#     qwen4_exp PLE op spins forever on sm_120 (40+ min startup wedge at
#     ~100 W). The patch honors the env var; V2 runner stays ON (V1 errors
#     "PLE inputs were not prepared" and cannot run this model at all).
#   * GDN decode: nightly default is the AOT fused CUDA kernel. If the op
#     is ever missing vLLM logs "Falling back to the Triton GDN decode
#     path" — watch for that line after image bumps.
#   * gdn_prefill_backend left auto -> FlashInfer prefill (AOT cubins,
#     head_k_dim=128 satisfied by this checkpoint).
#   * KV is BF16 (nightly QSA allowlist: auto|bfloat16 — no FP8 KV).
#   * MTP off: NOT validated on this pinned build (image 2026-09-05,
#     commit e962733). Upstream has moved since the pin — #55513 (block
#     FP8 MTP fix for ModelOpt checkpoints) merged 2026-09-08, and the
#     Qwen4Exp-specific MTP fixes are still open (e.g. #56742) as of
#     2026-09-25. The old "#55313/#55513 needed" note was wrong: #55313
#     does not exist. Revalidate on a bumped image before enabling.
#     MTP=1 keeps the switch ready for that day (~2.5 GiB/GPU cost).
#   * VRAM/GPU: ~63.4 GiB weights; KV gets ~28 GiB = 2,118,489 tokens
#     (live /metrics 2026-09-25, GPU_UTIL=0.94) = 4.03 concurrent
#     full-524288 requests.
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
# PINNED to the exact nightly the patches/ were cut against (vLLM
# 0.28.1rc1.dev437+ge962733e0, image built 2026-09-05). The two patch
# files are whole-file overrides: a NEWER nightly would silently get
# partially replaced by this older patched code. To try a newer build:
# IMAGE=vllm/vllm-openai:nightly ... then re-diff the patches against
# the new container's own files (patches/README.md) before trusting it.
IMAGE=${IMAGE:-vllm/vllm-openai@sha256:89dd8f442a3f4c08c6b3cd634c4f735cd709160651c296596673cf974ea6ee39}
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
  # Native UVA weight offload: park the FP8 n-gram table in pinned host RAM
  # behind a zero-copy device view. Frees ~23.8 GiB/GPU for KV, which is
  # what makes 524288 ctx affordable on 72 GB cards.
  --cpu-offload-gb 24
  --cpu-offload-params ngram_embedding.weight
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  --reasoning-parser qwen3
  --trust-remote-code
  --port "$PORT"
)
args+=("${ROPE_ARGS[@]}")
if [[ "${MTP}" == "1" ]]; then
  args+=(--speculative-config '{"method":"mtp","num_speculative_tokens":3}')
fi
if [[ -n "${MAX_BT}" ]]; then
  args+=(--max-num-batched-tokens "$MAX_BT")
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
  echo "docker rm -f ${NAME}   (skipped in DRY_RUN)"
  echo "docker run -d --name ${NAME} \\"
  echo "  ${VOL_ARGS[*]} ${DEV_ARGS[*]} \\"
  echo "  ${EXTRA_OPTS[*]:+${EXTRA_OPTS[*]}} \\"
  echo "  --ipc=host \\"
  echo "  -p ${BIND}:${PORT}:${PORT} \\"
  echo "  -v ${MODEL}:/model:ro \\"
  echo "  -v ${PATCH_DIR}/uva.py:.../vllm/model_executor/offloader/uva.py:ro \\"
  echo "  -v ${PATCH_DIR}/gpu_worker.py:.../vllm/v1/worker/gpu_worker.py:ro \\"
  echo "  -e VLLM_SKIP_WARMUP_KERNELS=1 -e NCCL_P2P_DISABLE=1 \\"
  if [[ "$CTX" == "524288" ]]; then echo "  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \\"; fi
  echo "  -e NCCL_DEBUG=WARN -e VLLM_RPC_TIMEOUT=900000 \\"
  echo "  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200 \\"
  echo "  ${IMAGE} \\"
  echo "  ${args[*]}"
  exit 0
fi

# --- (re)start ---------------------------------------------------------------
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
  -v "$PATCH_DIR/uva.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/offloader/uva.py:ro" \
  -v "$PATCH_DIR/gpu_worker.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py:ro" \
  -e VLLM_SKIP_WARMUP_KERNELS=1 \
  -v "${NAME}-vllm-cache:/root/.cache/vllm" \
  -v "${NAME}-triton-cache:/root/.triton" \
  -v "${NAME}-inductor-cache:/tmp/torchinductor_root" \
  -e NCCL_P2P_DISABLE=1 \
  $([[ "$CTX" == "524288" ]] && echo -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1) \
  -e NCCL_DEBUG=WARN \
  -e VLLM_RPC_TIMEOUT=900000 \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200 \
  "$IMAGE" \
  "${args[@]}"

echo "container ${NAME} started (MTP=${MTP}, CTX=${CTX}, NUM_SEQS=${NUM_SEQS}, port ${PORT})."
echo "tail logs:   docker logs -f ${NAME}"
echo "health:      curl -s http://localhost:${PORT}/health"
