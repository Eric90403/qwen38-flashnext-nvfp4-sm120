#!/usr/bin/env bash
#
# preflight.sh — environment preflight for the Qwen3.8-Flash-Next NVFP4 vLLM
# serving recipe (2x RTX PRO 5000 72GB Blackwell, TP2, 524288-token context).
# Run this BEFORE the launcher; it only reads state, never starts/stops anything.
#
# Checks and why each requirement exists:
#   1. nvidia-smi present, >=2 CUDA GPUs
#      The recipe runs tensor-parallel size 2; it needs two visible GPUs.
#   2. Per-GPU VRAM >= 52 GiB (derived), total across GPUs >= 104 GiB
#      The floor is computed from this recipe's measured per-card memory
#      split (live vLLM startup log, gpu_worker.py:879, 2026-09-25):
#      38.8 GiB weights + non-torch (the ~23.8 GiB PLE shard pins to
#      host RAM), ~1.7 GiB activation/graphs, 6.6 GiB KV for ONE full
#      524,288-token window, ~2 GiB prefill-indexer headroom (the
#      0.97-OOM finding) = 49.1 GiB at GMU 0.94 -> 52 GiB card floor.
#      Below the floor the recipe cannot hold even one full window.
#      At/above the floor the check prints the PROJECTED KV pool and
#      full-context concurrency from that split; the reference 71.7 GiB
#      card projects 26.9 GiB vs 26.5 GiB measured (2,118,489 tokens).
#   3. Compute capability == 12.0 (sm_120)
#      Kernels and patches were validated on sm_120 only; other archs may
#      run but are untested, so a mismatch is a WARN, not a FAIL.
#   4. Driver version >= 570
#      Blackwell (sm_120) support and the CUDA 12.8 runtime used by the
#      vLLM images require the 570.x branch or newer.
#   5. Host RAM available >= 100 GB (/proc/meminfo MemAvailable)
#      PLE (per-token embedding) offload pins ~24 GiB per TP rank in host
#      RAM (≈48 GiB for TP2) and the loader needs headroom on top of that.
#      If the vLLM stack is already running, its pinned pages are the
#      shortfall, so the check degrades to WARN (it is meant for a cold host).
#   6. docker info succeeds
#      The launcher runs vLLM inside a Docker container; the daemon must be
#      up and the current user must be allowed to talk to it.
#   7. Model path exists and contains config.json
#      The checkpoint dir (override with MODEL=...) must be mounted where the
#      launcher expects it, or the container will fail immediately on load.
#   8. A vLLM image is present locally
#      The launcher does not pull; a missing image is a WARN with a
#      `docker pull` hint so first-time users know how to fix it.
#
# Output: one line per check — [PASS]/[WARN]/[FAIL] name: measured (requirement).
# Exit code: 1 if any check FAILs; WARNs do not fail the run.

set -u

MODEL="${MODEL:-/mnt/4tb-nvme/4tb-nvme-models/qwen38-flash-next-nvfp4}"

FAILS=0
WARNS=0

pass() { printf '[PASS] %s: %s\n' "$1" "$2"; }
warn() { printf '[WARN] %s: %s\n' "$1" "$2"; WARNS=$((WARNS + 1)); }
fail() { printf '[FAIL] %s: %s\n' "$1" "$2"; FAILS=$((FAILS + 1)); }

gib() { awk -v m="$1" 'BEGIN { printf "%.1f", m / 1024 }'; }   # MiB -> GiB

# --- 1-4: NVIDIA GPU checks ---------------------------------------------------
if ! command -v nvidia-smi >/dev/null 2>&1; then
    fail gpu_count 'nvidia-smi not found (tool present, >=2 CUDA GPUs)'
    fail total_vram 'nvidia-smi not found (>=104 GiB across all GPUs)'
    fail per_gpu_vram 'nvidia-smi not found (>=52 GiB card floor)'
    fail compute_capability 'nvidia-smi not found (expected 12.0 / sm_120)'
    fail driver_version 'nvidia-smi not found (>=570)'
else
    gpu_rows="$(nvidia-smi --query-gpu=index,name,driver_version,compute_cap,memory.total \
        --format=csv,noheader,nounits 2>/dev/null | grep -v '^[[:space:]]*$' || true)"

    gpu_count="$(printf '%s\n' "$gpu_rows" | grep -c . || true)"
    if [ -z "$gpu_rows" ]; then gpu_count=0; fi

    if [ "$gpu_count" -ge 2 ]; then
        gpu_names="$(printf '%s\n' "$gpu_rows" | awk -F',' '{gsub(/^ +| +$/,"",$2); print $2}' | paste -sd ';')"
        pass gpu_count "${gpu_count} CUDA GPUs (${gpu_names}) (>=2 GPUs for TP2)"
    else
        fail gpu_count "${gpu_count} CUDA GPU(s) visible to nvidia-smi (>=2 required for TP2)"
    fi

    if [ "$gpu_count" -ge 1 ]; then
        total_mib="$(printf '%s\n' "$gpu_rows" | awk -F',' '{gsub(/ /,"",$5); s+=$5} END {printf "%.0f", s}')"
        min_mib="$(printf '%s\n' "$gpu_rows" | awk -F',' '{gsub(/ /,"",$5)} NR==1 || $5+0<m+0 {m=$5} END {print m+0}')"

        # Derived from this recipe's measured per-card memory split (header,
        # check 2): GMU-0.94 envelope must hold consumed 38.78 + act/graphs
        # 1.62 + one 524,288-token window 6.56 + ~2 GiB prefill-indexer
        # headroom = 49.0 GiB -> 52 GiB card floor (49.0 / 0.94 = 52.1).
        # KV density measured on the reference card: 2,118,489 tokens /
        # 26.51 GiB = 79,913 tokens per GiB of KV.
        if awk -v t="$total_mib" 'BEGIN { exit !(t >= 106496) }'; then   # 104 GiB in MiB
            pass total_vram "$(gib "$total_mib") GiB across ${gpu_count} GPUs (>=104 GiB total = 2x the 52 GiB card floor)"
        else
            fail total_vram "$(gib "$total_mib") GiB across ${gpu_count} GPUs (>=104 GiB total = 2x the 52 GiB card floor)"
        fi

        if awk -v m="$min_mib" 'BEGIN { exit !(m >= 53248) }'; then      # 52 GiB in MiB
            kv_gib="$(awk -v m="$min_mib" 'BEGIN { printf "%.1f", 0.94 * (m / 1024) - 40.34 }')"
            kv_tok="$(awk -v k="$kv_gib" 'BEGIN { printf "%d", k * 79913 }')"
            conc="$(awk -v k="$kv_tok" 'BEGIN { printf "%d", int(k / 524288) }')"
            pass per_gpu_vram "smallest GPU $(gib "$min_mib") GiB (>=52 GiB floor; projected at GMU 0.94: KV ${kv_gib} GiB ~ ${kv_tok} tokens ~ ${conc} full-524,288 request(s); reference card measures 26.5 GiB / 2,118,489 / 4)"
        else
            fail per_gpu_vram "smallest GPU $(gib "$min_mib") GiB (>=52 GiB card floor: consumed 38.8 + activation/graphs 1.6 + one full 524,288-token window 6.6 + ~2 GiB indexer headroom, at GMU 0.94)"
        fi

        caps="$(printf '%s\n' "$gpu_rows" | awk -F',' '{gsub(/^ +| +$/,"",$4); print $4}' | sort -u | paste -sd ';')"
        if [ "$caps" = "12.0" ]; then
            pass compute_capability "all GPUs report compute capability ${caps} (expected 12.0 / sm_120)"
        else
            warn compute_capability "GPU(s) report compute capability ${caps} (recipe validated on sm_120 / 12.0 only — other archs untested)"
        fi

        # Driver can differ per GPU only across machines; check each row, use the lowest major.
        drv="$(printf '%s\n' "$gpu_rows" | awk -F',' '{gsub(/^ +| +$/,"",$3); print $3}' | sort -V | head -n1)"
        drv_major="${drv%%.*}"
        if [ -n "$drv" ] && [ "$drv_major" -ge 570 ] 2>/dev/null; then
            pass driver_version "lowest driver ${drv} (>=570 for Blackwell/CUDA 12.8)"
        else
            fail driver_version "lowest driver ${drv:-unknown} (>=570 for Blackwell/CUDA 12.8)"
        fi
    fi
fi

# --- 5: host RAM ---------------------------------------------------------------
avail_kb="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
avail_gib="$(awk -v k="$avail_kb" 'BEGIN { printf "%.1f", k / 1048576 }')"
NEED_KB=$((100 * 1048576))   # 100 GiB

if [ "$avail_kb" -ge "$NEED_KB" ]; then
    pass host_ram_available "${avail_gib} GiB MemAvailable (>=100 GB; PLE offload pins ~24 GiB per TP rank in host RAM)"
else
    # If the vLLM stack is already up, its pinned PLE pages are the shortfall —
    # this check is meant for a cold host, so degrade to WARN (does not fail).
    running_stack=""
    if command -v docker >/dev/null 2>&1; then
        running_stack="$(docker ps --format '{{.Names}} ({{.Image}})' 2>/dev/null | grep -Ei 'vllm|qwen' || true)"
    fi
    if [ -n "$running_stack" ]; then
        warn host_ram_available "${avail_gib} GiB MemAvailable (>=100 GB on a cold host; PLE offload pins ~24 GiB per TP rank — shortfall expected while the stack is already running: $(printf '%s' "$running_stack" | paste -sd ';'))"
    else
        fail host_ram_available "${avail_gib} GiB MemAvailable (>=100 GB; PLE offload pins ~24 GiB per TP rank in host RAM)"
    fi
fi

# --- 6: docker ------------------------------------------------------------------
if ! command -v docker >/dev/null 2>&1; then
    fail docker_info 'docker CLI not found (daemon reachable via "docker info")'
elif docker info >/dev/null 2>&1; then
    docker_ver="$(docker info --format '{{.ServerVersion}}' 2>/dev/null || echo '?')"
    pass docker_info "docker daemon reachable (server ${docker_ver}) (docker info succeeds)"
else
    fail docker_info 'docker info failed (daemon down or user lacks permission)'
fi

# --- 7: model checkpoint ----------------------------------------------------------
if [ -f "${MODEL}/config.json" ]; then
    pass model_path "${MODEL}/config.json present (MODEL dir exists with config.json; override with MODEL=...)"
else
    fail model_path "${MODEL}/config.json not found (model dir must exist with config.json; set MODEL=/path/to/checkpoint)"
fi

# --- 8: local vLLM image -----------------------------------------------------------
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    vllm_images="$(docker images --format '{{.Repository}}:{{.Tag}}' | grep -i vllm || true)"
    if [ -n "$vllm_images" ]; then
        pass vllm_image "$(printf '%s' "$vllm_images" | paste -sd ';') (>=1 local image matching "vllm")"
    else
        warn vllm_image 'no local image matching "vllm" (pull one: docker pull vllm/vllm-openai:nightly — see README for the pinned tag)'
    fi
else
    warn vllm_image 'docker unavailable — could not list local images (docker pull vllm/vllm-openai:nightly after fixing docker access)'
fi

# --- summary ------------------------------------------------------------------------
printf -- '---\n'
if [ "$FAILS" -gt 0 ]; then
    printf '%d FAIL, %d WARN — fix the FAILs before launching.\n' "$FAILS" "$WARNS"
    exit 1
fi
printf '%d WARN, 0 FAIL — preflight OK%s.\n' "$WARNS" "$( [ "$WARNS" -gt 0 ] && echo ' (review WARNs)' || true )"
exit 0
