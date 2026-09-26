# patches/ — provenance and maintenance

> **SUPERSEDED (2026-09-26).** Both files are historical: the native Engram
> PLE path (#54371) makes `uva.py` irrelevant (tables are allocated pinned
> host-side by the model itself — the generic offloader is not used), and
> #55146 + #58197 upstream fixed the warmup wedge that `gpu_worker.py`
> worked around. The launcher's default `NATIVE=1` mode runs neither. Keep
> reading for the `NATIVE=0` legacy path (pinned Sep-5 image) and for what
> each edit did.

These two files are **complete copies of upstream vLLM modules with our
edits applied**, designed to bind-mount over the container's originals.
They are derived from the vLLM project (Apache-2.0); the unmodified
portions retain their original SPDX headers. The repo-root `LICENSE` is
Apache-2.0.

## What they were cut against

| | |
|---|---|
| vLLM version string | `0.28.1rc1.dev437+ge962733e0` |
| Upstream commit | [`vllm-project/vllm@e962733`](https://github.com/vllm-project/vllm/commit/e962733e0) ("[Security] Validate cache salts before they reach LMCache (#51444)") |
| Image | `vllm/vllm-openai:nightly`, pulled digest `sha256:89dd8f442a3f...`, image created 2026-09-05 |
| Validated on | 2026-09-22, 2× RTX PRO 5000 72GB (sm_120), driver 595.91.07, container torch 2.13.0+cu130 |

## The edits (and only the edits)

### `uva.py` → `vllm/model_executor/offloader/uva.py`
In `UVAOffloader`'s parameter-move path: upstream does
`p.data.to("cpu")` (allocating a **pageable** copy) then `.pin_memory()`
(allocating a **pinned** copy) — both alive simultaneously. For the
23.84 GiB FP8 n-gram shard that is ~51 GB per worker, ~102 GB across two
TP ranks loading concurrently, and the kernel OOM-kills workers on a
125 GB host (reproduced 2×; the vLLM log just stops, no traceback).
Our edit allocates the pinned buffer **directly** (`torch.empty(...,
pin_memory=True)`) and copies once. Marked in-file with `TRX50 patch`.

### `gpu_worker.py` → `vllm/v1/worker/gpu_worker.py`
Wraps the `warmup_kernels(...)` call in the model-runner warmup path with
`if os.environ.get("VLLM_SKIP_WARMUP_KERNELS", "0") != "1"`.
Reason: `warmup_kernels` invokes `execute_model` **without prepared PLE
inputs**, and the qwen4_exp PLE custom op spins forever on sm_120 in that
state (startup wedges 40+ min at ~100 W GPU draw). The V1 runner cannot
substitute — it raises "PLE inputs were not prepared" for this model — so
skipping the warmup (the first real request exercises the same kernels,
JIT cached in the persistent Triton volume) is the minimal fix.
Marked in-file with `TRX50 patch`.

## Not patched (deliberately)

`ple_layer.py` (a third file from our August pre-merge era, which moved
the PLE table to host RAM by hand via `VLLM_PLE_HOST_OFFLOAD`) is **not
part of this recipe and is not mounted** by the launcher. After qwen4_exp
merged upstream, the native UVA offloader replaced it entirely. Ship the
two files above, nothing else.

## Maintenance rules (read before bumping the image)

1. These are whole-file overrides: **a stale copy silently reverts any
   upstream fix to that module.** On every image bump, re-diff against the
   container's own files before booting:
   ```bash
   docker create --name probe <new-image> true
   for f in model_executor/offloader/uva.py v1/worker/gpu_worker.py; do
     docker cp probe:/usr/local/lib/python3.12/dist-packages/vllm/$f /tmp/$(basename $f)
     diff /tmp/$(basename $f) patches/$(basename $f)
   done
   docker rm probe
   ```
2. If the upstream diff touches our edited regions (or the edited lines
   moved), re-apply the two edits by hand and re-run the validation matrix
   in the top-level README before trusting the tag.
   The launcher also asserts at boot — before touching any running
   container — that the image's own interpreter imports both modules WITH
   the `TRX50 patch` markers. A moved dist-packages path (the mounts
   hardcode python3.12) fails the launch instead of silently serving
   unpatched code. That closes the mount-effectiveness gap only: it
   proves the mount took, NOT that the content is fresh — a newer
   upstream fix inside a module we override is still silently reverted by
   our stale copy, so the re-diff above stays mandatory on every bump.
3. If upstream ever merges the pinned-direct allocation or a PLE-safe
   warmup (watch PR #54371-adjacent work; for MTP, #55513 — block FP8
   MTP fix for ModelOpt checkpoints — merged 2026-09-08, and Qwen4Exp
   MTP fixes remain open, e.g. #56742; the old "#55313" reference was
   wrong, that PR does not exist),
   **delete the corresponding patch** and drop its bind mount — these
   files exist to work around upstream gaps, not to own them.
