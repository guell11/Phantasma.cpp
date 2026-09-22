# wackMall Triton research lane

This directory is an optional research and AOT lane for expert scheduling experiments. It is not part of the llama.cpp runtime build and it does not add a Triton or Python dependency to the production path.

The first prototype is a grouped expert matmul for FP16 inputs and FP16 expert weights. It consumes a packed token assignment list plus expert prefix bounds:

```text
x               [n_tokens, K]
weight          [n_experts, N, K]
token_ids       [n_assignments]
expert_offsets  [n_experts + 1]
output          [n_assignments, N]
```

For expert `e`, assignments live in `token_ids[expert_offsets[e]:expert_offsets[e + 1]]`. The output keeps the same packed assignment order. This layout matches the scheduling problem needed by an expert cache without requiring a dense token-by-expert matrix.

The prototype deliberately does not implement IQ4_XS or Q4_K bit unpacking. The production CUDA backend already has validated block layouts, Q8 activation quantization, and optimized MMVQ/MMQ vec-dot kernels for those formats. A Triton quantized kernel should only be added after it can be checked against that path on the target GPU.

## Files

- `kernels/grouped_expert.py`: FP16 grouped expert matmul kernel and launch helpers.
- `configs/profiles.json`: small-M/decode and prefill launch candidates.
- `verify.py`: prefix-bound reference checks and optional CUDA/Triton numerical verification.
- `autotune.py`: benchmarks the candidate profiles and writes the fastest measured profile.
- `compile_aot.py`: warms up a fixed-shape kernel and exports the compiled cubin/PTX when the installed Triton exposes it.

## Quick checks

The pure scheduling check has no Triton dependency:

```powershell
python tools/wackmall-triton/verify.py --cpu-only
```

With PyTorch, Triton, and a CUDA device available:

```powershell
python tools/wackmall-triton/verify.py --require-triton
python tools/wackmall-triton/autotune.py --mode small_m
python tools/wackmall-triton/compile_aot.py --profile small_m_m1
```

`autotune.py` and `compile_aot.py` exit with a clear dependency error when Triton, PyTorch, or CUDA is unavailable. `verify.py` reports a skip unless `--require-triton` is requested.

## Scope

This lane is useful for testing grouped scheduling geometry, prefix-bound routing, launch shapes, and AOT packaging. Production quantized cold-expert acceleration should stay CUDA-first and retain the current CPU cold path as the reference fallback until a Triton path is validated for the exact IQ4_XS/Q4_K layouts and numerical behavior.

## Host-to-device staging infrastructure

The production expert-tier subsystem exposes a small backend-owned staging primitive in `src/llama-expert-tier.{h,cpp}`. It accepts the existing compute `ggml_backend_t`, creates a second backend context on the same device for copy submission, and allocates three pinned host slots through that device's host-buffer type. Each slot has its own completion event. Uploads advance round-robin: caller bytes are copied unchanged into the selected pinned slot, `ggml_backend_tensor_set_async()` submits the H2D copy, and `ggml_backend_event_wait()` orders later work on the compute backend behind that slot's event.

The primitive deliberately does not choose experts, cache slots, prefetch distance, or eviction policy. The caller retains those decisions and must call `h2d_stage_upload()` before enqueueing work that consumes the destination. The caller also owns the compute backend and destination tensors, which must outlive the stage and pending uploads, and must not reuse a destination range while earlier queued compute can still access it. Independent earlier compute may overlap H2D traffic; later compute is held behind the corresponding event waits. Up to three pinned sources can be in flight before slot reuse requires synchronizing that slot's completion event. Waiting on or destroying the asynchronous stage drains only its per-slot events; it does not issue a device-wide synchronization.

Pinned allocation, async submission, and events are capability-gated. If any required backend capability is unavailable, if any of the three pinned slots or events cannot be created, if the destination does not use the backend's default buffer type, or if a transfer exceeds the per-slot staging capacity, `h2d_stage_upload()` uses the existing synchronous `ggml_backend_tensor_set()` path. When stage events are available, synchronous fallback fences the compute stream with a backend event. Without stage events it synchronizes only the compute backend before the tensor set so an earlier queued consumer cannot race the fallback write; CUDA implements this as a stream synchronization, not a device-wide synchronization. No CUDA runtime API is called directly from `libllama`, and staged data remains in its original packed representation with no FP16 conversion or dequantization.

The staging API exposes an explicit dedupe seam through `h2d_stage_upload_once()`. Hot-expert promotion assigns one stable identity to each `(store, hot-slot)` destination and keeps a generation counter on the owning layer hot slot. The generation increments before every repin rewrite, and all matrices written for that repin use the new generation. Only an in-flight upload with the same non-zero identity and generation can be suppressed. Destination address, offset, size, and byte equality are never the dedupe identity; destination metadata only rejects inconsistent reuse of the same explicit key. This bookkeeping follows cache ownership only and does not participate in expert selection, scoring, target/MTP behavior, dispatcher policy, or logits. Packed source bytes still pass directly through the pinned staging ring with no intermediate dequantization.

With `LLAMA_EXPERT_STATS`, the existing PP/TG PERF block also reports staging-specific counters: async submitted bytes/calls, synchronous stage-fallback bytes/calls, ring slot-reuse stall count/time, and promotion publish/wait count/time. Slot stall time measures only the host wait needed before reusing an in-flight pinned slot; drain/free waits are excluded. Publish/wait time is wall time in the promotion publish step, including either same-device event-wait enqueueing or the host-wait fallback, and is not presented as pure device-copy duration.

The PERF block also keeps policy-neutral PP/TG dispatcher evidence as exponential moving costs. The compact `MUL_MAT_ID_COLD` CPU path has its own backend timer, separate from the aggregate fused/compact cold timer. Hot-GPU route cost is intentionally reported as a graph-wait-per-hot-route proxy because cold CPU work may overlap GPU work; cold time is never subtracted from graph wait to manufacture a GPU-only duration. Graph-sync plus promotion-publish wait and H2D submit cost per MiB are tracked separately. `dispatcher_cost_ready` only reports whether each signal has accumulated the minimum sample count; `dispatcher_policy=current_static` remains the active behavior and no measured threshold changes placement yet.

`LLAMA_EXPERT_TMAX` now remains a guard for the fused `MOE_COLD` path, whose temporary gate/up/activation storage grows with the prefill token count. Larger prefills continue through the generic hot plus `MUL_MAT_ID_COLD` path instead of falling back to the fully stock expert MMID path. The cold MMID uses compact selected-row buckets, while the hot MMID is bounded by the configured hot-slot count; unsupported tensor/layout cases keep their existing stock fallbacks.
