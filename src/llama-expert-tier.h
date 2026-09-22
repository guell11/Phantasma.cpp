#pragma once

#include <cstdint>

#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"

struct llama_model;

// expert hot-tiering: pins the most-used routed experts of each MoE layer in
// GPU memory; the remaining (cold) experts are computed on the CPU.
//
// enabled via env:
//   LLAMA_EXPERT_HOT   - path to heat csv with header "layer,expert,count"
//                        (seed; optional if LLAMA_EXPERT_ADAPT=1)
//   LLAMA_EXPERT_S     - hot slots per layer (unset = auto-fit from free VRAM)
//   LLAMA_EXPERT_TMAX  - max n_tokens for the hot/cold path (default 16)
//   LLAMA_EXPERT_STATS - dump cold-hit stats at exit ("1" = stderr, else path)
//   LLAMA_EXPERT_ADAPT - 1: online repin of hot slots (decay + hysteresis)
//   LLAMA_EXPERT_USAGE - dump learned hot set at exit (heat csv format)

namespace llama_expert_tier {

// Infrastructure-only H2D staging primitive. The caller supplies the compute
// backend that will consume the destination tensor. The stage creates a second
// backend context on the same device for copies and keeps three pinned upload
// slots, each with its own completion event. If pinned host memory, async copies
// or events are unavailable, uploads fall back to the synchronous tensor-set path.
// The caller owns the compute backend and destination tensors; both must outlive
// the stage and any upload that references them.
struct h2d_stage;

// Binds tier graph hooks to one model during graph construction. The tier is
// process-global, so a scope for another model always uses stock operators.
struct graph_scope {
    explicit graph_scope(const llama_model & model);
    ~graph_scope();

    graph_scope(const graph_scope &) = delete;
    graph_scope & operator=(const graph_scope &) = delete;

private:
    const llama_model * previous = nullptr;
};

struct h2d_load_key {
    uint64_t identity = 0;
    uint64_t generation = 0;
};

h2d_stage * h2d_stage_create(ggml_backend_t backend, size_t capacity);
void        h2d_stage_free(h2d_stage * stage);
bool        h2d_stage_is_async(const h2d_stage * stage);
size_t      h2d_stage_capacity(const h2d_stage * stage);
bool        h2d_stage_upload(h2d_stage * stage, ggml_tensor * dst, size_t offset, const void * src, size_t size);
// A non-zero identity together with its generation names one immutable upload request.
// Matching keys can suppress only an in-flight duplicate. Destination metadata
// validates key reuse but is never the dedupe identity. identity == 0 disables
// dedupe and always submits a new upload.
bool        h2d_stage_upload_once(h2d_stage * stage, ggml_tensor * dst, size_t offset, const void * src, size_t size, h2d_load_key key);
// Queue waits for active uploads on a same-device consumer backend. Returns
// false when the backend cannot consume this stage's events.
bool        h2d_stage_wait_backend(h2d_stage * stage, ggml_backend_t backend);
void        h2d_stage_wait(h2d_stage * stage);

// initialize after model tensors and context runtime buffers are allocated;
// auto-fit sizes the hot tier from the remaining free VRAM, and adaptation can
// enable tiering without an explicit LLAMA_EXPERT_HOT seed
void init(const llama_model & model);
void context_constructing(const llama_model & model);
void model_destroying(const llama_model & model);

// drop-in replacement for ggml_mul_mat_id on MoE expert weights:
// hot part on the GPU via the pinned store, cold part on the CPU via
// ggml_mul_mat_id_cold; falls back to plain ggml_mul_mat_id when the weight
// has no store. LLAMA_EXPERT_TMAX limits only the fused MOE_COLD path.
ggml_tensor * build_mul_mat_id(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x, ggml_tensor * ids);

// consume accumulated expert selection counts for the owning model, update
// scores, and (LLAMA_EXPERT_ADAPT) repin hot slots after backend synchronization.
// Returns true when routing counts were consumed.
bool update(const llama_model & model, ggml_backend_t promotion_backend = nullptr, int64_t n_tokens = 0);

// Stateless score primitive used by aggregate and PP/TG-specific accounting.
float score_ema_update(float score, float decay, uint64_t count);

// Low-overhead performance accounting is active only with LLAMA_EXPERT_STATS.
bool stats_enabled();
void record_perf(int64_t n_tokens, uint64_t total_us, uint64_t wait_us, uint64_t update_us);

// total bytes of routed-expert weight tensors (for dense-fit estimates)
LLAMA_API size_t expert_weight_bytes(const llama_model & model);

// fused cold-expert path for one MoE layer, two calls per layer:
// begin (before the expert matmuls): validates the layer and makes
//   build_mul_mat_id return hot-only results.
//   eligible must be false unless the graph uses separate gate/up with SILU or
//   merged gate_up with GELU. Weight-before-FFN, expert biases/scales, and other
//   activation/layout combinations must use the generic per-matrix path.
// end (after the down matmul, late in node order so the CPU cold op overlaps
//   GPU hot work): returns the cold contribution [embd, n_used, n_tokens]
//   (add it to the down result) or nullptr. x must be the layer input
//   captured before the matmuls.
bool begin_moe_cold(bool eligible,
        ggml_tensor * gate_w, ggml_tensor * up_w, ggml_tensor * down_w,
        ggml_tensor * ids);

ggml_tensor * end_moe_cold(ggml_context * ctx,
        ggml_tensor * gate_w, ggml_tensor * up_w, ggml_tensor * down_w,
        ggml_tensor * x, ggml_tensor * ids);

// count-only path for any non-fused layer: returns a scalar f32 tensor (add it
// to the layer output to keep the op in the graph) or nullptr
ggml_tensor * build_moe_count(ggml_context * ctx, ggml_tensor * down_w, ggml_tensor * ids);

}
