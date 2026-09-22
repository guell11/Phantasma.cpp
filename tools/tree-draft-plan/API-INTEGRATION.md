# Tree-Draft API Integration Map

This note maps the early Pilar G API contracts onto the current public llama C API and `tools/server` implementation. It is intentionally a read-only integration plan: the manifests and core/server code are unchanged.

## Current execution path

The existing server already provides the main control plane needed by Tree-Draft:

```text
HTTP/OpenAI request
    |
    v
server_routes::handle_completions_impl()
    |
    v
server_task
    |
    v
server_queue
    |
    v
server_context::launch_slot_with_task()
    |
    v
server_slot(s) -> shared llama_batch -> llama_decode()
    |
    v
server_task_result_{cmpl_partial,cmpl_final,error}
    |
    v
server_response
    |
    v
server_response_reader
    |
    v
HTTP JSON / SSE
```

The important implication is that Tree-Draft does not need a new request queue, HTTP stack, SSE implementation, or OpenAI compatibility layer to satisfy the root and early G modules. The smallest integration is an execution seam underneath `server_slot` / `server_context::update_slots()`, while preserving the existing task and response path.

## Public llama C API surface that Tree-Draft can build on

The public API in `include/llama.h` already exposes the primitives required by an internal Tree-Draft executor:

| Public API | Current role | Tree-Draft use |
| --- | --- | --- |
| `llama_context` | Owns model execution and memory state | Target/draft execution contexts remain opaque to server/API code |
| `llama_context_params` | Configures context, batching, callbacks, optional related context | Carries batch limits and any target/draft context relationship configured below the server layer |
| `llama_batch` | Multi-token, multi-sequence decode input | Reuse as the physical batch submitted by Tree-Draft execution |
| `llama_batch_init/free` | Batch storage lifecycle | Reuse if Tree-Draft owns temporary batch storage |
| `llama_decode()` | Main decoder execution primitive | Remains the compute boundary for target/draft passes unless an existing lower-level Tree-Draft runtime wraps it |
| `llama_get_logits*()` | Retrieves decode logits | Useful at the execution layer for verification/sampling when not delegated to backend samplers |
| `llama_get_sampled_token_ith()` and sampled probability APIs | Backend sampler result access | Reuse if Tree-Draft uses configured sampler chains |
| `llama_sampler_*` | Sampling configuration and state | Keep sampling semantics aligned with ordinary server requests |
| `llama_get_memory()` plus `llama_memory_seq_*` | Sequence memory/KV inspection and mutation | Required for sequence-local rollback, acceptance, and slot lifecycle integration |
| `llama_state_seq_*` | Sequence state serialization | Optional later session/checkpoint integration; not required for the first API seam |
| `llama_set_abort_callback()` | Aborts `llama_decode()` when callback returns true | Only a coarse compute abort primitive today, and documented as CPU-only |

The public header documents `llama_decode()` return code `2` as an abort. It also states that ubatches already processed before an abort remain in context memory. Any Tree-Draft use of decode abortion therefore needs an explicit rollback/reconciliation policy before exposing it as per-request cancellation.

The current abort callback is not sufficient as the primary cancellation mechanism for Tree-Draft because it is attached to the whole `llama_context`, while the server batches multiple slots into one decode. Aborting the shared decode for one request can affect unrelated slots. The first integration should keep per-request cancellation at the slot/task layer and use the abort callback only if a future execution design can prove context-wide abort safety.

## Minimal interfaces needed by root and early Pilar G modules

The early G contracts can be implemented with a small adapter around structures that already exist.

### Canonical request: ID_301, ID_309

The server-side equivalent already exists as:

- `server_task` for identity, task type, tokens, slot affinity, and child tasks;
- `task_params` for stream mode, token limits, sampling, speculative settings, LoRA configuration, response type, and timing options;
- `server_tokens` for tokenized text/multimodal input.

The minimal Tree-Draft host request does not need to replace these types inside `tools/server`. A narrow conversion step can map the canonical G request into a completion `server_task`, or the canonical host contract can be represented by a transport-neutral subset embedded by/adapted into `server_task`.

Fields that should stay above Tree-Draft execution:

- JSON/OpenAI field names;
- chat template state;
- tool-call parsing state;
- SSE formatting;
- HTTP headers.

Fields that Tree-Draft execution needs:

- task/request identity;
- sequence/slot identity;
- input token span;
- max generated tokens;
- sampling/speculative configuration;
- cancellation state;
- deadline if added;
- batching compatibility data.

This follows the existing server design rule in `README-dev.md`: parse JSON into native C++ types before data reaches `server_slot`.

### Lifecycle and terminal semantics: ID_302, ID_318

The current implementation already has an implicit lifecycle:

```text
constructed
  -> queued
  -> deferred or launched in slot
  -> prompt processing / generation
  -> partial result(s)
  -> final result

or

constructed/queued/launched
  -> cancel task
  -> pending task removal or slot.release()
```

Relevant existing types/functions:

- `server_queue::post()`, `defer()`, and `pop_deferred_task()`;
- `server_context::launch_slot_with_task()`;
- `server_context::send_partial_response()`;
- `server_context::send_final_response()`;
- `server_task_result_cmpl_partial::is_stop() == false`;
- `server_task_result_cmpl_final::is_stop() == true`;
- `server_response_reader::next()` and `has_next()`.

For Tree-Draft, the minimal new lifecycle state can stay internal to the host adapter. Existing server callers only need the same observable property they already rely on: zero or more partial results followed by one final result or error.

The current result path does not carry an explicit monotonically increasing event sequence number. Early Tree-Draft integration can preserve current ordering through `server_response` and `server_response_reader`; adding an explicit canonical sequence number is useful for IPC/replay/conformance later, but it is not required to replace the server queue before Tree-Draft execution is wired in.

### Error contract: ID_303

Current server errors are represented by:

- `server_task_result_error`;
- `error_type` in `server-common.h`;
- `format_error_response()`.

Existing categories include invalid request, authentication, server, not found, permission, unavailable, not supported, and context-size exceeded.

Tree-Draft-specific failures should first map into the existing server error path:

| Tree-Draft failure | Existing server mapping |
| --- | --- |
| Invalid configuration/input | `ERROR_TYPE_INVALID_REQUEST` |
| Unsupported Tree-Draft mode/model/backend | `ERROR_TYPE_NOT_SUPPORTED` |
| Temporary runtime/backend unavailable | `ERROR_TYPE_UNAVAILABLE` |
| Context/KV capacity failure | `ERROR_TYPE_EXCEED_CONTEXT_SIZE` when applicable |
| Internal execution failure | `ERROR_TYPE_SERVER` |

The richer Pilar G domain/code taxonomy can be added behind or alongside this mapping without bypassing the existing HTTP error formatter.

### Native ownership and handles: ID_304..ID_308

The public llama API already uses opaque `llama_context *`, model pointers, sampler pointers, and `llama_memory_t`. Tree-Draft should keep those opaque below its host interface.

For server integration, a new public C handle layer is not required immediately. `tools/server` can call a C++ Tree-Draft host interface directly while the stable C ABI is specified separately for external consumers.

The minimal server-facing ownership rule is:

- `server_context` owns long-lived target/draft execution state;
- `server_slot` owns request-local Tree-Draft state or a small opaque request/session object;
- partial/final result objects own copied output needed after execution advances;
- no HTTP object owns a raw `llama_context` or Tree-Draft internal pointer.

This matches the current server architecture and avoids coupling the C ABI design to the first server integration.

### Cancellation: ID_310

The current cancellation path is already strong enough for the first Tree-Draft integration:

1. `server_response_reader::stop()` creates high-priority `SERVER_TASK_TYPE_CANCEL` tasks.
2. `server_queue::post()` removes matching pending/deferred work through `cleanup_pending_task()`.
3. If the request is already running, `server_context` handles `SERVER_TASK_TYPE_CANCEL` and calls `slot.release()`.
4. HTTP disconnect is observed through the response reader stop path for normal streams.
5. Resumable SSE uses an atomic cancel flag; explicit DELETE eventually unwinds the producer, whose response reader cancels the generation.

Tree-Draft should reuse this path and make request-local Tree-Draft state destructible/releasable when `server_slot::release()` runs.

The important semantic boundary is decode granularity. If a cancel arrives while a shared `llama_decode()` is executing, the current server cannot safely stop only one slot mid-call. Cancellation becomes effective when control returns to the server loop and the cancel task is processed. This is consistent with cooperative cancellation.

Do not wire one request's cancel token directly to `llama_set_abort_callback()` on the shared target context unless the executor is isolated to that request or can roll back all affected sequences correctly.

### Deadlines: ID_311

`task_params` currently contains generation timing limits such as `t_max_predict_ms`, while the queue/response reader polls for stop conditions. There is no general canonical absolute deadline propagated through the server task path.

The minimal future seam is one optional monotonic deadline field on the canonical host request / native task metadata. The server can translate an expired deadline into the existing cancellation path. No HTTP transport rewrite is needed.

### Backpressure and event queue: ID_312, ID_313

The current implementation provides boundedness at several layers but does not expose the credit protocol defined by Pilar G:

- `server_response` is the cross-thread result queue;
- the HTTP layer pulls one chunk at a time through `server_http_res::next()`;
- `httplib::DataSink::write()` determines whether the socket accepted a chunk;
- resumable streaming stores a bounded 4 MiB ring per stream session and drops the oldest bytes on overflow.

For initial Tree-Draft server integration, reuse these mechanisms. Tree-Draft should emit partial results at the same granularity as ordinary generation and must not build a second unbounded token queue.

Explicit credit-based backpressure becomes important for the later C ABI, Python IPC, and gRPC adapters. It can be introduced at the canonical host stream boundary without changing how `server_slot` batches compute.

### Admission and batching: ID_314, ID_317

The existing scheduler model is already compatible with Tree-Draft:

- `server_queue` accepts tasks;
- `server_context` defers tasks when no compatible slot is available;
- active `server_slot` instances contribute tokens into one shared `llama_batch`;
- `update_slots()` performs the shared decode;
- incompatible LoRA configurations are kept out of the same effective batching path.

Tree-Draft should add only the compatibility information its executor requires. Examples include target/draft model identity, Tree-Draft mode/configuration, and any state-sharing constraint that makes two requests unsafe to co-batch.

The preferred integration point is slot/batch selection before `llama_decode()`, not the HTTP route layer. The server routes should remain unaware of target/draft execution mechanics beyond validated request options.

### Session/context binding: ID_315, ID_316

The current server's closest session primitive is `server_slot` plus its sequence id and cached prompt/KV state. It also supports slot save/restore and sequence-state operations through llama memory/state APIs.

For the first Tree-Draft integration, bind request-local Tree-Draft state to the slot/sequence that already owns the target sequence. Do not create a parallel session registry just to run Tree-Draft.

The richer cross-request session API from Pilar G should later wrap or reference this sequence state with generation-safe IDs. Resumable SSE conversation IDs are transport/session-replay identifiers and should not be treated as the engine session identity.

## Where Tree-Draft should enter the server

The narrowest server seam is around the current decode stage in `server-context.cpp`.

Today:

```text
update_slots()
    -> build shared batch from active server_slot instances
    -> decode(...)
        -> llama_decode(ctx_tgt, batch_view)
    -> sample / advance slot
    -> send_partial_response() or send_final_response()
```

Recommended Tree-Draft shape:

```text
update_slots()
    -> build/select compatible active slot work
    -> execution adapter
         ordinary mode -> existing llama_decode path
         tree-draft mode -> Tree-Draft executor using llama contexts/memory/samplers
    -> commit accepted token(s) back to existing slot state
    -> existing sampling/result accounting as applicable
    -> existing send_partial_response()/send_final_response()
```

The execution adapter should return native data, not JSON. At minimum it needs:

```cpp
struct tree_draft_step_input {
    // logical active sequences/slots
    // token positions/input tokens needed for this step
    // per-request Tree-Draft and sampling configuration
    // cooperative cancellation/deadline view
};

struct tree_draft_step_output {
    // accepted/committed token(s) per sequence
    // updated sampling/verification data required by server_slot
    // speculative counters
    // execution status/error
};
```

These are conceptual fields, not a proposed patch. The actual implementation should reuse existing server and common types wherever they already express the same data.

The key invariant is that `server_slot` remains the owner of server-visible generation state. Tree-Draft may speculate internally, but only committed tokens become visible through `completion_token_output`, partial results, metrics, and KV state that ordinary server code treats as accepted history.

## Existing server paths to reuse

### OpenAI and native HTTP

Reuse unchanged:

- `POST /completion`;
- `POST /v1/completions`;
- `POST /v1/chat/completions`;
- `POST /v1/responses`;
- existing request conversion functions such as `oaicompat_chat_params_parse()`;
- `handle_completions_impl()`;
- existing `task_response_type` formatting.

Tree-Draft request options, if exposed at all, should enter through parsed native parameters and then become `task_params` / canonical host metadata. The existing OpenAI response format should remain identical unless optional Tree-Draft telemetry is deliberately exposed through extension fields.

### Streaming

Reuse unchanged:

- `server_task_result_cmpl_partial`;
- `server_task_result_cmpl_final`;
- `server_response_reader::next()`;
- HTTP chunked response handling in `server-http.cpp`;
- existing SSE serialization and final sentinel behavior in the response formatter;
- resumable streaming in `server-stream.{h,cpp}`.

Tree-Draft can accept multiple speculative tokens internally in one step, but externally it must preserve the same committed-token order. Chunk coalescing is allowed only if the existing response contract still reconstructs the identical committed output.

### Cancellation

Reuse unchanged:

- `server_response_reader::stop()`;
- high-priority `SERVER_TASK_TYPE_CANCEL`;
- pending queue cleanup;
- active `slot.release()`;
- normal HTTP disconnect stop checks;
- resumable stream explicit DELETE/cancel path.

Tree-Draft only needs cleanup hooks tied to slot/request release.

### Batching and slot lifecycle

Reuse:

- `server_queue`;
- slot allocation/defer logic;
- `server_slot` as the API-visible sequence owner;
- the shared `update_slots()` scheduling loop;
- existing per-slot metrics and timing surfaces.

Tree-Draft should extend compatibility/execution decisions within this layer rather than add an independent scheduler.

### Metrics and speculative accounting

`result_timings` already includes:

- `draft_n`;
- `draft_n_accepted`.

This is an immediate reuse point for Tree-Draft acceptance telemetry. Existing Prometheus/server metrics infrastructure should remain the publication path for later Pilar G observability work.

## Gaps that require a real integration seam

The current server and public C API do not fully provide these early-G concepts:

1. A transport-neutral canonical host request/result type independent of `tools/server`.
2. Explicit lifecycle enum and event sequence numbers.
3. Stable cross-language error codes richer than the current HTTP-oriented `error_type`.
4. Per-request cancellation visible inside a long shared decode without affecting other sequences.
5. General monotonic request deadline propagation.
6. Explicit credit-based stream backpressure for non-HTTP consumers.
7. A stable public C ABI for Tree-Draft runtime/session/request handles.
8. A Tree-Draft execution interface that can return committed tokens and acceptance metadata while hiding speculative intermediate state.

Only item 8 is required to integrate Tree-Draft into the current server compute path. The other items can be layered around the same execution seam as Pilar G progresses.

## Cancellation semantics for Tree-Draft

The server/API contract should define cancellation at commit boundaries:

- Tokens already committed into the server slot before cancellation remain part of the request history.
- Speculative tokens not yet accepted/committed must be discarded.
- A pending request can be removed without entering a slot.
- A running request becomes cancelled when the server regains control and releases its slot/request state.
- No partial response may be emitted for uncommitted speculative tokens.
- Cancellation must be idempotent.
- Cancellation of one slot must not abort unrelated slots sharing the same target decode.

If Tree-Draft maintains draft-side KV/state ahead of the committed target position, request teardown may simply discard that request-local speculative state. If it mutates shared target sequence state speculatively, it must restore the target sequence to the last committed position before slot reuse.

## Streaming semantics for Tree-Draft

Tree-Draft changes how tokens are produced internally, not what the public stream means.

The server should continue to expose:

1. optional begin/progress chunks;
2. ordered committed content/token chunks;
3. exactly one final result;
4. existing OpenAI/SSE terminal framing.

For a speculative step that accepts tokens `[t0,t1,t2]`, the adapter may emit one chunk per token or a coalesced chunk if current formatter semantics permit it. In either case:

```text
concatenate(public committed chunks) == committed target sequence
```

Rejected draft tokens never appear in public output, token counts, stop matching, or generated-text state.

Stop words, EOS, max-token limits, tool parsing, and response formatting should continue to operate on committed output in the same server layer that handles them today.

## Suggested integration order

The minimal implementation order for early Pilar G is:

1. Define one internal Tree-Draft execution adapter at the `server_context` decode boundary.
2. Bind its per-request state to existing `server_slot` lifecycle.
3. Preserve existing `server_task`, `server_queue`, `server_response`, and result types.
4. Reuse current cancellation by making slot release tear down Tree-Draft state.
5. Feed committed tokens through existing partial/final response functions.
6. Populate existing `draft_n` and `draft_n_accepted` timing/accounting fields.
7. Add only the smallest request option/compatibility metadata required to select Tree-Draft execution.
8. After the compute path is stable, layer the stable C ABI, Python, IPC, and richer error/backpressure contracts on the same host execution interface.

This order keeps Tree-Draft inside the current llama.cpp scheduling and HTTP architecture. The only new mandatory boundary is the execution adapter that converts slot work into committed Tree-Draft output; everything above that boundary can initially reuse existing server behavior.

## Source locations reviewed

- `include/llama.h`: public context, batch, decode, memory, sampler, state, and abort callback APIs.
- `tools/server/server-task.h`: `server_task`, `task_params`, partial/final/error result types, speculative timing counters.
- `tools/server/server-queue.h` and `server-queue.cpp`: submission, result delivery, response reader, cancellation task creation.
- `tools/server/server-context.h` and `server-context.cpp`: slots, scheduling, task launch, shared batching, `llama_decode()`, result emission, active cancellation.
- `tools/server/server-http.cpp`: chunked streaming transport and connection completion.
- `tools/server/server-stream.h` and `README-dev.md`: resumable SSE buffering, explicit cancellation, replay lifetime.
- `tools/server/server-common.h`: current error categories.
- `tools/server/server.cpp`: route registration for native and OpenAI-compatible APIs.

