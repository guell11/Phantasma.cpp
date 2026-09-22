# Dependency Audit - Pilar G (ID_301..ID_350)

Scope: semantic cross-pillar audit for Pilar G against the existing Pilar C draft handoff, Pilar D verification result, Pilar F model compatibility/preflight contracts, Pilar H conformance modules, and the current tools/server ownership model. This file proposes dependency changes only. No manifest or core/server source is edited.

## Executive finding

Pilar G should remain an API and host-contract layer around the request runtime that tools/server already owns. The current server already owns request parsing, server_task creation, queueing, slot allocation, shared batching, speculative execution, cancellation, committed-token accounting, partial/final results, and HTTP/SSE delivery.

The missing G dependencies are therefore narrow. API configuration and cancellation consume C contracts directly, while the final host integration boundary consumes the sealed C proposal and immutable D verification result. Public event and transport modules must see committed output only; they should not depend on draft-tree nodes, verifier logits, rejected branches, or KV internals.

Pilar F is different. In the current server, draft/target models and speculative compatibility are established when the server runtime is loaded, before individual requests are admitted. No current HTTP/OpenAI request dynamically constructs or validates a draft/target model pair. For that reason, this audit finds no high-confidence direct G -> F manifest edge under the current ownership model.

Use notation `consumer -> dependency`.

## High-confidence missing edges from G into execution contracts

| Suggested edge | Reason |
| --- | --- |
| `ID_309 -> ID_101` | G entry validation explicitly covers speculative parameters. C ID_101 is the immutable draft-request configuration contract for tree/node/depth/candidate limits, temperature, and seed. G should normalize public speculative options into that contract instead of defining a second set of Tree-Draft limits/defaults. |
| `ID_310 -> ID_109` | G defines the frontend-visible cooperative cancellation contract. C ID_109 defines the safe points at which a live draft expansion stops creating children while preserving a valid sealable tree. The host cancellation path must ultimately obey those engine safe-point semantics. |
| `ID_350 -> ID_150` | The final G integration boundary is where existing server slot work enters Tree-Draft execution. C ID_150 is the immutable sealed proposal that leaves the draft side. The host adapter must consume that exact handoff rather than invent another request-local tree representation. |
| `ID_350 -> ID_191` | After target verification, the host adapter needs the accepted path/count, replacement or bonus token, fallback status, and related immutable decision data to update the existing server slot and expose committed output. D ID_191 is the canonical verification result carrying those fields. |

These edges do not make G the owner of draft expansion or target verification. They define the two places where the existing server-owned request lifecycle crosses into and back out of Tree-Draft execution.

## Pilar F compatibility boundary

No direct G -> F edge is required for the current server integration.

The current runtime loads the target context and optional draft context before slots are initialized. `common_speculative_init()` then constructs the speculative implementation, and the existing draft path checks draft/target vocabulary and special-token compatibility before it can be used. The initialized `server_context` owns those contexts and gives each `server_slot` references to the already-selected execution state.

That means the API consumes an activated runtime capability, not a loader/preflight object. In particular:

- `ID_338` should not depend on `ID_292`. An OpenAI `model` field is currently validated/routed as a server model name or alias; it does not choose a new draft/target pair for the request.
- `ID_346` should not depend on `ID_292` or `ID_299`. A batching compatibility key may contain an opaque activated-runtime/model-pair identity, but the transport normalizer does not need to re-run loader compatibility.
- `ID_301`, `ID_306`, and `ID_316` should keep model/runtime internals opaque. Exposing a runtime/session handle does not imply exposing an F ModelHandle or preflight report.

Two F edges become reasonable only if Pilar G later expands its public contract:

| Candidate edge | Add it only if |
| --- | --- |
| `ID_309 -> ID_299` | Per-request Tree-Draft mode selection is validated directly against F's preflight capability/fallback report rather than against an opaque already-activated runtime capability. |
| `ID_337 -> ID_299` | The HTTP models/runtime-info surface explicitly promises to expose F's preflight capability report or its normalized fallback reasons. |

Neither condition is true in the current tools/server path, so these are not part of the conservative manifest change set.

## High-confidence downstream H edges into G

H should consume the observable G contracts. G must not depend on H.

| Suggested edge | Reason |
| --- | --- |
| `ID_356 -> ID_318` | H defines TTFT from request start to the first externally visible token. G ID_318 defines the ordered committed-token event stream and terminal publication boundary, so it defines what counts as the first visible token. |
| `ID_357 -> ID_318` | H inter-token latency and end-to-end latency require externally visible token-event timestamps plus one terminal completion. G ID_318 owns that observable sequence and exactly-once terminal rule. |
| `ID_380 -> ID_317` | H's batched/concurrent-request integration test needs the single API-to-scheduler submission contract. G ID_317 maps accepted requests onto scheduler tickets while preserving server ownership of queueing/slots; this is the correct concurrency ingress boundary to test. |
| `ID_382 -> ID_318` | H requires that no token after committed EOS/stop be externally emitted. That assertion is specifically about G's committed event ordering and terminal delivery, not about speculative nodes that never leave the engine. |
| `ID_400 -> ID_350` | H's final release-readiness matrix includes end-to-end integration evidence. G ID_350 is the final cross-binding host/API integration contract whose invariants must have passing evidence before the full Tree-Draft surface is release-ready. |

I do not recommend `ID_379 -> ID_350`. ID_379 tests core Tree-Draft behavior across tree shapes. Requiring all native/Python/IPC/HTTP/gRPC integration contracts first would turn a core execution test into a full public-binding gate and unnecessarily lengthen the functional critical path.

## Edges I do not recommend adding

- Do not add `ID_301 -> ID_150` or `ID_301 -> ID_191`. The canonical host envelope represents request input and committed result output. It should not expose draft proposal or verifier-internal records.
- Do not add `ID_318 -> ID_191`. Event sequencing consumes committed server output after verification. Rejected nodes, winner-branch internals, and verifier RNG data must remain below the public stream boundary.
- Do not add `ID_319 -> ID_191`. Existing server accounting already aggregates speculative counters such as `draft_n` and `draft_n_accepted`. The metrics schema can consume those stable counters without coupling every metrics implementation to the verifier result layout.
- Do not add `ID_337 -> ID_150`, `ID_338 -> ID_150`, or equivalent transport-to-draft-tree edges. HTTP/OpenAI parsing remains above `server_task` and must not understand proposal topology.
- Do not add `ID_338 -> ID_292` simply because both discuss a model. Request model naming and loader-time draft/target compatibility are different contracts in the current server.
- Do not add `ID_346 -> ID_292`. Dynamic batching needs an equality key for already-admitted execution state, not the algorithm used to prove that state was loadable.
- Do not add `ID_316 -> ID_191`. A session binds to opaque engine state/KV ownership. A verifier result is a per-step value, not session identity.
- Do not add generic G dependencies on B KV or A kernel modules. G should treat KV, packed masks, logits, and backend kernels as engine-private state behind the Tree-Draft execution adapter.

## Current server seams that justify the audit

The present implementation already supplies the ownership model the dependency graph should preserve:

1. `server_task` plus `task_params` carry request identity, token input, stream mode, limits, sampling, speculative configuration, LoRA state, and response mode into the existing queue.
2. `server_context::launch_slot_with_task()` binds admitted work to a `server_slot`. The slot owns request-local generation state, sampler state, prompt state, speculative draft/index/checkpoint data, and speculative counters.
3. `server_context::update_slots()` selects compatible live slots, calls the existing `common_speculative_draft()` seam, builds the shared target batch, and executes target decode. Tree-Draft belongs inside this stage.
4. `post_decode()` currently verifies linear speculation, restores/removes rejected speculative state, calls `common_speculative_accept()`, and sends only accepted/committed tokens through `process_token()`. Tree verification should preserve the same visibility boundary.
5. `send_partial_response()` and `send_final_response()` publish server-owned result objects. G streaming/HTTP/Python/IPC/gRPC adapters should consume this committed-result boundary, not verifier internals.
6. Cancellation already enters as `SERVER_TASK_TYPE_CANCEL`: pending work is removed from the queue and a live task releases its slot when the server regains control. One request must not directly abort the shared target `llama_context`.
7. The server owns `ctx_tgt`, optional `ctx_dft`, speculative initialization, and the slot array. Draft/target compatibility is established before request execution, so transport adapters do not own model compatibility checks.

## Required runtime ownership shape

```text
public adapters / canonical G request
              |
              v
existing server_task -> server_queue -> server_slot
                                      |
                                      v
                            Tree-Draft execution seam
                                      |
                         C ID_150 sealed proposal
                                      |
                                      v
                              D target verifier
                                      |
                         D ID_191 verifier result
                                      |
                                      v
                     existing server slot commit/accounting
                                      |
                                      v
                         G ID_318 committed events
                                      |
                                      v
                          existing public transports
```

The server remains the request-runtime owner throughout this flow. Pilar G does not need a second request queue, scheduler, session registry, HTTP stack, SSE implementation, or target/draft model executor.

## Cross-binding conformance gap

G ID_349 defines language-neutral request/stream/error/cancellation vectors, but Pilar H currently has no module whose objective is specifically to run those vectors through every G binding. ID_379 is a core Tree-Draft integration suite, and ID_380 is a batch/concurrency suite; forcing either to own Python, IPC, HTTP, and gRPC round trips would distort their existing scope.

A future manifest revision can either expand an H module explicitly to own cross-binding conformance or add a dedicated consumer of ID_349. Until that ownership decision is made, this audit does not invent an artificial H -> ID_349 edge.

## Recommended conservative manifest change set

Add these nine high-confidence edges:

1. `ID_309 -> ID_101`
2. `ID_310 -> ID_109`
3. `ID_350 -> ID_150`
4. `ID_350 -> ID_191`
5. `ID_356 -> ID_318`
6. `ID_357 -> ID_318`
7. `ID_380 -> ID_317`
8. `ID_382 -> ID_318`
9. `ID_400 -> ID_350`

No direct Pilar G -> Pilar F edge is recommended under the current server ownership model.

## DAG effect

For this delta alone, the G execution edges point back to C/D and the H conformance edges point back to G. No edge points from G into H, and no F dependency is introduced. Relative to the current published manifests, these additions do not create a reciprocal G path and are consistent with the intended execution order:

```text
C proposal -> D verification -> G committed host/public result -> H observable conformance
```

The prime should still rerun the global SCC/topological validator after reconciling all pillar audits, because other audits may add cross-pillar edges that are not present in the current manifests.
