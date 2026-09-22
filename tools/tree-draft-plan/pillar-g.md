# Pilar G - API, C++ Bindings, and Python IPC

Pilar G defines the public and inter-process contract around the tree-draft runtime. The design uses one canonical host request/result model and makes every language or transport adapter preserve the same lifecycle, ordering, cancellation, error, batching, session, and observability semantics.

The 50 modules are intentionally split so implementation can proceed in layers after dependencies are satisfied:

| IDs | Area | Contract focus |
| --- | --- | --- |
| ID_301..ID_309 | Core host API and native ABI | Canonical request/result records, lifecycle, errors, buffer ownership, ABI versioning, opaque handles, callbacks, C++ RAII, validation |
| ID_310..ID_321 | Runtime control semantics | Cancellation, deadlines, backpressure, bounded event queues, admission, sessions, scheduler submission, event sequencing, metrics, tracing, logging |
| ID_322..ID_329 | Python / pybind11 | Python object ownership, typed conversion, futures, async streaming, cancellation, GIL rules, zero-copy views, exception mapping |
| ID_330..ID_336 | IPC | Framing, negotiation, shared memory, flow control, multiplexing, cancellation, disconnect and recovery semantics |
| ID_337..ID_342 | HTTP, OpenAI compatibility, and SSE | HTTP schema, OpenAI request/response translation, SSE framing, disconnect/backpressure, HTTP error mapping |
| ID_343..ID_345 | gRPC | Protobuf service contracts, streaming, deadlines and cancellation |
| ID_346..ID_350 | Cross-transport integration | Batching metadata, session equivalence, observability, conformance vectors, end-to-end invariants |

## Canonical contract

All ingress paths normalize into the host request defined by ID_301 before scheduler submission. All egress paths consume the canonical event/result/error model rather than defining transport-specific engine behavior. This keeps the following observable properties equivalent across C, C++, Python, IPC, HTTP/OpenAI, SSE, and gRPC:

- request identity and lifecycle transitions;
- token/event order and exactly-once terminal delivery;
- cancellation and deadline races;
- bounded buffering and backpressure;
- batching and admission metadata;
- session generation and close behavior;
- stable error domain/code values;
- usage accounting and finish reasons;
- trace, metric, and log correlation.

## Dependency direction

The intended dependency direction is host contract -> control/lifecycle -> language and transport adapters -> cross-transport conformance. Transport modules may depend on canonical host modules, but canonical host modules never depend on HTTP, gRPC, Python, or IPC.

The native ABI is anchored by ID_304..ID_308. Python builds on the C++ RAII facade in ID_322 rather than creating a separate runtime ownership model. IPC uses the common error, cancellation, event ordering, and backpressure contracts. HTTP/OpenAI/SSE and gRPC translate to the same host request and consume the same ordered events.

## Backpressure and cancellation

ID_312 defines the transport-neutral credit model. Native callbacks, Python async iteration, IPC byte credits, SSE socket writability, and gRPC HTTP/2 flow control all reduce to that mechanism. A slow or disconnected consumer must therefore bound producer memory and can propagate cancellation without changing engine-visible semantics.

ID_310 and ID_311 define cancellation and deadline races once. Bindings only translate their local cancellation source into the shared token. Once a request reaches a terminal lifecycle state, later cancellation is idempotent and cannot alter the result.

## Sessions and batching

ID_315 and ID_316 define session identity, generation, engine-state binding, and close semantics. ID_347 requires Python, IPC, HTTP, and gRPC session operations to preserve those rules.

ID_314 defines admission inputs and limits, while ID_346 normalizes per-transport batching metadata. Transport choice must not create different batching eligibility for semantically equivalent requests.

## Error and observability semantics

ID_303 is the only source of cross-language error identity. Python exceptions, HTTP status/error bodies, IPC terminal failures, and gRPC statuses map from this taxonomy.

ID_319..ID_321 define metrics, trace context, and structured logs. ID_348 exposes them at API boundaries while keeping high-cardinality identifiers out of metric labels and prompt/token content out of logs by default.

## Completion gate

ID_349 defines language-neutral conformance vectors. ID_350 is the integration gate: a request issued through any supported binding must normalize to the same canonical semantics and reconstruct the same ordered output, terminal status, usage, and error identity after transport framing is removed. Native handles and exported buffers must also return to zero live ownership after teardown.

The machine-readable source of truth is `pillar-g.jsonl`. It contains exactly one JSON object per module, with IDs ID_301 through ID_350.
