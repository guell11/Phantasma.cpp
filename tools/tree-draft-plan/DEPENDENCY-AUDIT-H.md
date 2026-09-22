# Dependency Audit - Pilar H (ID_351..ID_400)

Scope: semantic audit of Pilar H against implementation contracts in Pilar A through Pilar G and the current llama.cpp speculative/server seams. This document proposes dependency changes only. No pillar manifest or core source was edited.

## Executive finding

Pilar H currently has zero cross-pillar dependencies. Every dependency in `pillar-h.jsonl` points to another H module even though H measures and tests behavior owned by A through G. That makes the global DAG materially dishonest: benchmark and correctness nodes can become ready before the runtime contracts they claim to observe exist.

The strongest missing seams are:

- A owns deterministic attention/sampling primitives, tolerance policy, and integrated GPU kernel acceptance.
- B owns tree-KV rollback, commit, partial acceptance, context shifting, consistency, and concurrent cache publication.
- C owns draft-tree expansion, proposal RNG/probabilities, telemetry, and the sealed proposal handoff.
- D owns target verification, greedy/stochastic acceptance, residual fallback, accepted-path compaction, discard sets, and verifier conformance.
- E owns asynchronous CUDA stream/event ordering and allocator/device telemetry used by GPU checks.
- F owns activated-model preflight and draft/target compatibility.
- G owns request lifecycle, committed event ordering, batching/submission, and transport-neutral metrics.

Notation below is `consumer -> dependency`.

## High-confidence missing edges: timing, throughput, TTFT, and memory

| Suggested edge | Reason |
| --- | --- |
| `ID_352 -> ID_149` | H phase timing names draft expansion as a measured phase. C ID_149 defines the per-committed-expansion telemetry record and its model/postprocess timing inputs. H should instrument the producer contract rather than invent a second draft timing boundary. |
| `ID_352 -> ID_164` | H explicitly times target verification. D ID_164 is the single target verification invocation boundary, so its begin/end is the semantic verify phase. |
| `ID_352 -> ID_318` | H includes first externally visible token and request end. G ID_318 defines ordered committed token events and exactly-once terminal delivery, which are the correct externally visible boundaries. |
| `ID_355 -> ID_318` | Generated-token throughput must count committed emitted output, not proposed draft tokens. G ID_318 owns the event stream whose token deltas reconstruct committed output exactly. |
| `ID_356 -> ID_318` | TTFT is defined by first externally visible committed token. That event is owned by G ID_318. |
| `ID_357 -> ID_318` | ITL and E2E distributions need ordered committed emission timestamps plus one terminal event. G ID_318 defines that sequence. |
| `ID_358 -> ID_149` | Draft-phase cost must use C's expansion-step telemetry boundary, especially when draft model work and postprocessing overlap. |
| `ID_358 -> ID_164` | Verify-phase cost must wrap the target verification decode contract rather than arbitrary surrounding host work. |
| `ID_364 -> ID_077` | KV memory attribution requires B's shared-versus-exclusive per-branch accounting so shared physical pages are not double counted. This edge was also found independently in the Pilar B audit. |
| `ID_365 -> ID_238` | E ID_238 owns live/reserved/peak and fragmentation telemetry for speculative temporary allocators. H scratch high-water measurement should consume those counters where the E allocator path is active. |

I do not recommend making `ID_351` depend on a runtime pillar. The benchmark scenario schema is useful as an independent root contract; execution modules can validate a scenario against F/G later.

## High-confidence missing edges: tree metrics and seed provenance

| Suggested edge | Reason |
| --- | --- |
| `ID_359 -> ID_150` | Proposed-token/node counts originate in C's sealed draft-tree handoff, including topological count and proposal provenance. |
| `ID_359 -> ID_191` | Accepted counts and mode/fallback result data originate in D's immutable verification result record. H acceptance rate needs both proposal and acceptance producers. |
| `ID_360 -> ID_150` | H promises raw accepted topology identifiers. The stable node/path identities come from C's sealed tree. |
| `ID_360 -> ID_191` | D ID_191 returns accepted nodes and winning branch metadata needed to compute accepted depth/branch structure. |
| `ID_361 -> ID_164` | The denominator is the number of target verification invocations; D ID_164 is exactly one target call per speculative verification pass. |
| `ID_361 -> ID_191` | The numerator is committed/accepted output from the verification result. D ID_191 is the immutable per-sequence producer for that information. |
| `ID_366 -> ID_044` | H's root/sub-seed scheme must ultimately feed the shared counter-based RNG ABI instead of defining a benchmark-only RNG family. |
| `ID_366 -> ID_111` | C ID_111 defines draft per-node RNG coordinates and therefore one consumer domain for H-derived seeds. |
| `ID_366 -> ID_174` | D ID_174 defines verifier uniform coordinates and is the second independent consumer domain H must seed reproducibly. |
| `ID_376 -> ID_191` | H's synthetic verification records should mirror D's real immutable verification result schema so counter tests cannot pass on a shape the runtime never emits. |

The current server already exposes `n_draft_total`, `n_draft_accepted`, verification-step counts, and accepted-per-position counters. Those are useful implementation reuse points, but the Tree-Draft DAG should define the producer contracts above before H standardizes the derived metrics.

## High-confidence missing edges: greedy and logit equivalence

| Suggested edge | Reason |
| --- | --- |
| `ID_367 -> ID_172` | D ID_172 closes the greedy verifier rule by selecting the target replacement token on rejection or bonus token after full acceptance. Token-for-token greedy equivalence cannot be tested before that rule is defined. |
| `ID_367 -> ID_318` | The compared Tree-Draft output must be the committed public sequence. G ID_318 guarantees token deltas concatenate to exactly that output. |
| `ID_368 -> ID_197` | D ID_197 already defines the exact one-pass target-logit versus linear-replay comparison and first-divergence diagnostics H wants to promote into an integration gate. |
| `ID_368 -> ID_035` | A ID_035 owns backend/path-specific numerical tolerances. H should consume that matrix instead of inventing independent `atol/rtol` values. |

### Artificial edge to remove

`ID_367 -> ID_366` is not a real prerequisite. ID_367 is explicitly temperature-zero greedy equivalence. Its expected output must be independent of stochastic seed propagation. A test runner may still record a seed field for scenario reproducibility, but the greedy correctness contract does not consume H's stochastic seed derivation. Replace this edge with the D/G edges above.

## High-confidence missing edges: stochastic correctness

| Suggested edge | Reason |
| --- | --- |
| `ID_369 -> ID_115` | C ID_115 defines deterministic categorical selection from the filtered draft distribution. Distribution-level Tree-Draft tests must cover the actual proposal sampler semantics, not only verifier correction. |
| `ID_369 -> ID_184` | D ID_184 defines the exact post-temperature target and draft distributions and the temperature-zero dispatch boundary used by stochastic verification. |
| `ID_369 -> ID_198` | D ID_198 cross-checks optimized stochastic verification against the scalar acceptance/residual reference. H's repeated-sampling goodness-of-fit test should sit above that semantic oracle. |
| `ID_370 -> ID_318` | Pairwise sampled-output comparison needs the same committed-output event contract for both Tree-Draft and reference runs. |
| `ID_371 -> ID_173` | The hand-computable acceptance probability under test is exactly D ID_173. |
| `ID_371 -> ID_175` | Boundary uniform decisions are implemented by D ID_175's log-space accept predicate. |
| `ID_371 -> ID_180` | ID_371 also promises residual-distribution normalization checks; D ID_180 defines stable residual construction. |
| `ID_372 -> ID_188` | Multiple overlapping branches must reduce to one policy-valid winning branch/path. D ID_188 is the per-sequence winner reduction over greedy or stochastic paths. |
| `ID_373 -> ID_181` | Normal rejection fallback token selection is D ID_181's residual sampler. |
| `ID_373 -> ID_182` | Numerically degenerate residual mass uses D ID_182's target-distribution fallback. |

### Artificial edge to remove

`ID_371 -> ID_366` is unnecessary for the stated unit test. ID_371 injects hand-computable probability vectors and boundary uniforms; it tests acceptance math, not benchmark seed derivation. D ID_174 remains relevant to end-to-end seeded stochastic tests through ID_366, but the scalar rule test can directly supply `u`.

## High-confidence missing edges: rollback, commit, context, and stop semantics

| Suggested edge | Reason |
| --- | --- |
| `ID_374 -> ID_067` | H directly tests rollback to an accepted ancestor. B ID_067 owns page-table truncation, refcount release, and reader-safe reclamation for that operation. |
| `ID_374 -> ID_190` | D ID_190 defines exactly which speculative nodes must be discarded after verification. Rollback must remove the cache state corresponding to that discard set. |
| `ID_375 -> ID_070` | B ID_070 defines full/partial acceptance at page boundaries and already depends on canonical commit ID_069. H's zero/partial/maximal acceptance cases need that contract. |
| `ID_375 -> ID_189` | D ID_189 compacts accepted node IDs into causal commit order. H must compare sequence/cache state against the same order the verifier produces. |
| `ID_381 -> ID_089` | Long-context and post-shift tests directly exercise B's context-shift behavior for canonical plus speculative branches. |
| `ID_382 -> ID_109` | C ID_109 owns cancellation and early-stop propagation through the draft runtime. Once EOS/stop is committed, no further proposal work should survive as live request work. |
| `ID_382 -> ID_190` | Descendants after a committed stop boundary must be in the verifier discard set rather than becoming commit candidates. |
| `ID_382 -> ID_318` | The externally observable condition is no committed token event after terminal stop/EOS. G ID_318 owns that event/terminal invariant. |

Current `tools/server/server-context.cpp` already demonstrates why these boundaries are separate: target/draft state rollback uses checkpoints plus sequence removal, while accepted tokens are processed and emitted afterward through the normal result path. H should test the B/D/G contracts rather than treating acceptance, cache rollback, and visible output as one opaque action.

## High-confidence missing edges: end-to-end integration and concurrency

| Suggested edge | Reason |
| --- | --- |
| `ID_379 -> ID_050` | Representative tree shapes must run only after A's kernel/dispatcher correctness, fallback, deterministic, sampling, and benchmark acceptance contract is available. |
| `ID_379 -> ID_100` | End-to-end generation with tree state requires B's KV/attention integration and acceptance suite. |
| `ID_379 -> ID_150` | C's sealed draft-tree handoff is the proposal object the integration path must feed into verification. |
| `ID_379 -> ID_200` | D ID_200 closes verifier conformance across tree shapes, batches, greedy/stochastic modes, failures, and KV equivalence. H's broader E2E suite should build on that contract. |
| `ID_379 -> ID_299` | F ID_299 is the activation preflight guaranteeing loaded target/draft models and fallback plan are executable before an E2E Tree-Draft run starts. |
| `ID_380 -> ID_093` | Concurrent-request integration must respect B's RESERVE/WRITE/PUBLISH/ATTEND/VERIFY/COMMIT reader-writer phase boundaries. |
| `ID_380 -> ID_317` | G ID_317 owns the request submission contract into batching/scheduling, including one ticket per accepted request. |
| `ID_380 -> ID_318` | Per-request isolation is observed through ordered committed event streams and exactly one terminal per request. |

I do not recommend `ID_379 -> ID_350`. ID_379 is an engine/runtime integration test and can run below public bindings. G's full cross-binding completion contract belongs in release readiness and API-specific integration evidence, not as a prerequisite for the core engine test.

## High-confidence missing edges: fuzzing, sanitizers, and GPU checks

| Suggested edge | Reason |
| --- | --- |
| `ID_383 -> ID_005` | A ID_005 is the packed-tree topology validator whose parent/range/forest invariants the topology fuzzer should attack. |
| `ID_383 -> ID_150` | C's sealed handoff is the in-memory/serialized producer-side topology and proposal metadata shape consumed downstream. |
| `ID_383 -> ID_160` | D ID_160 validates packed target token/sequence/position/ancestor mappings. H explicitly fuzzes verification metadata parsers, so this is the verifier-side validation target. |
| `ID_384 -> ID_070` | Property fuzzing state transitions must include partial/full commit behavior, not rollback alone. B ID_070 closes partial acceptance semantics. |
| `ID_384 -> ID_191` | Generated acceptance decision streams should be interpreted through the same immutable verification result shape used by production commit/discard logic. |
| `ID_386 -> ID_094` | H's TSan scope names shared cache metadata. B ID_094 defines versioned snapshot consistency for asynchronously consumed page tables. |
| `ID_386 -> ID_319` | H also names shared metrics state. G ID_319 owns metric update points/accounting, which is the state the TSan job should exercise concurrently. |
| `ID_387 -> ID_050` | GPU memory checking must cover A's production tree attention/top-k/sampling/gather paths, not only a host integration harness. |
| `ID_387 -> ID_100` | The GPU check scope includes tree-KV and paged-attention metadata; B's acceptance suite closes those contracts. |
| `ID_387 -> ID_222` | Asynchronous GPU races depend on E's explicit event-based cross-stream happens-before rules. Memory/race checking should execute work under those rules. |

### Artificial edge to remove

`ID_383 -> ID_378` couples unrelated schemas. ID_378 validates benchmark scenario canonicalization; ID_383 fuzzes tree topology and verification metadata. The fuzzer should depend on the A/C/D parser and validator contracts above. Keep benchmark-scenario fuzzing as a separate target if desired.

`ID_385` does not need additional direct A-G edges if `ID_379`, `ID_383`, and `ID_384` gain the dependencies above. Its job is to run ASan/UBSan over those already-defined test surfaces.

## High-confidence missing edges: artifacts and CI gates

| Suggested edge | Reason |
| --- | --- |
| `ID_388 -> ID_319` | Result artifacts should use the transport-neutral metric names/accounting rules so benchmark output and runtime observability do not report incompatible definitions for throughput, latency, acceptance, or cancellation. |
| `ID_395 -> ID_392` | Artifact retention says failed gates must preserve reproducer inputs/logs, but its current dependencies include only performance/tree-quality gates. Correctness-gate failures need the same mandatory failure bundle. |
| `ID_392 -> ID_386` | The correctness gate promises required sanitizer coverage. TSan is the concurrency sanitizer for shared Tree-Draft state and must be included on supported builds, with platform exceptions handled by the gate policy. |
| `ID_399 -> ID_386` | CI tier scheduling explicitly maps sanitizer workloads. TSan needs an assigned supported tier rather than existing outside the schedule graph. |
| `ID_400 -> ID_050` | Release readiness must include current evidence that Pilar A's integrated kernel/sampling path is accepted. |
| `ID_400 -> ID_100` | Release readiness must include Pilar B KV/paged-attention integration and stress evidence. |
| `ID_400 -> ID_150` | Release readiness must include the finalized C draft-tree handoff contract used by the runtime. |
| `ID_400 -> ID_200` | Release readiness must include D's target-verifier conformance suite. |
| `ID_400 -> ID_250` | Hardware-specific readiness for the SM89 path must include E's tuning acceptance matrix when that backend/device is in release scope. |
| `ID_400 -> ID_300` | Release readiness needs F loader/model-preflight conformance fixtures so benchmark success is not based on unvalidated model activation. |
| `ID_400 -> ID_350` | Public integration readiness must include G's cross-binding lifecycle/output/cancellation/error invariants. |

For `ID_400 -> ID_250`, the readiness matrix should mark the edge/evidence conditional on SM89 being a supported release target. The dependency remains semantically correct for the complete project plan, while the evidence row can be scoped by backend/device.

## Intra-H coverage edges worth adding

These are independent of A-G but close obvious evidence gaps in H itself:

| Suggested edge | Reason |
| --- | --- |
| `ID_388 -> ID_357` | Artifact schema should preserve ITL/E2E raw samples and summaries. |
| `ID_388 -> ID_358` | Artifact schema should preserve draft/verify/sample phase costs and residual overhead. |
| `ID_388 -> ID_360` | Artifact schema should preserve accepted-depth/branch distributions. |
| `ID_388 -> ID_361` | Artifact schema should preserve committed tokens per target invocation and verified/proposed counts. |
| `ID_388 -> ID_364` | Artifact schema should preserve Tree-Draft KV memory attribution. |
| `ID_388 -> ID_365` | Artifact schema should preserve scratch/workspace high-water measurements. |

Without these edges, ID_388 can be implemented and frozen before half of H's benchmark metric payloads have a stable definition.

## Medium-confidence edges to review before manifest edits

| Candidate edge | Why it may be needed | Why it may remain indirect |
| --- | --- | --- |
| `ID_352 -> ID_319` | G metric names/update points overlap H timing counters. | ID_352 can remain a benchmark-local instrumentation contract while ID_388 maps final metrics to G names. |
| `ID_362 -> ID_318` | Baseline/tree runs should compare the same committed-output policy. | If ID_355 already consumes ID_318 and ID_362 only combines completed benchmark summaries, the dependency is inherited. |
| `ID_367 -> ID_299` | E2E greedy comparison requires an activated compatible model pair. | If the greedy test uses a fixture that enters through ID_379's preflight-backed harness, this can stay inherited at the harness layer. |
| `ID_369 -> ID_185` | Stateful grammar/repetition/bias transforms affect exact target/proposal distributions. | ID_369 says fixed context categorical goodness-of-fit and can intentionally use a sampler configuration without stateful constraints. Add this edge only if the statistical suite claims grammar coverage. |
| `ID_380 -> ID_133` | C defines fairness-aware selection across concurrent draft requests. | H380 asserts isolation/equivalence, not fairness. A separate scheduler-fairness test should depend on ID_133. |
| `ID_387 -> ID_250` | E's SM89 acceptance path should be exercised under GPU memory/race checking on Ada. | ID_387 is accelerator-generic; forcing a direct dependency on SM89-specific tuning would make non-Ada GPU coverage dishonest. Scope this in ID_399/ID_400 instead. |
| `ID_399 -> ID_286` | Hardware-specific CI matrices can use F's device/quant kernel capability matrix to select supported jobs. | CI scheduling can also be static per runner and keep runtime capability probing inside the test binary. |

## Edges I do not recommend adding

- Do not make every H benchmark depend on `ID_350`. Core benchmark execution should remain usable without Python/IPC/HTTP/gRPC completion.
- Do not make generic H GPU tests depend on `ID_250`; E is specifically SM89/Ada tuning. Keep generic A/B GPU correctness separate and make E evidence conditional in hardware-specific readiness.
- Do not make H tree-quality metrics depend directly on C adaptive policy controllers ID_146/ID_147. The metrics are observations; the policy consumes them, not the reverse.
- Do not make ID_359 depend on server counters as the normative source. Existing `draft_n`/`draft_n_accepted` are useful reuse points, but Tree-Draft should derive semantics from C/D proposal and verification records and then map them into server metrics.
- Do not add H dependencies to F parser/quantization internals individually. ID_299 is the correct activation boundary for runtime integration; ID_300 is the correct loader conformance boundary for release readiness.

## Current llama.cpp seams that support the audit

1. `tools/server/server-context.cpp` measures prompt/generation time and publishes `draft_n` plus `draft_n_accepted`, but it does not expose H's tree depth, verified-node, or phase-separated accounting. H needs C/D records for those metrics.
2. The server samples ordinary tokens only after target decode and treats the sampling call as a synchronization point for timing. This confirms H phase timing must account for asynchronous GPU completion rather than only host enqueue time.
3. In speculative server execution, `common_sampler_sample_and_accept_n()` returns accepted draft tokens plus the replacement/bonus token. The server then computes rollback depth, may restore a target/draft checkpoint, trims sequence memory, and only afterward emits committed tokens. This directly separates D acceptance, B rollback/commit, and G output semantics.
4. `common_speculative_draft()` records draft-generation timing and `common_speculative_accept()` records accept timing/statistics per implementation. These are useful implementation hooks for C/H telemetry, but the current counters are linear-draft oriented and do not define Tree-Draft topology metrics.
5. `examples/speculative/speculative.cpp` implements stochastic `p_target/p_draft` acceptance and residual sampling explicitly, then prunes/copies/removes sequence memory. The Tree-Draft H stochastic tests should therefore consume D's formal acceptance/residual contracts and B's cache contracts rather than infer correctness from final text alone.
6. Existing server speculative rollback has capability-dependent checkpoint/sequence-removal paths. That reinforces `ID_374 -> ID_067` and `ID_375 -> ID_070`: H must test the abstract rollback/commit contracts across supported memory modes, not one concrete removal mechanism.

## Recommended conservative manifest change set

If the prime wants one conservative correction pass, apply these edges/removals first:

1. `ID_352 -> ID_149`
2. `ID_352 -> ID_164`
3. `ID_352 -> ID_318`
4. `ID_355 -> ID_318`
5. `ID_356 -> ID_318`
6. `ID_357 -> ID_318`
7. `ID_358 -> ID_149`
8. `ID_358 -> ID_164`
9. `ID_359 -> ID_150`
10. `ID_359 -> ID_191`
11. `ID_360 -> ID_150`
12. `ID_360 -> ID_191`
13. `ID_361 -> ID_164`
14. `ID_361 -> ID_191`
15. `ID_364 -> ID_077`
16. `ID_365 -> ID_238`
17. `ID_366 -> ID_044`
18. `ID_366 -> ID_111`
19. `ID_366 -> ID_174`
20. `ID_367 -> ID_172`
21. `ID_367 -> ID_318`
22. remove `ID_367 -> ID_366`
23. `ID_368 -> ID_197`
24. `ID_368 -> ID_035`
25. `ID_369 -> ID_115`
26. `ID_369 -> ID_184`
27. `ID_369 -> ID_198`
28. `ID_370 -> ID_318`
29. `ID_371 -> ID_173`
30. `ID_371 -> ID_175`
31. `ID_371 -> ID_180`
32. remove `ID_371 -> ID_366`
33. `ID_372 -> ID_188`
34. `ID_373 -> ID_181`
35. `ID_373 -> ID_182`
36. `ID_374 -> ID_067`
37. `ID_374 -> ID_190`
38. `ID_375 -> ID_070`
39. `ID_375 -> ID_189`
40. `ID_376 -> ID_191`
41. `ID_379 -> ID_050`
42. `ID_379 -> ID_100`
43. `ID_379 -> ID_150`
44. `ID_379 -> ID_200`
45. `ID_379 -> ID_299`
46. `ID_380 -> ID_093`
47. `ID_380 -> ID_317`
48. `ID_380 -> ID_318`
49. `ID_381 -> ID_089`
50. `ID_382 -> ID_109`
51. `ID_382 -> ID_190`
52. `ID_382 -> ID_318`
53. `ID_383 -> ID_005`
54. `ID_383 -> ID_150`
55. `ID_383 -> ID_160`
56. remove `ID_383 -> ID_378`
57. `ID_384 -> ID_070`
58. `ID_384 -> ID_191`
59. `ID_386 -> ID_094`
60. `ID_386 -> ID_319`
61. `ID_387 -> ID_050`
62. `ID_387 -> ID_100`
63. `ID_387 -> ID_222`
64. `ID_388 -> ID_319`
65. `ID_395 -> ID_392`
66. `ID_392 -> ID_386`
67. `ID_399 -> ID_386`
68. `ID_400 -> ID_050`
69. `ID_400 -> ID_100`
70. `ID_400 -> ID_150`
71. `ID_400 -> ID_200`
72. `ID_400 -> ID_250`
73. `ID_400 -> ID_300`
74. `ID_400 -> ID_350`

Then add the six ID_388 intra-H metric-schema edges listed above if the artifact schema is meant to cover every H benchmark metric from its first version.

All proposed cross-pillar edges point from H to lower-numbered A-G modules, so they cannot create a cycle by themselves. The proposed intra-H additions also point backward. The three removals eliminate artificial ordering only; they do not remove a runtime producer contract because replacement dependencies are supplied explicitly.
