# Dependency Audit - Pilar D (ID_151..ID_200)

Scope: audit Pilar D against Pilar A tree attention/sampling, Pilar B tree KV, Pilar C sealed draft handoff, Pilar F draft/target model compatibility, Pilar H tests, and the current llama.cpp speculative/target execution path. This file proposes dependency changes only. No manifest was edited.

## Executive finding

The current global catalog contains no cross-pillar dependency edges at all. For Pilar D this is materially incorrect: target verification consumes topology/mask/sampling contracts from A, tree/KV lifetime rules from B, immutable proposal data from C, token/position compatibility from F, and then becomes the behavior exercised by H.

The conservative correction set below contains 44 direct semantic edges and remains acyclic when applied to the current 400-module catalog.

Use notation `consumer -> dependency`.

## High-confidence edges between D and Pilar A

| Suggested edge | Reason |
| --- | --- |
| `ID_151 -> ID_001` | The target verification input carries parent/depth/tree identity fields that must use the same packed topology ABI consumed by tree attention. Otherwise C-to-D conversion can create a second incompatible topology representation. |
| `ID_155 -> ID_006` | D computes `P + depth` target positions. A already owns the packed-tree position mapping, including heterogeneous prefixes, shared sibling positions, and overflow rules. |
| `ID_156 -> ID_002` | D's ancestor predicate is exactly the logical ancestry relation whose closure is defined by A. D should consume that definition instead of independently defining ancestry. |
| `ID_157 -> ID_010` | The dense D reference mask needs the same committed-prefix plus ancestor-only visibility rule as A's composite tree mask. This is the semantic source for sibling isolation. |
| `ID_159 -> ID_014` | D encodes ancestor visibility as packed bitsets. A already defines the portable bit numbering, word type, and layout consumed by kernels. |
| `ID_160 -> ID_005` | D preflight validates parent order, roots, depths, and tree consistency. Those topology failure rules are already defined by A's topology validator. |
| `ID_161 -> ID_011` | The packed one-pass target batch must be convertible to the ragged attention descriptor used by A kernels. Query/tree offsets and prefix lengths cannot be invented independently in D. |
| `ID_164 -> ID_032` | The production one-pass verifier needs the runtime attention dispatcher to select an exact tree-capable kernel or report a fallback condition. |
| `ID_165 -> ID_047` | D's drafted-token logit gather is the direct consumer of A's verification-logit gather primitive and its explicit row-map semantics. |
| `ID_169 -> ID_038` | Greedy verification uses target argmax with stable tie behavior. A's exact top-k reference with k=1 is the existing canonical tie/NaN ordering contract. |
| `ID_174 -> ID_044` | Verification uniforms must use the same schedule-independent counter RNG ABI as A sampling. Keeping ID_174 as an independent root would permit two incompatible RNG engines. |
| `ID_192 -> ID_033` | D's unsupported-tree-mask fallback must consume A's backend capability/fallback matrix rather than infer backend support locally. |

These edges intentionally avoid making D depend on A performance or benchmark modules. D needs A semantic and dispatch contracts, not A's entire acceptance suite.

## High-confidence edges between D and Pilar B

| Suggested edge | Reason |
| --- | --- |
| `ID_155 -> ID_088` | Target positions become stored KV/RoPE positions. B defines when logical tree positions are shareable and when position transforms invalidate shared KV. |
| `ID_090 -> ID_161` | B's packed Tree PagedAttention query descriptor contains the verifier query row and stable ordering back to sampler logits. D first defines the single-pass packed target rows, so B's query descriptor should consume D's row-layout contract. |
| `ID_164 -> ID_093` | The target decode executes inside B's RESERVE -> WRITE -> PUBLISH -> ATTEND -> VERIFY -> COMMIT/PRUNE phase model. D cannot safely execute asynchronous tree verification while ignoring reader/writer lifetime rules. |
| `ID_194 -> ID_096` | A failed or partially completed target pass may already have written speculative KV. D's restore-on-failure rule requires B's multi-branch failure atomicity and rollback record semantics. |
| `ID_199 -> ID_070` | D's correctness invariant states that committed KV equals ordinary decoding after zero/full/partial acceptance. B ID_070 owns partial acceptance inside shared/COW pages and already includes canonical commit semantics. |
| `ID_099 -> ID_191` | B's end-to-end scheduler/KV/attention integration contract explicitly says the verifier returns accepted path A before commit/prune. D ID_191 is the immutable verification result record that provides that interface. |

The direction on `ID_090 -> ID_161` and `ID_099 -> ID_191` matters. Making D depend on those higher-level B integration records would hide the fact that B is consuming D's query/result mapping, and can create artificial ownership of verifier state inside the KV manager.

## High-confidence edges between D and Pilar C

| Suggested edge | Reason |
| --- | --- |
| `ID_151 -> ID_150` | ID_151 is the bridge from the sealed draft tree into target verification. Its token, parent, depth, local probability, path identity, active/leaf metadata, prefix provenance, and lifetime all originate in the sealed C handoff. |
| `ID_167 -> ID_150` | Draft proposal log probabilities must be validated against the exact immutable proposal provenance handed off by C. D cannot reconstruct their meaning from token IDs alone. |
| `ID_174 -> ID_111` | D verification RNG needs C's stable path identity and per-node counter-addressing semantics so replay is independent of frontier compaction and scheduling. Together with ID_044 this defines one RNG family with domain-separated draws. |
| `ID_177 -> ID_116` | Stochastic sibling verification depends on the exact ordered multi-child proposal process. C ID_116 defines sampling without replacement, draw ordinals, and returned child probabilities. |
| `ID_183 -> ID_113` | Exact truncated proposal support depends on C's deterministic top-k semantics and tie-breaking. |
| `ID_183 -> ID_114` | Exact truncated proposal support and renormalization depend on C's top-p cutoff semantics. |
| `ID_183 -> ID_150` | Even with filter semantics defined, D must receive the actual support/probability data or replay provenance from the sealed proposal that generated the node. |
| `ID_184 -> ID_112` | D temperature-mode dispatch and stochastic ratios must use the same post-temperature proposal distribution that C used to sample the draft token. |

Several of these C edges are transitively reachable once `ID_151 -> ID_150` is added. They are still direct semantic dependencies: D_174 directly consumes C RNG/path semantics, D_177 directly consumes sibling proposal semantics, and D_183/D_184 directly consume the proposal transforms whose probabilities enter acceptance math.

## High-confidence edges between D and Pilar F

| Suggested edge | Reason |
| --- | --- |
| `ID_153 -> ID_293` | "Pack target token ids" is only correct when draft token IDs are already target IDs or have been mapped through the exact proven draft-to-target bijection. ID_293 depends on the draft/target compatibility check in ID_292, so this edge also gates incompatible vocabularies. |
| `ID_155 -> ID_296` | Target absolute positions must respect the target model's normalized positional encoding and context metadata. F owns RoPE/scaling/max-context normalization used to determine whether a computed position is legal. |

I do not recommend a blanket `ID_151 -> ID_299` or `ID_164 -> ID_299` edge yet. The current host/server already owns model activation. If ID_299 becomes the sole runtime capability object presented to the verifier, then `ID_192 -> ID_299` is a reasonable follow-up edge; otherwise it would over-couple the verification ABI to the loader implementation.

## High-confidence H test edges into D

H should consume D. D must not depend on H.

| Suggested edge | Reason |
| --- | --- |
| `ID_359 -> ID_191` | Acceptance-rate accounting consumes accepted/proposed counts from the verifier result boundary. |
| `ID_360 -> ID_191` | Accepted depth/branch structure is derived from accepted nodes and winner branch in the verifier result. |
| `ID_361 -> ID_164` | "Target invocations" are the one-pass target verification calls defined by D_164. |
| `ID_361 -> ID_191` | Committed/verified token counts per invocation require the corresponding verifier result. |
| `ID_366 -> ID_174` | Benchmark seed propagation must feed the deterministic verification RNG contract used by stochastic acceptance. |
| `ID_367 -> ID_191` | Greedy end-to-end equivalence must compare the actual committed verifier result, including replacement/bonus behavior. |
| `ID_368 -> ID_197` | H logits-equivalence testing directly exercises D's single-pass versus linear-replay differential checker. |
| `ID_369 -> ID_198` | Statistical sampled-output equivalence depends on a stochastic verifier already checked against the scalar acceptance/residual oracle. |
| `ID_371 -> ID_198` | The single-path acceptance test is the public test-layer consumer of D's scalar stochastic reference equivalence. |
| `ID_372 -> ID_199` | Overlapping-branch tests assert the same one-path/prefix/rejection invariants codified by D_199. |
| `ID_373 -> ID_198` | Residual replacement tests need D's scalar residual construction and sampling oracle. |
| `ID_374 -> ID_190` | KV rollback tests need the verifier's exact rejected-node discard set in addition to B's rollback contract. |
| `ID_375 -> ID_191` | Commit correctness tests need the exact accepted path and replacement/bonus result passed to the KV commit layer. |
| `ID_379 -> ID_200` | End-to-end representative tree-shape integration tests should start after the Pilar D conformance surface is defined. |
| `ID_383 -> ID_160` | Metadata fuzzing directly targets D's target-packing validator and its fail-closed parser/invariant behavior. |
| `ID_384 -> ID_199` | Property fuzzing of acceptance/rollback transitions should assert D's end-to-end verification invariants. |

These H edges complement the B audit edges such as `ID_374 -> ID_067` and `ID_375 -> ID_070`: H needs both the verifier decision and the KV operation that applies that decision.

## Contract gaps that dependency edges alone do not fix

### 1. ID_150 does not currently carry enough information for exact residual sampling

The sealed C handoff now promises exact node-local `local_p/local_logp`, topology, path identity, active/leaf metadata, base position, and sampler/RNG provenance. C_115/C_116 also define exact selected-token and ordered sibling conditional probabilities. D_173 can use the selected proposal probability `q(x)`, but D_180/D_183 still need the full transformed proposal distribution `q(k)`, or at least its exact finite support plus normalized masses, after top-k/top-p/temperature and any other proposal transforms.

For stochastic rejection to be implementable, ID_150 must expose one of:

- the exact filtered support and probabilities for every proposal context that may reject; or
- an immutable replay handle/state that can reproduce that proposal distribution bit-for-bit from the sealed path.

Without one of those, `ID_183 -> ID_150` is necessary but insufficient. Re-running a draft model from token IDs alone is not an equivalent contract when sampler state or transforms are stateful.

### 2. Vocabulary remapping must cover proposal support, not only the selected token

If F ID_293 supplies a non-identity draft-to-target vocabulary bijection, D_153 must map the selected proposal token into target ID space. Stochastic residual sampling also requires every token ID in draft support `q` to be remapped through the same bijection before subtracting it from target `p`.

Therefore the implementation of ID_183 must consume the same mapping already required by ID_153, even if no additional direct manifest edge is added because ID_183 reaches ID_153 through the D chain.

### 3. Stateful sampler transforms need an explicit per-path replay rule

D_185 names grammar, repetition, bias, min-p, and similar transforms. In current llama.cpp, ordinary sampling state lives in `common_sampler`, and the linear server verifier mutates/restores that state while walking draft tokens. A tree has multiple divergent sampler histories.

The sealed topology is enough to replay token history only if D_185 explicitly defines how target/draft sampler state is cloned or replayed from the committed prefix for each path. Otherwise a tree can have correct logits and still apply the wrong grammar/repetition state.

### 4. Target decode failure is not logically read-only

The current target execution path writes memory as decode progresses. Server speculative rollback already uses sequence removal or checkpoints after partial acceptance, and the public integration note documents that an aborted decode can leave already-processed ubatches in context memory.

That is why `ID_194 -> ID_096` is a correctness dependency, not just error-handling bookkeeping.

## Current llama.cpp constraints supporting these edges

The present source makes the ownership boundaries concrete:

1. `examples/speculative/speculative.cpp` already verifies multiple draft branches in one target `llama_decode()` by cloning sequence memory, sharing ancestor batch membership, and storing explicit target batch indices per branch. This is the current correctness oracle for D_161/D_162/D_197.
2. `llm_graph_input_attn_kv::set_input()` delegates cached-attention mask construction to the memory context. `llama_kv_cache::set_input_kq_mask()` currently uses sequence membership, causal positions, and SWA. Packed tree ancestry must therefore enter through the A/B graph-memory seam rather than a second verifier-only attention stack.
3. `llama_context::decode()` preserves user batch-to-output mapping through `output_ids`; output rows may be reordered across ubatches. D_162/D_165 must keep explicit node-to-batch mappings, which matches A_047.
4. `common_sampler_sample_and_accept_n()` is linear and mutates one sampler state while walking draft tokens until the first mismatch. Tree verification must preserve those sampler semantics per path instead of treating sibling rows as independent ordinary samples.
5. `tools/server/server-context.cpp` already restores checkpoints or removes speculative sequence state after partial acceptance. D's logical accepted/discard sets therefore have to feed B's commit/rollback semantics, not directly mutate a second KV cache.

## Edges I do not recommend adding

- Do not add a generic `ID_164 -> ID_100` or `ID_200 -> ID_100`. D needs specific B contracts for execution/commit correctness; waiting for the entire B acceptance suite would make validation artifacts into runtime prerequisites.
- Do not add `ID_200 -> ID_400` or any other D-to-H edge. H is downstream test/benchmark evidence and must consume D.
- Do not make D_161 own B page tables or physical addresses. The current integration map places physical KV resolution in the target memory context. B_090 can consume D's query-row layout while B keeps KV ownership.
- Do not make D_166 depend directly on A_042/A_045 until the owner of the production sampler-transform boundary is settled. Current `common_sampler` includes transforms that are broader than A's fused GPU subset. A direct dependency would be correct only if those A modules become the canonical sampler distribution contract.
- Do not model C adaptive-width/depth feedback as `ID_146/ID_147 -> ID_191` in the current DAG. C_150 depends transitively on C_146/C_147, while D_151 depends on C_150; adding the reverse edge would create a compile-time cycle for what is actually runtime feedback across speculative steps. The feedback ABI should be an earlier neutral contract, with D_191 producing values into it and C consuming them on the next step.

## Medium-confidence edges to review with the owning pillars

| Candidate edge | Why it may be needed | Why it may stay implicit |
| --- | --- | --- |
| `ID_166 -> ID_042` | Stable target probabilities should use the same temperature/bias/mask transform semantics as fused target sampling. | A_042 currently describes a fused optimization subset, while `common_sampler` is the broader semantic owner. |
| `ID_183 -> ID_045` | If target residual math applies top-k/top-p on the target side, A_045 defines the exact retained/renormalized target distribution. | The production target sampler may remain outside the A fused path and D_166 may already receive post-transform probabilities. |
| `ID_185 -> ID_150` | Per-path grammar/repetition replay consumes sealed path/provenance data directly. | D_185 already reaches C_150 through D_167 after the proposed edges; a direct edge is useful only if the handoff grows explicit sampler-state fields. |
| `ID_196 -> ID_092` | Production packed verification may use B Tree PagedAttention, whose dense-equivalence oracle is B_092. | D_196 can remain an A-mask checker while B_092 independently validates the paged path, with H integration joining both. |
| `ID_192 -> ID_299` | F preflight may be the runtime source of "tree verification unsupported" capability/fallback reasons. | If runtime support is owned by A_032/A_033 plus existing context capabilities, binding D fallback to the loader report is unnecessary. |

## Recommended conservative manifest change set

Add these 44 edges first:

1. `ID_151 -> ID_001`
2. `ID_155 -> ID_006`
3. `ID_156 -> ID_002`
4. `ID_157 -> ID_010`
5. `ID_159 -> ID_014`
6. `ID_160 -> ID_005`
7. `ID_161 -> ID_011`
8. `ID_164 -> ID_032`
9. `ID_165 -> ID_047`
10. `ID_169 -> ID_038`
11. `ID_174 -> ID_044`
12. `ID_192 -> ID_033`
13. `ID_155 -> ID_088`
14. `ID_090 -> ID_161`
15. `ID_164 -> ID_093`
16. `ID_194 -> ID_096`
17. `ID_199 -> ID_070`
18. `ID_099 -> ID_191`
19. `ID_151 -> ID_150`
20. `ID_167 -> ID_150`
21. `ID_174 -> ID_111`
22. `ID_177 -> ID_116`
23. `ID_183 -> ID_113`
24. `ID_183 -> ID_114`
25. `ID_183 -> ID_150`
26. `ID_184 -> ID_112`
27. `ID_153 -> ID_293`
28. `ID_155 -> ID_296`
29. `ID_359 -> ID_191`
30. `ID_360 -> ID_191`
31. `ID_361 -> ID_164`
32. `ID_361 -> ID_191`
33. `ID_366 -> ID_174`
34. `ID_367 -> ID_191`
35. `ID_368 -> ID_197`
36. `ID_369 -> ID_198`
37. `ID_371 -> ID_198`
38. `ID_372 -> ID_199`
39. `ID_373 -> ID_198`
40. `ID_374 -> ID_190`
41. `ID_375 -> ID_191`
42. `ID_379 -> ID_200`
43. `ID_383 -> ID_160`
44. `ID_384 -> ID_199`

A dry graph simulation over the current catalog with exactly these 44 additional edges remains acyclic.

The main implementation blocker after the DAG fix is not graph structure: stochastic correctness still requires C_150 to expose exact proposal support/probabilities or a replayable proposal-distribution handle. That contract should be fixed before treating D_173..D_185 as implementable.
