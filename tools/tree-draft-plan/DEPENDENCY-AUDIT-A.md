# Dependency Audit - Pilar A (ID_001..ID_050)

Scope: semantic audit of Pilar A against Pilar B/C/D/E/H and the current llama.cpp attention and sampling seams. This document proposes dependency changes only. No manifest was edited.

Notation is `consumer -> dependency`.

## Executive finding

The current 400-module catalog has zero cross-pillar dependency edges. That is not an honest implementation DAG. Pilar A owns shared topology, ancestry/mask, attention, exact top-k, RNG, fused sampling, and verifier-gather contracts that are already redefined or consumed by B, C, D, E, and H.

The current runtime also gives these contracts concrete attachment points:

- cached attention mask construction enters through `llama_kv_cache::set_input_kq_mask()` and is consumed by `ggml_flash_attn_ext(..., kq_mask, ...)`;
- CUDA dispatch already has `GGML_OP_FLASH_ATTN_EXT` and `GGML_OP_TOP_K`;
- backend sampling already builds graph operations through `llama_sampler_i::backend_apply`, including top-k, softmax, cumulative sum, candidate gather, and sampled-token output;
- `common/speculative.cpp` already attaches backend sampler chains to draft contexts.

Pilar A should therefore define shared semantics and backend primitives used by those existing paths. It should not become a second attention or sampling runtime.

## High-confidence cross-pillar edges

### Pilar B consumes Pilar A semantics

| Suggested edge | Reason |
| --- | --- |
| `ID_051 -> ID_001` | B's logical tree-KV identity contains node parent/position/branch identity. It must start from the same packed tree topology convention used by A, rather than define a parallel tree identity model. |
| `ID_055 -> ID_002` | B's interval/branch ancestry metadata is an acceleration structure for the ancestor relation defined by A. The optimized predicate must preserve A's inclusive ancestor closure. |
| `ID_083 -> ID_010` | B's paged-attention mask allows ancestry plus causal/window restrictions. A ID_010 owns the committed-prefix plus tree visibility rule that prevents sibling leakage. |
| `ID_086 -> ID_021` | B's paged MHA/GQA/MQA head indexing must use the same query-to-KV head mapping as the attention kernels. Two head-grouping formulas would make dense and paged paths disagree. |
| `ID_087 -> ID_023` | B's quantized page compatibility specializes the generic quantized-KV capability/fallback boundary from A. Unsupported page types must resolve to the same fallback decision. |
| `ID_088 -> ID_006` | B's RoPE/shared-page semantics depend on the logical tree position formula `prefix + depth` and on siblings sharing an absolute position. |
| `ID_091 -> ID_016` | B's CPU Tree PagedAttention oracle should implement the exact attention math already defined by A, changing only how visible K/V are addressed. |
| `ID_092 -> ID_003` | B explicitly compares paged gathering with a dense-tree mask. A ID_003 is the canonical dense visibility oracle for that comparison. |

The physical indirection seam still needs one ownership decision before adding another hard edge. A ID_009 exposes per-node `kv_slot` indirection, while B ID_081/ID_082 expose page-segment tables. Current `GGML_OP_FLASH_ATTN_EXT` consumes ordinary tensor operands plus an additive mask and has no page-table input. Before a production paged A kernel is claimed, one adapter must be designated: either B materializes A's `kv_slot` ABI, or A consumes B's page-table resolution contract. Do not add both directions and create duplicate ownership.

### Pilar C consumes Pilar A topology and sampling primitives

| Suggested edge | Reason |
| --- | --- |
| `ID_102 -> ID_001` | C's draft node record carries parent/depth topology that is eventually handed to A attention and D verification. It must be representable by the canonical packed topology ABI. |
| `ID_103 -> ID_004` | C's contiguous node arena and packed frontier use global packed indices across requests. A ID_004 owns ragged forest offsets and local/global index conversion. |
| `ID_111 -> ID_044` | C's per-node stateless RNG is a domain-specific addressing scheme over the shared schedule-independent counter RNG ABI. Keeping C ID_111 as a separate RNG family would make GPU/CPU replay fragile. |
| `ID_113 -> ID_038` | C chooses top-k candidates as policy input, but A ID_038 owns exact ranking, tie, and non-finite semantics. |
| `ID_143 -> ID_045` | C decides when the post-logit stages may be fused; A ID_045 defines the exact fused top-k/top-p/RNG sampling behavior and fallback validity. |

This agrees with the producer/primitive split already visible in the current runtime: `common/speculative` owns proposal lifecycle, while the sampler/backend graph owns exact sampling operations.

### Pilar D consumes Pilar A verification-facing contracts

| Suggested edge | Reason |
| --- | --- |
| `ID_151 -> ID_001` | D's verifier input repeats parent/depth/tree fields and must use the same topology ABI as the attention path. |
| `ID_155 -> ID_006` | D computes the same `prefix + depth` absolute position mapping as A. |
| `ID_156 -> ID_002` | D's ancestor predicate is the same logical ancestor closure used by tree attention. |
| `ID_157 -> ID_010` | D's dense verifier mask must have exactly the same prefix-plus-ancestor visibility semantics as production attention. |
| `ID_159 -> ID_014` | D's packed ancestor bitsets must use A's word width, bit numbering, alignment, and padding convention. |
| `ID_160 -> ID_005` | D preflight validates roots, parent order, depth consistency, and tree membership. Those failure rules are already A's topology validator contract. |
| `ID_161 -> ID_011` | D's one-pass target layout must be convertible to A's ragged attention descriptor without redefining q/tree/prefix range conventions. |
| `ID_164 -> ID_032` | Production single-pass target verification needs A's dispatcher to select an exact tree-capable attention path or return an explicit fallback reason. |
| `ID_165 -> ID_047` | D's drafted-token target-logit gather is the direct consumer of A's explicit row-map/token gather primitive. |
| `ID_169 -> ID_038` | Greedy verifier argmax is top-k with k=1 and must inherit the same stable tie and non-finite policy. |
| `ID_174 -> ID_044` | Verification uniforms must share A's schedule-independent counter RNG family, with a distinct verification domain/counter namespace. |
| `ID_192 -> ID_033` | D's unsupported-tree-mask fallback must consume the backend capability/fallback matrix rather than infer support independently. |

These are semantic dependencies, not a request to make D depend on all of Pilar A. D can still use the existing branch-per-sequence or materialized-mask path where A's optimized kernel is unavailable.

### Pilar E specializes Pilar A for SM89

| Suggested edge | Reason |
| --- | --- |
| `ID_214 -> ID_001` | E's SM89 tree-node metadata layout is a physical layout specialization of A's canonical node fields. It must not rename or reinterpret parent/depth/token identity. |
| `ID_216 -> ID_022` | E's tensor-core datatype policy must stay inside A's allowed input, accumulation, cast, and numerical-precision policy. |
| `ID_219 -> ID_029` | E's Ada FP8 eligibility is the SM89 specialization of A's guarded FP8 attention contract. This edge is only honest after the ID_029 dependency correction below removes its artificial Hopper-only prerequisite. |

`ID_243 -> ID_030` is a reasonable second-pass edge if E's measured SM89 autotuner becomes the implementation behind A's attention autotune key space. It is not required while E ID_243 remains a reusable hardware tuner for kernels beyond A.

### Pilar H validates Pilar A behavior

| Suggested edge | Reason |
| --- | --- |
| `ID_383 -> ID_005` | H fuzzes topology/verifier metadata and must exercise the canonical topology validator, not only a second parser in D. |
| `ID_387 -> ID_018` | GPU memory checking for tree-mask/attention workloads requires at least the baseline accelerator tree-attention implementation to exist. |
| `ID_387 -> ID_045` | The same GPU memory-check job explicitly covers sampling workloads, so the fused sampling path is a direct test dependency. |
| `ID_387 -> ID_047` | H also names verification kernels; A's verifier-logit gather is the concrete GPU primitive in that category. |
| `ID_400 -> ID_050` | Release-readiness cannot claim complete Tree-Draft evidence while the Pilar A integration acceptance contract is absent. Other pillar audits should add their own corresponding acceptance edges into H. |

I do not recommend a direct `ID_366 -> ID_044` edge if H ID_366 already consumes the domain-specific C/D RNG contracts. Seed propagation should feed C ID_111 and D ID_174, which then depend on A ID_044; this keeps benchmark seed derivation separate from RNG implementation details.

## High-confidence corrections inside Pilar A

The cross-pillar edges expose several internal A edges that are backwards or incomplete.

### Reverse the GQA/MQA dependency

Current:

`ID_021 -> ID_018`

Required:

`ID_021 -> ID_007`

`ID_018 -> ID_021`

ID_018 is the baseline attention implementation and explicitly needs head-mapping hooks. The head map is a semantic input to the kernel, so it must be defined before the kernel. ID_021 only needs the Q/K/V head layout from ID_007; it does not need an attention kernel in order to define `h_q -> h_kv`.

### Remove the Hopper-only prerequisite from generic FP8 semantics

Current:

`ID_029 -> ID_027`

Required:

`ID_029 -> ID_026`

Keep the existing `ID_029 -> ID_022`.

ID_027 is explicitly a Hopper warp-specialized schedule, while ID_029 describes a guarded FP8 attention path based on hardware/dtype/policy capability. Leaving the Hopper edge makes the generic FP8 contract unusable as the parent of E's SM89/Ada FP8 eligibility. ID_026 is the architecture-neutral performance-oriented attention schedule that can host capability-specific variants.

### Make the dispatcher consume the quantized-KV boundary

Add:

`ID_032 -> ID_023`

The dispatcher claims to select on dtype/layout/capability and return fallback when a variant is unsupported. ID_023 is the module that defines exactly that support/fallback boundary for quantized K/V. Without the edge, ID_023 is disconnected from actual dispatch.

### Make fused sampling consume fused logit-transform semantics

Add:

`ID_045 -> ID_042`

ID_042 defines which temperature/scale/bias/mask transforms may legally fuse before top-k while preserving sampler-chain semantics. ID_045 cannot claim exact joint top-k/top-p sampling with transform support while bypassing that contract.

### Include branch expansion in Pilar A completion

Add:

`ID_050 -> ID_046`

ID_046 is part of Pilar A's declared GPU sampling/tree-expansion surface but is currently a terminal module outside the Pilar A acceptance path. The final A handoff should not be considered complete without either validating this interface or explicitly removing it from Pilar A scope.

## Current runtime seams that justify the direction

### Attention

`llm_graph_context::build_attn_mha()` passes the existing `kq_mask` directly into `ggml_flash_attn_ext()`. Cached mask population is delegated to the memory context, and `llama_kv_cache::set_input_kq_mask()` currently evaluates sequence membership, causal position, and sliding-window rules.

That makes A ID_002/ID_010 the semantic extension of an existing mask seam. D should feed the same semantics into target verification, and B should preserve them when replacing dense logical KV traversal with paged addressing. A second verifier-only attention stack would be the wrong dependency direction.

`ggml_cuda_flash_attn_ext()` currently dispatches among vector, tile, and MMA implementations and only sees ordinary Q/K/V tensors plus the mask. This is why page-table ownership between A ID_009 and B ID_081/ID_082 must be made explicit before claiming a paged custom kernel.

### Top-k and sampling

CUDA already implements `GGML_OP_TOP_K`, and backend samplers already compose `ggml_top_k`, `ggml_get_rows`, `ggml_soft_max`, `ggml_cumsum`, and sampled-token outputs. A sampling work should plug into this backend sampler seam.

There is one important semantic mismatch: the current CUDA CUB top-k path requests `determinism::not_guaranteed` and `output_ordering::unsorted`. A ID_038 and C ID_113 require stable deterministic ranking/tie behavior. Therefore `ID_113 -> ID_038` is necessary, but implementation cannot simply alias that contract to today's CUB result. It needs deterministic post-ordering/tie handling or a different exact path where required.

The current distribution sampler uses `std::mt19937` on the host, and the backend graph receives a host-generated uniform input. A ID_044 is therefore new shared infrastructure, not a label for an already counter-based llama.cpp RNG. C ID_111 and D ID_174 should consume it only after one counter RNG ABI is implemented.

### Triton

There is no materialized Triton runtime/backend integration in the current ggml/llama source tree for these operations. A ID_018 and later Triton modules therefore represent new backend integration work. Their dependency chain should still terminate at the existing ggml operation/dispatcher seams instead of creating a second model execution loop.

## Edges I do not recommend adding

- Do not add blanket `ID_164 -> ID_050` or `ID_200 -> ID_050`. D needs specific A semantic/dispatch contracts and can validate its fallback path before every A optimization is complete.
- Do not make D probability construction depend directly on A ID_042/ID_045 yet. The current `common_sampler` transform surface includes grammar, repetition, min-p, bias, and other stateful transforms outside A's fused subset.
- Do not make every C policy module depend on A CUDA kernels. C chooses width, pruning, budgets, and scheduling; A implements exact primitives. The direct joins are C ID_113, ID_143, and the shared RNG contract.
- Do not connect A ID_009 and B ID_081/ID_082 in both directions. The current contracts describe different physical metadata. Pick one adapter owner after the paged-attention execution interface is fixed.
- Do not make H end-to-end correctness modules wait on A performance gates. Existing fallback/oracle execution is valuable before optimized attention and sampling pass performance acceptance.

## Conservative manifest change set

The first correction pass for this audit is 39 additions and 2 removals.

Remove:

1. `ID_021 -> ID_018`
2. `ID_029 -> ID_027`

Add:

1. `ID_021 -> ID_007`
2. `ID_018 -> ID_021`
3. `ID_029 -> ID_026`
4. `ID_032 -> ID_023`
5. `ID_045 -> ID_042`
6. `ID_050 -> ID_046`
7. `ID_051 -> ID_001`
8. `ID_055 -> ID_002`
9. `ID_083 -> ID_010`
10. `ID_086 -> ID_021`
11. `ID_087 -> ID_023`
12. `ID_088 -> ID_006`
13. `ID_091 -> ID_016`
14. `ID_092 -> ID_003`
15. `ID_102 -> ID_001`
16. `ID_103 -> ID_004`
17. `ID_111 -> ID_044`
18. `ID_113 -> ID_038`
19. `ID_143 -> ID_045`
20. `ID_151 -> ID_001`
21. `ID_155 -> ID_006`
22. `ID_156 -> ID_002`
23. `ID_157 -> ID_010`
24. `ID_159 -> ID_014`
25. `ID_160 -> ID_005`
26. `ID_161 -> ID_011`
27. `ID_164 -> ID_032`
28. `ID_165 -> ID_047`
29. `ID_169 -> ID_038`
30. `ID_174 -> ID_044`
31. `ID_192 -> ID_033`
32. `ID_214 -> ID_001`
33. `ID_216 -> ID_022`
34. `ID_219 -> ID_029`
35. `ID_383 -> ID_005`
36. `ID_387 -> ID_018`
37. `ID_387 -> ID_045`
38. `ID_387 -> ID_047`
39. `ID_400 -> ID_050`

A dry graph simulation with this set remains acyclic. The global roots become `ID_001, ID_038, ID_044, ID_101, ID_201, ID_251, ID_301, ID_351`; in particular B ID_051, D ID_151, and D ID_174 stop appearing as artificial independent roots.

The remaining important unresolved seam is A/B physical KV indirection. It should be settled when the packed target-memory adapter is specified, because the current ggml flash-attention API does not yet consume B's page tables directly. That interface choice should happen before adding a hard A<->B physical-layout edge.
