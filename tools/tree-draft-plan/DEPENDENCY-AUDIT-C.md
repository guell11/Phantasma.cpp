# Dependency Audit - Pilar C (ID_101..ID_150)

Scope: semantic audit of Pilar C against Pilar A topology/sampling/RNG, Pilar B KV contracts, Pilar D target-verifier handoff, Pilar F draft/target model compatibility, and the current `common/speculative` runtime seams. This document proposes dependency edges only. No manifest was edited.

## Executive finding

Pilar C currently behaves too much like a self-contained speculative runtime. In reality, several of its contracts are consumers or producers of interfaces already owned by other pillars:

- A owns packed tree topology plus exact top-k/sampling/RNG semantics.
- B owns the meaning and lifetime of tree KV references.
- D consumes the sealed C tree and must validate the exact proposal distribution that C materializes.
- F owns model-handle validity and draft/target compatibility before C may execute a draft model.

The current C manifest has no cross-pillar dependencies. That makes the global DAG dishonest around the most important seams.

## High-confidence missing edges

Notation is `consumer -> dependency`.

| Suggested edge | Rationale |
| --- | --- |
| `ID_102 -> ID_001` | C's compact node record contains parent/depth/path topology that must be representable in A's packed tree topology ABI. Without this edge, C can define parent/depth conventions that the tree-attention/verifier packing path cannot consume. |
| `ID_103 -> ID_004` | C's contiguous arena and packed frontier use stable packed node indices. A ID_004 owns packed-forest offsets and local/global index conversion across ragged trees. C's arena layout must preserve that indexing contract when more than one request/tree is packed. |
| `ID_107 -> ID_056` | C's frontier descriptor explicitly contains `kv_ref`. B ID_056 defines generation-aware logical-position to physical-page resolution; this is the concrete meaning the reference must carry or resolve through. This edge was already identified in the Pilar B audit and remains high confidence from C's side. |
| `ID_111 -> ID_044` | C defines counter-based per-node RNG semantics, while A ID_044 already owns the scheduling-independent RNG ABI used by GPU sampling. C should derive its `(seed,path,draw,depth)` addressing on that ABI instead of inventing a second RNG family. |
| `ID_113 -> ID_038` | C's deterministic top-k filtering must match A's exact top-k reference, including ties and non-finite handling. The policy layer may choose `k`, but it should not redefine what top-k means. |
| `ID_143 -> ID_045` | C's fused temperature/top-k/top-p/RNG CUDA pipeline is an orchestration/fusion boundary over the exact sampling semantics A ID_045 defines. This prevents C's fused path from drifting from the kernel-level sampler contract. |
| `ID_106 -> ID_299` | The draft-model execution adapter must only activate a model after F's complete preflight, which includes loader validity, kernel/placement feasibility, draft-target compatibility, tree assets, KV geometry, and positional metadata. Otherwise C can legally schedule a draft model F has declared incompatible or non-executable. |
| `ID_150 -> ID_292` | The sealed handoff contains draft token IDs/probabilities intended for a target verifier. F ID_292 defines whether draft and target vocab/logit semantics are compatible at all. C cannot promise a verifier-consumable handoff without this compatibility result. |
| `ID_151 -> ID_150` | D's target-verification input contract is the direct consumer of C's immutable sealed draft-tree view. The edge should point from D into C so the verifier schema is derived from the actual producer contract instead of duplicating a parallel tree schema. |
| `ID_167 -> ID_138` | D validates that stored `log p_D(v)` equals the exact proposal probability that generated the token. C ID_138 is the point where sampled local probability/log-probability is materialized into the tree. D's validation contract therefore depends directly on that producer contract. |
| `ID_177 -> ID_116` | D's stochastic sibling semantics require knowledge of the exact ordered multi-child proposal process. C ID_116 defines sampling without replacement and draw order for sibling creation. Treating sibling proposals as independent would break exact stochastic correction. |
| `ID_174 -> ID_044` | D's verification RNG is another scheduling-independent counter-based stream. It should share A's RNG ABI and use a distinct domain/counter namespace rather than define an unrelated PRNG contract. This is relevant to C because C records RNG/config provenance in ID_150 for verifier replay. |

## High-confidence missing edges inside Pilar C

| Suggested edge | Rationale |
| --- | --- |
| `ID_150 -> ID_138` | The sealed verifier handoff includes `local_p`, but its current dependency chain reaches topology/frontier/telemetry without explicitly consuming the module that defines and stores exact local proposal probabilities. |
| `ID_150 -> ID_111` | ID_150 promises RNG/config provenance required for replay. ID_111 defines the per-node draft RNG addressing scheme, so the sealed provenance contract must name/encode that scheme. |
| `ID_140 -> ID_106` | One expansion step is defined mathematically as `Sample(Model(Batch(S_k)))`, but the dependency list only reaches batch formation and append/frontier logic. The actual draft-model adapter is a direct execution-stage dependency. |
| `ID_140 -> ID_115` | The same expansion-step transaction includes sampling, yet no sampling-selection module is a direct dependency. ID_115 is the canonical categorical selection operation; multi-child ID_116 may be used depending on branch width, but the transaction must at least consume the sampling result contract. |

## Edges that should not be added

- Do not add `ID_106 -> ID_292` in addition to `ID_106 -> ID_299`. F ID_299 already includes ID_292 plus the rest of the activation preflight. A second direct edge would add no semantic information.
- Do not add C policy modules such as ID_123/ID_130 directly to A's GPU top-k kernels. C decides width/budget/utility; A implements exact primitives. The semantic join is at ID_113/ID_143, not every policy stage.
- Do not add `ID_140 -> ID_096` merely because both modules use transactional language. C's transaction commits tree-arena expansion, while B ID_096 commits KV reservation/publication. They are coordinated by the later runtime integration contract, but one is not required to implement the other.
- Do not add direct C dependencies on B rollback/commit modules ID_067-ID_070. C constructs proposals; target verification plus the integration layer decides accepted state and KV rollback/commit.
- Do not add `ID_150 -> ID_151`. That direction would invert producer/consumer ownership and can create an artificial C<->D schema cycle. D ID_151 should consume C ID_150.

## Current `common/speculative` seams that support these edges

The existing runtime already separates several responsibilities that the Tree-Draft DAG should preserve:

1. `common_speculative_draft()` delegates generation to method-specific implementations, while `common_speculative_accept()` later informs the draft implementation how many tokens the target accepted. C is therefore naturally the proposal producer, not the owner of target verification/commit semantics.
2. Draft implementations call `llama_decode(ctx_dft, ...)`, sample through `common_sampler_sample(...)`, and advance sampler state through `common_sampler_accept(...)`. C's draft-model adapter and sampling contracts must preserve the same sampler semantics rather than invent a parallel probability pipeline.
3. `common_speculative_are_compatible()` already checks target/draft vocabulary and special-token compatibility before ordinary speculative decoding. F ID_292 formalizes this concern. C therefore needs a validated compatibility/preflight result before emitting verifier-consumable token IDs.
4. Existing Eagle/MTP-style draft implementations remove rejected or pending draft KV with `llama_memory_seq_rm(...)`. That confirms KV lifetime is a separate memory concern from proposal topology; C should carry B-defined KV references but should not own rollback/commit policy.
5. Existing sampling state is stateful: `common_sampler_sample`, `common_sampler_accept`, sampler cloning, grammar transforms, and seeded RNG can affect the exact proposal distribution. D ID_167's validation of draft probabilities therefore depends on C storing the probability actually used to produce a child, not a recomputed raw softmax.
6. The server and common speculative paths already distinguish target and draft contexts. F's draft/target compatibility and C's draft execution adapter should preserve this distinction; sharing a tree scheduler does not imply architecture dimensions or KV layouts are identical.

## A/C overlap that should be resolved by dependency, not duplicate semantics

Several C modules intentionally describe policy-level versions of primitives that A also specifies at kernel/reference level. They can coexist if the dependency direction is explicit:

- C ID_113 chooses/represents candidate top-k for policy; A ID_038 defines exact top-k semantics.
- C ID_111 assigns per-node RNG coordinates; A ID_044 defines the counter-based RNG ABI and scheduling-independence contract.
- C ID_143 decides when stages may be fused; A ID_045 defines exact fused top-k/top-p sampling semantics and validity behavior.
- C ID_116 defines multi-child proposal policy. A ID_046 defines GPU-facing branch expansion output, but I do not recommend a direct edge yet because C's multi-child semantics are not necessarily implemented by A ID_046; they can meet later through an execution adapter.

## D/C handoff ownership

The clean semantic boundary is:

`C ID_150 sealed tree -> D ID_151 verification input -> D probability/acceptance pipeline`

The minimum information crossing that boundary must include topology, token ids, depths/branch identities, exact local proposal probability or log-probability, stable proposal order for stochastic siblings, and RNG/config provenance sufficient to reproduce proposal semantics. D may repack or canonicalize this data, but it should not redefine the producer-side meaning.

This implies two verifier-side edges beyond the root handoff:

- `ID_167 -> ID_138` for exact proposal probability provenance.
- `ID_177 -> ID_116` for ordered sibling proposal semantics.

These edges are more honest than making all of D depend only on ID_151, because they document the specific producer contracts whose correctness D is checking.

## Recommended conservative change set

For a first manifest correction pass, add these sixteen high-confidence edges:

1. `ID_102 -> ID_001`
2. `ID_103 -> ID_004`
3. `ID_107 -> ID_056`
4. `ID_111 -> ID_044`
5. `ID_113 -> ID_038`
6. `ID_143 -> ID_045`
7. `ID_106 -> ID_299`
8. `ID_150 -> ID_292`
9. `ID_151 -> ID_150`
10. `ID_167 -> ID_138`
11. `ID_177 -> ID_116`
12. `ID_174 -> ID_044`
13. `ID_150 -> ID_138`
14. `ID_150 -> ID_111`
15. `ID_140 -> ID_106`
16. `ID_140 -> ID_115`

The change set intentionally avoids policy-to-kernel overcoupling and avoids making C responsible for B rollback/commit. After applying manifest changes, rerun the global DAG validator because cross-pillar edges to F (higher numeric IDs) can expose cycles if F later gains dependencies back into C. Under the currently published manifests, no such F->C edge exists.
