# Dependency Audit - Pilar B (ID_051..ID_100)

Scope: audit Pilar B against the available Pilar C/E/F/G/H manifests and the current llama.cpp KV/cache interfaces. This file proposes dependency changes only. No manifest was edited.

## Executive finding

The current global DAG materially under-represents Pilar B integration dependencies. None of the available C/E/F/G/H manifests currently declares a dependency on ID_051..ID_100, even though several modules explicitly consume paged-KV metadata, KV rollback/commit behavior, context shifting, or concurrency rules defined by B.

The strongest corrections are additive edges from later pillars into B. I do not recommend mechanically removing B edges merely because they are transitively reachable. Several such edges are direct semantic dependencies and accurately document which contract a module consumes.

## High-confidence missing edges

Use notation `consumer -> dependency`.

| Suggested edge | Reason |
| --- | --- |
| `ID_107 -> ID_056` | C's packed frontier descriptor explicitly carries a `kv_ref`. The page-table/address-resolution contract in ID_056 is the concrete definition of what that reference can identify and how stale physical mappings are rejected. Without this edge, C can be implemented against an incompatible KV reference model. |
| `ID_213 -> ID_056` | E's module is specifically "Lay out paged KV cache metadata". It cannot honestly choose a hardware-local layout until B defines the semantic page-table fields and generation-aware mapping. ID_056 already brings page descriptors and page geometry transitively. |
| `ID_215 -> ID_081` | E's verifier gather ordering consumes irregular tree-KV reads. ID_081 defines the ordered visible-page segment set for a query/tree path. The locality optimization must preserve that semantic gather set rather than invent a second representation. |
| `ID_221 -> ID_093` | E's CUDA stream roles for speculative decode must obey B's RESERVE -> WRITE -> PUBLISH -> ATTEND -> VERIFY -> COMMIT/PRUNE phase ordering and reader-quiescence rules. Otherwise stream overlap can violate cache publication/reclamation safety. |
| `ID_295 -> ID_052` | F validates attention/KV geometry required by tree kernels. ID_052 owns the fixed-page geometry, per-layer K/V layout, and alignment requirements being validated. F should validate B's contract rather than independently redefine it. |
| `ID_364 -> ID_077` | H measures KV-cache memory attributable to Tree-Draft state. ID_077 defines per-branch shared-versus-exclusive residency accounting without double-counting shared physical pages. H needs that accounting contract to report attributable KV memory honestly. |
| `ID_374 -> ID_067` | H's KV rollback test directly tests the rollback operation defined by ID_067. Acceptance semantics alone (ID_372) is insufficient to define cache truncation, refcount release, and reader-safe reclamation. |
| `ID_375 -> ID_070` | H explicitly tests full and partial tree acceptance. ID_070 defines the partial-page acceptance rules and already depends on the canonical commit path ID_069, so this one edge covers both commit and partial-page behavior. |
| `ID_381 -> ID_089` | H's long-context/context-boundary tests exercise shift or eviction behavior. ID_089 owns the safe interaction between context shifting and live speculative descendants/shared pages. |

## High-confidence missing edges inside Pilar B

| Suggested edge | Reason |
| --- | --- |
| `ID_078 -> ID_060` | The checkpoint format must preserve or deterministically reconstruct sharing ownership. Page tables alone do not define correct live-page reference counts and transient ownership semantics. ID_060 is the ownership/refcount contract required for a restorable shared tree. |
| `ID_089 -> ID_085` | Context shifting cannot be specified solely as token-KV position mutation on current llama.cpp. Hybrid/recurrent memory has distinct state and capability constraints. ID_085 explicitly separates ordinary attention KV from recurrent state, so ID_089 should consume that distinction before declaring descendants shiftable. |
| `ID_099 -> ID_085` | The end-to-end scheduler/KV/attention integration contract currently omits the hybrid/recurrent boundary even though current llama.cpp supports memory modes that are not ordinary token KV. ID_085 is needed to make fallback/unsupported behavior explicit at the integration boundary. |

## Medium-confidence edges to review before changing manifests

These are real coupling points, but whether they deserve a direct edge depends on which module owns the final interface.

| Candidate edge | Why it may be needed | Why it may remain transitive/implicit |
| --- | --- | --- |
| `ID_239 -> ID_094` | CUDA Graph capture eligibility depends on metadata pointers/snapshots remaining valid for asynchronous execution. ID_094 defines versioned page-table snapshots. | If E's graph layer only captures compute after ID_221 has established safe stream phases and pointer lifetime is fully owned by a separate graph parameter module, a direct edge may be unnecessary. |
| `ID_295 -> ID_087` | F's KV-geometry validation may need to reject quantized K/V page layouts that violate B's quant-block constraints. | If ID_295 validates only architectural head/embedding geometry and quantized-KV compatibility is validated later at dispatch, ID_052 alone is the honest dependency. |
| `ID_380 -> ID_093` | Concurrent-request integration should exercise B's cache mutation/read phase rules. | If ID_380 treats each request as an opaque engine call and concurrency safety is covered by a lower-level dedicated test, the edge can be inherited elsewhere. |
| `ID_386 -> ID_094` | H's TSan scope explicitly includes shared cache metadata; snapshot/version publication is the core shared-metadata contract. | If `ID_380 -> ID_093` is added and the implementation places all snapshot access behind scheduler serialization, ID_386 may not need a direct dependency. |

## Edges I do not recommend adding

- No Pilar G module currently needs a direct B dependency. The available G modules describe bindings, streaming, IPC, and protocol/session surfaces. They can treat engine/KV state as opaque unless a public API begins exposing branch/page handles.
- Do not add `ID_130 -> ID_077` merely because C's cost-adjusted utility has a memory-cost term. ID_130 explicitly consumes scheduler-provided normalized cost estimates; binding it directly to KV residency accounting would couple policy to one estimator.
- Do not add `ID_140 -> ID_096` solely because both modules use transaction language. C's transaction boundary protects draft-tree arena append, while B's ID_096 protects multi-branch KV reservation/publication. They become coordinated at ID_099, but they are independently implementable contracts.
- Do not add generic B dependencies to E's device allocator modules (`ID_231`, `ID_234`, `ID_238`). Those modules manage temporary device allocations, while B's ID_057 manages logical fixed KV pages. Conflating them would make the DAG imply one allocator implementation owns both concerns.

## Redundant-edge review inside Pilar B

A transitive-reduction scan flags many B edges as graph-theoretically redundant, including examples such as `ID_054 -> ID_051`, `ID_059 -> ID_053`, `ID_062 -> ID_057`, `ID_083 -> ID_055`, `ID_087 -> ID_052`, `ID_093 -> ID_059`, and `ID_095 -> ID_053`. I do not recommend removing these just because another dependency also reaches the same ancestor.

Those edges document direct interface consumption:

- ID_054 directly consumes the logical branch/token identity contract from ID_051 and page descriptors from ID_053.
- ID_059 directly consumes page lifecycle state from ID_053 while publishing a reservation produced through ID_058.
- ID_062 directly calls for allocation and refcount/COW behavior, so ID_057 and ID_060 remain real direct dependencies even though ID_061 reaches them.
- ID_083 directly consumes ancestry semantics from ID_055 even though ID_081 also depends on ancestry.
- ID_087 directly validates page geometry from ID_052 even though GPU block-table packing reaches that geometry transitively.
- ID_093 directly coordinates publication (ID_059), reader lifetime (ID_066), and commit/prune (ID_069); these are independent phase contracts.
- ID_095 directly validates descriptor generations from ID_053 in addition to snapshot/quiescence behavior from ID_094.

Therefore the audit finds no high-confidence artificial B edge to remove. A future policy that insists on strict transitive reduction would produce a smaller graph, but it would no longer show which interfaces are directly consumed by each module.

## Current llama.cpp API constraints that justify the audit

The present implementation has concrete constraints that the DAG should reflect during implementation planning:

1. `llama_kv_cache::seq_cp()` currently asserts that sequence copy is supported only for full KV buffers. ID_061's "seq_cp-style" fork sharing cannot be treated as a universal implementation primitive for sliding-window or hybrid memory. The B design should keep page sharing as its own contract and use existing sequence-copy support only where capability checks allow it.
2. `llama_kv_cache::seq_add()` and `seq_div()` assert `n_pos_per_embd() == 1`. Context-position transforms are therefore capability-limited, which is why ID_089 needs the memory-mode distinction from ID_085 rather than assuming every cache can shift identically.
3. `llama_memory_seq_rm/cp/add/div` are memory-interface operations, while server code first probes `common_context_can_seq_rm()` and has multiple rollback paths depending on memory capability. B rollback/commit/context-shift modules must integrate through capability-aware memory APIs rather than assuming the ordinary full KV cache path.
4. State save/restore exists through `llama_state_seq_get_*`, `llama_state_seq_set_*`, and KV `state_write/state_read`; fragmented restore is already a supported concern. Tree checkpoints add branch/page-sharing metadata on top, so ID_078/ID_079 must preserve ownership equivalence independently of original physical cell/page ids.
5. `llama_kv_cells` already tracks used cells, per-cell sequence membership, sequence position ranges, and cell moves used during defrag. B's page descriptors/page tables are therefore an additional tree-paging abstraction that must map deliberately onto these facilities rather than silently duplicating incompatible ownership state.
6. Server speculative rollback uses sequence removal/checkpoint restoration and computes rollback depth after verification. H's rollback/commit tests therefore need explicit dependencies on B's rollback/commit contracts, otherwise the DAG allows the tests to be implemented before the behavior they claim to validate is defined.

## Recommended manifest change set

If the prime wants a conservative first correction pass, add only these twelve edges:

1. `ID_107 -> ID_056`
2. `ID_213 -> ID_056`
3. `ID_215 -> ID_081`
4. `ID_221 -> ID_093`
5. `ID_295 -> ID_052`
6. `ID_364 -> ID_077`
7. `ID_374 -> ID_067`
8. `ID_375 -> ID_070`
9. `ID_381 -> ID_089`
10. `ID_078 -> ID_060`
11. `ID_089 -> ID_085`
12. `ID_099 -> ID_085`

After applying them, rerun the global DAG validator. The proposed intra-B edges point to lower-numbered modules, and the proposed C/E/F/H edges point back into B; none creates a cycle under the current published topology. The medium-confidence candidates should be decided after the owners of E/F/H confirm which layer owns snapshot lifetime, quantized-KV validation, and concurrency-test contracts.
