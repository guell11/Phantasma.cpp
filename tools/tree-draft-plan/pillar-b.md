# Pilar B - KV-Cache Management & Tree PagedAttention

Scope: IDs ID_051 through ID_100. This pillar specifies tree/graph KV metadata, fixed-size paging, allocation, sharing, copy-on-write, rollback/pruning, fragmentation control, Tree PagedAttention metadata, concurrency, and validation. It is specification-only: each JSONL entry is independently implementable once its listed dependencies are satisfied.

The design maps onto existing llama.cpp memory concepts where possible: KV cache/cells, sequence copy/remove operations, state save/restore, context shift, and scheduler-owned decode phases. Physical pages are fixed-size, ancestry sharing is reference-counted, divergent writes use copy-on-write, and asynchronous readers are protected by generations plus pins/epochs.

## Module index

| ID | Objective | Dependencies |
| --- | --- | --- |
| ID_051 | Define the logical tree-KV address space and immutable identity model for tokens, branches, and physical pages. | - |
| ID_052 | Specify fixed-size KV page geometry for tree decoding across layers and K/V tensor layouts. | ID_051 |
| ID_053 | Define page descriptor metadata and lifecycle states. | ID_052 |
| ID_054 | Define branch descriptors linking tree topology to ordered page spans. | ID_051, ID_053 |
| ID_055 | Specify ancestry intervals for constant-time visibility checks in tree attention metadata. | ID_054 |
| ID_056 | Define the page table mapping branch-local logical positions to physical KV pages. | ID_053, ID_054 |
| ID_057 | Define free-page pool semantics and allocation ordering. | ID_053 |
| ID_058 | Specify batched page reservation for a draft tree expansion step. | ID_054, ID_057 |
| ID_059 | Define page publication from RESERVED to LIVE after KV writes complete. | ID_053, ID_058 |
| ID_060 | Define reference counting for ancestry-shared pages. | ID_056, ID_059 |
| ID_061 | Specify fork-time page sharing without KV copying. | ID_054, ID_060 |
| ID_062 | Define copy-on-write for divergent writes into a shared tail page. | ID_057, ID_060, ID_061 |
| ID_063 | Specify append allocation for branches that cross page boundaries. | ID_056, ID_058, ID_062 |
| ID_064 | Define multi-branch append planning for one packed draft microbatch. | ID_058, ID_063 |
| ID_065 | Define physical KV write indexing from packed draft tokens. | ID_064 |
| ID_066 | Specify page pinning during asynchronous attention reads. | ID_060 |
| ID_067 | Define branch rollback to an ancestor position after speculative rejection. | ID_056, ID_060, ID_066 |
| ID_068 | Define pruning of rejected sibling branches. | ID_060, ID_067 |
| ID_069 | Define commit of one accepted speculative branch into the canonical sequence. | ID_061, ID_068 |
| ID_070 | Define partial acceptance when a verified path ends inside a shared/COW page. | ID_062, ID_067, ID_069 |
| ID_071 | Specify branch-id and generation reuse without stale-handle aliasing. | ID_054 |
| ID_072 | Define allocator fragmentation metrics for paged tree KV. | ID_053, ID_057 |
| ID_073 | Define tail-page compaction opportunities across branches. | ID_060, ID_072 |
| ID_074 | Define a relocation plan for page compaction/defragmentation. | ID_056, ID_066, ID_073 |
| ID_075 | Define safe publication of a completed page relocation. | ID_066, ID_074 |
| ID_076 | Specify memory-pressure policy for tree KV page reservations. | ID_058, ID_068, ID_072 |
| ID_077 | Define per-branch KV residency accounting and quota signals. | ID_060, ID_076 |
| ID_078 | Define serialization format for tree-KV metadata checkpoints. | ID_052, ID_056, ID_071 |
| ID_079 | Define restore into a fragmented destination arena. | ID_057, ID_078 |
| ID_080 | Define tree-KV consistency checking for debug and tests. | ID_056, ID_060, ID_062, ID_071 |
| ID_081 | Define token-to-visible-page gather metadata for Tree PagedAttention. | ID_055, ID_056 |
| ID_082 | Specify block table packing for GPU Tree PagedAttention. | ID_081 |
| ID_083 | Define causal and ancestry masking semantics inside gathered pages. | ID_055, ID_081 |
| ID_084 | Extend visibility semantics for sliding-window attention. | ID_083 |
| ID_085 | Extend paged tree attention metadata for hybrid recurrent/cache layers. | ID_083 |
| ID_086 | Define GQA/MQA head mapping over paged KV storage. | ID_052, ID_082 |
| ID_087 | Define quantized KV page compatibility for Tree PagedAttention. | ID_052, ID_082 |
| ID_088 | Define RoPE position semantics for shared tree KV. | ID_051, ID_061 |
| ID_089 | Define context-shift behavior for canonical and speculative branches. | ID_067, ID_088 |
| ID_090 | Define packed Tree PagedAttention query descriptors. | ID_071, ID_082, ID_084 |
| ID_091 | Define a CPU reference Tree PagedAttention algorithm for correctness. | ID_083, ID_086, ID_090 |
| ID_092 | Define numerical equivalence tests between dense-tree masking and paged gathering. | ID_084, ID_087, ID_091 |
| ID_093 | Define concurrent reader/writer phase boundaries for one decode step. | ID_059, ID_066, ID_069 |
| ID_094 | Define snapshot consistency for page tables consumed asynchronously by GPU work. | ID_066, ID_093 |
| ID_095 | Define ABA-resistant page reuse under asynchronous work. | ID_053, ID_066, ID_094 |
| ID_096 | Define failure atomicity for multi-branch reservation and publication. | ID_058, ID_059, ID_094 |
| ID_097 | Define invariant-focused stress tests for fork/append/rollback/prune concurrency. | ID_080, ID_093, ID_095, ID_096 |
| ID_098 | Define performance counters for tree paged KV management. | ID_072, ID_082, ID_093 |
| ID_099 | Define end-to-end integration contract between speculative tree scheduler, KV manager, and attention planner. | ID_064, ID_069, ID_090, ID_093, ID_096 |
| ID_100 | Define the Pilar B acceptance suite and formal completion criteria. | ID_079, ID_092, ID_097, ID_098, ID_099 |

## Core invariants

1. Every published page-table reference targets a LIVE page descriptor with a matching generation.
2. Stored page refcounts equal published branch references plus documented transient pins.
3. A branch page table resolves each materialized logical position to exactly one physical page/offset pair in path order.
4. Forked branches may share only immutable ancestry; divergent writes into a shared tail pass through copy-on-write.
5. Pruning and rollback make rejected suffixes unreachable before their pages are reclaimed.
6. A page id cannot be reused for a new generation until readers of the old generation have quiesced.
7. Tree PagedAttention exposes exactly the ancestor-and-causal key set represented by the dense tree mask, including sliding-window clipping where applicable.
8. Reservation/publication for one draft step is atomic from the reader perspective: either the new metadata snapshot is published or the previous snapshot remains authoritative.
9. Checkpoint restore preserves logical topology and sharing equivalence even when physical page ids differ.
10. Debug reconciliation can derive ownership from branch page tables and detect refcount, generation, ordering, or COW violations.

## Validation contract

The companion pillar-b.jsonl must contain exactly 50 JSON objects, one per line, with sequential IDs ID_051 through ID_100. Every object has exactly five fields: module_id, objective, mathematical_spec, dependencies, and implementation_prompt. Dependencies are module IDs only and intentionally minimal.
