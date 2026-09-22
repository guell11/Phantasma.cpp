# ID_051 logical tree-KV identity

ID_051 defines identity only. A logical token is the accepted ID_001 packed node index together with its logical position, branch handle and optional physical page address. It does not allocate pages, choose page capacity, publish KV writes, or change llama memory ownership.

`common_tree_draft_branch_handle` and `common_tree_draft_page_handle` are `(id,generation)` pairs. Equality includes the generation so a recycled numeric id cannot silently compare equal to an older handle. The generation lifecycle itself is specified by later Pilar B modules; ID_051 only makes the generation part of immutable identity from the start.

`common_tree_draft_page_address` is `(page_handle,offset)`. `offset` is a token slot within that physical page. ID_051 intentionally does not impose an upper bound because fixed page capacity, layout and alignment belong to ID_052. `UINT32_MAX` is reserved as the invalid id and invalid offset sentinel.

Each `common_tree_draft_kv_token_identity` occupies the same packed node order as ID_001. The logical node field must equal its array index. In ID_051, roots have logical position zero and every child position is its parent position plus one, which is equivalent to the validated ID_001 depth. A materialized token must have a valid page id and offset. An unmaterialized token carries no physical address. `shared_from_node` is `-1` for ordinary ownership; a non-negative value explicitly names an already materialized ancestor whose exact page address is being shared.

Two distinct materialized logical tokens cannot silently own the same complete `(page id,page generation,offset)` address. An exact duplicate is accepted only when the later token explicitly names the earlier token through `shared_from_node` and that node is an actual topology ancestor. Tokens may freely share one physical page at different offsets. Later fork/page-table modules refine how ancestry-shared pages are published and made copy-on-write.

The mapping to current llama.cpp is deliberate but non-owning. A page address is a higher-level grouping over physical KV cells; it does not replace `llama_kv_cache`, `llama_kv_cells`, sequence membership, or existing memory APIs. Future page geometry can map `(page,offset)` to an existing cell/slot index while the underlying llama memory object continues to own K/V storage and lifetime.

Serialization stores logical node/position, branch `(id,generation)`, materialization state, and physical `(page id,generation,offset)` identity as integer fields. Raw pointers and backend buffer addresses are never part of the serialized identity. Restoring physical page ids into a different arena is deferred to ID_078/ID_079.
