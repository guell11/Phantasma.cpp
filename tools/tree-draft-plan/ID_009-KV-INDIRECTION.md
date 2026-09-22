# ID_009 branch KV indirection

ID_009 maps each accepted ID_001 logical proposal node `g` to one physical cell index in the existing llama KV cache. It does not allocate, own, resize, or replace that cache.

The logical mapping is a contiguous `uint32_t kv_slot[N]` array in the same packed global-node order used by ID_007. `kv_slot[g]` is measured in KV cache cells/slots, matching `llama_kv_cache::slot_info::idxs`; it is not a byte offset and not a token position.

`COMMON_TREE_DRAFT_KV_SLOT_INVALID` is `UINT32_MAX`. An invalid entry means logical node `g` currently has no readable physical KV cell. A valid slot must satisfy `kv_slot[g] < kv_size`. Multiple logical nodes may reference the same valid physical slot. Such duplicates are aliases for reads only; they do not imply duplicate physical storage and they do not merge the logical nodes for ancestry or masking.

Packed batching does not reset slot numbering per tree or per request. The entire packed forest shares the physical slot namespace exposed by the active llama KV cache. Logical identity remains the ID_001 global node index, while physical identity is `kv_slot[g]`. Cross-tree or sibling visibility is therefore never inferred from equal slot numbers.

For byte addressing into an existing K or V tensor, ID_009 uses an explicit layout:

```text
byte_offset(slot,h,d) = slot*stride_slot + h*stride_head + d*stride_dim
```

All three strides are in bytes, as are ggml tensor `nb[]` values. `slot` itself remains a cell index. `element_size` is also in bytes and is used only to validate that the final addressed element does not overflow `size_t`. K and V may use different layout descriptors; ID_009 does not require their strides to match.

Validation checks the complete ID_001 topology, one indirection entry per logical node, every valid slot against `kv_size`, and address arithmetic overflow. Duplicate valid slots are intentionally accepted. The sentinel is intentionally accepted by the table validator but rejected when a caller requests a readable physical slot/address for that logical node.

ID_009 is only the logical-to-physical read indirection contract. It does not create a second runtime/cache, allocate KV cells, alter llama KV eviction/write policy, or implement attention/mask successors.
