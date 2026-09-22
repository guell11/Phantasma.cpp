# ID_007 packed QKV gather layout

ID_007 exposes a CPU/reference byte-strided view for proposal Q/K/V tensors while preserving the accepted ID_004 packed global-node order and consuming the ID_006 position vector for the same packed nodes.

Each `common_tree_draft_qkv_view` has logical shape `[g, h, d]` with independent byte strides:

```text
address(g,h,d) = data + g*stride_node + h*stride_head + d*stride_dim
```

Q uses `[N, Hq, Dq]`; K uses `[N, Hkv, Dk]`; V uses `[N, Hkv, Dv]`. The interface does not require equal Q/K/V head counts or head dimensions. `element_size` is explicit, so the gather is layout-only and does not reinterpret element values.

The gathered reference buffer is tightly packed in global-node-major order, then head-major, then dimension-major. This keeps global node `g` aligned with ID_004 packed entry/local/global mapping and ID_006 `position[g]` even when the ggml source is a non-contiguous view. The source may therefore use padded or permuted byte strides as long as the maximum addressed byte offset is representable by `size_t`.

The gather validates ID_004 offset monotonicity and derives `N` from the final packed offset. `source.n_nodes` and the ID_006 position count must both equal `N`. Non-empty inputs require source and position pointers. Shape products, source stride address arithmetic, and packed output byte counts are checked for overflow before copying. Output capacity is checked before the first write.

ID_007 defines only the packed gather/layout contract and CPU correctness reference. It does not implement Triton kernels, head mapping, tree attention, KV indirection, or successor modules.
