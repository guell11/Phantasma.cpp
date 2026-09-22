# ID_023 quantized KV compatibility boundary

ID_023 consumes the accepted ID_009 logical-to-physical KV indirection and defines only how a Tree-Draft caller reports K/V cache compatibility and selects an exact execution path. It does not create another cache, convert cache contents, or implement a new attention kernel.

The K and V cache types are the existing `ggml_type` values exposed by `llama_kv_cache::type_k()` and `llama_kv_cache::type_v()`. Quantization reporting uses the existing `ggml_is_quantized()` predicate; ID_023 does not maintain a second type table.

The caller reports four semantic capabilities for the active execution context:

```text
direct_k               direct Tree-Draft path can read the current K cache type exactly
direct_v               direct Tree-Draft path can read the current V cache type exactly
existing_backend_exact an existing llama/ggml backend path supports this exact attention operation
staging_exact          an explicit existing staging path is available and preserves the same attention inputs
```

`existing_backend_exact` is intended to be derived from the normal ggml backend/scheduler capability checks such as `ggml_backend_supports_op()` for the real attention graph. `staging_exact` describes an already available staging route selected by the caller; ID_023 deliberately does not implement dequantization or staging copies.

Selection is deterministic and ordered:

```text
1. DIRECT            when direct_k && direct_v
2. EXISTING_BACKEND  otherwise when existing_backend_exact
3. STAGING           otherwise when staging_exact
4. UNSUPPORTED       otherwise
```

K and V are treated independently for direct capability. A mixed cache such as quantized K with F16 V is direct only if both individual type checks pass. Quantized status alone never forces staging: if the current backend already supports the exact K/V types and attention semantics, the existing backend path is preferred over staging.

Every non-direct path must preserve the same logical attention problem defined by ID_009 and the normal llama graph: the same `kv_slot[g]` physical cell references, the same logical proposal-node ordering, the same K/V values represented by the cache types, and the same attention mask/position semantics. A path that changes those semantics must report its `*_exact` capability as false and cannot be selected by ID_023.

The runtime already enforces important cache constraints outside ID_023, including the existing requirement that quantized V cache use Flash Attention and the ggml block-size divisibility checks for quantized K/V cache heads. ID_023 does not duplicate or weaken those checks.
