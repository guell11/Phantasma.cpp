# ID_086 paged KV GQA/MQA head mapping

ID_086 reuses the canonical ID_021 query-head to KV-head mapping for paged storage. For a valid divisible head geometry, query head h maps to KV head floor(h / (n_q / n_kv)); MHA, GQA, and MQA therefore use one rule and do not duplicate page-table topology per head.

For standard unquantized pages, each K and V tensor is token-major within the layer. The byte strides are dim = element_size, head = head_dim * dim_stride, and token = n_kv * head_stride. The resulting token stride must exactly match the ID_052 per-layer K/V byte geometry before the paged layout is accepted.

ID_082 page-table segments remain head-independent. Address calculation takes an existing segment, selects a token within its [lo, hi) page range, maps the query head through the canonical head grouping, and returns an offset within that page's K or V tensor. Page id and generation stay unchanged and no per-head copy of row or segment metadata is created.

Quantized page element addressing is outside this manifest and remains gated by ID_087.
