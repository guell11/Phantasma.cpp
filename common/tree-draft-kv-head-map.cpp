#include "tree-draft-kv-head-map.h"

#include <limits>

static bool common_tree_draft_paged_kv_mul(uint64_t a, uint64_t b, uint64_t * out) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) return false;
    *out = a * b;
    return true;
}

static bool common_tree_draft_paged_kv_add(uint64_t a, uint64_t b, uint64_t * out) {
    if (b > std::numeric_limits<uint64_t>::max() - a) return false;
    *out = a + b;
    return true;
}

common_tree_draft_paged_kv_head_status common_tree_draft_paged_kv_head_layout_build(
        const common_tree_draft_head_map_plan & heads,
        const common_tree_draft_kv_page_geometry & pages,
        uint32_t layer,
        uint32_t head_dim,
        uint32_t k_element_size,
        uint32_t v_element_size,
        common_tree_draft_paged_kv_head_layout * out) {
    if (out == nullptr) return COMMON_TREE_DRAFT_PAGED_KV_HEAD_NULL_OUTPUT;
    if (heads.query_heads == 0 || heads.kv_heads == 0 || heads.query_heads_per_kv == 0 ||
        heads.query_heads % heads.kv_heads != 0 ||
        heads.query_heads / heads.kv_heads != heads.query_heads_per_kv ||
        heads.groups.size() != heads.kv_heads ||
        pages.tokens_per_page == 0 || pages.layers == nullptr || layer >= pages.layer_count ||
        head_dim == 0 || k_element_size == 0 || v_element_size == 0) {
        return COMMON_TREE_DRAFT_PAGED_KV_HEAD_GEOMETRY;
    }
    for (uint32_t kv = 0; kv < heads.kv_heads; ++kv) {
        const auto & group = heads.groups[kv];
        const uint64_t begin = static_cast<uint64_t>(kv) * heads.query_heads_per_kv;
        const uint64_t end = begin + heads.query_heads_per_kv;
        if (group.kv_head != kv || group.query_head_begin != begin || group.query_head_end != end) {
            return COMMON_TREE_DRAFT_PAGED_KV_HEAD_GEOMETRY;
        }
    }

    const auto & layer_geometry = pages.layers[layer];
    if (layer_geometry.layout != COMMON_TREE_DRAFT_KV_LAYOUT_STANDARD) {
        return COMMON_TREE_DRAFT_PAGED_KV_HEAD_LAYOUT;
    }

    uint64_t k_stride_head = 0;
    uint64_t v_stride_head = 0;
    uint64_t k_stride_token = 0;
    uint64_t v_stride_token = 0;
    if (!common_tree_draft_paged_kv_mul(head_dim, k_element_size, &k_stride_head) ||
        !common_tree_draft_paged_kv_mul(head_dim, v_element_size, &v_stride_head) ||
        !common_tree_draft_paged_kv_mul(heads.kv_heads, k_stride_head, &k_stride_token) ||
        !common_tree_draft_paged_kv_mul(heads.kv_heads, v_stride_head, &v_stride_token)) {
        return COMMON_TREE_DRAFT_PAGED_KV_HEAD_OVERFLOW;
    }
    if (layer_geometry.k_bytes_per_token != k_stride_token ||
        layer_geometry.v_bytes_per_token != v_stride_token) {
        return COMMON_TREE_DRAFT_PAGED_KV_HEAD_LAYOUT;
    }

    *out = {
        heads.query_heads,
        heads.kv_heads,
        heads.query_heads_per_kv,
        head_dim,
        pages.tokens_per_page,
        k_stride_token,
        k_stride_head,
        k_element_size,
        v_stride_token,
        v_stride_head,
        v_element_size,
    };
    return COMMON_TREE_DRAFT_PAGED_KV_HEAD_OK;
}

common_tree_draft_paged_kv_head_status common_tree_draft_paged_kv_head_offset(
        const common_tree_draft_paged_kv_head_layout & layout,
        const common_tree_draft_kv_gpu_segment & segment,
        uint32_t segment_token,
        uint32_t query_head,
        uint32_t dim,
        bool value,
        uint64_t * byte_offset) {
    if (byte_offset == nullptr) return COMMON_TREE_DRAFT_PAGED_KV_HEAD_NULL_OUTPUT;
    if (layout.query_heads == 0 || layout.kv_heads == 0 || layout.query_heads_per_kv == 0 ||
        layout.head_dim == 0 || layout.tokens_per_page == 0 ||
        layout.query_heads % layout.kv_heads != 0 ||
        layout.query_heads / layout.kv_heads != layout.query_heads_per_kv) {
        return COMMON_TREE_DRAFT_PAGED_KV_HEAD_GEOMETRY;
    }
    if (segment.page_id == COMMON_TREE_DRAFT_KV_ID_INVALID || segment.lo >= segment.hi ||
        segment.hi > layout.tokens_per_page ||
        segment_token >= segment.hi - segment.lo ||
        query_head >= layout.query_heads || dim >= layout.head_dim) {
        return COMMON_TREE_DRAFT_PAGED_KV_HEAD_RANGE;
    }

    const uint32_t kv_head = query_head / layout.query_heads_per_kv;
    if (kv_head >= layout.kv_heads) return COMMON_TREE_DRAFT_PAGED_KV_HEAD_RANGE;

    const uint64_t stride_token = value ? layout.v_stride_token : layout.k_stride_token;
    const uint64_t stride_head = value ? layout.v_stride_head : layout.k_stride_head;
    const uint64_t stride_dim = value ? layout.v_stride_dim : layout.k_stride_dim;
    const uint64_t page_token = static_cast<uint64_t>(segment.lo) + segment_token;
    uint64_t token_offset = 0;
    uint64_t head_offset = 0;
    uint64_t dim_offset = 0;
    uint64_t offset = 0;
    if (!common_tree_draft_paged_kv_mul(page_token, stride_token, &token_offset) ||
        !common_tree_draft_paged_kv_mul(kv_head, stride_head, &head_offset) ||
        !common_tree_draft_paged_kv_mul(dim, stride_dim, &dim_offset) ||
        !common_tree_draft_paged_kv_add(token_offset, head_offset, &offset) ||
        !common_tree_draft_paged_kv_add(offset, dim_offset, &offset)) {
        return COMMON_TREE_DRAFT_PAGED_KV_HEAD_OVERFLOW;
    }
    *byte_offset = offset;
    return COMMON_TREE_DRAFT_PAGED_KV_HEAD_OK;
}
