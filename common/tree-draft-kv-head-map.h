#pragma once

#include "tree-draft-head-map.h"
#include "tree-draft-kv-block-table.h"
#include "tree-draft-kv-page.h"

#include <cstdint>

struct common_tree_draft_paged_kv_head_layout {
    uint32_t query_heads = 0;
    uint32_t kv_heads = 0;
    uint32_t query_heads_per_kv = 0;
    uint32_t head_dim = 0;
    uint32_t tokens_per_page = 0;
    uint64_t k_stride_token = 0;
    uint64_t k_stride_head = 0;
    uint64_t k_stride_dim = 0;
    uint64_t v_stride_token = 0;
    uint64_t v_stride_head = 0;
    uint64_t v_stride_dim = 0;
};

enum common_tree_draft_paged_kv_head_status : uint32_t {
    COMMON_TREE_DRAFT_PAGED_KV_HEAD_OK = 0,
    COMMON_TREE_DRAFT_PAGED_KV_HEAD_NULL_OUTPUT,
    COMMON_TREE_DRAFT_PAGED_KV_HEAD_GEOMETRY,
    COMMON_TREE_DRAFT_PAGED_KV_HEAD_LAYOUT,
    COMMON_TREE_DRAFT_PAGED_KV_HEAD_RANGE,
    COMMON_TREE_DRAFT_PAGED_KV_HEAD_OVERFLOW,
};

common_tree_draft_paged_kv_head_status common_tree_draft_paged_kv_head_layout_build(
        const common_tree_draft_head_map_plan & heads,
        const common_tree_draft_kv_page_geometry & pages,
        uint32_t layer,
        uint32_t head_dim,
        uint32_t k_element_size,
        uint32_t v_element_size,
        common_tree_draft_paged_kv_head_layout * out);

common_tree_draft_paged_kv_head_status common_tree_draft_paged_kv_head_offset(
        const common_tree_draft_paged_kv_head_layout & layout,
        const common_tree_draft_kv_gpu_segment & segment,
        uint32_t segment_token,
        uint32_t query_head,
        uint32_t dim,
        bool value,
        uint64_t * byte_offset);
