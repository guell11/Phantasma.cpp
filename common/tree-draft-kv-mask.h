#pragma once

#include "tree-draft-composite-mask.h"
#include "tree-draft-kv-gather.h"

#include <cstdint>

enum common_tree_draft_kv_mask_status : uint32_t {
    COMMON_TREE_DRAFT_KV_MASK_OK = 0,
    COMMON_TREE_DRAFT_KV_MASK_NULL_OUTPUT,
    COMMON_TREE_DRAFT_KV_MASK_METADATA,
    COMMON_TREE_DRAFT_KV_MASK_QUERY_RANGE,
    COMMON_TREE_DRAFT_KV_MASK_KEY_RANGE,
    COMMON_TREE_DRAFT_KV_MASK_OVERFLOW,
};

common_tree_draft_kv_mask_status common_tree_draft_kv_mask_visible(
        const common_tree_draft_kv_gather_metadata & gather,
        uint32_t query_index,
        uint32_t segment_index,
        uint32_t page_offset,
        bool * visible,
        int64_t * logical_position);

float common_tree_draft_kv_mask_value(bool visible);
