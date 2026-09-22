#pragma once

#include "tree-draft-kv-gather.h"

#include <cstddef>
#include <cstdint>

struct common_tree_draft_kv_window_query {
    int64_t query_position = -1;
    int64_t window_start = 0;
};

enum common_tree_draft_kv_window_status : uint32_t {
    COMMON_TREE_DRAFT_KV_WINDOW_OK = 0,
    COMMON_TREE_DRAFT_KV_WINDOW_NULL_BUFFER,
    COMMON_TREE_DRAFT_KV_WINDOW_METADATA,
    COMMON_TREE_DRAFT_KV_WINDOW_QUERY_RANGE,
    COMMON_TREE_DRAFT_KV_WINDOW_OUTPUT_TOO_SMALL,
    COMMON_TREE_DRAFT_KV_WINDOW_OVERFLOW,
};

common_tree_draft_kv_window_status common_tree_draft_kv_window_clip(
        const common_tree_draft_kv_gather_metadata & source,
        const common_tree_draft_kv_window_query * queries,
        size_t query_count,
        common_tree_draft_kv_gather_metadata * output);
