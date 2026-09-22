#pragma once

#include "tree-draft-kv-visibility.h"

#include <cstddef>
#include <cstdint>

struct common_tree_draft_kv_query {
    uint32_t branch_index = 0;
    int64_t position = -1;
};

struct common_tree_draft_kv_gather_segment {
    common_tree_draft_page_handle page = {};
    uint32_t lo = 0;
    uint32_t hi = 0;
};

struct common_tree_draft_kv_gather_metadata {
    uint32_t * row_ptr = nullptr;
    size_t row_ptr_capacity = 0;
    common_tree_draft_kv_gather_segment * segments = nullptr;
    size_t segment_capacity = 0;
    uint32_t query_count = 0;
    uint32_t segment_count = 0;
};

enum common_tree_draft_kv_gather_status : uint32_t {
    COMMON_TREE_DRAFT_KV_GATHER_OK = 0,
    COMMON_TREE_DRAFT_KV_GATHER_NULL_BUFFER,
    COMMON_TREE_DRAFT_KV_GATHER_QUERY_RANGE,
    COMMON_TREE_DRAFT_KV_GATHER_BRANCH,
    COMMON_TREE_DRAFT_KV_GATHER_VISIBILITY,
    COMMON_TREE_DRAFT_KV_GATHER_OUTPUT_TOO_SMALL,
    COMMON_TREE_DRAFT_KV_GATHER_OVERFLOW,
};

common_tree_draft_kv_gather_status common_tree_draft_kv_gather_build(
        const common_tree_draft_kv_query * queries,
        size_t query_count,
        const common_tree_draft_kv_branch_descriptor * branches,
        uint32_t branch_count,
        const common_tree_draft_kv_visibility_table & visibility,
        const common_tree_draft_kv_page_geometry & geometry,
        common_tree_draft_kv_gather_metadata * output);
