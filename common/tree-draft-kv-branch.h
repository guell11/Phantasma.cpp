#pragma once

#include "tree-draft-kv-page.h"

#include <cstddef>
#include <cstdint>

struct common_tree_draft_kv_branch_span {
    common_tree_draft_page_handle page;
    uint32_t lo;
    uint32_t hi;
};

struct common_tree_draft_kv_branch_descriptor {
    common_tree_draft_branch_handle branch;
    common_tree_draft_branch_handle parent_branch;
    int64_t fork_position;
    int64_t tip_position;
    uint64_t epoch;
    const common_tree_draft_kv_branch_span * spans;
    uint32_t span_count;
};

enum common_tree_draft_kv_branch_status : uint32_t {
    COMMON_TREE_DRAFT_KV_BRANCH_OK = 0,
    COMMON_TREE_DRAFT_KV_BRANCH_NULL_BUFFER,
    COMMON_TREE_DRAFT_KV_BRANCH_HANDLE,
    COMMON_TREE_DRAFT_KV_BRANCH_RANGE,
    COMMON_TREE_DRAFT_KV_BRANCH_PAGE,
    COMMON_TREE_DRAFT_KV_BRANCH_LENGTH,
    COMMON_TREE_DRAFT_KV_BRANCH_NOT_FOUND,
};

common_tree_draft_kv_branch_status common_tree_draft_kv_branch_validate(
        const common_tree_draft_kv_branch_descriptor & branch,
        const common_tree_draft_kv_page_geometry & geometry);

common_tree_draft_kv_branch_status common_tree_draft_kv_branch_lookup(
        const common_tree_draft_kv_branch_descriptor & branch,
        int64_t logical_position,
        common_tree_draft_page_address * address);

