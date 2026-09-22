#pragma once

#include "tree-draft-kv-branch.h"

#include <cstddef>
#include <cstdint>

constexpr int64_t COMMON_TREE_DRAFT_KV_VISIBILITY_NONE = -1;

struct common_tree_draft_kv_visibility_table {
    int64_t * cutoffs;
    size_t cutoff_count;
    uint32_t branch_count;
};

enum common_tree_draft_kv_visibility_status : uint32_t {
    COMMON_TREE_DRAFT_KV_VISIBILITY_OK = 0,
    COMMON_TREE_DRAFT_KV_VISIBILITY_NULL_BUFFER,
    COMMON_TREE_DRAFT_KV_VISIBILITY_OUTPUT_TOO_SMALL,
    COMMON_TREE_DRAFT_KV_VISIBILITY_SIZE_OVERFLOW,
    COMMON_TREE_DRAFT_KV_VISIBILITY_BRANCH_HANDLE,
    COMMON_TREE_DRAFT_KV_VISIBILITY_DUPLICATE_BRANCH,
    COMMON_TREE_DRAFT_KV_VISIBILITY_PARENT_ORDER,
    COMMON_TREE_DRAFT_KV_VISIBILITY_FORK_RANGE,
};

size_t common_tree_draft_kv_visibility_cutoff_count(uint32_t branch_count, bool * overflow);

common_tree_draft_kv_visibility_status common_tree_draft_kv_visibility_build(
        const common_tree_draft_kv_branch_descriptor * branches,
        uint32_t branch_count,
        common_tree_draft_kv_visibility_table & output);

bool common_tree_draft_kv_visible(
        const common_tree_draft_kv_visibility_table & table,
        uint32_t query_branch,
        int64_t query_position,
        uint32_t key_branch,
        int64_t key_position);
