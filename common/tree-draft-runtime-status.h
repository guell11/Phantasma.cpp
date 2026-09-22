#pragma once

#include "tree-draft-arena.h"

#include <cstdint>

enum common_tree_draft_runtime_status : uint32_t {
    COMMON_TREE_DRAFT_RUNTIME_OK = 0,
    COMMON_TREE_DRAFT_RUNTIME_CAPACITY,
    COMMON_TREE_DRAFT_RUNTIME_INVALID,
    COMMON_TREE_DRAFT_RUNTIME_MODEL,
    COMMON_TREE_DRAFT_RUNTIME_DEVICE,
    COMMON_TREE_DRAFT_RUNTIME_CANCELLED,
    COMMON_TREE_DRAFT_RUNTIME_UNSUPPORTED,
};

struct common_tree_draft_runtime_result {
    common_tree_draft_runtime_status status;
    uint32_t committed_node_count;
    uint32_t committed_frontier_count;
};

common_tree_draft_runtime_result common_tree_draft_runtime_snapshot(
        common_tree_draft_runtime_status status,
        const common_tree_draft_arena & arena);

bool common_tree_draft_runtime_can_fallback(common_tree_draft_runtime_status status);
const char * common_tree_draft_runtime_status_name(common_tree_draft_runtime_status status);

