#pragma once

#include "tree-draft-kv-gather.h"

#include <cstdint>

enum common_tree_draft_memory_mode : uint32_t {
    COMMON_TREE_DRAFT_MEMORY_ATTENTION = 0,
    COMMON_TREE_DRAFT_MEMORY_RECURRENT,
};

struct common_tree_draft_layer_memory_metadata {
    common_tree_draft_memory_mode mode = COMMON_TREE_DRAFT_MEMORY_ATTENTION;
};

struct common_tree_draft_layer_memory_view {
    const common_tree_draft_kv_gather_metadata * paged_kv = nullptr;
    uint32_t sequence_index = UINT32_MAX;
    uint32_t branch_index = UINT32_MAX;
    bool requires_state_fork = false;
    bool requires_state_rollback = false;
};

enum common_tree_draft_memory_mode_status : uint32_t {
    COMMON_TREE_DRAFT_MEMORY_MODE_OK = 0,
    COMMON_TREE_DRAFT_MEMORY_MODE_NULL_OUTPUT,
    COMMON_TREE_DRAFT_MEMORY_MODE_INVALID_MODE,
    COMMON_TREE_DRAFT_MEMORY_MODE_MISSING_PAGED_KV,
    COMMON_TREE_DRAFT_MEMORY_MODE_RECURRENT_USES_PAGED_KV,
    COMMON_TREE_DRAFT_MEMORY_MODE_MISSING_RECURRENT_IDENTITY,
};

common_tree_draft_memory_mode_status common_tree_draft_memory_mode_bind(
        const common_tree_draft_layer_memory_metadata & layer,
        const common_tree_draft_kv_gather_metadata * paged_kv,
        uint32_t sequence_index,
        uint32_t branch_index,
        common_tree_draft_layer_memory_view * output);
