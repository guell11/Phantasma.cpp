#include "tree-draft-runtime-status.h"

common_tree_draft_runtime_result common_tree_draft_runtime_snapshot(
        common_tree_draft_runtime_status status,
        const common_tree_draft_arena & arena) {
    return { status, arena.node_count, arena.frontier_count };
}

bool common_tree_draft_runtime_can_fallback(common_tree_draft_runtime_status status) {
    return status != COMMON_TREE_DRAFT_RUNTIME_OK && status != COMMON_TREE_DRAFT_RUNTIME_CANCELLED;
}

const char * common_tree_draft_runtime_status_name(common_tree_draft_runtime_status status) {
    switch (status) {
        case COMMON_TREE_DRAFT_RUNTIME_OK:          return "ok";
        case COMMON_TREE_DRAFT_RUNTIME_CAPACITY:    return "capacity";
        case COMMON_TREE_DRAFT_RUNTIME_INVALID:     return "invalid";
        case COMMON_TREE_DRAFT_RUNTIME_MODEL:       return "model";
        case COMMON_TREE_DRAFT_RUNTIME_DEVICE:      return "device";
        case COMMON_TREE_DRAFT_RUNTIME_CANCELLED:   return "cancelled";
        case COMMON_TREE_DRAFT_RUNTIME_UNSUPPORTED: return "unsupported";
    }
    return "unknown";
}

