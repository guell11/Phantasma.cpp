#include "tree-draft-lifecycle.h"

bool common_tree_draft_lifecycle_transition_is_valid(
        common_tree_draft_lifecycle_state from,
        common_tree_draft_lifecycle_state to) {
    switch (from) {
        case COMMON_TREE_DRAFT_LIFECYCLE_CREATED:
            return to == COMMON_TREE_DRAFT_LIFECYCLE_QUEUED;
        case COMMON_TREE_DRAFT_LIFECYCLE_QUEUED:
            return to == COMMON_TREE_DRAFT_LIFECYCLE_RUNNING ||
                   to == COMMON_TREE_DRAFT_LIFECYCLE_COMPLETED ||
                   to == COMMON_TREE_DRAFT_LIFECYCLE_CANCELLED ||
                   to == COMMON_TREE_DRAFT_LIFECYCLE_FAILED;
        case COMMON_TREE_DRAFT_LIFECYCLE_RUNNING:
            return to == COMMON_TREE_DRAFT_LIFECYCLE_STREAMING ||
                   to == COMMON_TREE_DRAFT_LIFECYCLE_COMPLETED ||
                   to == COMMON_TREE_DRAFT_LIFECYCLE_CANCELLED ||
                   to == COMMON_TREE_DRAFT_LIFECYCLE_FAILED;
        case COMMON_TREE_DRAFT_LIFECYCLE_STREAMING:
            return to == COMMON_TREE_DRAFT_LIFECYCLE_COMPLETED ||
                   to == COMMON_TREE_DRAFT_LIFECYCLE_CANCELLED ||
                   to == COMMON_TREE_DRAFT_LIFECYCLE_FAILED;
        case COMMON_TREE_DRAFT_LIFECYCLE_COMPLETED:
        case COMMON_TREE_DRAFT_LIFECYCLE_CANCELLED:
        case COMMON_TREE_DRAFT_LIFECYCLE_FAILED:
            return false;
    }

    return false;
}

const char * common_tree_draft_lifecycle_state_name(common_tree_draft_lifecycle_state state) {
    switch (state) {
        case COMMON_TREE_DRAFT_LIFECYCLE_CREATED:   return "created";
        case COMMON_TREE_DRAFT_LIFECYCLE_QUEUED:    return "queued";
        case COMMON_TREE_DRAFT_LIFECYCLE_RUNNING:   return "running";
        case COMMON_TREE_DRAFT_LIFECYCLE_STREAMING: return "streaming";
        case COMMON_TREE_DRAFT_LIFECYCLE_COMPLETED: return "completed";
        case COMMON_TREE_DRAFT_LIFECYCLE_CANCELLED: return "cancelled";
        case COMMON_TREE_DRAFT_LIFECYCLE_FAILED:    return "failed";
    }

    return "unknown";
}
