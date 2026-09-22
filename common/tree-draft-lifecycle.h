#pragma once

#include <cstdint>

enum common_tree_draft_lifecycle_state : int32_t {
    COMMON_TREE_DRAFT_LIFECYCLE_CREATED = 0,
    COMMON_TREE_DRAFT_LIFECYCLE_QUEUED,
    COMMON_TREE_DRAFT_LIFECYCLE_RUNNING,
    COMMON_TREE_DRAFT_LIFECYCLE_STREAMING,
    COMMON_TREE_DRAFT_LIFECYCLE_COMPLETED,
    COMMON_TREE_DRAFT_LIFECYCLE_CANCELLED,
    COMMON_TREE_DRAFT_LIFECYCLE_FAILED,
};

bool common_tree_draft_lifecycle_transition_is_valid(
        common_tree_draft_lifecycle_state from,
        common_tree_draft_lifecycle_state to);

const char * common_tree_draft_lifecycle_state_name(common_tree_draft_lifecycle_state state);
