#pragma once

#include <cstddef>
#include <cstdint>

enum common_tree_draft_action_flag : uint32_t {
    COMMON_TREE_DRAFT_ACTION_NONE = 0,
    COMMON_TREE_DRAFT_ACTION_CARRY = 1u << 0,
    COMMON_TREE_DRAFT_ACTION_FORCE_MINIMUM = 1u << 1,
};

struct common_tree_draft_expansion_action {
    uint64_t request_id;
    uint32_t node_index;
    uint32_t child_budget;
    uint32_t depth;
    double utility;
    uint32_t cost_class;
    uint32_t flags;
};

struct common_tree_draft_fair_request_state {
    uint64_t request_id;
    double virtual_finish;
    double weight;
};

enum common_tree_draft_scheduler_status : uint32_t {
    COMMON_TREE_DRAFT_SCHEDULER_OK = 0,
    COMMON_TREE_DRAFT_SCHEDULER_NULL_BUFFER,
    COMMON_TREE_DRAFT_SCHEDULER_INVALID_ACTION,
    COMMON_TREE_DRAFT_SCHEDULER_INVALID_WEIGHT,
    COMMON_TREE_DRAFT_SCHEDULER_INVALID_COST,
    COMMON_TREE_DRAFT_SCHEDULER_REQUEST_NOT_FOUND,
    COMMON_TREE_DRAFT_SCHEDULER_EMPTY,
    COMMON_TREE_DRAFT_SCHEDULER_NUMERIC,
};

common_tree_draft_scheduler_status common_tree_draft_action_validate(
        const common_tree_draft_expansion_action & action);

common_tree_draft_scheduler_status common_tree_draft_fair_select(
        const common_tree_draft_expansion_action * actions,
        const double * costs,
        size_t action_count,
        common_tree_draft_fair_request_state * requests,
        size_t request_count,
        double * virtual_time,
        size_t * selected_action);

