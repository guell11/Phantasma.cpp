#pragma once

#include "tree-draft-ranking.h"

#include <cstddef>
#include <cstdint>

struct common_tree_draft_child_budget_input {
    double weight;
    uint32_t cap;
    uint32_t minimum;
    common_tree_draft_rank_entry rank;
};

struct common_tree_draft_child_budget_output {
    uint32_t requested;
    uint32_t granted;
};

enum common_tree_draft_child_budget_status : uint32_t {
    COMMON_TREE_DRAFT_CHILD_BUDGET_OK = 0,
    COMMON_TREE_DRAFT_CHILD_BUDGET_NULL_BUFFER,
    COMMON_TREE_DRAFT_CHILD_BUDGET_INVALID_WEIGHT,
    COMMON_TREE_DRAFT_CHILD_BUDGET_MINIMUM_RANGE,
    COMMON_TREE_DRAFT_CHILD_BUDGET_INFEASIBLE_MINIMUM,
    COMMON_TREE_DRAFT_CHILD_BUDGET_OVERFLOW,
};

common_tree_draft_child_budget_status common_tree_draft_allocate_child_budget(
        const common_tree_draft_child_budget_input * inputs,
        size_t count,
        uint32_t total_budget,
        common_tree_draft_score_mode rank_mode,
        common_tree_draft_depth_preference depth_preference,
        common_tree_draft_child_budget_output * outputs,
        size_t out_capacity);

