#pragma once

#include <cstdint>

struct common_tree_draft_budget_vector {
    uint32_t nodes;
    uint32_t depth;
    uint32_t draft_tokens;
    uint32_t steps;
};

struct common_tree_draft_budget_state {
    common_tree_draft_budget_vector limit;
    common_tree_draft_budget_vector committed;
    common_tree_draft_budget_vector reserved;
    uint64_t next_reservation_id;
};

struct common_tree_draft_budget_reservation {
    uint64_t id;
    common_tree_draft_budget_vector reserved;
    bool active;
};

enum common_tree_draft_budget_status : uint32_t {
    COMMON_TREE_DRAFT_BUDGET_OK = 0,
    COMMON_TREE_DRAFT_BUDGET_LIMIT,
    COMMON_TREE_DRAFT_BUDGET_OVERFLOW,
    COMMON_TREE_DRAFT_BUDGET_RESERVATION,
    COMMON_TREE_DRAFT_BUDGET_REALIZED_RANGE,
};

common_tree_draft_budget_state common_tree_draft_budget_make(common_tree_draft_budget_vector limit);

common_tree_draft_budget_status common_tree_draft_budget_reserve(
        common_tree_draft_budget_state * state,
        common_tree_draft_budget_vector delta,
        common_tree_draft_budget_reservation * reservation);

common_tree_draft_budget_status common_tree_draft_budget_commit(
        common_tree_draft_budget_state * state,
        common_tree_draft_budget_reservation * reservation,
        common_tree_draft_budget_vector realized);

common_tree_draft_budget_status common_tree_draft_budget_release(
        common_tree_draft_budget_state * state,
        common_tree_draft_budget_reservation * reservation);

