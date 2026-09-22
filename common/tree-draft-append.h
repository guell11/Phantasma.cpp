#pragma once

#include "tree-draft-arena.h"
#include "tree-draft-budget.h"

#include <cstddef>
#include <cstdint>

enum common_tree_draft_append_phase : uint32_t {
    COMMON_TREE_DRAFT_APPEND_IDLE = 0,
    COMMON_TREE_DRAFT_APPEND_RESERVED,
    COMMON_TREE_DRAFT_APPEND_COMMITTED,
    COMMON_TREE_DRAFT_APPEND_ABORTED,
};

struct common_tree_draft_append_transaction {
    common_tree_draft_append_phase phase;
    uint32_t base_node_count;
    uint32_t total_children;
    uint32_t written_children;
    common_tree_draft_budget_reservation budget_reservation;
    uint8_t * written_mask;
};

enum common_tree_draft_append_status : uint32_t {
    COMMON_TREE_DRAFT_APPEND_OK = 0,
    COMMON_TREE_DRAFT_APPEND_NULL_BUFFER,
    COMMON_TREE_DRAFT_APPEND_CAPACITY,
    COMMON_TREE_DRAFT_APPEND_OVERFLOW,
    COMMON_TREE_DRAFT_APPEND_PHASE,
    COMMON_TREE_DRAFT_APPEND_RANGE,
    COMMON_TREE_DRAFT_APPEND_DUPLICATE_WRITE,
    COMMON_TREE_DRAFT_APPEND_INCOMPLETE,
    COMMON_TREE_DRAFT_APPEND_BUDGET,
};

common_tree_draft_append_status common_tree_draft_append_reserve(
        common_tree_draft_arena * arena,
        common_tree_draft_budget_state * budget,
        const uint32_t * child_counts,
        size_t parent_count,
        uint32_t * child_offsets,
        size_t offset_capacity,
        uint8_t * written_mask,
        size_t written_capacity,
        common_tree_draft_append_transaction * transaction);

common_tree_draft_append_status common_tree_draft_append_write(
        common_tree_draft_arena * arena,
        common_tree_draft_append_transaction * transaction,
        uint32_t relative_child_index,
        const common_tree_draft_node & node);

common_tree_draft_append_status common_tree_draft_append_commit(
        common_tree_draft_arena * arena,
        common_tree_draft_budget_state * budget,
        common_tree_draft_append_transaction * transaction);

common_tree_draft_append_status common_tree_draft_append_abort(
        common_tree_draft_budget_state * budget,
        common_tree_draft_append_transaction * transaction);

