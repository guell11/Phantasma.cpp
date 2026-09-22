#include "tree-draft-append.h"

#include <algorithm>
#include <limits>

common_tree_draft_append_status common_tree_draft_append_reserve(
        common_tree_draft_arena * arena,
        common_tree_draft_budget_state * budget,
        const uint32_t * child_counts,
        size_t parent_count,
        uint32_t * child_offsets,
        size_t offset_capacity,
        uint8_t * written_mask,
        size_t written_capacity,
        common_tree_draft_append_transaction * transaction) {
    if (arena == nullptr || budget == nullptr || transaction == nullptr ||
        (parent_count > 0 && (child_counts == nullptr || child_offsets == nullptr))) {
        return COMMON_TREE_DRAFT_APPEND_NULL_BUFFER;
    }
    if (offset_capacity < parent_count + 1) return COMMON_TREE_DRAFT_APPEND_CAPACITY;

    uint64_t total = 0;
    child_offsets[0] = 0;
    for (size_t i = 0; i < parent_count; ++i) {
        total += child_counts[i];
        if (total > UINT32_MAX) return COMMON_TREE_DRAFT_APPEND_OVERFLOW;
        child_offsets[i + 1] = static_cast<uint32_t>(total);
    }
    if (written_capacity < total || (total > 0 && written_mask == nullptr)) return COMMON_TREE_DRAFT_APPEND_CAPACITY;
    if (total > static_cast<uint64_t>(arena->capacity - arena->node_count)) return COMMON_TREE_DRAFT_APPEND_CAPACITY;

    common_tree_draft_budget_reservation reservation = {};
    const common_tree_draft_budget_vector delta = {
        static_cast<uint32_t>(total), 0, static_cast<uint32_t>(total), 1
    };
    if (common_tree_draft_budget_reserve(budget, delta, &reservation) != COMMON_TREE_DRAFT_BUDGET_OK) {
        return COMMON_TREE_DRAFT_APPEND_BUDGET;
    }
    if (total > 0) std::fill_n(written_mask, static_cast<size_t>(total), uint8_t{0});
    *transaction = {
        COMMON_TREE_DRAFT_APPEND_RESERVED,
        arena->node_count,
        static_cast<uint32_t>(total),
        0,
        reservation,
        written_mask,
    };
    return COMMON_TREE_DRAFT_APPEND_OK;
}

common_tree_draft_append_status common_tree_draft_append_write(
        common_tree_draft_arena * arena,
        common_tree_draft_append_transaction * transaction,
        uint32_t relative_child_index,
        const common_tree_draft_node & node) {
    if (arena == nullptr || transaction == nullptr) return COMMON_TREE_DRAFT_APPEND_NULL_BUFFER;
    if (transaction->phase != COMMON_TREE_DRAFT_APPEND_RESERVED) return COMMON_TREE_DRAFT_APPEND_PHASE;
    if (relative_child_index >= transaction->total_children) return COMMON_TREE_DRAFT_APPEND_RANGE;
    if (transaction->written_mask[relative_child_index] != 0) return COMMON_TREE_DRAFT_APPEND_DUPLICATE_WRITE;
    const uint32_t absolute = transaction->base_node_count + relative_child_index;
    if (absolute >= arena->capacity) return COMMON_TREE_DRAFT_APPEND_CAPACITY;
    arena->nodes[absolute] = node;
    transaction->written_mask[relative_child_index] = 1;
    ++transaction->written_children;
    return COMMON_TREE_DRAFT_APPEND_OK;
}

common_tree_draft_append_status common_tree_draft_append_commit(
        common_tree_draft_arena * arena,
        common_tree_draft_budget_state * budget,
        common_tree_draft_append_transaction * transaction) {
    if (arena == nullptr || budget == nullptr || transaction == nullptr) return COMMON_TREE_DRAFT_APPEND_NULL_BUFFER;
    if (transaction->phase != COMMON_TREE_DRAFT_APPEND_RESERVED) return COMMON_TREE_DRAFT_APPEND_PHASE;
    if (arena->node_count != transaction->base_node_count || transaction->written_children != transaction->total_children) {
        return COMMON_TREE_DRAFT_APPEND_INCOMPLETE;
    }
    for (uint32_t i = 0; i < transaction->total_children; ++i) {
        if (transaction->written_mask[i] == 0) return COMMON_TREE_DRAFT_APPEND_INCOMPLETE;
    }
    const common_tree_draft_budget_vector realized = {
        transaction->total_children, 0, transaction->total_children, 1
    };
    if (common_tree_draft_budget_commit(budget, &transaction->budget_reservation, realized) != COMMON_TREE_DRAFT_BUDGET_OK) {
        return COMMON_TREE_DRAFT_APPEND_BUDGET;
    }
    arena->node_count = transaction->base_node_count + transaction->total_children;
    transaction->phase = COMMON_TREE_DRAFT_APPEND_COMMITTED;
    return COMMON_TREE_DRAFT_APPEND_OK;
}

common_tree_draft_append_status common_tree_draft_append_abort(
        common_tree_draft_budget_state * budget,
        common_tree_draft_append_transaction * transaction) {
    if (budget == nullptr || transaction == nullptr) return COMMON_TREE_DRAFT_APPEND_NULL_BUFFER;
    if (transaction->phase != COMMON_TREE_DRAFT_APPEND_RESERVED) return COMMON_TREE_DRAFT_APPEND_PHASE;
    if (common_tree_draft_budget_release(budget, &transaction->budget_reservation) != COMMON_TREE_DRAFT_BUDGET_OK) {
        return COMMON_TREE_DRAFT_APPEND_BUDGET;
    }
    transaction->phase = COMMON_TREE_DRAFT_APPEND_ABORTED;
    return COMMON_TREE_DRAFT_APPEND_OK;
}

