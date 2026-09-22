#include "tree-draft-budget.h"

#include <limits>

static bool common_tree_draft_budget_add(uint32_t a, uint32_t b, uint32_t * out) {
    if (b > std::numeric_limits<uint32_t>::max() - a) return false;
    *out = a + b;
    return true;
}

static bool common_tree_draft_budget_leq(common_tree_draft_budget_vector a, common_tree_draft_budget_vector b) {
    return a.nodes <= b.nodes && a.depth <= b.depth && a.draft_tokens <= b.draft_tokens && a.steps <= b.steps;
}

static bool common_tree_draft_budget_sum(common_tree_draft_budget_vector a, common_tree_draft_budget_vector b,
        common_tree_draft_budget_vector * out) {
    return common_tree_draft_budget_add(a.nodes, b.nodes, &out->nodes) &&
           common_tree_draft_budget_add(a.depth, b.depth, &out->depth) &&
           common_tree_draft_budget_add(a.draft_tokens, b.draft_tokens, &out->draft_tokens) &&
           common_tree_draft_budget_add(a.steps, b.steps, &out->steps);
}

static void common_tree_draft_budget_sub(common_tree_draft_budget_vector * a, common_tree_draft_budget_vector b) {
    a->nodes -= b.nodes;
    a->depth -= b.depth;
    a->draft_tokens -= b.draft_tokens;
    a->steps -= b.steps;
}

common_tree_draft_budget_state common_tree_draft_budget_make(common_tree_draft_budget_vector limit) {
    return { limit, {0,0,0,0}, {0,0,0,0}, 1 };
}

common_tree_draft_budget_status common_tree_draft_budget_reserve(
        common_tree_draft_budget_state * state,
        common_tree_draft_budget_vector delta,
        common_tree_draft_budget_reservation * reservation) {
    if (state == nullptr || reservation == nullptr || state->next_reservation_id == 0) {
        return COMMON_TREE_DRAFT_BUDGET_RESERVATION;
    }
    common_tree_draft_budget_vector used_plus_reserved = {};
    common_tree_draft_budget_vector projected = {};
    if (!common_tree_draft_budget_sum(state->committed, state->reserved, &used_plus_reserved) ||
        !common_tree_draft_budget_sum(used_plus_reserved, delta, &projected)) {
        return COMMON_TREE_DRAFT_BUDGET_OVERFLOW;
    }
    if (!common_tree_draft_budget_leq(projected, state->limit)) {
        return COMMON_TREE_DRAFT_BUDGET_LIMIT;
    }
    common_tree_draft_budget_vector new_reserved = {};
    if (!common_tree_draft_budget_sum(state->reserved, delta, &new_reserved)) {
        return COMMON_TREE_DRAFT_BUDGET_OVERFLOW;
    }
    const uint64_t id = state->next_reservation_id++;
    if (state->next_reservation_id == 0) state->next_reservation_id = 1;
    state->reserved = new_reserved;
    *reservation = { id, delta, true };
    return COMMON_TREE_DRAFT_BUDGET_OK;
}

common_tree_draft_budget_status common_tree_draft_budget_commit(
        common_tree_draft_budget_state * state,
        common_tree_draft_budget_reservation * reservation,
        common_tree_draft_budget_vector realized) {
    if (state == nullptr || reservation == nullptr || !reservation->active || reservation->id == 0) {
        return COMMON_TREE_DRAFT_BUDGET_RESERVATION;
    }
    if (!common_tree_draft_budget_leq(realized, reservation->reserved)) {
        return COMMON_TREE_DRAFT_BUDGET_REALIZED_RANGE;
    }
    common_tree_draft_budget_vector new_committed = {};
    if (!common_tree_draft_budget_sum(state->committed, realized, &new_committed)) {
        return COMMON_TREE_DRAFT_BUDGET_OVERFLOW;
    }
    if (!common_tree_draft_budget_leq(new_committed, state->limit) ||
        !common_tree_draft_budget_leq(reservation->reserved, state->reserved)) {
        return COMMON_TREE_DRAFT_BUDGET_RESERVATION;
    }
    common_tree_draft_budget_sub(&state->reserved, reservation->reserved);
    state->committed = new_committed;
    reservation->active = false;
    return COMMON_TREE_DRAFT_BUDGET_OK;
}

common_tree_draft_budget_status common_tree_draft_budget_release(
        common_tree_draft_budget_state * state,
        common_tree_draft_budget_reservation * reservation) {
    if (state == nullptr || reservation == nullptr || !reservation->active || reservation->id == 0 ||
        !common_tree_draft_budget_leq(reservation->reserved, state->reserved)) {
        return COMMON_TREE_DRAFT_BUDGET_RESERVATION;
    }
    common_tree_draft_budget_sub(&state->reserved, reservation->reserved);
    reservation->active = false;
    return COMMON_TREE_DRAFT_BUDGET_OK;
}

