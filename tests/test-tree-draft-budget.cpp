#include "tree-draft-budget.h"

#include <cassert>

int main() {
    common_tree_draft_budget_state state = common_tree_draft_budget_make({ 10, 4, 20, 5 });
    common_tree_draft_budget_reservation a = {};
    assert(common_tree_draft_budget_reserve(&state, { 4, 2, 8, 1 }, &a) == COMMON_TREE_DRAFT_BUDGET_OK);
    assert(a.active && a.id != 0);
    assert(state.reserved.nodes == 4 && state.committed.nodes == 0);

    common_tree_draft_budget_reservation b = {};
    assert(common_tree_draft_budget_reserve(&state, { 7, 1, 1, 1 }, &b) == COMMON_TREE_DRAFT_BUDGET_LIMIT);
    assert(state.reserved.nodes == 4);

    assert(common_tree_draft_budget_commit(&state, &a, { 3, 1, 6, 1 }) == COMMON_TREE_DRAFT_BUDGET_OK);
    assert(!a.active);
    assert(state.reserved.nodes == 0 && state.committed.nodes == 3);
    assert(common_tree_draft_budget_commit(&state, &a, { 1, 1, 1, 1 }) == COMMON_TREE_DRAFT_BUDGET_RESERVATION);
    assert(common_tree_draft_budget_release(&state, &a) == COMMON_TREE_DRAFT_BUDGET_RESERVATION);

    assert(common_tree_draft_budget_reserve(&state, { 2, 1, 4, 1 }, &b) == COMMON_TREE_DRAFT_BUDGET_OK);
    assert(common_tree_draft_budget_commit(&state, &b, { 3, 1, 1, 1 }) == COMMON_TREE_DRAFT_BUDGET_REALIZED_RANGE);
    assert(b.active && state.reserved.nodes == 2);
    assert(common_tree_draft_budget_release(&state, &b) == COMMON_TREE_DRAFT_BUDGET_OK);
    assert(!b.active && state.reserved.nodes == 0);

    common_tree_draft_budget_reservation c = {};
    assert(common_tree_draft_budget_reserve(&state, { 7, 3, 14, 4 }, &c) == COMMON_TREE_DRAFT_BUDGET_OK);
    assert(common_tree_draft_budget_commit(&state, &c, { 7, 3, 14, 4 }) == COMMON_TREE_DRAFT_BUDGET_OK);
    assert(state.committed.nodes == 10 && state.committed.depth == 4 && state.committed.draft_tokens == 20 && state.committed.steps == 5);
    common_tree_draft_budget_reservation d = {};
    assert(common_tree_draft_budget_reserve(&state, { 1, 0, 0, 0 }, &d) == COMMON_TREE_DRAFT_BUDGET_LIMIT);
    return 0;
}

