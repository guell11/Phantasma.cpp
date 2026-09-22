#include "tree-draft-child-budget.h"

#include <cassert>

static common_tree_draft_rank_entry rank(uint64_t path, uint32_t node) {
    return { -1.0, -1.0, -1.0, 1, path, node };
}

int main() {
    const common_tree_draft_child_budget_input weighted[] = {
        { 3.0, 4, 1, rank(10, 0) },
        { 1.0, 4, 1, rank(20, 1) },
    };
    common_tree_draft_child_budget_output out[3] = {};
    assert(common_tree_draft_allocate_child_budget(weighted, 2, 6, COMMON_TREE_DRAFT_SCORE_CUMULATIVE,
                COMMON_TREE_DRAFT_DEPTH_NEUTRAL, out, 3) == COMMON_TREE_DRAFT_CHILD_BUDGET_OK);
    assert(out[0].requested == 4 && out[1].requested == 4);
    assert(out[0].granted == 4 && out[1].granted == 2);

    const common_tree_draft_child_budget_input equal[] = {
        { 0.0, 3, 0, rank(10, 0) },
        { 0.0, 3, 0, rank(20, 1) },
        { 0.0, 3, 0, rank(30, 2) },
    };
    assert(common_tree_draft_allocate_child_budget(equal, 3, 2, COMMON_TREE_DRAFT_SCORE_CUMULATIVE,
                COMMON_TREE_DRAFT_DEPTH_NEUTRAL, out, 3) == COMMON_TREE_DRAFT_CHILD_BUDGET_OK);
    assert(out[0].granted == 1 && out[1].granted == 1 && out[2].granted == 0);

    const common_tree_draft_child_budget_input capped[] = {
        { 100.0, 1, 0, rank(10, 0) },
        { 1.0, 5, 0, rank(20, 1) },
    };
    assert(common_tree_draft_allocate_child_budget(capped, 2, 4, COMMON_TREE_DRAFT_SCORE_CUMULATIVE,
                COMMON_TREE_DRAFT_DEPTH_NEUTRAL, out, 3) == COMMON_TREE_DRAFT_CHILD_BUDGET_OK);
    assert(out[0].granted == 1 && out[1].granted == 3);

    const common_tree_draft_child_budget_input infeasible[] = {
        { 1.0, 3, 2, rank(10, 0) },
        { 1.0, 3, 2, rank(20, 1) },
    };
    assert(common_tree_draft_allocate_child_budget(infeasible, 2, 3, COMMON_TREE_DRAFT_SCORE_CUMULATIVE,
                COMMON_TREE_DRAFT_DEPTH_NEUTRAL, out, 3) == COMMON_TREE_DRAFT_CHILD_BUDGET_INFEASIBLE_MINIMUM);
    return 0;
}

