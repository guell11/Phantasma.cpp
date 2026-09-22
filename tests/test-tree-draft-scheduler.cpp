#include "tree-draft-scheduler.h"

#include <cassert>
#include <type_traits>

int main() {
    static_assert(std::is_standard_layout<common_tree_draft_expansion_action>::value, "action must be POD-style");
    static_assert(std::is_trivially_copyable<common_tree_draft_expansion_action>::value, "action must be trivially copyable");

    const common_tree_draft_expansion_action actions[] = {
        { 1, 4, 2, 2, 1.0, 0, 0 },
        { 2, 8, 3, 1, 2.0, 1, COMMON_TREE_DRAFT_ACTION_CARRY },
        { 1, 5, 1, 2, 0.5, 0, 0 },
    };
    const double costs[] = { 2.0, 2.0, 1.0 };
    common_tree_draft_fair_request_state requests[] = {
        { 1, 0.0, 1.0 },
        { 2, 0.0, 2.0 },
    };
    double virtual_time = 0.0;
    size_t selected = SIZE_MAX;
    assert(common_tree_draft_fair_select(actions, costs, 3, requests, 2, &virtual_time, &selected) == COMMON_TREE_DRAFT_SCHEDULER_OK);
    assert(selected == 1);
    assert(requests[1].virtual_finish == 1.0 && virtual_time == 1.0);

    assert(common_tree_draft_fair_select(actions, costs, 3, requests, 2, &virtual_time, &selected) == COMMON_TREE_DRAFT_SCHEDULER_OK);
    assert(selected == 1);
    assert(requests[1].virtual_finish == 2.0 && virtual_time == 2.0);

    const common_tree_draft_expansion_action tie_actions[] = {
        { 9, 10, 1, 1, 0.0, 0, 0 },
        { 8, 11, 1, 1, 0.0, 0, 0 },
    };
    const double tie_costs[] = { 1.0, 1.0 };
    common_tree_draft_fair_request_state tie_requests[] = { {9,0.0,1.0}, {8,0.0,1.0} };
    virtual_time = 0.0;
    assert(common_tree_draft_fair_select(tie_actions, tie_costs, 2, tie_requests, 2, &virtual_time, &selected) == COMMON_TREE_DRAFT_SCHEDULER_OK);
    assert(selected == 1);

    common_tree_draft_expansion_action invalid = actions[0];
    invalid.child_budget = 0;
    assert(common_tree_draft_action_validate(invalid) == COMMON_TREE_DRAFT_SCHEDULER_INVALID_ACTION);
    return 0;
}

