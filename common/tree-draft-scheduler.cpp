#include "tree-draft-scheduler.h"

#include <algorithm>
#include <cmath>
#include <limits>

static constexpr uint32_t COMMON_TREE_DRAFT_ACTION_FLAG_MASK =
        COMMON_TREE_DRAFT_ACTION_CARRY | COMMON_TREE_DRAFT_ACTION_FORCE_MINIMUM;

common_tree_draft_scheduler_status common_tree_draft_action_validate(
        const common_tree_draft_expansion_action & action) {
    if (action.request_id == 0 || action.child_budget == 0 || !std::isfinite(action.utility) ||
        (action.flags & ~COMMON_TREE_DRAFT_ACTION_FLAG_MASK) != 0) {
        return COMMON_TREE_DRAFT_SCHEDULER_INVALID_ACTION;
    }
    return COMMON_TREE_DRAFT_SCHEDULER_OK;
}

common_tree_draft_scheduler_status common_tree_draft_fair_select(
        const common_tree_draft_expansion_action * actions,
        const double * costs,
        size_t action_count,
        common_tree_draft_fair_request_state * requests,
        size_t request_count,
        double * virtual_time,
        size_t * selected_action) {
    if (virtual_time == nullptr || selected_action == nullptr ||
        (action_count > 0 && (actions == nullptr || costs == nullptr)) ||
        (request_count > 0 && requests == nullptr)) {
        return COMMON_TREE_DRAFT_SCHEDULER_NULL_BUFFER;
    }
    if (action_count == 0) return COMMON_TREE_DRAFT_SCHEDULER_EMPTY;
    if (!std::isfinite(*virtual_time) || *virtual_time < 0.0) return COMMON_TREE_DRAFT_SCHEDULER_NUMERIC;

    size_t best_action = SIZE_MAX;
    size_t best_request = SIZE_MAX;
    double best_finish = std::numeric_limits<double>::infinity();

    for (size_t i = 0; i < action_count; ++i) {
        if (common_tree_draft_action_validate(actions[i]) != COMMON_TREE_DRAFT_SCHEDULER_OK) {
            return COMMON_TREE_DRAFT_SCHEDULER_INVALID_ACTION;
        }
        if (!(costs[i] >= 0.0) || !std::isfinite(costs[i])) {
            return COMMON_TREE_DRAFT_SCHEDULER_INVALID_COST;
        }
        size_t request_index = SIZE_MAX;
        for (size_t r = 0; r < request_count; ++r) {
            if (requests[r].request_id == actions[i].request_id) {
                request_index = r;
                break;
            }
        }
        if (request_index == SIZE_MAX) return COMMON_TREE_DRAFT_SCHEDULER_REQUEST_NOT_FOUND;
        const auto & request = requests[request_index];
        if (!(request.weight > 0.0) || !std::isfinite(request.weight) ||
            !(request.virtual_finish >= 0.0) || !std::isfinite(request.virtual_finish)) {
            return COMMON_TREE_DRAFT_SCHEDULER_INVALID_WEIGHT;
        }
        const double finish = std::max(request.virtual_finish, *virtual_time) + costs[i] / request.weight;
        if (!std::isfinite(finish)) return COMMON_TREE_DRAFT_SCHEDULER_NUMERIC;
        bool better = finish < best_finish;
        if (finish == best_finish && best_action != SIZE_MAX) {
            if (actions[i].request_id != actions[best_action].request_id) {
                better = actions[i].request_id < actions[best_action].request_id;
            } else if (actions[i].node_index != actions[best_action].node_index) {
                better = actions[i].node_index < actions[best_action].node_index;
            } else {
                better = i < best_action;
            }
        }
        if (better || best_action == SIZE_MAX) {
            best_action = i;
            best_request = request_index;
            best_finish = finish;
        }
    }

    if (best_action == SIZE_MAX) return COMMON_TREE_DRAFT_SCHEDULER_EMPTY;
    requests[best_request].virtual_finish = best_finish;
    *virtual_time = best_finish;
    *selected_action = best_action;
    return COMMON_TREE_DRAFT_SCHEDULER_OK;
}

