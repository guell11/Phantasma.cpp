#include "tree-draft-child-budget.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

struct common_tree_draft_child_budget_remainder {
    size_t index;
    double remainder;
};

common_tree_draft_child_budget_status common_tree_draft_allocate_child_budget(
        const common_tree_draft_child_budget_input * inputs,
        size_t count,
        uint32_t total_budget,
        common_tree_draft_score_mode rank_mode,
        common_tree_draft_depth_preference depth_preference,
        common_tree_draft_child_budget_output * outputs,
        size_t out_capacity) {
    if ((count > 0 && (inputs == nullptr || outputs == nullptr)) || out_capacity < count) {
        return COMMON_TREE_DRAFT_CHILD_BUDGET_NULL_BUFFER;
    }

    uint64_t mandatory_total = 0;
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(inputs[i].weight) || inputs[i].weight < 0.0) {
            return COMMON_TREE_DRAFT_CHILD_BUDGET_INVALID_WEIGHT;
        }
        if (inputs[i].minimum > inputs[i].cap) {
            return COMMON_TREE_DRAFT_CHILD_BUDGET_MINIMUM_RANGE;
        }
        mandatory_total += inputs[i].minimum;
        if (mandatory_total > std::numeric_limits<uint32_t>::max()) {
            return COMMON_TREE_DRAFT_CHILD_BUDGET_OVERFLOW;
        }
    }
    if (mandatory_total > total_budget) {
        return COMMON_TREE_DRAFT_CHILD_BUDGET_INFEASIBLE_MINIMUM;
    }

    for (size_t i = 0; i < count; ++i) {
        outputs[i] = { inputs[i].cap, inputs[i].minimum };
    }
    uint32_t remaining = total_budget - static_cast<uint32_t>(mandatory_total);

    while (remaining > 0) {
        double weight_sum = 0.0;
        size_t eligible = 0;
        for (size_t i = 0; i < count; ++i) {
            if (outputs[i].granted < inputs[i].cap) {
                weight_sum += inputs[i].weight;
                ++eligible;
            }
        }
        if (eligible == 0) {
            break;
        }

        std::vector<common_tree_draft_child_budget_remainder> remainders;
        remainders.reserve(eligible);
        uint32_t assigned_floor = 0;
        for (size_t i = 0; i < count; ++i) {
            const uint32_t room = inputs[i].cap - outputs[i].granted;
            if (room == 0) {
                continue;
            }
            const double share = weight_sum > 0.0 ?
                    static_cast<double>(remaining) * inputs[i].weight / weight_sum :
                    static_cast<double>(remaining) / static_cast<double>(eligible);
            const uint32_t floor_share = std::min<uint32_t>(room, static_cast<uint32_t>(std::floor(share)));
            outputs[i].granted += floor_share;
            assigned_floor += floor_share;
            remainders.push_back({ i, share - std::floor(share) });
        }
        if (assigned_floor > remaining) {
            return COMMON_TREE_DRAFT_CHILD_BUDGET_OVERFLOW;
        }
        remaining -= assigned_floor;
        if (remaining == 0) {
            break;
        }

        std::sort(remainders.begin(), remainders.end(), [&](const auto & a, const auto & b) {
            if (a.remainder != b.remainder) {
                return a.remainder > b.remainder;
            }
            return common_tree_draft_rank_compare(inputs[a.index].rank, inputs[b.index].rank, rank_mode, depth_preference) < 0;
        });

        uint32_t awarded = 0;
        for (const auto & entry : remainders) {
            if (remaining == 0) {
                break;
            }
            if (outputs[entry.index].granted < inputs[entry.index].cap) {
                ++outputs[entry.index].granted;
                --remaining;
                ++awarded;
            }
        }
        if (awarded == 0 && assigned_floor == 0) {
            break;
        }
    }
    return COMMON_TREE_DRAFT_CHILD_BUDGET_OK;
}

