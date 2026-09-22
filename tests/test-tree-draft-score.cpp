#include "tree-draft-score.h"

#include <cassert>
#include <cmath>

static common_tree_draft_config config() {
    const common_tree_draft_request_params params = { 8, 4, 4, 1.0f, 1 };
    const common_tree_draft_limits limits = { 8, 4, 4 };
    auto admitted = common_tree_draft_config::admit(params, limits);
    assert(admitted.has_value());
    return *admitted;
}

int main() {
    const common_tree_draft_node nodes[] = {
        { -1, -1, 0, 0.0f, 1, 0 },
        { 0, 10, 1, std::log(0.5f), 2, 0 },
        { 0, 20, 1, std::log(0.25f), 3, 0 },
        { 1, 30, 2, std::log(0.2f), 4, 0 },
    };
    double scores[4] = {};
    assert(common_tree_draft_score_cumulative_logp(nodes, 4, config(), 1e-6f, scores, 4) == COMMON_TREE_DRAFT_SCORE_OK);
    assert(scores[0] == 0.0);
    assert(std::fabs(scores[1] - std::log(0.5)) < 1e-6);
    assert(std::fabs(scores[2] - std::log(0.25)) < 1e-6);
    assert(std::fabs(scores[3] - (std::log(0.5) + std::log(0.2))) < 1e-6);

    common_tree_draft_node floored[] = {
        nodes[0],
        { 0, 10, 1, -1000.0f, 2, 0 },
    };
    double floor_scores[2] = {};
    assert(common_tree_draft_score_cumulative_logp(floored, 2, config(), 1e-3f, floor_scores, 2) == COMMON_TREE_DRAFT_SCORE_OK);
    assert(std::fabs(floor_scores[1] - std::log(1e-3)) < 1e-5);
    assert(common_tree_draft_score_cumulative_logp(nodes, 4, config(), 0.0f, scores, 4) == COMMON_TREE_DRAFT_SCORE_EPSILON);
    assert(common_tree_draft_score_cumulative_logp(nodes, 4, config(), 1e-6f, scores, 3) == COMMON_TREE_DRAFT_SCORE_NULL_BUFFER);
    return 0;
}

