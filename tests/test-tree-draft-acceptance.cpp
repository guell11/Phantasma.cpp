#include "tree-draft-acceptance.h"

#include <cassert>
#include <cmath>

static double square_proxy(double p, void *) { return p * p; }

static common_tree_draft_config config() {
    const common_tree_draft_request_params params = { 8, 4, 4, 1.0f, 1 };
    const common_tree_draft_limits limits = { 8, 4, 4 };
    auto admitted = common_tree_draft_config::admit(params, limits);
    assert(admitted.has_value());
    return *admitted;
}

int main() {
    double estimate = 0.0;
    const common_tree_draft_acceptance_proxy identity = { nullptr, nullptr };
    const common_tree_draft_acceptance_proxy square = { square_proxy, nullptr };
    assert(common_tree_draft_acceptance_proxy_probability(0.5, identity, &estimate) == COMMON_TREE_DRAFT_ACCEPTANCE_OK && estimate == 0.5);
    assert(common_tree_draft_acceptance_proxy_probability(0.5, square, &estimate) == COMMON_TREE_DRAFT_ACCEPTANCE_OK && estimate == 0.25);

    common_tree_draft_acceptance_calibration calibration = {};
    assert(common_tree_draft_acceptance_calibration_init(&calibration, 4, 1.0, 1.0) == COMMON_TREE_DRAFT_ACCEPTANCE_OK);
    bool calibrated = true;
    assert(common_tree_draft_acceptance_calibration_lookup(calibration, 0.75, identity, &estimate, &calibrated) == COMMON_TREE_DRAFT_ACCEPTANCE_OK);
    assert(!calibrated && estimate == 0.75);
    assert(common_tree_draft_acceptance_calibration_update(&calibration, 0.75, true) == COMMON_TREE_DRAFT_ACCEPTANCE_OK);
    assert(common_tree_draft_acceptance_calibration_update(&calibration, 0.75, false) == COMMON_TREE_DRAFT_ACCEPTANCE_OK);
    assert(common_tree_draft_acceptance_calibration_lookup(calibration, 0.75, identity, &estimate, &calibrated) == COMMON_TREE_DRAFT_ACCEPTANCE_OK);
    assert(calibrated && std::fabs(estimate - 0.5) < 1e-12);

    const common_tree_draft_node nodes[] = {
        { -1, -1, 0, 0.0f, 1, 0 },
        { 0, 10, 1, std::log(0.75f), 2, 0 },
        { 1, 11, 2, std::log(0.25f), 3, 0 },
    };
    double scores[3] = {};
    assert(common_tree_draft_acceptance_branch_scores(nodes, 3, config(), calibration, identity, 1e-6, scores, 3) == COMMON_TREE_DRAFT_ACCEPTANCE_OK);
    assert(scores[0] == 0.0);
    assert(std::fabs(scores[1] - std::log(0.5)) < 1e-6);
    assert(std::fabs(scores[2] - (std::log(0.5) + std::log(0.25))) < 1e-5);

    const double q[] = { 0.8, 0.5, 0.25 };
    common_tree_draft_acceptance_gain gain = {};
    assert(common_tree_draft_expected_prefix_gain(q, 3, &gain) == COMMON_TREE_DRAFT_ACCEPTANCE_OK);
    assert(std::fabs(gain.expected_prefix - (0.8 + 0.4 + 0.1)) < 1e-12);
    assert(std::fabs(gain.marginal_gain - 0.1) < 1e-12);

    double utility = 0.0;
    assert(common_tree_draft_expansion_utility(gain.expected_prefix, {0.5, 0.25}, {0.2, 0.4}, &utility) == COMMON_TREE_DRAFT_ACCEPTANCE_OK);
    assert(std::fabs(utility - (1.3 - 0.1 - 0.1)) < 1e-12);
    return 0;
}

