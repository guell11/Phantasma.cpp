#include "tree-draft-acceptance.h"

#include <algorithm>
#include <cmath>
#include <limits>

static bool common_tree_draft_probability_valid(double p) {
    return std::isfinite(p) && p >= 0.0 && p <= 1.0;
}

common_tree_draft_acceptance_status common_tree_draft_acceptance_proxy_probability(
        double draft_probability,
        const common_tree_draft_acceptance_proxy & proxy,
        double * estimate) {
    if (estimate == nullptr) return COMMON_TREE_DRAFT_ACCEPTANCE_NULL_BUFFER;
    if (!common_tree_draft_probability_valid(draft_probability)) return COMMON_TREE_DRAFT_ACCEPTANCE_PROBABILITY;
    const double value = proxy.fn == nullptr ? draft_probability : proxy.fn(draft_probability, proxy.user_data);
    if (!common_tree_draft_probability_valid(value)) return COMMON_TREE_DRAFT_ACCEPTANCE_PROBABILITY;
    *estimate = value;
    return COMMON_TREE_DRAFT_ACCEPTANCE_OK;
}

common_tree_draft_acceptance_status common_tree_draft_acceptance_calibration_init(
        common_tree_draft_acceptance_calibration * calibration,
        size_t buckets,
        double alpha,
        double beta) {
    if (calibration == nullptr) return COMMON_TREE_DRAFT_ACCEPTANCE_NULL_BUFFER;
    if (buckets == 0 || !(alpha > 0.0) || !(beta > 0.0) || !std::isfinite(alpha) || !std::isfinite(beta)) {
        return COMMON_TREE_DRAFT_ACCEPTANCE_CONFIG;
    }
    calibration->accepted.assign(buckets, 0);
    calibration->total.assign(buckets, 0);
    calibration->alpha = alpha;
    calibration->beta = beta;
    return COMMON_TREE_DRAFT_ACCEPTANCE_OK;
}

static size_t common_tree_draft_acceptance_bucket(const common_tree_draft_acceptance_calibration & c, double p) {
    const size_t n = c.total.size();
    if (p >= 1.0) return n - 1;
    return std::min(n - 1, static_cast<size_t>(p * static_cast<double>(n)));
}

common_tree_draft_acceptance_status common_tree_draft_acceptance_calibration_update(
        common_tree_draft_acceptance_calibration * calibration,
        double draft_probability,
        bool was_accepted) {
    if (calibration == nullptr) return COMMON_TREE_DRAFT_ACCEPTANCE_NULL_BUFFER;
    if (!common_tree_draft_probability_valid(draft_probability)) return COMMON_TREE_DRAFT_ACCEPTANCE_PROBABILITY;
    if (calibration->total.empty() || calibration->accepted.size() != calibration->total.size()) return COMMON_TREE_DRAFT_ACCEPTANCE_CONFIG;
    const size_t bucket = common_tree_draft_acceptance_bucket(*calibration, draft_probability);
    if (calibration->total[bucket] == UINT64_MAX || (was_accepted && calibration->accepted[bucket] == UINT64_MAX)) {
        return COMMON_TREE_DRAFT_ACCEPTANCE_OVERFLOW;
    }
    ++calibration->total[bucket];
    if (was_accepted) ++calibration->accepted[bucket];
    return COMMON_TREE_DRAFT_ACCEPTANCE_OK;
}

common_tree_draft_acceptance_status common_tree_draft_acceptance_calibration_lookup(
        const common_tree_draft_acceptance_calibration & calibration,
        double draft_probability,
        const common_tree_draft_acceptance_proxy & cold_proxy,
        double * estimate,
        bool * calibrated) {
    if (estimate == nullptr) return COMMON_TREE_DRAFT_ACCEPTANCE_NULL_BUFFER;
    if (!common_tree_draft_probability_valid(draft_probability)) return COMMON_TREE_DRAFT_ACCEPTANCE_PROBABILITY;
    if (calibration.total.empty() || calibration.accepted.size() != calibration.total.size() ||
        !(calibration.alpha > 0.0) || !(calibration.beta > 0.0)) return COMMON_TREE_DRAFT_ACCEPTANCE_CONFIG;
    const size_t bucket = common_tree_draft_acceptance_bucket(calibration, draft_probability);
    if (calibration.total[bucket] == 0) {
        if (calibrated != nullptr) *calibrated = false;
        return common_tree_draft_acceptance_proxy_probability(draft_probability, cold_proxy, estimate);
    }
    const double value = (static_cast<double>(calibration.accepted[bucket]) + calibration.alpha) /
            (static_cast<double>(calibration.total[bucket]) + calibration.alpha + calibration.beta);
    if (!common_tree_draft_probability_valid(value)) return COMMON_TREE_DRAFT_ACCEPTANCE_NUMERIC;
    *estimate = value;
    if (calibrated != nullptr) *calibrated = true;
    return COMMON_TREE_DRAFT_ACCEPTANCE_OK;
}

common_tree_draft_acceptance_status common_tree_draft_acceptance_branch_scores(
        const common_tree_draft_node * nodes,
        size_t n_nodes,
        const common_tree_draft_config & config,
        const common_tree_draft_acceptance_calibration & calibration,
        const common_tree_draft_acceptance_proxy & cold_proxy,
        double epsilon,
        double * scores,
        size_t out_capacity) {
    if ((n_nodes > 0 && (nodes == nullptr || scores == nullptr)) || out_capacity < n_nodes) return COMMON_TREE_DRAFT_ACCEPTANCE_NULL_BUFFER;
    if (!(epsilon > 0.0) || epsilon > 1.0 || !std::isfinite(epsilon)) return COMMON_TREE_DRAFT_ACCEPTANCE_CONFIG;
    if (common_tree_draft_nodes_validate(nodes, n_nodes, config) != COMMON_TREE_DRAFT_NODE_OK) return COMMON_TREE_DRAFT_ACCEPTANCE_RANGE;
    if (n_nodes == 0) return COMMON_TREE_DRAFT_ACCEPTANCE_OK;
    scores[0] = 0.0;
    for (size_t i = 1; i < n_nodes; ++i) {
        const double draft_p = std::exp(static_cast<double>(nodes[i].logp));
        if (!common_tree_draft_probability_valid(draft_p)) return COMMON_TREE_DRAFT_ACCEPTANCE_NUMERIC;
        double q = 0.0;
        const auto status = common_tree_draft_acceptance_calibration_lookup(calibration, draft_p, cold_proxy, &q);
        if (status != COMMON_TREE_DRAFT_ACCEPTANCE_OK) return status;
        scores[i] = scores[nodes[i].parent] + std::log(std::max(q, epsilon));
        if (!std::isfinite(scores[i])) return COMMON_TREE_DRAFT_ACCEPTANCE_NUMERIC;
    }
    return COMMON_TREE_DRAFT_ACCEPTANCE_OK;
}

common_tree_draft_acceptance_status common_tree_draft_expected_prefix_gain(
        const double * acceptance_probabilities,
        size_t depth,
        common_tree_draft_acceptance_gain * gain) {
    if (gain == nullptr || (depth > 0 && acceptance_probabilities == nullptr)) return COMMON_TREE_DRAFT_ACCEPTANCE_NULL_BUFFER;
    double prefix = 1.0;
    double expected = 0.0;
    for (size_t i = 0; i < depth; ++i) {
        if (!common_tree_draft_probability_valid(acceptance_probabilities[i])) return COMMON_TREE_DRAFT_ACCEPTANCE_PROBABILITY;
        prefix *= acceptance_probabilities[i];
        expected += prefix;
    }
    if (!std::isfinite(expected) || !std::isfinite(prefix)) return COMMON_TREE_DRAFT_ACCEPTANCE_NUMERIC;
    *gain = { expected, depth == 0 ? 0.0 : prefix };
    return COMMON_TREE_DRAFT_ACCEPTANCE_OK;
}

common_tree_draft_acceptance_status common_tree_draft_expansion_utility(
        double acceptance_gain,
        common_tree_draft_expansion_cost cost,
        common_tree_draft_expansion_utility_params params,
        double * utility) {
    if (utility == nullptr) return COMMON_TREE_DRAFT_ACCEPTANCE_NULL_BUFFER;
    if (!std::isfinite(acceptance_gain) || !std::isfinite(cost.compute) || !std::isfinite(cost.memory) ||
        !std::isfinite(params.lambda_compute) || !std::isfinite(params.lambda_memory) ||
        cost.compute < 0.0 || cost.memory < 0.0 || params.lambda_compute < 0.0 || params.lambda_memory < 0.0) {
        return COMMON_TREE_DRAFT_ACCEPTANCE_CONFIG;
    }
    const double value = acceptance_gain - params.lambda_compute * cost.compute - params.lambda_memory * cost.memory;
    if (!std::isfinite(value)) return COMMON_TREE_DRAFT_ACCEPTANCE_NUMERIC;
    *utility = value;
    return COMMON_TREE_DRAFT_ACCEPTANCE_OK;
}

