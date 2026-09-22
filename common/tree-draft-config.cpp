#include "tree-draft-config.h"

#include <cmath>

std::optional<common_tree_draft_config> common_tree_draft_config::admit(
        const common_tree_draft_request_params & params,
        const common_tree_draft_limits & limits,
        common_tree_draft_config_error * error) {
    const auto reject = [error](common_tree_draft_config_error value) {
        if (error != nullptr) {
            *error = value;
        }
        return std::optional<common_tree_draft_config>{};
    };

    if (params.tree_budget == 0 || params.tree_budget > limits.tree_budget) {
        return reject(common_tree_draft_config_error::tree_budget);
    }
    if (params.max_depth == 0 || params.max_depth > limits.max_depth) {
        return reject(common_tree_draft_config_error::max_depth);
    }
    if (params.candidate_cap == 0 || params.candidate_cap > limits.candidate_cap) {
        return reject(common_tree_draft_config_error::candidate_cap);
    }
    if (!(params.temperature > 0.0f) || !std::isfinite(params.temperature)) {
        return reject(common_tree_draft_config_error::temperature);
    }

    if (error != nullptr) {
        *error = common_tree_draft_config_error::none;
    }
    return common_tree_draft_config(params, limits);
}

common_tree_draft_config::common_tree_draft_config(
        const common_tree_draft_request_params & params,
        const common_tree_draft_limits & limits)
    : params_(params), limits_(limits) {
}

uint32_t common_tree_draft_config::tree_budget() const {
    return params_.tree_budget;
}

uint32_t common_tree_draft_config::max_depth() const {
    return params_.max_depth;
}

uint32_t common_tree_draft_config::candidate_cap() const {
    return params_.candidate_cap;
}

float common_tree_draft_config::temperature() const {
    return params_.temperature;
}

uint64_t common_tree_draft_config::seed() const {
    return params_.seed;
}

const common_tree_draft_limits & common_tree_draft_config::limits() const {
    return limits_;
}
