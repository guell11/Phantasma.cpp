#include "tree-draft-config.h"

#include <cassert>
#include <cstdint>
#include <limits>
#include <type_traits>

static const common_tree_draft_limits limits = {
    /* .tree_budget   = */ 64,
    /* .max_depth     = */ 8,
    /* .candidate_cap = */ 16,
};

static common_tree_draft_request_params valid_params() {
    return {
        /* .tree_budget   = */ 32,
        /* .max_depth     = */ 4,
        /* .candidate_cap = */ 8,
        /* .temperature   = */ 0.75f,
        /* .seed          = */ UINT64_MAX,
    };
}

static void test_valid_config_is_immutable_and_preserves_values() {
    static_assert(!std::is_default_constructible<common_tree_draft_config>::value, "config must require admission");
    static_assert(!std::is_copy_assignable<common_tree_draft_config>::value, "admitted config must be immutable");
    static_assert(!std::is_move_assignable<common_tree_draft_config>::value, "admitted config must be immutable");

    common_tree_draft_config_error error = common_tree_draft_config_error::temperature;
    auto config = common_tree_draft_config::admit(valid_params(), limits, &error);

    assert(config.has_value());
    assert(error == common_tree_draft_config_error::none);
    assert(config->tree_budget() == 32);
    assert(config->max_depth() == 4);
    assert(config->candidate_cap() == 8);
    assert(config->temperature() == 0.75f);
    assert(config->seed() == UINT64_MAX);
    assert(config->limits().tree_budget == 64);
    assert(config->limits().max_depth == 8);
    assert(config->limits().candidate_cap == 16);
}

static void expect_rejected(
        const common_tree_draft_request_params & params,
        common_tree_draft_config_error expected) {
    common_tree_draft_config_error error = common_tree_draft_config_error::none;
    auto config = common_tree_draft_config::admit(params, limits, &error);
    assert(!config.has_value());
    assert(error == expected);
}

static void test_rejects_zero_values() {
    auto params = valid_params();
    params.tree_budget = 0;
    expect_rejected(params, common_tree_draft_config_error::tree_budget);

    params = valid_params();
    params.max_depth = 0;
    expect_rejected(params, common_tree_draft_config_error::max_depth);

    params = valid_params();
    params.candidate_cap = 0;
    expect_rejected(params, common_tree_draft_config_error::candidate_cap);

    params = valid_params();
    params.temperature = 0.0f;
    expect_rejected(params, common_tree_draft_config_error::temperature);
}

static void test_rejects_device_limit_violations() {
    auto params = valid_params();
    params.tree_budget = limits.tree_budget + 1;
    expect_rejected(params, common_tree_draft_config_error::tree_budget);

    params = valid_params();
    params.max_depth = limits.max_depth + 1;
    expect_rejected(params, common_tree_draft_config_error::max_depth);

    params = valid_params();
    params.candidate_cap = limits.candidate_cap + 1;
    expect_rejected(params, common_tree_draft_config_error::candidate_cap);
}

static void test_rejects_non_finite_temperature() {
    auto params = valid_params();
    params.temperature = std::numeric_limits<float>::infinity();
    expect_rejected(params, common_tree_draft_config_error::temperature);

    params.temperature = std::numeric_limits<float>::quiet_NaN();
    expect_rejected(params, common_tree_draft_config_error::temperature);
}

int main() {
    test_valid_config_is_immutable_and_preserves_values();
    test_rejects_zero_values();
    test_rejects_device_limit_violations();
    test_rejects_non_finite_temperature();
    return 0;
}
