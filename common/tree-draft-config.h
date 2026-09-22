#pragma once

#include <cstdint>
#include <optional>

struct common_tree_draft_request_params {
    uint32_t tree_budget   = 0;
    uint32_t max_depth     = 0;
    uint32_t candidate_cap = 0;
    float    temperature   = 0.0f;
    uint64_t seed          = 0;
};

struct common_tree_draft_limits {
    uint32_t tree_budget   = 0;
    uint32_t max_depth     = 0;
    uint32_t candidate_cap = 0;
};

enum class common_tree_draft_config_error {
    none,
    tree_budget,
    max_depth,
    candidate_cap,
    temperature,
};

class common_tree_draft_config {
public:
    common_tree_draft_config(const common_tree_draft_config &) = default;
    common_tree_draft_config(common_tree_draft_config &&) = default;

    common_tree_draft_config & operator=(const common_tree_draft_config &) = delete;
    common_tree_draft_config & operator=(common_tree_draft_config &&) = delete;

    static std::optional<common_tree_draft_config> admit(
            const common_tree_draft_request_params & params,
            const common_tree_draft_limits & limits,
            common_tree_draft_config_error * error = nullptr);

    uint32_t tree_budget() const;
    uint32_t max_depth() const;
    uint32_t candidate_cap() const;
    float temperature() const;
    uint64_t seed() const;

    const common_tree_draft_limits & limits() const;

private:
    common_tree_draft_config(
            const common_tree_draft_request_params & params,
            const common_tree_draft_limits & limits);

    const common_tree_draft_request_params params_;
    const common_tree_draft_limits limits_;
};
