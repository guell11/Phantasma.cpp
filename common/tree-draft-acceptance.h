#pragma once

#include "tree-draft-node.h"

#include <cstddef>
#include <cstdint>
#include <vector>

using common_tree_draft_acceptance_proxy_fn = double (*)(double draft_probability, void * user_data);

struct common_tree_draft_acceptance_proxy {
    common_tree_draft_acceptance_proxy_fn fn;
    void * user_data;
};

struct common_tree_draft_acceptance_calibration {
    std::vector<uint64_t> accepted;
    std::vector<uint64_t> total;
    double alpha;
    double beta;
};

struct common_tree_draft_acceptance_gain {
    double expected_prefix;
    double marginal_gain;
};

struct common_tree_draft_expansion_cost {
    double compute;
    double memory;
};

struct common_tree_draft_expansion_utility_params {
    double lambda_compute;
    double lambda_memory;
};

enum common_tree_draft_acceptance_status : uint32_t {
    COMMON_TREE_DRAFT_ACCEPTANCE_OK = 0,
    COMMON_TREE_DRAFT_ACCEPTANCE_NULL_BUFFER,
    COMMON_TREE_DRAFT_ACCEPTANCE_PROBABILITY,
    COMMON_TREE_DRAFT_ACCEPTANCE_CONFIG,
    COMMON_TREE_DRAFT_ACCEPTANCE_RANGE,
    COMMON_TREE_DRAFT_ACCEPTANCE_OVERFLOW,
    COMMON_TREE_DRAFT_ACCEPTANCE_NUMERIC,
};

common_tree_draft_acceptance_status common_tree_draft_acceptance_proxy_probability(
        double draft_probability,
        const common_tree_draft_acceptance_proxy & proxy,
        double * estimate);

common_tree_draft_acceptance_status common_tree_draft_acceptance_calibration_init(
        common_tree_draft_acceptance_calibration * calibration,
        size_t buckets,
        double alpha,
        double beta);

common_tree_draft_acceptance_status common_tree_draft_acceptance_calibration_update(
        common_tree_draft_acceptance_calibration * calibration,
        double draft_probability,
        bool accepted);

common_tree_draft_acceptance_status common_tree_draft_acceptance_calibration_lookup(
        const common_tree_draft_acceptance_calibration & calibration,
        double draft_probability,
        const common_tree_draft_acceptance_proxy & cold_proxy,
        double * estimate,
        bool * calibrated = nullptr);

common_tree_draft_acceptance_status common_tree_draft_acceptance_branch_scores(
        const common_tree_draft_node * nodes,
        size_t n_nodes,
        const common_tree_draft_config & config,
        const common_tree_draft_acceptance_calibration & calibration,
        const common_tree_draft_acceptance_proxy & cold_proxy,
        double epsilon,
        double * scores,
        size_t out_capacity);

common_tree_draft_acceptance_status common_tree_draft_expected_prefix_gain(
        const double * acceptance_probabilities,
        size_t depth,
        common_tree_draft_acceptance_gain * gain);

common_tree_draft_acceptance_status common_tree_draft_expansion_utility(
        double acceptance_gain,
        common_tree_draft_expansion_cost cost,
        common_tree_draft_expansion_utility_params params,
        double * utility);

