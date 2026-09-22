#pragma once

#include "tree-draft-sample.h"

#include <cstddef>
#include <cstdint>

struct common_tree_draft_child_sample {
    int32_t token;
    float conditional_probability;
    float log_conditional_probability;
    uint32_t draw_ordinal;
    common_tree_draft_rng_address rng_address;
};

enum common_tree_draft_multi_sample_status : uint32_t {
    COMMON_TREE_DRAFT_MULTI_SAMPLE_OK = 0,
    COMMON_TREE_DRAFT_MULTI_SAMPLE_NULL_BUFFER,
    COMMON_TREE_DRAFT_MULTI_SAMPLE_COUNT_RANGE,
    COMMON_TREE_DRAFT_MULTI_SAMPLE_DUPLICATE_TOKEN,
    COMMON_TREE_DRAFT_MULTI_SAMPLE_INVALID_PROBABILITY,
    COMMON_TREE_DRAFT_MULTI_SAMPLE_RNG,
};

common_tree_draft_multi_sample_status common_tree_draft_sample_without_replacement(
        const common_tree_draft_candidate * candidates,
        size_t n_candidates,
        size_t child_count,
        uint64_t seed,
        uint64_t request_id,
        uint64_t parent_path_id,
        uint32_t child_depth,
        common_tree_draft_child_sample * out,
        size_t out_capacity,
        size_t * out_count);

