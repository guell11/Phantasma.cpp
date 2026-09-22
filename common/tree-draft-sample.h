#pragma once

#include "tree-draft-sampling-rng.h"
#include "tree-draft-top-p.h"

#include <cstddef>
#include <cstdint>

struct common_tree_draft_sample_result {
    int32_t token;
    float probability;
    float log_probability;
    common_tree_draft_rng_address rng_address;
};

enum common_tree_draft_sample_status : uint32_t {
    COMMON_TREE_DRAFT_SAMPLE_OK = 0,
    COMMON_TREE_DRAFT_SAMPLE_NULL_BUFFER,
    COMMON_TREE_DRAFT_SAMPLE_EMPTY,
    COMMON_TREE_DRAFT_SAMPLE_INVALID_PROBABILITY,
    COMMON_TREE_DRAFT_SAMPLE_RNG,
};

common_tree_draft_sample_status common_tree_draft_sample_one(
        const common_tree_draft_candidate * candidates,
        size_t n_candidates,
        uint64_t seed,
        uint64_t request_id,
        uint64_t path_id,
        uint32_t depth,
        uint32_t draw_ordinal,
        common_tree_draft_sample_result * result);

