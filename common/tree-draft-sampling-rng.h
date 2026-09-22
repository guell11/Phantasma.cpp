#pragma once

#include "tree-draft-rng.h"

#include <cstdint>

static constexpr uint32_t COMMON_TREE_DRAFT_RNG_DOMAIN_DRAFT = UINT32_C(0x44524654);

enum common_tree_draft_sampling_rng_status : uint32_t {
    COMMON_TREE_DRAFT_SAMPLING_RNG_OK = 0,
    COMMON_TREE_DRAFT_SAMPLING_RNG_INVALID_PATH,
    COMMON_TREE_DRAFT_SAMPLING_RNG_NULL_OUTPUT,
};

common_tree_draft_sampling_rng_status common_tree_draft_sampling_rng_address(
        uint64_t seed,
        uint64_t request_id,
        uint64_t path_id,
        uint32_t depth,
        uint32_t draw_ordinal,
        common_tree_draft_rng_address * address);

common_tree_draft_sampling_rng_status common_tree_draft_sampling_rng_uniform(
        uint64_t seed,
        uint64_t request_id,
        uint64_t path_id,
        uint32_t depth,
        uint32_t draw_ordinal,
        float * value,
        common_tree_draft_rng_address * address = nullptr);

