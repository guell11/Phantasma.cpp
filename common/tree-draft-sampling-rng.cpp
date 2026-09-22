#include "tree-draft-sampling-rng.h"

common_tree_draft_sampling_rng_status common_tree_draft_sampling_rng_address(
        uint64_t seed,
        uint64_t request_id,
        uint64_t path_id,
        uint32_t depth,
        uint32_t draw_ordinal,
        common_tree_draft_rng_address * address) {
    if (address == nullptr) {
        return COMMON_TREE_DRAFT_SAMPLING_RNG_NULL_OUTPUT;
    }
    if (path_id == 0) {
        return COMMON_TREE_DRAFT_SAMPLING_RNG_INVALID_PATH;
    }

    *address = {
        seed,
        request_id,
        (static_cast<uint64_t>(COMMON_TREE_DRAFT_RNG_DOMAIN_DRAFT) << 32) | depth,
        path_id,
        draw_ordinal,
    };
    return COMMON_TREE_DRAFT_SAMPLING_RNG_OK;
}

common_tree_draft_sampling_rng_status common_tree_draft_sampling_rng_uniform(
        uint64_t seed,
        uint64_t request_id,
        uint64_t path_id,
        uint32_t depth,
        uint32_t draw_ordinal,
        float * value,
        common_tree_draft_rng_address * address) {
    if (value == nullptr) {
        return COMMON_TREE_DRAFT_SAMPLING_RNG_NULL_OUTPUT;
    }

    common_tree_draft_rng_address local = {};
    const auto status = common_tree_draft_sampling_rng_address(
            seed, request_id, path_id, depth, draw_ordinal, &local);
    if (status != COMMON_TREE_DRAFT_SAMPLING_RNG_OK) {
        return status;
    }
    *value = common_tree_draft_rng_uniform_f32(local);
    if (address != nullptr) {
        *address = local;
    }
    return COMMON_TREE_DRAFT_SAMPLING_RNG_OK;
}

