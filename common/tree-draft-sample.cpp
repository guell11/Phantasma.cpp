#include "tree-draft-sample.h"

#include <cmath>

common_tree_draft_sample_status common_tree_draft_sample_one(
        const common_tree_draft_candidate * candidates,
        size_t n_candidates,
        uint64_t seed,
        uint64_t request_id,
        uint64_t path_id,
        uint32_t depth,
        uint32_t draw_ordinal,
        common_tree_draft_sample_result * result) {
    if (result == nullptr || (n_candidates > 0 && candidates == nullptr)) {
        return COMMON_TREE_DRAFT_SAMPLE_NULL_BUFFER;
    }
    if (n_candidates == 0) {
        return COMMON_TREE_DRAFT_SAMPLE_EMPTY;
    }

    double total = 0.0;
    for (size_t i = 0; i < n_candidates; ++i) {
        if (!std::isfinite(candidates[i].probability) || candidates[i].probability < 0.0f) {
            return COMMON_TREE_DRAFT_SAMPLE_INVALID_PROBABILITY;
        }
        total += candidates[i].probability;
    }
    if (!(total > 0.0) || !std::isfinite(total)) {
        return COMMON_TREE_DRAFT_SAMPLE_INVALID_PROBABILITY;
    }

    float u = 0.0f;
    common_tree_draft_rng_address address = {};
    if (common_tree_draft_sampling_rng_uniform(
                seed, request_id, path_id, depth, draw_ordinal, &u, &address) != COMMON_TREE_DRAFT_SAMPLING_RNG_OK) {
        return COMMON_TREE_DRAFT_SAMPLE_RNG;
    }

    double cumulative = 0.0;
    size_t selected = n_candidates - 1;
    for (size_t i = 0; i < n_candidates; ++i) {
        const double q = static_cast<double>(candidates[i].probability) / total;
        cumulative += q;
        if (i + 1 == n_candidates) {
            cumulative = 1.0;
        }
        if (static_cast<double>(u) < cumulative) {
            selected = i;
            break;
        }
    }

    const float q = static_cast<float>(static_cast<double>(candidates[selected].probability) / total);
    if (!(q > 0.0f) || !std::isfinite(q)) {
        return COMMON_TREE_DRAFT_SAMPLE_INVALID_PROBABILITY;
    }
    *result = {
        candidates[selected].token,
        q,
        std::log(q),
        address,
    };
    return COMMON_TREE_DRAFT_SAMPLE_OK;
}

