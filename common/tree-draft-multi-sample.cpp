#include "tree-draft-multi-sample.h"

#include <array>
#include <cmath>
#include <vector>

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
        size_t * out_count) {
    if (out_count == nullptr || (n_candidates > 0 && candidates == nullptr) ||
        (out_capacity > 0 && out == nullptr)) {
        return COMMON_TREE_DRAFT_MULTI_SAMPLE_NULL_BUFFER;
    }
    *out_count = 0;
    if (child_count > n_candidates || out_capacity < child_count || child_count > UINT32_MAX) {
        return COMMON_TREE_DRAFT_MULTI_SAMPLE_COUNT_RANGE;
    }
    if (child_count == 0) {
        return COMMON_TREE_DRAFT_MULTI_SAMPLE_OK;
    }

    static constexpr size_t STACK_CANDIDATES = 64;
    static constexpr size_t TOKEN_HASH_SLOTS = 128;
    std::array<int32_t, TOKEN_HASH_SLOTS> token_hash = {};
    std::array<uint8_t, TOKEN_HASH_SLOTS> token_used = {};

    for (size_t i = 0; i < n_candidates; ++i) {
        if (!std::isfinite(candidates[i].probability) || candidates[i].probability < 0.0f) {
            return COMMON_TREE_DRAFT_MULTI_SAMPLE_INVALID_PROBABILITY;
        }
        if (n_candidates <= STACK_CANDIDATES) {
            uint32_t slot = static_cast<uint32_t>(candidates[i].token) * UINT32_C(0x9e3779b1);
            slot &= static_cast<uint32_t>(TOKEN_HASH_SLOTS - 1);
            while (token_used[slot]) {
                if (token_hash[slot] == candidates[i].token) {
                    return COMMON_TREE_DRAFT_MULTI_SAMPLE_DUPLICATE_TOKEN;
                }
                slot = (slot + 1) & static_cast<uint32_t>(TOKEN_HASH_SLOTS - 1);
            }
            token_used[slot] = 1;
            token_hash[slot] = candidates[i].token;
        } else {
            for (size_t j = 0; j < i; ++j) {
                if (candidates[j].token == candidates[i].token) {
                    return COMMON_TREE_DRAFT_MULTI_SAMPLE_DUPLICATE_TOKEN;
                }
            }
        }
    }

    // The speculative hot path overwhelmingly uses small top-k sets. Keep
    // their residual probabilities on the stack so every parent expansion
    // does not pay a heap allocation. Larger candidate sets retain the
    // general heap fallback.
    std::array<float, STACK_CANDIDATES> residual_stack = {};
    std::vector<float> residual_heap;
    float * residual = nullptr;
    if (n_candidates <= STACK_CANDIDATES) {
        residual = residual_stack.data();
    } else {
        residual_heap.resize(n_candidates);
        residual = residual_heap.data();
    }

    double residual_mass = 0.0;
    for (size_t i = 0; i < n_candidates; ++i) {
        residual[i] = candidates[i].probability;
        residual_mass += residual[i];
    }

    for (size_t draw = 0; draw < child_count; ++draw) {
        if (!(residual_mass > 0.0) || !std::isfinite(residual_mass)) {
            break;
        }

        float u = 0.0f;
        common_tree_draft_rng_address address = {};
        if (common_tree_draft_sampling_rng_uniform(
                    seed, request_id, parent_path_id, child_depth, static_cast<uint32_t>(draw),
                    &u, &address) != COMMON_TREE_DRAFT_SAMPLING_RNG_OK) {
            return COMMON_TREE_DRAFT_MULTI_SAMPLE_RNG;
        }

        double cumulative = 0.0;
        size_t selected = n_candidates;
        size_t last_positive = n_candidates;
        for (size_t i = 0; i < n_candidates; ++i) {
            if (residual[i] <= 0.0f) {
                continue;
            }
            last_positive = i;
            cumulative += static_cast<double>(residual[i]) / residual_mass;
            if (static_cast<double>(u) < cumulative) {
                selected = i;
                break;
            }
        }
        // Match sample_one's rounding-safe final-bin behavior while avoiding a
        // second token search through the candidate list.
        if (selected == n_candidates) {
            selected = last_positive;
        }
        if (selected == n_candidates || residual[selected] <= 0.0f) {
            return COMMON_TREE_DRAFT_MULTI_SAMPLE_INVALID_PROBABILITY;
        }

        const float conditional_probability =
                static_cast<float>(static_cast<double>(residual[selected]) / residual_mass);
        if (!(conditional_probability > 0.0f) || !std::isfinite(conditional_probability)) {
            return COMMON_TREE_DRAFT_MULTI_SAMPLE_INVALID_PROBABILITY;
        }
        out[*out_count] = {
            candidates[selected].token,
            conditional_probability,
            std::log(conditional_probability),
            static_cast<uint32_t>(draw),
            address,
        };
        ++*out_count;
        residual_mass -= residual[selected];
        residual[selected] = 0.0f;
    }
    return COMMON_TREE_DRAFT_MULTI_SAMPLE_OK;
}
