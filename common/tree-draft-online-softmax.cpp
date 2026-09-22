#include "tree-draft-online-softmax.h"

#include <algorithm>
#include <cmath>
#include <limits>

common_tree_draft_online_softmax_status common_tree_draft_online_softmax_init(
        common_tree_draft_online_softmax_state * state,
        float * accumulator,
        size_t value_dim) {
    if (state == nullptr || (value_dim > 0 && accumulator == nullptr)) {
        return COMMON_TREE_DRAFT_ONLINE_SOFTMAX_NULL_BUFFER;
    }
    state->max_score = -std::numeric_limits<float>::infinity();
    state->normalizer = 0.0f;
    state->accumulator = accumulator;
    state->value_dim = value_dim;
    state->initialized = false;
    for (size_t d = 0; d < value_dim; ++d) accumulator[d] = 0.0f;
    return COMMON_TREE_DRAFT_ONLINE_SOFTMAX_OK;
}

common_tree_draft_online_softmax_status common_tree_draft_online_softmax_update(
        common_tree_draft_online_softmax_state * state,
        const float * scores,
        const float * values,
        const uint8_t * masked,
        size_t rows) {
    if (state == nullptr || state->accumulator == nullptr ||
        (rows > 0 && (scores == nullptr || values == nullptr))) {
        return COMMON_TREE_DRAFT_ONLINE_SOFTMAX_NULL_BUFFER;
    }
    if (rows == 0) return COMMON_TREE_DRAFT_ONLINE_SOFTMAX_OK;

    float block_max = -std::numeric_limits<float>::infinity();
    bool any = false;
    for (size_t i = 0; i < rows; ++i) {
        if (masked != nullptr && masked[i] != 0) continue;
        if (!std::isfinite(scores[i])) return COMMON_TREE_DRAFT_ONLINE_SOFTMAX_INVALID_SCORE;
        block_max = std::max(block_max, scores[i]);
        any = true;
        for (size_t d = 0; d < state->value_dim; ++d) {
            if (!std::isfinite(values[i * state->value_dim + d])) {
                return COMMON_TREE_DRAFT_ONLINE_SOFTMAX_INVALID_VALUE;
            }
        }
    }
    if (!any) return COMMON_TREE_DRAFT_ONLINE_SOFTMAX_OK;

    const float new_max = state->initialized ? std::max(state->max_score, block_max) : block_max;
    const float old_scale = state->initialized ? std::exp(state->max_score - new_max) : 0.0f;

    float new_norm = state->normalizer * old_scale;
    for (size_t d = 0; d < state->value_dim; ++d) {
        state->accumulator[d] *= old_scale;
    }

    for (size_t i = 0; i < rows; ++i) {
        if (masked != nullptr && masked[i] != 0) continue;
        const float w = std::exp(scores[i] - new_max);
        new_norm += w;
        for (size_t d = 0; d < state->value_dim; ++d) {
            state->accumulator[d] += w * values[i * state->value_dim + d];
        }
    }
    if (!(new_norm > 0.0f) || !std::isfinite(new_norm)) {
        return COMMON_TREE_DRAFT_ONLINE_SOFTMAX_INVALID_STATE;
    }
    state->max_score = new_max;
    state->normalizer = new_norm;
    state->initialized = true;
    return COMMON_TREE_DRAFT_ONLINE_SOFTMAX_OK;
}

common_tree_draft_online_softmax_status common_tree_draft_online_softmax_finalize(
        const common_tree_draft_online_softmax_state & state,
        float * output,
        size_t output_count) {
    if (output == nullptr || output_count < state.value_dim) {
        return COMMON_TREE_DRAFT_ONLINE_SOFTMAX_NULL_BUFFER;
    }
    if (!state.initialized || !(state.normalizer > 0.0f) || !std::isfinite(state.normalizer)) {
        return COMMON_TREE_DRAFT_ONLINE_SOFTMAX_EMPTY;
    }
    for (size_t d = 0; d < state.value_dim; ++d) {
        const float v = state.accumulator[d] / state.normalizer;
        if (!std::isfinite(v)) return COMMON_TREE_DRAFT_ONLINE_SOFTMAX_INVALID_STATE;
        output[d] = v;
    }
    return COMMON_TREE_DRAFT_ONLINE_SOFTMAX_OK;
}
