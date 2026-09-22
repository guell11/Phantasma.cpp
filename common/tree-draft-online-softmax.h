#pragma once

#include <cstddef>
#include <cstdint>

struct common_tree_draft_online_softmax_state {
    float max_score = 0.0f;
    float normalizer = 0.0f;
    float * accumulator = nullptr;
    size_t value_dim = 0;
    bool initialized = false;
};

enum common_tree_draft_online_softmax_status : uint32_t {
    COMMON_TREE_DRAFT_ONLINE_SOFTMAX_OK = 0,
    COMMON_TREE_DRAFT_ONLINE_SOFTMAX_NULL_BUFFER,
    COMMON_TREE_DRAFT_ONLINE_SOFTMAX_INVALID_SCORE,
    COMMON_TREE_DRAFT_ONLINE_SOFTMAX_INVALID_VALUE,
    COMMON_TREE_DRAFT_ONLINE_SOFTMAX_INVALID_STATE,
    COMMON_TREE_DRAFT_ONLINE_SOFTMAX_EMPTY,
};

common_tree_draft_online_softmax_status common_tree_draft_online_softmax_init(
        common_tree_draft_online_softmax_state * state,
        float * accumulator,
        size_t value_dim);

// Adds one score/value block. masked[i]!=0 means the lane is invisible.
// Empty or fully masked blocks are exact no-ops.
common_tree_draft_online_softmax_status common_tree_draft_online_softmax_update(
        common_tree_draft_online_softmax_state * state,
        const float * scores,
        const float * values,
        const uint8_t * masked,
        size_t rows);

common_tree_draft_online_softmax_status common_tree_draft_online_softmax_finalize(
        const common_tree_draft_online_softmax_state & state,
        float * output,
        size_t output_count);
