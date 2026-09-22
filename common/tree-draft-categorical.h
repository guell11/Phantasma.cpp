#pragma once

#include <cstddef>
#include <cstdint>

enum common_tree_draft_categorical_status : uint32_t {
    COMMON_TREE_DRAFT_CATEGORICAL_OK = 0,
    COMMON_TREE_DRAFT_CATEGORICAL_NULL_BUFFER,
    COMMON_TREE_DRAFT_CATEGORICAL_TEMPERATURE,
    COMMON_TREE_DRAFT_CATEGORICAL_INVALID_LOGIT,
    COMMON_TREE_DRAFT_CATEGORICAL_NO_FINITE,
    COMMON_TREE_DRAFT_CATEGORICAL_NUMERIC,
};

common_tree_draft_categorical_status common_tree_draft_categorical_normalize(
        const float * logits,
        size_t n_vocab,
        float temperature,
        float * scaled_logits,
        float * probabilities);

