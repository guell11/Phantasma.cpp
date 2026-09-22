#pragma once

#include <cstddef>
#include <cstdint>

struct common_tree_draft_top_k_entry {
    int32_t token;
    float logit;
};

enum common_tree_draft_top_k_error : int32_t {
    COMMON_TREE_DRAFT_TOP_K_OK = 0,
    COMMON_TREE_DRAFT_TOP_K_NULL_LOGITS,
    COMMON_TREE_DRAFT_TOP_K_NULL_OUTPUT,
    COMMON_TREE_DRAFT_TOP_K_NULL_COUNT,
    COMMON_TREE_DRAFT_TOP_K_BUFFER_TOO_SMALL,
    COMMON_TREE_DRAFT_TOP_K_NAN_LOGIT,
};

constexpr int32_t COMMON_TREE_DRAFT_TOP_K_SENTINEL_TOKEN = -1;

// Reference ordering is logit descending, then token id ascending. Input
// logits are returned bit-for-bit as scores; +/-infinity are valid values.
// NaN invalidates the whole row. Every provided output slot is initialized to
// {SENTINEL_TOKEN, -infinity}, including unused slots and all slots on error.
common_tree_draft_top_k_error common_tree_draft_top_k_reference(
        const float * logits,
        size_t n_vocab,
        size_t k,
        common_tree_draft_top_k_entry * out,
        size_t out_capacity,
        size_t * out_count);

const char * common_tree_draft_top_k_error_name(common_tree_draft_top_k_error error);
