#pragma once

#include <cstddef>
#include <cstdint>

struct common_tree_draft_candidate {
    int32_t token;
    float score;
    float probability;
};

enum common_tree_draft_candidate_status : uint32_t {
    COMMON_TREE_DRAFT_CANDIDATE_OK = 0,
    COMMON_TREE_DRAFT_CANDIDATE_NULL_BUFFER,
    COMMON_TREE_DRAFT_CANDIDATE_BUFFER_TOO_SMALL,
    COMMON_TREE_DRAFT_CANDIDATE_INVALID_SCORE,
    COMMON_TREE_DRAFT_CANDIDATE_INVALID_PROBABILITY,
};

common_tree_draft_candidate_status common_tree_draft_candidate_top_k(
        const float * scores,
        const float * probabilities,
        size_t n_vocab,
        size_t k,
        common_tree_draft_candidate * out,
        size_t out_capacity,
        size_t * out_count);

