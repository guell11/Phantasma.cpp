#include "tree-draft-candidates.h"

#include "tree-draft-top-k.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

common_tree_draft_candidate_status common_tree_draft_candidate_top_k(
        const float * scores,
        const float * probabilities,
        size_t n_vocab,
        size_t k,
        common_tree_draft_candidate * out,
        size_t out_capacity,
        size_t * out_count) {
    if (out_count == nullptr || (n_vocab > 0 && (scores == nullptr || probabilities == nullptr)) ||
        (out_capacity > 0 && out == nullptr)) {
        return COMMON_TREE_DRAFT_CANDIDATE_NULL_BUFFER;
    }
    *out_count = 0;

    size_t n_finite = 0;
    for (size_t token = 0; token < n_vocab; ++token) {
        if (std::isnan(scores[token]) || scores[token] == std::numeric_limits<float>::infinity()) {
            return COMMON_TREE_DRAFT_CANDIDATE_INVALID_SCORE;
        }
        if (!std::isfinite(probabilities[token]) || probabilities[token] < 0.0f) {
            return COMMON_TREE_DRAFT_CANDIDATE_INVALID_PROBABILITY;
        }
        if (std::isfinite(scores[token])) {
            ++n_finite;
        } else if (probabilities[token] != 0.0f) {
            return COMMON_TREE_DRAFT_CANDIDATE_INVALID_PROBABILITY;
        }
    }

    const size_t count = std::min(k, n_finite);
    if (out_capacity < count) {
        return COMMON_TREE_DRAFT_CANDIDATE_BUFFER_TOO_SMALL;
    }
    if (count == 0) {
        return COMMON_TREE_DRAFT_CANDIDATE_OK;
    }

    std::vector<common_tree_draft_top_k_entry> selected(count);
    size_t selected_count = 0;
    const auto top_k_status = common_tree_draft_top_k_reference(
            scores, n_vocab, count, selected.data(), selected.size(), &selected_count);
    if (top_k_status != COMMON_TREE_DRAFT_TOP_K_OK || selected_count != count) {
        return COMMON_TREE_DRAFT_CANDIDATE_INVALID_SCORE;
    }

    for (size_t i = 0; i < count; ++i) {
        const int32_t token = selected[i].token;
        out[i] = { token, selected[i].logit, probabilities[token] };
    }
    *out_count = count;
    return COMMON_TREE_DRAFT_CANDIDATE_OK;
}

