#include "tree-draft-top-k.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

common_tree_draft_top_k_error common_tree_draft_top_k_reference(
        const float * logits,
        size_t n_vocab,
        size_t k,
        common_tree_draft_top_k_entry * out,
        size_t out_capacity,
        size_t * out_count) {
    if (out_count == nullptr) {
        return COMMON_TREE_DRAFT_TOP_K_NULL_COUNT;
    }
    *out_count = 0;

    if (out_capacity > 0 && out == nullptr) {
        return COMMON_TREE_DRAFT_TOP_K_NULL_OUTPUT;
    }

    const common_tree_draft_top_k_entry sentinel = {
        COMMON_TREE_DRAFT_TOP_K_SENTINEL_TOKEN,
        -std::numeric_limits<float>::infinity(),
    };
    std::fill_n(out, out_capacity, sentinel);

    if (n_vocab > 0 && logits == nullptr) {
        return COMMON_TREE_DRAFT_TOP_K_NULL_LOGITS;
    }

    const size_t n_select = std::min(k, n_vocab);
    if (out_capacity < n_select) {
        return COMMON_TREE_DRAFT_TOP_K_BUFFER_TOO_SMALL;
    }

    for (size_t token = 0; token < n_vocab; ++token) {
        if (std::isnan(logits[token])) {
            return COMMON_TREE_DRAFT_TOP_K_NAN_LOGIT;
        }
    }
    if (n_select == 0) {
        return COMMON_TREE_DRAFT_TOP_K_OK;
    }

    std::vector<size_t> order(n_vocab);
    std::iota(order.begin(), order.end(), size_t{0});
    std::sort(order.begin(), order.end(), [logits](size_t a, size_t b) {
        if (logits[a] != logits[b]) {
            return logits[a] > logits[b];
        }
        return a < b;
    });

    for (size_t i = 0; i < n_select; ++i) {
        out[i] = { static_cast<int32_t>(order[i]), logits[order[i]] };
    }
    *out_count = n_select;
    return COMMON_TREE_DRAFT_TOP_K_OK;
}

const char * common_tree_draft_top_k_error_name(common_tree_draft_top_k_error error) {
    switch (error) {
        case COMMON_TREE_DRAFT_TOP_K_OK:               return "ok";
        case COMMON_TREE_DRAFT_TOP_K_NULL_LOGITS:      return "null_logits";
        case COMMON_TREE_DRAFT_TOP_K_NULL_OUTPUT:      return "null_output";
        case COMMON_TREE_DRAFT_TOP_K_NULL_COUNT:       return "null_count";
        case COMMON_TREE_DRAFT_TOP_K_BUFFER_TOO_SMALL: return "buffer_too_small";
        case COMMON_TREE_DRAFT_TOP_K_NAN_LOGIT:        return "nan_logit";
    }
    return "unknown";
}
