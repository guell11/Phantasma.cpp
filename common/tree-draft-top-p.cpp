#include "tree-draft-top-p.h"

#include <cmath>

common_tree_draft_top_p_status common_tree_draft_top_p_filter(
        const common_tree_draft_candidate * candidates,
        size_t n_candidates,
        float threshold,
        common_tree_draft_candidate * out,
        size_t out_capacity,
        size_t * out_count) {
    if (out_count == nullptr || (n_candidates > 0 && candidates == nullptr) ||
        (out_capacity > 0 && out == nullptr)) {
        return COMMON_TREE_DRAFT_TOP_P_NULL_BUFFER;
    }
    *out_count = 0;
    if (!(threshold > 0.0f) || threshold > 1.0f || !std::isfinite(threshold)) {
        return COMMON_TREE_DRAFT_TOP_P_THRESHOLD;
    }
    if (n_candidates == 0) {
        return COMMON_TREE_DRAFT_TOP_P_ZERO_MASS;
    }

    double total = 0.0;
    for (size_t i = 0; i < n_candidates; ++i) {
        if (!std::isfinite(candidates[i].probability) || candidates[i].probability < 0.0f) {
            return COMMON_TREE_DRAFT_TOP_P_INVALID_PROBABILITY;
        }
        if (i > 0) {
            if (candidates[i - 1].score < candidates[i].score ||
                (candidates[i - 1].score == candidates[i].score && candidates[i - 1].token > candidates[i].token)) {
                return COMMON_TREE_DRAFT_TOP_P_ORDER;
            }
        }
        total += candidates[i].probability;
    }
    if (!(total > 0.0) || !std::isfinite(total)) {
        return COMMON_TREE_DRAFT_TOP_P_ZERO_MASS;
    }

    double cumulative = 0.0;
    size_t retained = 0;
    for (; retained < n_candidates; ++retained) {
        cumulative += static_cast<double>(candidates[retained].probability) / total;
        if (cumulative >= static_cast<double>(threshold)) {
            ++retained;
            break;
        }
    }
    if (retained == 0) {
        retained = 1;
    }
    if (out_capacity < retained) {
        return COMMON_TREE_DRAFT_TOP_P_NULL_BUFFER;
    }

    double retained_mass = 0.0;
    for (size_t i = 0; i < retained; ++i) {
        retained_mass += candidates[i].probability;
    }
    if (!(retained_mass > 0.0) || !std::isfinite(retained_mass)) {
        return COMMON_TREE_DRAFT_TOP_P_ZERO_MASS;
    }

    for (size_t i = 0; i < retained; ++i) {
        out[i] = candidates[i];
        out[i].probability = static_cast<float>(static_cast<double>(candidates[i].probability) / retained_mass);
    }
    *out_count = retained;
    return COMMON_TREE_DRAFT_TOP_P_OK;
}

