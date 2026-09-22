#include "tree-draft-width-policy.h"

#include <algorithm>
#include <cmath>

static common_tree_draft_width_status common_tree_draft_width_mass(
        const common_tree_draft_candidate * candidates,
        size_t count,
        double * total) {
    if (count > 0 && candidates == nullptr) return COMMON_TREE_DRAFT_WIDTH_NULL_BUFFER;
    *total = 0.0;
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(candidates[i].probability) || candidates[i].probability < 0.0f) {
            return COMMON_TREE_DRAFT_WIDTH_INVALID_PROBABILITY;
        }
        *total += candidates[i].probability;
    }
    if (!(*total > 0.0) || !std::isfinite(*total)) return COMMON_TREE_DRAFT_WIDTH_ZERO_MASS;
    return COMMON_TREE_DRAFT_WIDTH_OK;
}

common_tree_draft_width_status common_tree_draft_entropy_width(
        const common_tree_draft_candidate * candidates,
        size_t count,
        uint32_t b_min,
        uint32_t b_max,
        uint32_t * desired_width,
        double * normalized_entropy) {
    if (desired_width == nullptr) return COMMON_TREE_DRAFT_WIDTH_NULL_BUFFER;
    if (count == 0 || b_min == 0 || b_min > b_max) return COMMON_TREE_DRAFT_WIDTH_RANGE;
    double total = 0.0;
    const auto status = common_tree_draft_width_mass(candidates, count, &total);
    if (status != COMMON_TREE_DRAFT_WIDTH_OK) return status;

    double h = 0.0;
    for (size_t i = 0; i < count; ++i) {
        const double p = static_cast<double>(candidates[i].probability) / total;
        if (p > 0.0) h -= p * std::log(p);
    }
    const double h_norm = count == 1 ? 0.0 : h / std::log(static_cast<double>(count));
    if (!std::isfinite(h_norm)) return COMMON_TREE_DRAFT_WIDTH_INVALID_PROBABILITY;
    const double raw = static_cast<double>(b_min) + static_cast<double>(b_max - b_min) * h_norm;
    const uint32_t rounded = static_cast<uint32_t>(std::floor(raw + 0.5));
    *desired_width = std::min(b_max, std::max(b_min, rounded));
    if (normalized_entropy != nullptr) *normalized_entropy = h_norm;
    return COMMON_TREE_DRAFT_WIDTH_OK;
}

common_tree_draft_width_status common_tree_draft_confidence_width(
        const common_tree_draft_candidate * candidates,
        size_t count,
        double g_lo,
        double g_hi,
        uint32_t wide_width,
        common_tree_draft_width_override * result) {
    if (result == nullptr) return COMMON_TREE_DRAFT_WIDTH_NULL_BUFFER;
    if (count == 0 || !(g_lo >= 0.0) || g_lo > g_hi || g_hi > 1.0 || !std::isfinite(g_lo) || !std::isfinite(g_hi) || wide_width == 0) {
        return COMMON_TREE_DRAFT_WIDTH_RANGE;
    }
    double total = 0.0;
    const auto status = common_tree_draft_width_mass(candidates, count, &total);
    if (status != COMMON_TREE_DRAFT_WIDTH_OK) return status;
    const double p1 = static_cast<double>(candidates[0].probability) / total;
    const double p2 = count > 1 ? static_cast<double>(candidates[1].probability) / total : 0.0;
    if (p2 > p1) return COMMON_TREE_DRAFT_WIDTH_INVALID_PROBABILITY;
    const double gap = p1 - p2;
    if (gap >= g_hi) {
        *result = { COMMON_TREE_DRAFT_WIDTH_OVERRIDE, 1 };
    } else if (gap <= g_lo) {
        *result = { COMMON_TREE_DRAFT_WIDTH_OVERRIDE, wide_width };
    } else {
        *result = { COMMON_TREE_DRAFT_WIDTH_NO_OVERRIDE, 0 };
    }
    return COMMON_TREE_DRAFT_WIDTH_OK;
}

