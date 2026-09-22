#pragma once

#include "tree-draft-candidates.h"

#include <cstddef>
#include <cstdint>

enum common_tree_draft_width_status : uint32_t {
    COMMON_TREE_DRAFT_WIDTH_OK = 0,
    COMMON_TREE_DRAFT_WIDTH_NULL_BUFFER,
    COMMON_TREE_DRAFT_WIDTH_RANGE,
    COMMON_TREE_DRAFT_WIDTH_INVALID_PROBABILITY,
    COMMON_TREE_DRAFT_WIDTH_ZERO_MASS,
};

enum common_tree_draft_width_override_kind : uint32_t {
    COMMON_TREE_DRAFT_WIDTH_NO_OVERRIDE = 0,
    COMMON_TREE_DRAFT_WIDTH_OVERRIDE,
};

struct common_tree_draft_width_override {
    common_tree_draft_width_override_kind kind;
    uint32_t width;
};

common_tree_draft_width_status common_tree_draft_entropy_width(
        const common_tree_draft_candidate * candidates,
        size_t count,
        uint32_t b_min,
        uint32_t b_max,
        uint32_t * desired_width,
        double * normalized_entropy = nullptr);

common_tree_draft_width_status common_tree_draft_confidence_width(
        const common_tree_draft_candidate * candidates,
        size_t count,
        double g_lo,
        double g_hi,
        uint32_t wide_width,
        common_tree_draft_width_override * result);

