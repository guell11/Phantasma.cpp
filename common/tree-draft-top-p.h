#pragma once

#include "tree-draft-candidates.h"

#include <cstddef>
#include <cstdint>

enum common_tree_draft_top_p_status : uint32_t {
    COMMON_TREE_DRAFT_TOP_P_OK = 0,
    COMMON_TREE_DRAFT_TOP_P_NULL_BUFFER,
    COMMON_TREE_DRAFT_TOP_P_THRESHOLD,
    COMMON_TREE_DRAFT_TOP_P_INVALID_PROBABILITY,
    COMMON_TREE_DRAFT_TOP_P_ZERO_MASS,
    COMMON_TREE_DRAFT_TOP_P_ORDER,
};

common_tree_draft_top_p_status common_tree_draft_top_p_filter(
        const common_tree_draft_candidate * candidates,
        size_t n_candidates,
        float threshold,
        common_tree_draft_candidate * out,
        size_t out_capacity,
        size_t * out_count);

