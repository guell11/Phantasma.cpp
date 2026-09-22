#pragma once

#include "tree-draft-frontier-prune.h"

#include <cstddef>
#include <cstdint>

enum common_tree_draft_next_frontier_status : uint32_t {
    COMMON_TREE_DRAFT_NEXT_FRONTIER_OK = 0,
    COMMON_TREE_DRAFT_NEXT_FRONTIER_NULL_BUFFER,
    COMMON_TREE_DRAFT_NEXT_FRONTIER_RANGE,
    COMMON_TREE_DRAFT_NEXT_FRONTIER_DUPLICATE,
    COMMON_TREE_DRAFT_NEXT_FRONTIER_RANKING,
};

common_tree_draft_next_frontier_status common_tree_draft_build_next_frontier(
        common_tree_draft_arena * arena,
        const int32_t * new_children,
        size_t new_count,
        const int32_t * carry,
        size_t carry_count,
        const common_tree_draft_rank_entry * ranks,
        size_t rank_count,
        uint32_t beam_budget,
        common_tree_draft_score_mode mode,
        common_tree_draft_depth_preference depth_preference);

