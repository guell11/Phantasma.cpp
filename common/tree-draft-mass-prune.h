#pragma once

#include "tree-draft-arena.h"
#include "tree-draft-ranking.h"

#include <cstddef>
#include <cstdint>

enum common_tree_draft_mass_prune_status : uint32_t {
    COMMON_TREE_DRAFT_MASS_PRUNE_OK = 0,
    COMMON_TREE_DRAFT_MASS_PRUNE_NULL_BUFFER,
    COMMON_TREE_DRAFT_MASS_PRUNE_THRESHOLD,
    COMMON_TREE_DRAFT_MASS_PRUNE_RANGE,
    COMMON_TREE_DRAFT_MASS_PRUNE_NUMERIC,
};

common_tree_draft_mass_prune_status common_tree_draft_mass_prune(
        common_tree_draft_arena * arena,
        const common_tree_draft_rank_entry * rank_entries,
        const double * cumulative_scores,
        size_t count,
        double tau,
        uint32_t beam_budget,
        common_tree_draft_score_mode mode,
        common_tree_draft_depth_preference depth_preference);

