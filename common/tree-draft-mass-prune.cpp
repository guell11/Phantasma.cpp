#include "tree-draft-mass-prune.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

common_tree_draft_mass_prune_status common_tree_draft_mass_prune(
        common_tree_draft_arena * arena,
        const common_tree_draft_rank_entry * rank_entries,
        const double * cumulative_scores,
        size_t count,
        double tau,
        uint32_t beam_budget,
        common_tree_draft_score_mode mode,
        common_tree_draft_depth_preference depth_preference) {
    if (arena == nullptr || (count > 0 && (rank_entries == nullptr || cumulative_scores == nullptr))) {
        return COMMON_TREE_DRAFT_MASS_PRUNE_NULL_BUFFER;
    }
    if (!(tau > 0.0) || tau > 1.0 || !std::isfinite(tau)) {
        return COMMON_TREE_DRAFT_MASS_PRUNE_THRESHOLD;
    }
    if (count != arena->frontier_count) {
        return COMMON_TREE_DRAFT_MASS_PRUNE_RANGE;
    }
    if (count == 0) {
        return COMMON_TREE_DRAFT_MASS_PRUNE_OK;
    }

    double best = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < count; ++i) {
        if (rank_entries[i].node_index >= arena->node_count ||
            rank_entries[i].node_index != static_cast<uint32_t>(arena->frontier[i])) {
            return COMMON_TREE_DRAFT_MASS_PRUNE_RANGE;
        }
        if (!std::isfinite(cumulative_scores[i])) {
            return COMMON_TREE_DRAFT_MASS_PRUNE_NUMERIC;
        }
        best = std::max(best, cumulative_scores[i]);
    }

    std::vector<uint32_t> survivors;
    survivors.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        const double rel = std::exp(cumulative_scores[i] - best);
        if (!std::isfinite(rel)) {
            return COMMON_TREE_DRAFT_MASS_PRUNE_NUMERIC;
        }
        if (rel >= tau) {
            survivors.push_back(static_cast<uint32_t>(i));
        }
    }

    std::sort(survivors.begin(), survivors.end(), [&](uint32_t a, uint32_t b) {
        return common_tree_draft_rank_compare(rank_entries[a], rank_entries[b], mode, depth_preference) < 0;
    });
    if (survivors.size() > beam_budget) {
        survivors.resize(beam_budget);
    }

    std::vector<uint8_t> retained(arena->node_count, 0);
    for (uint32_t idx : survivors) {
        retained[rank_entries[idx].node_index] = 1;
    }
    for (size_t i = 0; i < count; ++i) {
        const uint32_t node = rank_entries[i].node_index;
        uint32_t flags = arena->nodes[node].flags;
        flags &= ~COMMON_TREE_DRAFT_NODE_FLAG_FRONTIER;
        flags &= ~COMMON_TREE_DRAFT_NODE_FLAG_PRUNED;
        if (retained[node]) {
            flags |= COMMON_TREE_DRAFT_NODE_FLAG_FRONTIER;
        } else {
            flags |= COMMON_TREE_DRAFT_NODE_FLAG_PRUNED;
        }
        arena->nodes[node].flags = flags;
    }

    arena->frontier_count = static_cast<uint32_t>(survivors.size());
    for (size_t i = 0; i < survivors.size(); ++i) {
        arena->frontier[i] = static_cast<int32_t>(rank_entries[survivors[i]].node_index);
    }
    return COMMON_TREE_DRAFT_MASS_PRUNE_OK;
}

