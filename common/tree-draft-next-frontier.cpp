#include "tree-draft-next-frontier.h"

#include <algorithm>
#include <vector>

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
        common_tree_draft_depth_preference depth_preference) {
    if (arena == nullptr || (new_count > 0 && new_children == nullptr) ||
        (carry_count > 0 && carry == nullptr) || (rank_count > 0 && ranks == nullptr)) {
        return COMMON_TREE_DRAFT_NEXT_FRONTIER_NULL_BUFFER;
    }

    std::vector<int32_t> candidates;
    candidates.reserve(new_count + carry_count);
    const auto append = [&](const int32_t * values, size_t count) -> common_tree_draft_next_frontier_status {
        for (size_t i = 0; i < count; ++i) {
            const int32_t node = values[i];
            if (node < 0 || static_cast<uint32_t>(node) >= arena->node_count) return COMMON_TREE_DRAFT_NEXT_FRONTIER_RANGE;
            if ((arena->nodes[node].flags & COMMON_TREE_DRAFT_NODE_FLAG_TERMINAL) != 0) continue;
            if (std::find(candidates.begin(), candidates.end(), node) != candidates.end()) return COMMON_TREE_DRAFT_NEXT_FRONTIER_DUPLICATE;
            candidates.push_back(node);
        }
        return COMMON_TREE_DRAFT_NEXT_FRONTIER_OK;
    };
    auto status = append(new_children, new_count);
    if (status != COMMON_TREE_DRAFT_NEXT_FRONTIER_OK) return status;
    status = append(carry, carry_count);
    if (status != COMMON_TREE_DRAFT_NEXT_FRONTIER_OK) return status;

    std::sort(candidates.begin(), candidates.end());
    std::vector<common_tree_draft_rank_entry> ordered_ranks;
    ordered_ranks.reserve(candidates.size());
    for (int32_t node : candidates) {
        const common_tree_draft_rank_entry * found = nullptr;
        for (size_t r = 0; r < rank_count; ++r) {
            if (ranks[r].node_index == static_cast<uint32_t>(node)) {
                if (found != nullptr) return COMMON_TREE_DRAFT_NEXT_FRONTIER_DUPLICATE;
                found = &ranks[r];
            }
        }
        if (found == nullptr) return COMMON_TREE_DRAFT_NEXT_FRONTIER_RANKING;
        ordered_ranks.push_back(*found);
    }

    for (uint32_t i = 0; i < arena->node_count; ++i) {
        arena->nodes[i].flags &= ~COMMON_TREE_DRAFT_NODE_FLAG_FRONTIER;
    }
    arena->frontier_count = static_cast<uint32_t>(candidates.size());
    for (size_t i = 0; i < candidates.size(); ++i) {
        arena->frontier[i] = candidates[i];
        arena->nodes[candidates[i]].flags &= ~COMMON_TREE_DRAFT_NODE_FLAG_PRUNED;
        arena->nodes[candidates[i]].flags |= COMMON_TREE_DRAFT_NODE_FLAG_FRONTIER;
    }

    if (common_tree_draft_frontier_prune(arena, ordered_ranks.data(), ordered_ranks.size(), beam_budget, mode, depth_preference) !=
            COMMON_TREE_DRAFT_FRONTIER_PRUNE_OK) {
        return COMMON_TREE_DRAFT_NEXT_FRONTIER_RANKING;
    }
    return COMMON_TREE_DRAFT_NEXT_FRONTIER_OK;
}

