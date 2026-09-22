#include "tree-draft-frontier-prune.h"

#include <algorithm>
#include <vector>

common_tree_draft_frontier_prune_status common_tree_draft_frontier_prune(
        common_tree_draft_arena * arena,
        const common_tree_draft_rank_entry * rank_entries,
        size_t rank_count,
        uint32_t beam_budget,
        common_tree_draft_score_mode mode,
        common_tree_draft_depth_preference depth_preference) {
    if (arena == nullptr || (rank_count > 0 && rank_entries == nullptr)) {
        return COMMON_TREE_DRAFT_FRONTIER_PRUNE_NULL_BUFFER;
    }
    if (rank_count != arena->frontier_count) {
        return COMMON_TREE_DRAFT_FRONTIER_PRUNE_RANGE;
    }

    std::vector<uint32_t> order(rank_count);
    for (size_t i = 0; i < rank_count; ++i) {
        if (rank_entries[i].node_index >= arena->node_count ||
            static_cast<uint32_t>(arena->frontier[i]) != rank_entries[i].node_index) {
            return COMMON_TREE_DRAFT_FRONTIER_PRUNE_RANGE;
        }
        for (size_t j = 0; j < i; ++j) {
            if (rank_entries[j].node_index == rank_entries[i].node_index) {
                return COMMON_TREE_DRAFT_FRONTIER_PRUNE_DUPLICATE;
            }
        }
        order[i] = static_cast<uint32_t>(i);
    }

    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        return common_tree_draft_rank_compare(rank_entries[a], rank_entries[b], mode, depth_preference) < 0;
    });

    const size_t keep = std::min<size_t>(beam_budget, order.size());
    std::vector<int32_t> selected;
    selected.reserve(keep);
    std::vector<uint8_t> retained(arena->node_count, 0);
    for (size_t i = 0; i < keep; ++i) {
        const uint32_t node = rank_entries[order[i]].node_index;
        selected.push_back(static_cast<int32_t>(node));
        retained[node] = 1;
    }

    for (size_t i = 0; i < rank_count; ++i) {
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

    arena->frontier_count = static_cast<uint32_t>(keep);
    for (size_t i = 0; i < keep; ++i) {
        arena->frontier[i] = selected[i];
    }
    return COMMON_TREE_DRAFT_FRONTIER_PRUNE_OK;
}

