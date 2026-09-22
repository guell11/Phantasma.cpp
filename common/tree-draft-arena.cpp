#include "tree-draft-arena.h"

#include <algorithm>

common_tree_draft_arena_status common_tree_draft_arena_init(
        common_tree_draft_arena * arena,
        common_tree_draft_node * nodes,
        int32_t * frontier,
        uint32_t capacity,
        const common_tree_draft_config & config) {
    if (arena == nullptr) {
        return COMMON_TREE_DRAFT_ARENA_NULL_NODES;
    }
    if (capacity > config.tree_budget()) {
        return COMMON_TREE_DRAFT_ARENA_CAPACITY;
    }
    if (capacity > 0 && nodes == nullptr) {
        return COMMON_TREE_DRAFT_ARENA_NULL_NODES;
    }
    if (capacity > 0 && frontier == nullptr) {
        return COMMON_TREE_DRAFT_ARENA_NULL_FRONTIER;
    }

    *arena = { nodes, frontier, capacity, 0, 0 };
    return COMMON_TREE_DRAFT_ARENA_OK;
}

common_tree_draft_arena_status common_tree_draft_arena_append(
        common_tree_draft_arena * arena,
        const common_tree_draft_node & node,
        const common_tree_draft_config & config,
        uint32_t * node_index) {
    if (arena == nullptr || (arena->capacity > 0 && arena->nodes == nullptr)) {
        return COMMON_TREE_DRAFT_ARENA_NULL_NODES;
    }
    if (arena->node_count >= arena->capacity || arena->node_count >= config.tree_budget()) {
        return COMMON_TREE_DRAFT_ARENA_CAPACITY;
    }

    const uint32_t index = arena->node_count;
    arena->nodes[index] = node;
    const common_tree_draft_node_error error =
            common_tree_draft_nodes_validate(arena->nodes, static_cast<size_t>(index) + 1, config);
    if (error != COMMON_TREE_DRAFT_NODE_OK) {
        return COMMON_TREE_DRAFT_ARENA_NODE_INVALID;
    }

    arena->node_count = index + 1;
    if (node_index != nullptr) {
        *node_index = index;
    }
    return COMMON_TREE_DRAFT_ARENA_OK;
}

common_tree_draft_arena_status common_tree_draft_arena_set_frontier(
        common_tree_draft_arena * arena,
        const int32_t * frontier,
        uint32_t frontier_count) {
    if (arena == nullptr || (arena->capacity > 0 && arena->frontier == nullptr)) {
        return COMMON_TREE_DRAFT_ARENA_NULL_FRONTIER;
    }
    if (frontier_count > arena->capacity) {
        return COMMON_TREE_DRAFT_ARENA_CAPACITY;
    }
    if (frontier_count > 0 && frontier == nullptr) {
        return COMMON_TREE_DRAFT_ARENA_NULL_FRONTIER;
    }

    for (uint32_t i = 0; i < frontier_count; ++i) {
        if (frontier[i] < 0 || static_cast<uint32_t>(frontier[i]) >= arena->node_count) {
            return COMMON_TREE_DRAFT_ARENA_FRONTIER_RANGE;
        }
        for (uint32_t j = 0; j < i; ++j) {
            if (frontier[j] == frontier[i]) {
                return COMMON_TREE_DRAFT_ARENA_FRONTIER_DUPLICATE;
            }
        }
    }

    if (frontier_count > 0) {
        std::copy_n(frontier, frontier_count, arena->frontier);
    }
    arena->frontier_count = frontier_count;
    return COMMON_TREE_DRAFT_ARENA_OK;
}

const char * common_tree_draft_arena_status_name(common_tree_draft_arena_status status) {
    switch (status) {
        case COMMON_TREE_DRAFT_ARENA_OK:                 return "ok";
        case COMMON_TREE_DRAFT_ARENA_NULL_NODES:         return "null_nodes";
        case COMMON_TREE_DRAFT_ARENA_NULL_FRONTIER:      return "null_frontier";
        case COMMON_TREE_DRAFT_ARENA_CAPACITY:           return "capacity";
        case COMMON_TREE_DRAFT_ARENA_NODE_INVALID:       return "node_invalid";
        case COMMON_TREE_DRAFT_ARENA_FRONTIER_RANGE:     return "frontier_range";
        case COMMON_TREE_DRAFT_ARENA_FRONTIER_DUPLICATE: return "frontier_duplicate";
    }
    return "unknown";
}

