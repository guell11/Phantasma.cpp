#pragma once

#include "tree-draft-node.h"

#include <cstddef>
#include <cstdint>

enum common_tree_draft_arena_status : int32_t {
    COMMON_TREE_DRAFT_ARENA_OK = 0,
    COMMON_TREE_DRAFT_ARENA_NULL_NODES,
    COMMON_TREE_DRAFT_ARENA_NULL_FRONTIER,
    COMMON_TREE_DRAFT_ARENA_CAPACITY,
    COMMON_TREE_DRAFT_ARENA_NODE_INVALID,
    COMMON_TREE_DRAFT_ARENA_FRONTIER_RANGE,
    COMMON_TREE_DRAFT_ARENA_FRONTIER_DUPLICATE,
};

struct common_tree_draft_arena {
    common_tree_draft_node * nodes;
    int32_t * frontier;
    uint32_t capacity;
    uint32_t node_count;
    uint32_t frontier_count;
};

common_tree_draft_arena_status common_tree_draft_arena_init(
        common_tree_draft_arena * arena,
        common_tree_draft_node * nodes,
        int32_t * frontier,
        uint32_t capacity,
        const common_tree_draft_config & config);

common_tree_draft_arena_status common_tree_draft_arena_append(
        common_tree_draft_arena * arena,
        const common_tree_draft_node & node,
        const common_tree_draft_config & config,
        uint32_t * node_index = nullptr);

common_tree_draft_arena_status common_tree_draft_arena_set_frontier(
        common_tree_draft_arena * arena,
        const int32_t * frontier,
        uint32_t frontier_count);

const char * common_tree_draft_arena_status_name(common_tree_draft_arena_status status);

