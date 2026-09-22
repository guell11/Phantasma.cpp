#pragma once

#include "tree-draft-ancestor.h"
#include "tree-draft-forest.h"

#include <cstddef>
#include <cstdint>

struct common_tree_draft_tile_visibility {
    uint8_t * active;
    size_t active_count;
    int32_t query_tile_size;
    int32_t key_tile_size;
    int32_t n_query_tiles;
    int32_t n_key_tiles;
};

enum common_tree_draft_tile_visibility_error : int32_t {
    COMMON_TREE_DRAFT_TILE_VISIBILITY_OK = 0,
    COMMON_TREE_DRAFT_TILE_VISIBILITY_INVALID_FOREST,
    COMMON_TREE_DRAFT_TILE_VISIBILITY_INVALID_TILE_SIZE,
    COMMON_TREE_DRAFT_TILE_VISIBILITY_NULL_ANCESTOR,
    COMMON_TREE_DRAFT_TILE_VISIBILITY_ANCESTOR_TOO_SMALL,
    COMMON_TREE_DRAFT_TILE_VISIBILITY_SIZE_OVERFLOW,
    COMMON_TREE_DRAFT_TILE_VISIBILITY_NULL_OUTPUT,
    COMMON_TREE_DRAFT_TILE_VISIBILITY_OUTPUT_TOO_SMALL,
    COMMON_TREE_DRAFT_TILE_VISIBILITY_NODE_RANGE,
};

common_tree_draft_tile_visibility_error common_tree_draft_tile_visibility_build(
        const common_tree_draft_forest_offsets & forest,
        const common_tree_draft_ancestor_bitset & ancestors,
        int32_t query_tile_size,
        int32_t key_tile_size,
        common_tree_draft_tile_visibility & output);

bool common_tree_draft_element_visible(
        const common_tree_draft_forest_offsets & forest,
        const common_tree_draft_ancestor_bitset & ancestors,
        int32_t query_node,
        int32_t key_node);

const char * common_tree_draft_tile_visibility_error_name(common_tree_draft_tile_visibility_error error);
