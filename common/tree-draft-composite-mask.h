#pragma once

#include "tree-draft-ancestor.h"
#include "tree-draft-position.h"

#include <cstdint>

enum common_tree_draft_composite_key_kind : int32_t {
    COMMON_TREE_DRAFT_COMPOSITE_KEY_PREFIX = 0,
    COMMON_TREE_DRAFT_COMPOSITE_KEY_PROPOSAL = 1,
};

enum common_tree_draft_composite_mask_error : int32_t {
    COMMON_TREE_DRAFT_COMPOSITE_MASK_OK = 0,
    COMMON_TREE_DRAFT_COMPOSITE_MASK_INVALID_TOPOLOGY,
    COMMON_TREE_DRAFT_COMPOSITE_MASK_INVALID_FOREST,
    COMMON_TREE_DRAFT_COMPOSITE_MASK_INVALID_PREFIX_LENGTH,
    COMMON_TREE_DRAFT_COMPOSITE_MASK_NULL_ANCESTOR,
    COMMON_TREE_DRAFT_COMPOSITE_MASK_ANCESTOR_TOO_SMALL,
    COMMON_TREE_DRAFT_COMPOSITE_MASK_ENTRY_RANGE,
    COMMON_TREE_DRAFT_COMPOSITE_MASK_QUERY_RANGE,
    COMMON_TREE_DRAFT_COMPOSITE_MASK_KEY_RANGE,
    COMMON_TREE_DRAFT_COMPOSITE_MASK_INVALID_KEY_KIND,
    COMMON_TREE_DRAFT_COMPOSITE_MASK_NULL_OUTPUT,
};

common_tree_draft_composite_mask_error common_tree_draft_composite_mask_visible(
        const common_tree_draft_topology & topology,
        const common_tree_draft_ancestor_bitset & ancestors,
        const common_tree_draft_forest_offsets & forest,
        const int32_t * prefix_lengths,
        int32_t entry,
        int32_t query_local,
        common_tree_draft_composite_key_kind key_kind,
        int32_t key_index,
        bool * visible);

float common_tree_draft_composite_mask_value(bool visible);

const char * common_tree_draft_composite_mask_error_name(common_tree_draft_composite_mask_error error);
