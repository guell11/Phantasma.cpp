#pragma once

#include "tree-draft-forest.h"

#include <cstddef>
#include <cstdint>

struct common_tree_draft_positions {
    int32_t * values;
    size_t value_count;
};

enum common_tree_draft_position_error : int32_t {
    COMMON_TREE_DRAFT_POSITION_OK = 0,
    COMMON_TREE_DRAFT_POSITION_INVALID_TOPOLOGY,
    COMMON_TREE_DRAFT_POSITION_INVALID_FOREST,
    COMMON_TREE_DRAFT_POSITION_NULL_PREFIX_LENGTHS,
    COMMON_TREE_DRAFT_POSITION_NEGATIVE_PREFIX_LENGTH,
    COMMON_TREE_DRAFT_POSITION_NULL_OUTPUT,
    COMMON_TREE_DRAFT_POSITION_OUTPUT_TOO_SMALL,
    COMMON_TREE_DRAFT_POSITION_OVERFLOW,
    COMMON_TREE_DRAFT_POSITION_ENTRY_RANGE,
    COMMON_TREE_DRAFT_POSITION_LOCAL_RANGE,
    COMMON_TREE_DRAFT_POSITION_NULL_MAPPING_OUTPUT,
};

common_tree_draft_position_error common_tree_draft_positions_build(
        const common_tree_draft_topology & topology,
        const common_tree_draft_forest_offsets & forest,
        const int32_t * prefix_lengths,
        common_tree_draft_positions output);

common_tree_draft_position_error common_tree_draft_position_for_local(
        const common_tree_draft_topology & topology,
        const common_tree_draft_forest_offsets & forest,
        const int32_t * prefix_lengths,
        int32_t entry,
        int32_t local,
        int32_t * global,
        int32_t * position);

const char * common_tree_draft_position_error_name(common_tree_draft_position_error error);
