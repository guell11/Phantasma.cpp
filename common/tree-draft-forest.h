#pragma once

#include "tree-draft-topology.h"

#include <cstddef>
#include <cstdint>

struct common_tree_draft_forest_offsets {
    const int32_t * offsets;
    int32_t n_entries;
};

enum common_tree_draft_forest_error : int32_t {
    COMMON_TREE_DRAFT_FOREST_OK = 0,
    COMMON_TREE_DRAFT_FOREST_INVALID_TOPOLOGY,
    COMMON_TREE_DRAFT_FOREST_NEGATIVE_ENTRY_COUNT,
    COMMON_TREE_DRAFT_FOREST_NULL_NODE_COUNTS,
    COMMON_TREE_DRAFT_FOREST_NULL_OFFSETS,
    COMMON_TREE_DRAFT_FOREST_NULL_OUTPUT,
    COMMON_TREE_DRAFT_FOREST_OFFSETS_BUFFER_TOO_SMALL,
    COMMON_TREE_DRAFT_FOREST_NEGATIVE_NODE_COUNT,
    COMMON_TREE_DRAFT_FOREST_NODE_COUNT_OVERFLOW,
    COMMON_TREE_DRAFT_FOREST_NODE_COUNT_MISMATCH,
    COMMON_TREE_DRAFT_FOREST_ENTRY_RANGE,
    COMMON_TREE_DRAFT_FOREST_LOCAL_RANGE,
    COMMON_TREE_DRAFT_FOREST_GLOBAL_RANGE,
};

// Builds int32 offsets with shape [n_entries + 1] from explicit ragged entry
// lengths. Empty entries are represented by repeated offsets. The final offset
// must equal topology.n_nodes.
common_tree_draft_forest_error common_tree_draft_forest_offsets_build(
        const common_tree_draft_topology & topology,
        const int32_t * node_counts,
        int32_t n_entries,
        int32_t * offsets,
        size_t offsets_capacity);

common_tree_draft_forest_error common_tree_draft_forest_local_to_global(
        const common_tree_draft_forest_offsets & forest,
        int32_t entry,
        int32_t local,
        int32_t * global);

common_tree_draft_forest_error common_tree_draft_forest_global_to_local(
        const common_tree_draft_forest_offsets & forest,
        int32_t global,
        int32_t * entry,
        int32_t * local);

const char * common_tree_draft_forest_error_name(common_tree_draft_forest_error error);
