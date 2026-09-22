#pragma once

#include "tree-draft-ancestor.h"

#include <cstddef>
#include <cstdint>

struct common_tree_draft_dense_mask {
    // Row-major [query_node][key_node] additive mask. The first n_nodes^2
    // values are written; visible entries are 0 and hidden entries are -inf.
    float * values;
    size_t value_count;
};

enum common_tree_draft_dense_mask_error : int32_t {
    COMMON_TREE_DRAFT_DENSE_MASK_OK = 0,
    COMMON_TREE_DRAFT_DENSE_MASK_INVALID_TOPOLOGY,
    COMMON_TREE_DRAFT_DENSE_MASK_NULL_ANCESTOR,
    COMMON_TREE_DRAFT_DENSE_MASK_ANCESTOR_TOO_SMALL,
    COMMON_TREE_DRAFT_DENSE_MASK_NULL_OUTPUT,
    COMMON_TREE_DRAFT_DENSE_MASK_OUTPUT_TOO_SMALL,
    COMMON_TREE_DRAFT_DENSE_MASK_SIZE_OVERFLOW,
};

common_tree_draft_dense_mask_error common_tree_draft_dense_mask_build(
        const common_tree_draft_topology & topology,
        const common_tree_draft_ancestor_bitset & ancestors,
        common_tree_draft_dense_mask output);

const char * common_tree_draft_dense_mask_error_name(common_tree_draft_dense_mask_error error);
