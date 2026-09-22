#pragma once

#include "tree-draft-topology.h"

#include <cstddef>
#include <cstdint>

// Row-major uint32 ancestor closure. Row i contains bit j iff j is node i or
// an ancestor of i. Bit j uses word j/32 and bit j%32.
struct common_tree_draft_ancestor_bitset {
    uint32_t * words;
    size_t word_count;
};

enum common_tree_draft_ancestor_error : int32_t {
    COMMON_TREE_DRAFT_ANCESTOR_OK = 0,
    COMMON_TREE_DRAFT_ANCESTOR_INVALID_TOPOLOGY,
    COMMON_TREE_DRAFT_ANCESTOR_NULL_OUTPUT,
    COMMON_TREE_DRAFT_ANCESTOR_OUTPUT_TOO_SMALL,
    COMMON_TREE_DRAFT_ANCESTOR_SIZE_OVERFLOW,
};

size_t common_tree_draft_ancestor_words_per_row(int32_t n_nodes);

common_tree_draft_ancestor_error common_tree_draft_ancestor_build(
        const common_tree_draft_topology & topology,
        common_tree_draft_ancestor_bitset output);

bool common_tree_draft_ancestor_contains(
        const common_tree_draft_ancestor_bitset & bitset,
        int32_t n_nodes,
        int32_t node,
        int32_t candidate);

const char * common_tree_draft_ancestor_error_name(common_tree_draft_ancestor_error error);
