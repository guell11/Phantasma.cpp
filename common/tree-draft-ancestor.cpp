#include "tree-draft-ancestor.h"

#include <algorithm>
#include <limits>

size_t common_tree_draft_ancestor_words_per_row(int32_t n_nodes) {
    if (n_nodes <= 0) {
        return 0;
    }
    return (static_cast<size_t>(n_nodes) + 31) / 32;
}

common_tree_draft_ancestor_error common_tree_draft_ancestor_build(
        const common_tree_draft_topology & topology,
        common_tree_draft_ancestor_bitset output) {
    if (common_tree_draft_topology_validate(topology) != COMMON_TREE_DRAFT_TOPOLOGY_OK) {
        return COMMON_TREE_DRAFT_ANCESTOR_INVALID_TOPOLOGY;
    }

    const size_t words_per_row = common_tree_draft_ancestor_words_per_row(topology.n_nodes);
    if (topology.n_nodes == 0) {
        return COMMON_TREE_DRAFT_ANCESTOR_OK;
    }

    const size_t n_nodes = static_cast<size_t>(topology.n_nodes);
    if (words_per_row > std::numeric_limits<size_t>::max() / n_nodes) {
        return COMMON_TREE_DRAFT_ANCESTOR_SIZE_OVERFLOW;
    }
    const size_t required_words = n_nodes * words_per_row;
    if (output.words == nullptr) {
        return COMMON_TREE_DRAFT_ANCESTOR_NULL_OUTPUT;
    }
    if (output.word_count < required_words) {
        return COMMON_TREE_DRAFT_ANCESTOR_OUTPUT_TOO_SMALL;
    }

    std::fill(output.words, output.words + required_words, 0u);

    for (int32_t i = 0; i < topology.n_nodes; ++i) {
        uint32_t * row = output.words + static_cast<size_t>(i) * words_per_row;
        const int32_t parent = topology.parent[i];
        if (parent >= 0) {
            const uint32_t * parent_row = output.words + static_cast<size_t>(parent) * words_per_row;
            std::copy(parent_row, parent_row + words_per_row, row);
        }
        row[static_cast<size_t>(i) / 32] |= uint32_t(1) << (static_cast<uint32_t>(i) & 31u);
    }

    return COMMON_TREE_DRAFT_ANCESTOR_OK;
}

bool common_tree_draft_ancestor_contains(
        const common_tree_draft_ancestor_bitset & bitset,
        int32_t n_nodes,
        int32_t node,
        int32_t candidate) {
    if (bitset.words == nullptr || n_nodes <= 0 || node < 0 || node >= n_nodes || candidate < 0 || candidate >= n_nodes) {
        return false;
    }
    const size_t words_per_row = common_tree_draft_ancestor_words_per_row(n_nodes);
    const size_t word_index = static_cast<size_t>(node) * words_per_row + static_cast<size_t>(candidate) / 32;
    if (word_index >= bitset.word_count) {
        return false;
    }
    return ((bitset.words[word_index] >> (static_cast<uint32_t>(candidate) & 31u)) & 1u) != 0;
}

const char * common_tree_draft_ancestor_error_name(common_tree_draft_ancestor_error error) {
    switch (error) {
        case COMMON_TREE_DRAFT_ANCESTOR_OK:               return "ok";
        case COMMON_TREE_DRAFT_ANCESTOR_INVALID_TOPOLOGY: return "invalid_topology";
        case COMMON_TREE_DRAFT_ANCESTOR_NULL_OUTPUT:       return "null_output";
        case COMMON_TREE_DRAFT_ANCESTOR_OUTPUT_TOO_SMALL:  return "output_too_small";
        case COMMON_TREE_DRAFT_ANCESTOR_SIZE_OVERFLOW:     return "size_overflow";
    }
    return "unknown";
}
