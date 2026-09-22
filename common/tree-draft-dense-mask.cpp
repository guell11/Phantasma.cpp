#include "tree-draft-dense-mask.h"

#include <limits>

common_tree_draft_dense_mask_error common_tree_draft_dense_mask_build(
        const common_tree_draft_topology & topology,
        const common_tree_draft_ancestor_bitset & ancestors,
        common_tree_draft_dense_mask output) {
    if (common_tree_draft_topology_validate(topology) != COMMON_TREE_DRAFT_TOPOLOGY_OK) {
        return COMMON_TREE_DRAFT_DENSE_MASK_INVALID_TOPOLOGY;
    }
    if (topology.n_nodes == 0) {
        return COMMON_TREE_DRAFT_DENSE_MASK_OK;
    }

    const size_t n_nodes = static_cast<size_t>(topology.n_nodes);
    const size_t words_per_row = common_tree_draft_ancestor_words_per_row(topology.n_nodes);
    if (words_per_row > std::numeric_limits<size_t>::max() / n_nodes) {
        return COMMON_TREE_DRAFT_DENSE_MASK_SIZE_OVERFLOW;
    }
    const size_t required_words = n_nodes * words_per_row;
    if (ancestors.words == nullptr) {
        return COMMON_TREE_DRAFT_DENSE_MASK_NULL_ANCESTOR;
    }
    if (ancestors.word_count < required_words) {
        return COMMON_TREE_DRAFT_DENSE_MASK_ANCESTOR_TOO_SMALL;
    }
    if (n_nodes > std::numeric_limits<size_t>::max() / n_nodes) {
        return COMMON_TREE_DRAFT_DENSE_MASK_SIZE_OVERFLOW;
    }
    const size_t required_values = n_nodes * n_nodes;
    if (output.values == nullptr) {
        return COMMON_TREE_DRAFT_DENSE_MASK_NULL_OUTPUT;
    }
    if (output.value_count < required_values) {
        return COMMON_TREE_DRAFT_DENSE_MASK_OUTPUT_TOO_SMALL;
    }

    const float negative_infinity = -std::numeric_limits<float>::infinity();
    for (int32_t i = 0; i < topology.n_nodes; ++i) {
        for (int32_t j = 0; j < topology.n_nodes; ++j) {
            const bool visible = topology.tree_id[i] == topology.tree_id[j] &&
                common_tree_draft_ancestor_contains(ancestors, topology.n_nodes, i, j);
            output.values[static_cast<size_t>(i) * n_nodes + static_cast<size_t>(j)] = visible ? 0.0f : negative_infinity;
        }
    }

    return COMMON_TREE_DRAFT_DENSE_MASK_OK;
}

const char * common_tree_draft_dense_mask_error_name(common_tree_draft_dense_mask_error error) {
    switch (error) {
        case COMMON_TREE_DRAFT_DENSE_MASK_OK:                 return "ok";
        case COMMON_TREE_DRAFT_DENSE_MASK_INVALID_TOPOLOGY:   return "invalid_topology";
        case COMMON_TREE_DRAFT_DENSE_MASK_NULL_ANCESTOR:      return "null_ancestor";
        case COMMON_TREE_DRAFT_DENSE_MASK_ANCESTOR_TOO_SMALL: return "ancestor_too_small";
        case COMMON_TREE_DRAFT_DENSE_MASK_NULL_OUTPUT:        return "null_output";
        case COMMON_TREE_DRAFT_DENSE_MASK_OUTPUT_TOO_SMALL:   return "output_too_small";
        case COMMON_TREE_DRAFT_DENSE_MASK_SIZE_OVERFLOW:      return "size_overflow";
    }
    return "unknown";
}
