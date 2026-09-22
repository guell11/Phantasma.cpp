#include "tree-draft-composite-mask.h"

#include <limits>

static common_tree_draft_composite_mask_error map_position_error(common_tree_draft_position_error error) {
    switch (error) {
        case COMMON_TREE_DRAFT_POSITION_OK:
            return COMMON_TREE_DRAFT_COMPOSITE_MASK_OK;
        case COMMON_TREE_DRAFT_POSITION_INVALID_TOPOLOGY:
            return COMMON_TREE_DRAFT_COMPOSITE_MASK_INVALID_TOPOLOGY;
        case COMMON_TREE_DRAFT_POSITION_INVALID_FOREST:
            return COMMON_TREE_DRAFT_COMPOSITE_MASK_INVALID_FOREST;
        case COMMON_TREE_DRAFT_POSITION_NULL_PREFIX_LENGTHS:
        case COMMON_TREE_DRAFT_POSITION_NEGATIVE_PREFIX_LENGTH:
        case COMMON_TREE_DRAFT_POSITION_OVERFLOW:
            return COMMON_TREE_DRAFT_COMPOSITE_MASK_INVALID_PREFIX_LENGTH;
        case COMMON_TREE_DRAFT_POSITION_ENTRY_RANGE:
            return COMMON_TREE_DRAFT_COMPOSITE_MASK_ENTRY_RANGE;
        case COMMON_TREE_DRAFT_POSITION_LOCAL_RANGE:
            return COMMON_TREE_DRAFT_COMPOSITE_MASK_QUERY_RANGE;
        case COMMON_TREE_DRAFT_POSITION_NULL_OUTPUT:
        case COMMON_TREE_DRAFT_POSITION_OUTPUT_TOO_SMALL:
        case COMMON_TREE_DRAFT_POSITION_NULL_MAPPING_OUTPUT:
            return COMMON_TREE_DRAFT_COMPOSITE_MASK_NULL_OUTPUT;
    }
    return COMMON_TREE_DRAFT_COMPOSITE_MASK_INVALID_FOREST;
}

common_tree_draft_composite_mask_error common_tree_draft_composite_mask_visible(
        const common_tree_draft_topology & topology,
        const common_tree_draft_ancestor_bitset & ancestors,
        const common_tree_draft_forest_offsets & forest,
        const int32_t * prefix_lengths,
        int32_t entry,
        int32_t query_local,
        common_tree_draft_composite_key_kind key_kind,
        int32_t key_index,
        bool * visible) {
    if (visible == nullptr) {
        return COMMON_TREE_DRAFT_COMPOSITE_MASK_NULL_OUTPUT;
    }

    int32_t query_global = -1;
    int32_t query_position = -1;
    const common_tree_draft_position_error position_error = common_tree_draft_position_for_local(
        topology, forest, prefix_lengths, entry, query_local, &query_global, &query_position);
    const common_tree_draft_composite_mask_error mapped_error = map_position_error(position_error);
    if (mapped_error != COMMON_TREE_DRAFT_COMPOSITE_MASK_OK) {
        return mapped_error;
    }
    (void) query_position;

    const size_t words_per_row = common_tree_draft_ancestor_words_per_row(topology.n_nodes);
    const size_t n_nodes = static_cast<size_t>(topology.n_nodes);
    if (n_nodes > 0 && ancestors.words == nullptr) {
        return COMMON_TREE_DRAFT_COMPOSITE_MASK_NULL_ANCESTOR;
    }
    if (words_per_row != 0 && n_nodes > std::numeric_limits<size_t>::max() / words_per_row) {
        return COMMON_TREE_DRAFT_COMPOSITE_MASK_ANCESTOR_TOO_SMALL;
    }
    if (ancestors.word_count < n_nodes * words_per_row) {
        return COMMON_TREE_DRAFT_COMPOSITE_MASK_ANCESTOR_TOO_SMALL;
    }

    switch (key_kind) {
        case COMMON_TREE_DRAFT_COMPOSITE_KEY_PREFIX:
            if (key_index < 0 || key_index >= prefix_lengths[entry]) {
                return COMMON_TREE_DRAFT_COMPOSITE_MASK_KEY_RANGE;
            }
            *visible = true;
            return COMMON_TREE_DRAFT_COMPOSITE_MASK_OK;

        case COMMON_TREE_DRAFT_COMPOSITE_KEY_PROPOSAL: {
            int32_t key_global = -1;
            const common_tree_draft_forest_error forest_error =
                common_tree_draft_forest_local_to_global(forest, entry, key_index, &key_global);
            if (forest_error == COMMON_TREE_DRAFT_FOREST_LOCAL_RANGE) {
                return COMMON_TREE_DRAFT_COMPOSITE_MASK_KEY_RANGE;
            }
            if (forest_error != COMMON_TREE_DRAFT_FOREST_OK) {
                return COMMON_TREE_DRAFT_COMPOSITE_MASK_INVALID_FOREST;
            }
            *visible = topology.tree_id[query_global] == topology.tree_id[key_global] &&
                common_tree_draft_ancestor_contains(ancestors, topology.n_nodes, query_global, key_global);
            return COMMON_TREE_DRAFT_COMPOSITE_MASK_OK;
        }
    }

    return COMMON_TREE_DRAFT_COMPOSITE_MASK_INVALID_KEY_KIND;
}

float common_tree_draft_composite_mask_value(bool visible) {
    return visible ? 0.0f : -std::numeric_limits<float>::infinity();
}

const char * common_tree_draft_composite_mask_error_name(common_tree_draft_composite_mask_error error) {
    switch (error) {
        case COMMON_TREE_DRAFT_COMPOSITE_MASK_OK:                    return "ok";
        case COMMON_TREE_DRAFT_COMPOSITE_MASK_INVALID_TOPOLOGY:      return "invalid_topology";
        case COMMON_TREE_DRAFT_COMPOSITE_MASK_INVALID_FOREST:        return "invalid_forest";
        case COMMON_TREE_DRAFT_COMPOSITE_MASK_INVALID_PREFIX_LENGTH: return "invalid_prefix_length";
        case COMMON_TREE_DRAFT_COMPOSITE_MASK_NULL_ANCESTOR:         return "null_ancestor";
        case COMMON_TREE_DRAFT_COMPOSITE_MASK_ANCESTOR_TOO_SMALL:    return "ancestor_too_small";
        case COMMON_TREE_DRAFT_COMPOSITE_MASK_ENTRY_RANGE:           return "entry_range";
        case COMMON_TREE_DRAFT_COMPOSITE_MASK_QUERY_RANGE:           return "query_range";
        case COMMON_TREE_DRAFT_COMPOSITE_MASK_KEY_RANGE:             return "key_range";
        case COMMON_TREE_DRAFT_COMPOSITE_MASK_INVALID_KEY_KIND:      return "invalid_key_kind";
        case COMMON_TREE_DRAFT_COMPOSITE_MASK_NULL_OUTPUT:           return "null_output";
    }
    return "unknown";
}
