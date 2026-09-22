#include "tree-draft-position.h"

#include <limits>

static common_tree_draft_position_error common_tree_draft_position_validate_inputs(
        const common_tree_draft_topology & topology,
        const common_tree_draft_forest_offsets & forest,
        const int32_t * prefix_lengths) {
    if (common_tree_draft_topology_validate(topology) != COMMON_TREE_DRAFT_TOPOLOGY_OK) {
        return COMMON_TREE_DRAFT_POSITION_INVALID_TOPOLOGY;
    }
    if (forest.n_entries < 0 || forest.offsets == nullptr || forest.offsets[0] != 0) {
        return COMMON_TREE_DRAFT_POSITION_INVALID_FOREST;
    }
    for (int32_t b = 0; b < forest.n_entries; ++b) {
        if (forest.offsets[b] < 0 || forest.offsets[b + 1] < forest.offsets[b]) {
            return COMMON_TREE_DRAFT_POSITION_INVALID_FOREST;
        }
        const int32_t begin = forest.offsets[b];
        const int32_t end = forest.offsets[b + 1];
        for (int32_t global = begin; global < end; ++global) {
            const int32_t parent = topology.parent[global];
            if (parent >= 0 && parent < begin) {
                return COMMON_TREE_DRAFT_POSITION_INVALID_FOREST;
            }
        }
    }
    if (forest.offsets[forest.n_entries] != topology.n_nodes) {
        return COMMON_TREE_DRAFT_POSITION_INVALID_FOREST;
    }
    if (forest.n_entries > 0 && prefix_lengths == nullptr) {
        return COMMON_TREE_DRAFT_POSITION_NULL_PREFIX_LENGTHS;
    }
    for (int32_t b = 0; b < forest.n_entries; ++b) {
        if (prefix_lengths[b] < 0) {
            return COMMON_TREE_DRAFT_POSITION_NEGATIVE_PREFIX_LENGTH;
        }
    }
    return COMMON_TREE_DRAFT_POSITION_OK;
}

static common_tree_draft_position_error common_tree_draft_position_checked_sum(
        int32_t prefix_length,
        int32_t depth,
        int32_t * position) {
    if (depth < 0 || prefix_length > std::numeric_limits<int32_t>::max() - depth) {
        return COMMON_TREE_DRAFT_POSITION_OVERFLOW;
    }
    *position = prefix_length + depth;
    return COMMON_TREE_DRAFT_POSITION_OK;
}

common_tree_draft_position_error common_tree_draft_positions_build(
        const common_tree_draft_topology & topology,
        const common_tree_draft_forest_offsets & forest,
        const int32_t * prefix_lengths,
        common_tree_draft_positions output) {
    const common_tree_draft_position_error input_error =
        common_tree_draft_position_validate_inputs(topology, forest, prefix_lengths);
    if (input_error != COMMON_TREE_DRAFT_POSITION_OK) {
        return input_error;
    }
    if (topology.n_nodes == 0) {
        return COMMON_TREE_DRAFT_POSITION_OK;
    }
    if (output.values == nullptr) {
        return COMMON_TREE_DRAFT_POSITION_NULL_OUTPUT;
    }
    const size_t required = static_cast<size_t>(topology.n_nodes);
    if (output.value_count < required) {
        return COMMON_TREE_DRAFT_POSITION_OUTPUT_TOO_SMALL;
    }

    for (int32_t b = 0; b < forest.n_entries; ++b) {
        const int32_t begin = forest.offsets[b];
        const int32_t end = forest.offsets[b + 1];
        for (int32_t global = begin; global < end; ++global) {
            int32_t position = 0;
            const common_tree_draft_position_error error =
                common_tree_draft_position_checked_sum(prefix_lengths[b], topology.depth[global], &position);
            if (error != COMMON_TREE_DRAFT_POSITION_OK) {
                return error;
            }
        }
    }

    for (int32_t b = 0; b < forest.n_entries; ++b) {
        const int32_t begin = forest.offsets[b];
        const int32_t end = forest.offsets[b + 1];
        for (int32_t global = begin; global < end; ++global) {
            output.values[global] = prefix_lengths[b] + topology.depth[global];
        }
    }
    return COMMON_TREE_DRAFT_POSITION_OK;
}

common_tree_draft_position_error common_tree_draft_position_for_local(
        const common_tree_draft_topology & topology,
        const common_tree_draft_forest_offsets & forest,
        const int32_t * prefix_lengths,
        int32_t entry,
        int32_t local,
        int32_t * global,
        int32_t * position) {
    const common_tree_draft_position_error input_error =
        common_tree_draft_position_validate_inputs(topology, forest, prefix_lengths);
    if (input_error != COMMON_TREE_DRAFT_POSITION_OK) {
        return input_error;
    }
    if (global == nullptr || position == nullptr) {
        return COMMON_TREE_DRAFT_POSITION_NULL_MAPPING_OUTPUT;
    }
    if (entry < 0 || entry >= forest.n_entries) {
        return COMMON_TREE_DRAFT_POSITION_ENTRY_RANGE;
    }

    int32_t mapped_global = -1;
    const common_tree_draft_forest_error map_error =
        common_tree_draft_forest_local_to_global(forest, entry, local, &mapped_global);
    if (map_error == COMMON_TREE_DRAFT_FOREST_LOCAL_RANGE) {
        return COMMON_TREE_DRAFT_POSITION_LOCAL_RANGE;
    }
    if (map_error != COMMON_TREE_DRAFT_FOREST_OK) {
        return COMMON_TREE_DRAFT_POSITION_INVALID_FOREST;
    }

    int32_t mapped_position = 0;
    const common_tree_draft_position_error sum_error =
        common_tree_draft_position_checked_sum(prefix_lengths[entry], topology.depth[mapped_global], &mapped_position);
    if (sum_error != COMMON_TREE_DRAFT_POSITION_OK) {
        return sum_error;
    }

    *global = mapped_global;
    *position = mapped_position;
    return COMMON_TREE_DRAFT_POSITION_OK;
}

const char * common_tree_draft_position_error_name(common_tree_draft_position_error error) {
    switch (error) {
        case COMMON_TREE_DRAFT_POSITION_OK:                     return "ok";
        case COMMON_TREE_DRAFT_POSITION_INVALID_TOPOLOGY:       return "invalid_topology";
        case COMMON_TREE_DRAFT_POSITION_INVALID_FOREST:         return "invalid_forest";
        case COMMON_TREE_DRAFT_POSITION_NULL_PREFIX_LENGTHS:    return "null_prefix_lengths";
        case COMMON_TREE_DRAFT_POSITION_NEGATIVE_PREFIX_LENGTH: return "negative_prefix_length";
        case COMMON_TREE_DRAFT_POSITION_NULL_OUTPUT:            return "null_output";
        case COMMON_TREE_DRAFT_POSITION_OUTPUT_TOO_SMALL:       return "output_too_small";
        case COMMON_TREE_DRAFT_POSITION_OVERFLOW:               return "overflow";
        case COMMON_TREE_DRAFT_POSITION_ENTRY_RANGE:            return "entry_range";
        case COMMON_TREE_DRAFT_POSITION_LOCAL_RANGE:            return "local_range";
        case COMMON_TREE_DRAFT_POSITION_NULL_MAPPING_OUTPUT:     return "null_mapping_output";
    }
    return "unknown";
}
