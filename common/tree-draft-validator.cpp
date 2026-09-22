#include "tree-draft-validator.h"

common_tree_draft_validation_result common_tree_draft_validate(
        const common_tree_draft_topology & topology,
        const common_tree_draft_forest_offsets & forest) {
    const common_tree_draft_topology_error topology_error = common_tree_draft_topology_validate(topology);
    if (topology_error != COMMON_TREE_DRAFT_TOPOLOGY_OK) {
        return { COMMON_TREE_DRAFT_VALIDATOR_INVALID_TOPOLOGY, topology_error };
    }
    if (forest.n_entries < 0) {
        return { COMMON_TREE_DRAFT_VALIDATOR_NEGATIVE_ENTRY_COUNT, COMMON_TREE_DRAFT_TOPOLOGY_OK };
    }
    if (forest.offsets == nullptr) {
        return { COMMON_TREE_DRAFT_VALIDATOR_NULL_OFFSETS, COMMON_TREE_DRAFT_TOPOLOGY_OK };
    }
    if (forest.offsets[0] != 0) {
        return { COMMON_TREE_DRAFT_VALIDATOR_OFFSET_START, COMMON_TREE_DRAFT_TOPOLOGY_OK };
    }

    int32_t previous = 0;
    for (int32_t b = 0; b < forest.n_entries; ++b) {
        const int32_t next = forest.offsets[b + 1];
        if (next < 0) {
            return { COMMON_TREE_DRAFT_VALIDATOR_NEGATIVE_OFFSET, COMMON_TREE_DRAFT_TOPOLOGY_OK };
        }
        if (next < previous) {
            return { COMMON_TREE_DRAFT_VALIDATOR_OFFSET_ORDER, COMMON_TREE_DRAFT_TOPOLOGY_OK };
        }
        if (next > topology.n_nodes) {
            return { COMMON_TREE_DRAFT_VALIDATOR_OFFSET_RANGE, COMMON_TREE_DRAFT_TOPOLOGY_OK };
        }
        previous = next;
    }

    if (previous != topology.n_nodes) {
        return { COMMON_TREE_DRAFT_VALIDATOR_OFFSET_END, COMMON_TREE_DRAFT_TOPOLOGY_OK };
    }

    return { COMMON_TREE_DRAFT_VALIDATOR_OK, COMMON_TREE_DRAFT_TOPOLOGY_OK };
}

const char * common_tree_draft_validator_error_name(common_tree_draft_validator_error error) {
    switch (error) {
        case COMMON_TREE_DRAFT_VALIDATOR_OK:                   return "ok";
        case COMMON_TREE_DRAFT_VALIDATOR_INVALID_TOPOLOGY:     return "invalid_topology";
        case COMMON_TREE_DRAFT_VALIDATOR_NEGATIVE_ENTRY_COUNT: return "negative_entry_count";
        case COMMON_TREE_DRAFT_VALIDATOR_NULL_OFFSETS:         return "null_offsets";
        case COMMON_TREE_DRAFT_VALIDATOR_OFFSET_START:         return "offset_start";
        case COMMON_TREE_DRAFT_VALIDATOR_NEGATIVE_OFFSET:      return "negative_offset";
        case COMMON_TREE_DRAFT_VALIDATOR_OFFSET_ORDER:         return "offset_order";
        case COMMON_TREE_DRAFT_VALIDATOR_OFFSET_RANGE:         return "offset_range";
        case COMMON_TREE_DRAFT_VALIDATOR_OFFSET_END:           return "offset_end";
    }
    return "unknown";
}
