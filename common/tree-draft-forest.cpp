#include "tree-draft-forest.h"

#include <limits>

common_tree_draft_forest_error common_tree_draft_forest_offsets_build(
        const common_tree_draft_topology & topology,
        const int32_t * node_counts,
        int32_t n_entries,
        int32_t * offsets,
        size_t offsets_capacity) {
    if (topology.n_nodes < 0) {
        return COMMON_TREE_DRAFT_FOREST_INVALID_TOPOLOGY;
    }
    if (n_entries < 0) {
        return COMMON_TREE_DRAFT_FOREST_NEGATIVE_ENTRY_COUNT;
    }
    if (n_entries > 0 && node_counts == nullptr) {
        return COMMON_TREE_DRAFT_FOREST_NULL_NODE_COUNTS;
    }
    if (offsets == nullptr) {
        return COMMON_TREE_DRAFT_FOREST_NULL_OFFSETS;
    }

    const size_t required = static_cast<size_t>(n_entries) + 1;
    if (offsets_capacity < required) {
        return COMMON_TREE_DRAFT_FOREST_OFFSETS_BUFFER_TOO_SMALL;
    }

    offsets[0] = 0;
    int32_t total = 0;
    for (int32_t b = 0; b < n_entries; ++b) {
        const int32_t count = node_counts[b];
        if (count < 0) {
            return COMMON_TREE_DRAFT_FOREST_NEGATIVE_NODE_COUNT;
        }
        if (count > std::numeric_limits<int32_t>::max() - total) {
            return COMMON_TREE_DRAFT_FOREST_NODE_COUNT_OVERFLOW;
        }
        total += count;
        offsets[b + 1] = total;
    }

    if (total != topology.n_nodes) {
        return COMMON_TREE_DRAFT_FOREST_NODE_COUNT_MISMATCH;
    }
    return COMMON_TREE_DRAFT_FOREST_OK;
}

common_tree_draft_forest_error common_tree_draft_forest_local_to_global(
        const common_tree_draft_forest_offsets & forest,
        int32_t entry,
        int32_t local,
        int32_t * global) {
    if (forest.n_entries < 0 || forest.offsets == nullptr) {
        return COMMON_TREE_DRAFT_FOREST_NULL_OFFSETS;
    }
    if (entry < 0 || entry >= forest.n_entries) {
        return COMMON_TREE_DRAFT_FOREST_ENTRY_RANGE;
    }

    const int32_t begin = forest.offsets[entry];
    const int32_t end = forest.offsets[entry + 1];
    if (local < 0 || local >= end - begin) {
        return COMMON_TREE_DRAFT_FOREST_LOCAL_RANGE;
    }
    if (global == nullptr) {
        return COMMON_TREE_DRAFT_FOREST_NULL_OUTPUT;
    }
    *global = begin + local;
    return COMMON_TREE_DRAFT_FOREST_OK;
}

common_tree_draft_forest_error common_tree_draft_forest_global_to_local(
        const common_tree_draft_forest_offsets & forest,
        int32_t global,
        int32_t * entry,
        int32_t * local) {
    if (forest.n_entries < 0 || forest.offsets == nullptr) {
        return COMMON_TREE_DRAFT_FOREST_NULL_OFFSETS;
    }
    if (entry == nullptr || local == nullptr) {
        return COMMON_TREE_DRAFT_FOREST_NULL_OUTPUT;
    }
    if (global < 0 || global >= forest.offsets[forest.n_entries]) {
        return COMMON_TREE_DRAFT_FOREST_GLOBAL_RANGE;
    }

    int32_t lo = 0;
    int32_t hi = forest.n_entries;
    while (lo < hi) {
        const int32_t mid = lo + (hi - lo) / 2;
        if (forest.offsets[mid + 1] <= global) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    *entry = lo;
    *local = global - forest.offsets[lo];
    return COMMON_TREE_DRAFT_FOREST_OK;
}

const char * common_tree_draft_forest_error_name(common_tree_draft_forest_error error) {
    switch (error) {
        case COMMON_TREE_DRAFT_FOREST_OK:                       return "ok";
        case COMMON_TREE_DRAFT_FOREST_INVALID_TOPOLOGY:         return "invalid_topology";
        case COMMON_TREE_DRAFT_FOREST_NEGATIVE_ENTRY_COUNT:     return "negative_entry_count";
        case COMMON_TREE_DRAFT_FOREST_NULL_NODE_COUNTS:         return "null_node_counts";
        case COMMON_TREE_DRAFT_FOREST_NULL_OFFSETS:             return "null_offsets";
        case COMMON_TREE_DRAFT_FOREST_NULL_OUTPUT:              return "null_output";
        case COMMON_TREE_DRAFT_FOREST_OFFSETS_BUFFER_TOO_SMALL: return "offsets_buffer_too_small";
        case COMMON_TREE_DRAFT_FOREST_NEGATIVE_NODE_COUNT:      return "negative_node_count";
        case COMMON_TREE_DRAFT_FOREST_NODE_COUNT_OVERFLOW:      return "node_count_overflow";
        case COMMON_TREE_DRAFT_FOREST_NODE_COUNT_MISMATCH:      return "node_count_mismatch";
        case COMMON_TREE_DRAFT_FOREST_ENTRY_RANGE:              return "entry_range";
        case COMMON_TREE_DRAFT_FOREST_LOCAL_RANGE:              return "local_range";
        case COMMON_TREE_DRAFT_FOREST_GLOBAL_RANGE:             return "global_range";
    }
    return "unknown";
}
