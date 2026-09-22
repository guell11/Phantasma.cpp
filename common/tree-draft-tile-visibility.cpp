#include "tree-draft-tile-visibility.h"

#include <algorithm>
#include <limits>

static bool common_tree_draft_tile_visibility_validate_forest(
        const common_tree_draft_forest_offsets & forest,
        int32_t * n_nodes) {
    if (forest.n_entries < 0 || forest.offsets == nullptr || forest.offsets[0] != 0) {
        return false;
    }
    for (int32_t b = 0; b < forest.n_entries; ++b) {
        if (forest.offsets[b] < 0 || forest.offsets[b + 1] < forest.offsets[b]) {
            return false;
        }
    }
    *n_nodes = forest.offsets[forest.n_entries];
    return *n_nodes >= 0;
}

static int32_t common_tree_draft_tile_visibility_entry_for_node(
        const common_tree_draft_forest_offsets & forest,
        int32_t node) {
    int32_t lo = 0;
    int32_t hi = forest.n_entries;
    while (lo < hi) {
        const int32_t mid = lo + (hi - lo) / 2;
        if (forest.offsets[mid + 1] <= node) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

bool common_tree_draft_element_visible(
        const common_tree_draft_forest_offsets & forest,
        const common_tree_draft_ancestor_bitset & ancestors,
        int32_t query_node,
        int32_t key_node) {
    int32_t n_nodes = 0;
    if (!common_tree_draft_tile_visibility_validate_forest(forest, &n_nodes)) {
        return false;
    }
    if (query_node < 0 || query_node >= n_nodes || key_node < 0 || key_node >= n_nodes) {
        return false;
    }
    const int32_t query_entry = common_tree_draft_tile_visibility_entry_for_node(forest, query_node);
    const int32_t key_entry = common_tree_draft_tile_visibility_entry_for_node(forest, key_node);
    if (query_entry != key_entry || query_entry >= forest.n_entries) {
        return false;
    }
    return common_tree_draft_ancestor_contains(ancestors, n_nodes, query_node, key_node);
}

common_tree_draft_tile_visibility_error common_tree_draft_tile_visibility_build(
        const common_tree_draft_forest_offsets & forest,
        const common_tree_draft_ancestor_bitset & ancestors,
        int32_t query_tile_size,
        int32_t key_tile_size,
        common_tree_draft_tile_visibility & output) {
    int32_t n_nodes = 0;
    if (!common_tree_draft_tile_visibility_validate_forest(forest, &n_nodes)) {
        return COMMON_TREE_DRAFT_TILE_VISIBILITY_INVALID_FOREST;
    }
    if (query_tile_size <= 0 || key_tile_size <= 0) {
        return COMMON_TREE_DRAFT_TILE_VISIBILITY_INVALID_TILE_SIZE;
    }

    const size_t words_per_row = common_tree_draft_ancestor_words_per_row(n_nodes);
    if (n_nodes > 0) {
        if (ancestors.words == nullptr) {
            return COMMON_TREE_DRAFT_TILE_VISIBILITY_NULL_ANCESTOR;
        }
        if (words_per_row > std::numeric_limits<size_t>::max() / static_cast<size_t>(n_nodes)) {
            return COMMON_TREE_DRAFT_TILE_VISIBILITY_SIZE_OVERFLOW;
        }
        if (ancestors.word_count < words_per_row * static_cast<size_t>(n_nodes)) {
            return COMMON_TREE_DRAFT_TILE_VISIBILITY_ANCESTOR_TOO_SMALL;
        }
    }

    const int32_t n_query_tiles = n_nodes == 0 ? 0 : 1 + (n_nodes - 1) / query_tile_size;
    const int32_t n_key_tiles = n_nodes == 0 ? 0 : 1 + (n_nodes - 1) / key_tile_size;
    if (n_query_tiles > 0 && static_cast<size_t>(n_key_tiles) >
            std::numeric_limits<size_t>::max() / static_cast<size_t>(n_query_tiles)) {
        return COMMON_TREE_DRAFT_TILE_VISIBILITY_SIZE_OVERFLOW;
    }
    const size_t required = static_cast<size_t>(n_query_tiles) * static_cast<size_t>(n_key_tiles);
    if (required > 0 && output.active == nullptr) {
        return COMMON_TREE_DRAFT_TILE_VISIBILITY_NULL_OUTPUT;
    }
    if (output.active_count < required) {
        return COMMON_TREE_DRAFT_TILE_VISIBILITY_OUTPUT_TOO_SMALL;
    }

    for (int32_t qt = 0; qt < n_query_tiles; ++qt) {
        const int32_t q_begin = qt * query_tile_size;
        const int32_t q_end = std::min(n_nodes, q_begin + query_tile_size);
        for (int32_t kt = 0; kt < n_key_tiles; ++kt) {
            const int32_t k_begin = kt * key_tile_size;
            const int32_t k_end = std::min(n_nodes, k_begin + key_tile_size);
            bool active = false;
            for (int32_t i = q_begin; i < q_end && !active; ++i) {
                for (int32_t j = k_begin; j < k_end; ++j) {
                    if (common_tree_draft_element_visible(forest, ancestors, i, j)) {
                        active = true;
                        break;
                    }
                }
            }
            output.active[static_cast<size_t>(qt) * static_cast<size_t>(n_key_tiles) + static_cast<size_t>(kt)] =
                active ? 1 : 0;
        }
    }
    output.query_tile_size = query_tile_size;
    output.key_tile_size = key_tile_size;
    output.n_query_tiles = n_query_tiles;
    output.n_key_tiles = n_key_tiles;
    return COMMON_TREE_DRAFT_TILE_VISIBILITY_OK;
}

const char * common_tree_draft_tile_visibility_error_name(common_tree_draft_tile_visibility_error error) {
    switch (error) {
        case COMMON_TREE_DRAFT_TILE_VISIBILITY_OK:                 return "ok";
        case COMMON_TREE_DRAFT_TILE_VISIBILITY_INVALID_FOREST:     return "invalid_forest";
        case COMMON_TREE_DRAFT_TILE_VISIBILITY_INVALID_TILE_SIZE:  return "invalid_tile_size";
        case COMMON_TREE_DRAFT_TILE_VISIBILITY_NULL_ANCESTOR:      return "null_ancestor";
        case COMMON_TREE_DRAFT_TILE_VISIBILITY_ANCESTOR_TOO_SMALL: return "ancestor_too_small";
        case COMMON_TREE_DRAFT_TILE_VISIBILITY_SIZE_OVERFLOW:      return "size_overflow";
        case COMMON_TREE_DRAFT_TILE_VISIBILITY_NULL_OUTPUT:        return "null_output";
        case COMMON_TREE_DRAFT_TILE_VISIBILITY_OUTPUT_TOO_SMALL:   return "output_too_small";
        case COMMON_TREE_DRAFT_TILE_VISIBILITY_NODE_RANGE:         return "node_range";
    }
    return "unknown";
}
