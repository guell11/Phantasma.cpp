#include "tree-draft-ragged-attention.h"

#include <limits>
#include <type_traits>

static_assert(std::is_standard_layout<common_tree_draft_ragged_attention_entry>::value, "ragged attention entry must be POD-like");
static_assert(sizeof(common_tree_draft_ragged_attention_entry) == 24, "ragged attention entry ABI changed");
static_assert(alignof(common_tree_draft_ragged_attention_entry) == 8, "ragged attention entry alignment changed");

static common_tree_draft_ragged_attention_error common_tree_draft_ragged_attention_validate_forest(
        const common_tree_draft_forest_offsets & forest,
        int32_t * n_nodes) {
    if (forest.n_entries < 0 || forest.offsets == nullptr || forest.offsets[0] != 0) {
        return COMMON_TREE_DRAFT_RAGGED_ATTENTION_INVALID_FOREST;
    }

    int32_t previous = 0;
    for (int32_t entry = 0; entry < forest.n_entries; ++entry) {
        const int32_t next = forest.offsets[entry + 1];
        if (next < previous || next < 0) {
            return COMMON_TREE_DRAFT_RAGGED_ATTENTION_INVALID_FOREST;
        }
        previous = next;
    }

    *n_nodes = previous;
    return COMMON_TREE_DRAFT_RAGGED_ATTENTION_OK;
}

common_tree_draft_ragged_attention_error common_tree_draft_ragged_attention_build(
        const common_tree_draft_forest_offsets & forest,
        const int32_t * prefix_lengths,
        const int32_t * kv_bases,
        common_tree_draft_ragged_attention_batch & output) {
    int32_t n_nodes = 0;
    const common_tree_draft_ragged_attention_error forest_error =
        common_tree_draft_ragged_attention_validate_forest(forest, &n_nodes);
    if (forest_error != COMMON_TREE_DRAFT_RAGGED_ATTENTION_OK) {
        return forest_error;
    }
    if (forest.n_entries > 0 && prefix_lengths == nullptr) {
        return COMMON_TREE_DRAFT_RAGGED_ATTENTION_NULL_PREFIX_LENGTHS;
    }
    if (forest.n_entries > 0 && kv_bases == nullptr) {
        return COMMON_TREE_DRAFT_RAGGED_ATTENTION_NULL_KV_BASES;
    }
    if (forest.n_entries > 0 && output.entries == nullptr) {
        return COMMON_TREE_DRAFT_RAGGED_ATTENTION_NULL_OUTPUT;
    }
    if (output.entry_capacity < static_cast<size_t>(forest.n_entries)) {
        return COMMON_TREE_DRAFT_RAGGED_ATTENTION_OUTPUT_TOO_SMALL;
    }

    for (int32_t entry = 0; entry < forest.n_entries; ++entry) {
        if (prefix_lengths[entry] < 0) {
            return COMMON_TREE_DRAFT_RAGGED_ATTENTION_NEGATIVE_PREFIX_LENGTH;
        }
        if (kv_bases[entry] < 0) {
            return COMMON_TREE_DRAFT_RAGGED_ATTENTION_NEGATIVE_KV_BASE;
        }

        const int32_t tree_offset = forest.offsets[entry];
        const int32_t tree_end = forest.offsets[entry + 1];
        const int32_t tree_len = tree_end - tree_offset;
        if (prefix_lengths[entry] > std::numeric_limits<int32_t>::max() - tree_len) {
            return COMMON_TREE_DRAFT_RAGGED_ATTENTION_RANGE_OVERFLOW;
        }
        const int32_t kv_len = prefix_lengths[entry] + tree_len;
        if (kv_bases[entry] > std::numeric_limits<int32_t>::max() - kv_len) {
            return COMMON_TREE_DRAFT_RAGGED_ATTENTION_RANGE_OVERFLOW;
        }
    }

    for (int32_t entry = 0; entry < forest.n_entries; ++entry) {
        const int32_t tree_offset = forest.offsets[entry];
        const int32_t tree_len = forest.offsets[entry + 1] - tree_offset;
        output.entries[entry] = {
            tree_offset,
            tree_len,
            prefix_lengths[entry],
            tree_offset,
            tree_len,
            kv_bases[entry],
        };
    }
    output.n_entries = forest.n_entries;
    output.n_queries = n_nodes;
    output.n_tree_nodes = n_nodes;
    return COMMON_TREE_DRAFT_RAGGED_ATTENTION_OK;
}

common_tree_draft_ragged_attention_error common_tree_draft_ragged_attention_validate(
        const common_tree_draft_ragged_attention_batch & batch) {
    if (batch.n_entries < 0 || batch.n_queries < 0 || batch.n_tree_nodes < 0) {
        return COMMON_TREE_DRAFT_RAGGED_ATTENTION_INVALID_FOREST;
    }
    if (batch.n_entries > 0 && batch.entries == nullptr) {
        return COMMON_TREE_DRAFT_RAGGED_ATTENTION_NULL_OUTPUT;
    }
    if (batch.entry_capacity < static_cast<size_t>(batch.n_entries)) {
        return COMMON_TREE_DRAFT_RAGGED_ATTENTION_OUTPUT_TOO_SMALL;
    }

    int32_t expected_q_offset = 0;
    int32_t expected_tree_offset = 0;
    for (int32_t entry = 0; entry < batch.n_entries; ++entry) {
        const common_tree_draft_ragged_attention_entry & descriptor = batch.entries[entry];
        if (descriptor.q_offset != expected_q_offset || descriptor.tree_offset != expected_tree_offset ||
            descriptor.q_len < 0 || descriptor.tree_len < 0) {
            return COMMON_TREE_DRAFT_RAGGED_ATTENTION_INVALID_FOREST;
        }
        if (descriptor.prefix_len < 0) {
            return COMMON_TREE_DRAFT_RAGGED_ATTENTION_NEGATIVE_PREFIX_LENGTH;
        }
        if (descriptor.kv_base < 0) {
            return COMMON_TREE_DRAFT_RAGGED_ATTENTION_NEGATIVE_KV_BASE;
        }
        if (descriptor.q_len > std::numeric_limits<int32_t>::max() - expected_q_offset ||
            descriptor.tree_len > std::numeric_limits<int32_t>::max() - expected_tree_offset ||
            descriptor.prefix_len > std::numeric_limits<int32_t>::max() - descriptor.tree_len) {
            return COMMON_TREE_DRAFT_RAGGED_ATTENTION_RANGE_OVERFLOW;
        }
        const int32_t kv_len = descriptor.prefix_len + descriptor.tree_len;
        if (descriptor.kv_base > std::numeric_limits<int32_t>::max() - kv_len) {
            return COMMON_TREE_DRAFT_RAGGED_ATTENTION_RANGE_OVERFLOW;
        }
        expected_q_offset += descriptor.q_len;
        expected_tree_offset += descriptor.tree_len;
    }

    if (expected_q_offset != batch.n_queries || expected_tree_offset != batch.n_tree_nodes) {
        return COMMON_TREE_DRAFT_RAGGED_ATTENTION_INVALID_FOREST;
    }
    return COMMON_TREE_DRAFT_RAGGED_ATTENTION_OK;
}

const char * common_tree_draft_ragged_attention_error_name(common_tree_draft_ragged_attention_error error) {
    switch (error) {
        case COMMON_TREE_DRAFT_RAGGED_ATTENTION_OK:                     return "ok";
        case COMMON_TREE_DRAFT_RAGGED_ATTENTION_INVALID_FOREST:         return "invalid_forest";
        case COMMON_TREE_DRAFT_RAGGED_ATTENTION_NULL_PREFIX_LENGTHS:    return "null_prefix_lengths";
        case COMMON_TREE_DRAFT_RAGGED_ATTENTION_NULL_KV_BASES:          return "null_kv_bases";
        case COMMON_TREE_DRAFT_RAGGED_ATTENTION_NEGATIVE_PREFIX_LENGTH: return "negative_prefix_length";
        case COMMON_TREE_DRAFT_RAGGED_ATTENTION_NEGATIVE_KV_BASE:       return "negative_kv_base";
        case COMMON_TREE_DRAFT_RAGGED_ATTENTION_NULL_OUTPUT:            return "null_output";
        case COMMON_TREE_DRAFT_RAGGED_ATTENTION_OUTPUT_TOO_SMALL:       return "output_too_small";
        case COMMON_TREE_DRAFT_RAGGED_ATTENTION_RANGE_OVERFLOW:         return "range_overflow";
    }
    return "unknown";
}
