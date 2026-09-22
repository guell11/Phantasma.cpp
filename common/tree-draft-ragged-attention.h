#pragma once

#include "tree-draft-forest.h"

#include <cstddef>
#include <cstdint>

struct alignas(8) common_tree_draft_ragged_attention_entry {
    int32_t q_offset;
    int32_t q_len;
    int32_t prefix_len;
    int32_t tree_offset;
    int32_t tree_len;
    int32_t kv_base;
};

struct common_tree_draft_ragged_attention_batch {
    common_tree_draft_ragged_attention_entry * entries;
    size_t entry_capacity;
    int32_t n_entries;
    int32_t n_queries;
    int32_t n_tree_nodes;
};

enum common_tree_draft_ragged_attention_error : int32_t {
    COMMON_TREE_DRAFT_RAGGED_ATTENTION_OK = 0,
    COMMON_TREE_DRAFT_RAGGED_ATTENTION_INVALID_FOREST,
    COMMON_TREE_DRAFT_RAGGED_ATTENTION_NULL_PREFIX_LENGTHS,
    COMMON_TREE_DRAFT_RAGGED_ATTENTION_NULL_KV_BASES,
    COMMON_TREE_DRAFT_RAGGED_ATTENTION_NEGATIVE_PREFIX_LENGTH,
    COMMON_TREE_DRAFT_RAGGED_ATTENTION_NEGATIVE_KV_BASE,
    COMMON_TREE_DRAFT_RAGGED_ATTENTION_NULL_OUTPUT,
    COMMON_TREE_DRAFT_RAGGED_ATTENTION_OUTPUT_TOO_SMALL,
    COMMON_TREE_DRAFT_RAGGED_ATTENTION_RANGE_OVERFLOW,
};

common_tree_draft_ragged_attention_error common_tree_draft_ragged_attention_build(
        const common_tree_draft_forest_offsets & forest,
        const int32_t * prefix_lengths,
        const int32_t * kv_bases,
        common_tree_draft_ragged_attention_batch & output);

common_tree_draft_ragged_attention_error common_tree_draft_ragged_attention_validate(
        const common_tree_draft_ragged_attention_batch & batch);

const char * common_tree_draft_ragged_attention_error_name(common_tree_draft_ragged_attention_error error);
