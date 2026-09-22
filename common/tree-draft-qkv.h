#pragma once

#include "tree-draft-position.h"

#include <cstddef>
#include <cstdint>

struct common_tree_draft_qkv_view {
    const uint8_t * data;
    int32_t n_nodes;
    int32_t n_heads;
    int32_t head_dim;
    size_t stride_node;
    size_t stride_head;
    size_t stride_dim;
    size_t element_size;
};

struct common_tree_draft_qkv_output {
    uint8_t * data;
    size_t byte_count;
};

enum common_tree_draft_qkv_error : int32_t {
    COMMON_TREE_DRAFT_QKV_OK = 0,
    COMMON_TREE_DRAFT_QKV_INVALID_FOREST,
    COMMON_TREE_DRAFT_QKV_NULL_POSITIONS,
    COMMON_TREE_DRAFT_QKV_POSITION_COUNT_MISMATCH,
    COMMON_TREE_DRAFT_QKV_INVALID_SHAPE,
    COMMON_TREE_DRAFT_QKV_NULL_SOURCE,
    COMMON_TREE_DRAFT_QKV_ZERO_ELEMENT_SIZE,
    COMMON_TREE_DRAFT_QKV_STRIDE_OVERFLOW,
    COMMON_TREE_DRAFT_QKV_NULL_OUTPUT,
    COMMON_TREE_DRAFT_QKV_OUTPUT_TOO_SMALL,
};

size_t common_tree_draft_qkv_packed_bytes(const common_tree_draft_qkv_view & view, bool * overflow);

common_tree_draft_qkv_error common_tree_draft_qkv_gather(
        const common_tree_draft_forest_offsets & forest,
        const int32_t * positions,
        size_t position_count,
        const common_tree_draft_qkv_view & source,
        common_tree_draft_qkv_output output);

const char * common_tree_draft_qkv_error_name(common_tree_draft_qkv_error error);
