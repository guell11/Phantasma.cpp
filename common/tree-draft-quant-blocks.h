#pragma once

#include "tree-draft-gguf-tensors.h"
#include "tree-draft-quant-registry.h"

#include <cstdint>

struct common_tree_draft_quant_block_plan {
    uint64_t logical_elements = 0;
    uint64_t block_count = 0;
    uint32_t block_elems = 0;
    uint32_t block_bytes = 0;
    uint64_t encoded_bytes = 0;
};

struct common_tree_draft_quant_block_span {
    uint64_t logical_begin = 0;
    uint64_t logical_end = 0;
    uint64_t byte_begin = 0;
    uint64_t byte_end = 0;
    uint64_t first_block = 0;
    uint64_t block_count = 0;
};

enum common_tree_draft_quant_block_status : uint32_t {
    COMMON_TREE_DRAFT_QUANT_BLOCK_OK = 0,
    COMMON_TREE_DRAFT_QUANT_BLOCK_UNREGISTERED,
    COMMON_TREE_DRAFT_QUANT_BLOCK_DIMENSION,
    COMMON_TREE_DRAFT_QUANT_BLOCK_OVERFLOW,
    COMMON_TREE_DRAFT_QUANT_BLOCK_SIZE_MISMATCH,
    COMMON_TREE_DRAFT_QUANT_BLOCK_RANGE,
    COMMON_TREE_DRAFT_QUANT_BLOCK_ALIGNMENT,
};

common_tree_draft_quant_block_status common_tree_draft_quant_block_plan_build(
        const common_tree_draft_gguf_tensor_descriptor & tensor,
        common_tree_draft_quant_block_plan * plan);

common_tree_draft_quant_block_status common_tree_draft_quant_block_span_map(
        const common_tree_draft_quant_block_plan & plan,
        uint64_t logical_begin,
        uint64_t logical_end,
        common_tree_draft_quant_block_span * span);

