#pragma once

#include "tree-draft-kv-gather.h"

#include <cstddef>
#include <cstdint>

struct alignas(16) common_tree_draft_kv_gpu_segment {
    uint32_t page_id;
    uint32_t generation;
    uint32_t lo;
    uint32_t hi;
};

static_assert(sizeof(common_tree_draft_kv_gpu_segment) == 16, "GPU segment ABI must be 16 bytes");
static_assert(alignof(common_tree_draft_kv_gpu_segment) == 16, "GPU segment ABI must be 16-byte aligned");

struct common_tree_draft_kv_block_table {
    uint32_t * row_ptr = nullptr;
    size_t row_ptr_capacity = 0;
    common_tree_draft_kv_gpu_segment * segments = nullptr;
    size_t segment_capacity = 0;
    uint32_t query_count = 0;
    uint32_t segment_count = 0;
};

enum common_tree_draft_kv_block_table_status : uint32_t {
    COMMON_TREE_DRAFT_KV_BLOCK_TABLE_OK = 0,
    COMMON_TREE_DRAFT_KV_BLOCK_TABLE_NULL_BUFFER,
    COMMON_TREE_DRAFT_KV_BLOCK_TABLE_RANGE,
    COMMON_TREE_DRAFT_KV_BLOCK_TABLE_OUTPUT_TOO_SMALL,
};

common_tree_draft_kv_block_table_status common_tree_draft_kv_block_table_pack(
        const common_tree_draft_kv_gather_metadata & gather,
        common_tree_draft_kv_block_table * output);
