#pragma once

#include "tree-draft-qkv.h"
#include "tree-draft-topology.h"

#include <cstddef>
#include <cstdint>

static constexpr uint32_t COMMON_TREE_DRAFT_KV_SLOT_INVALID = UINT32_MAX;

struct common_tree_draft_kv_indirection {
    const uint32_t * slots;
    size_t slot_count;
    uint32_t kv_size;
};

struct common_tree_draft_kv_layout {
    const uint8_t * data;
    uint32_t kv_size;
    int32_t n_heads;
    int32_t head_dim;
    size_t stride_slot;
    size_t stride_head;
    size_t stride_dim;
    size_t element_size;
};

enum common_tree_draft_kv_indirection_error : int32_t {
    COMMON_TREE_DRAFT_KV_INDIRECTION_OK = 0,
    COMMON_TREE_DRAFT_KV_INDIRECTION_INVALID_TOPOLOGY,
    COMMON_TREE_DRAFT_KV_INDIRECTION_NULL_SLOTS,
    COMMON_TREE_DRAFT_KV_INDIRECTION_SLOT_COUNT_MISMATCH,
    COMMON_TREE_DRAFT_KV_INDIRECTION_SLOT_RANGE,
    COMMON_TREE_DRAFT_KV_INDIRECTION_LOGICAL_RANGE,
    COMMON_TREE_DRAFT_KV_INDIRECTION_INVALID_SLOT,
    COMMON_TREE_DRAFT_KV_INDIRECTION_INVALID_LAYOUT,
    COMMON_TREE_DRAFT_KV_INDIRECTION_ADDRESS_OVERFLOW,
    COMMON_TREE_DRAFT_KV_INDIRECTION_NULL_OUTPUT,
};

common_tree_draft_kv_indirection_error common_tree_draft_kv_indirection_validate(
        const common_tree_draft_topology & topology,
        const common_tree_draft_kv_indirection & indirection);

common_tree_draft_kv_indirection_error common_tree_draft_kv_slot_for_logical(
        const common_tree_draft_topology & topology,
        const common_tree_draft_kv_indirection & indirection,
        int32_t logical_node,
        uint32_t * physical_slot);

common_tree_draft_kv_indirection_error common_tree_draft_kv_address(
        const common_tree_draft_kv_layout & layout,
        uint32_t physical_slot,
        int32_t head,
        int32_t dim,
        size_t * byte_offset);

const char * common_tree_draft_kv_indirection_error_name(common_tree_draft_kv_indirection_error error);
