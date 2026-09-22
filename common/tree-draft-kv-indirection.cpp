#include "tree-draft-kv-indirection.h"

#include <limits>

static bool common_tree_draft_kv_mul_overflow(size_t a, size_t b, size_t * result) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        return true;
    }
    *result = a * b;
    return false;
}

static bool common_tree_draft_kv_add_overflow(size_t a, size_t b, size_t * result) {
    if (b > std::numeric_limits<size_t>::max() - a) {
        return true;
    }
    *result = a + b;
    return false;
}

common_tree_draft_kv_indirection_error common_tree_draft_kv_indirection_validate(
        const common_tree_draft_topology & topology,
        const common_tree_draft_kv_indirection & indirection) {
    if (common_tree_draft_topology_validate(topology) != COMMON_TREE_DRAFT_TOPOLOGY_OK) {
        return COMMON_TREE_DRAFT_KV_INDIRECTION_INVALID_TOPOLOGY;
    }
    if (indirection.slot_count != static_cast<size_t>(topology.n_nodes)) {
        return COMMON_TREE_DRAFT_KV_INDIRECTION_SLOT_COUNT_MISMATCH;
    }
    if (topology.n_nodes > 0 && indirection.slots == nullptr) {
        return COMMON_TREE_DRAFT_KV_INDIRECTION_NULL_SLOTS;
    }
    for (int32_t g = 0; g < topology.n_nodes; ++g) {
        const uint32_t slot = indirection.slots[g];
        if (slot != COMMON_TREE_DRAFT_KV_SLOT_INVALID && slot >= indirection.kv_size) {
            return COMMON_TREE_DRAFT_KV_INDIRECTION_SLOT_RANGE;
        }
    }
    return COMMON_TREE_DRAFT_KV_INDIRECTION_OK;
}

common_tree_draft_kv_indirection_error common_tree_draft_kv_slot_for_logical(
        const common_tree_draft_topology & topology,
        const common_tree_draft_kv_indirection & indirection,
        int32_t logical_node,
        uint32_t * physical_slot) {
    const common_tree_draft_kv_indirection_error validation =
        common_tree_draft_kv_indirection_validate(topology, indirection);
    if (validation != COMMON_TREE_DRAFT_KV_INDIRECTION_OK) {
        return validation;
    }
    if (logical_node < 0 || logical_node >= topology.n_nodes) {
        return COMMON_TREE_DRAFT_KV_INDIRECTION_LOGICAL_RANGE;
    }
    if (physical_slot == nullptr) {
        return COMMON_TREE_DRAFT_KV_INDIRECTION_NULL_OUTPUT;
    }
    const uint32_t slot = indirection.slots[logical_node];
    if (slot == COMMON_TREE_DRAFT_KV_SLOT_INVALID) {
        return COMMON_TREE_DRAFT_KV_INDIRECTION_INVALID_SLOT;
    }
    *physical_slot = slot;
    return COMMON_TREE_DRAFT_KV_INDIRECTION_OK;
}

common_tree_draft_kv_indirection_error common_tree_draft_kv_address(
        const common_tree_draft_kv_layout & layout,
        uint32_t physical_slot,
        int32_t head,
        int32_t dim,
        size_t * byte_offset) {
    if (layout.element_size == 0 || layout.n_heads < 0 || layout.head_dim < 0 ||
        (layout.kv_size > 0 && layout.data == nullptr)) {
        return COMMON_TREE_DRAFT_KV_INDIRECTION_INVALID_LAYOUT;
    }
    if (physical_slot == COMMON_TREE_DRAFT_KV_SLOT_INVALID) {
        return COMMON_TREE_DRAFT_KV_INDIRECTION_INVALID_SLOT;
    }
    if (physical_slot >= layout.kv_size || head < 0 || head >= layout.n_heads || dim < 0 || dim >= layout.head_dim) {
        return COMMON_TREE_DRAFT_KV_INDIRECTION_SLOT_RANGE;
    }
    if (byte_offset == nullptr) {
        return COMMON_TREE_DRAFT_KV_INDIRECTION_NULL_OUTPUT;
    }

    size_t slot_offset = 0;
    size_t head_offset = 0;
    size_t dim_offset = 0;
    size_t offset = 0;
    size_t end_offset = 0;
    if (common_tree_draft_kv_mul_overflow(static_cast<size_t>(physical_slot), layout.stride_slot, &slot_offset) ||
        common_tree_draft_kv_mul_overflow(static_cast<size_t>(head), layout.stride_head, &head_offset) ||
        common_tree_draft_kv_mul_overflow(static_cast<size_t>(dim), layout.stride_dim, &dim_offset) ||
        common_tree_draft_kv_add_overflow(slot_offset, head_offset, &offset) ||
        common_tree_draft_kv_add_overflow(offset, dim_offset, &offset) ||
        common_tree_draft_kv_add_overflow(offset, layout.element_size, &end_offset)) {
        return COMMON_TREE_DRAFT_KV_INDIRECTION_ADDRESS_OVERFLOW;
    }
    *byte_offset = offset;
    return COMMON_TREE_DRAFT_KV_INDIRECTION_OK;
}

const char * common_tree_draft_kv_indirection_error_name(common_tree_draft_kv_indirection_error error) {
    switch (error) {
        case COMMON_TREE_DRAFT_KV_INDIRECTION_OK:                  return "ok";
        case COMMON_TREE_DRAFT_KV_INDIRECTION_INVALID_TOPOLOGY:    return "invalid_topology";
        case COMMON_TREE_DRAFT_KV_INDIRECTION_NULL_SLOTS:          return "null_slots";
        case COMMON_TREE_DRAFT_KV_INDIRECTION_SLOT_COUNT_MISMATCH: return "slot_count_mismatch";
        case COMMON_TREE_DRAFT_KV_INDIRECTION_SLOT_RANGE:          return "slot_range";
        case COMMON_TREE_DRAFT_KV_INDIRECTION_LOGICAL_RANGE:       return "logical_range";
        case COMMON_TREE_DRAFT_KV_INDIRECTION_INVALID_SLOT:        return "invalid_slot";
        case COMMON_TREE_DRAFT_KV_INDIRECTION_INVALID_LAYOUT:      return "invalid_layout";
        case COMMON_TREE_DRAFT_KV_INDIRECTION_ADDRESS_OVERFLOW:    return "address_overflow";
        case COMMON_TREE_DRAFT_KV_INDIRECTION_NULL_OUTPUT:         return "null_output";
    }
    return "unknown";
}
