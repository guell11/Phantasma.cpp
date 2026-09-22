#include "tree-draft-alignment.h"

#include <limits>
#include <numeric>

static bool valid_alignment(uint64_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

static bool lcm_checked(uint64_t a, uint64_t b, uint64_t * out) {
    const uint64_t g = std::gcd(a, b);
    const uint64_t left = a / g;
    if (left != 0 && b > std::numeric_limits<uint64_t>::max() / left) return false;
    *out = left * b;
    return true;
}

static common_tree_draft_alignment_status combine(
        std::initializer_list<uint64_t> values,
        uint64_t * out) {
    uint64_t current = 1;
    for (uint64_t value : values) {
        if (!valid_alignment(value)) return COMMON_TREE_DRAFT_ALIGNMENT_INVALID;
        if (!lcm_checked(current, value, &current)) return COMMON_TREE_DRAFT_ALIGNMENT_OVERFLOW;
    }
    *out = current;
    return COMMON_TREE_DRAFT_ALIGNMENT_OK;
}

common_tree_draft_alignment_status common_tree_draft_alignment_build(
        const common_tree_draft_alignment_constraints & c,
        common_tree_draft_alignment_table * table) {
    if (table == nullptr) return COMMON_TREE_DRAFT_ALIGNMENT_INVALID;
    for (uint64_t value : {c.vector_alignment, c.dma_alignment, c.tensor_alignment, c.allocator_alignment}) {
        if (!valid_alignment(value)) return COMMON_TREE_DRAFT_ALIGNMENT_INVALID;
    }

    auto status = combine({c.vector_alignment, c.dma_alignment, c.allocator_alignment}, &table->required[COMMON_TREE_DRAFT_BUFFER_HOST_STAGING]);
    if (status != COMMON_TREE_DRAFT_ALIGNMENT_OK) return status;
    status = combine({c.vector_alignment, c.tensor_alignment, c.allocator_alignment}, &table->required[COMMON_TREE_DRAFT_BUFFER_DEVICE_SCRATCH]);
    if (status != COMMON_TREE_DRAFT_ALIGNMENT_OK) return status;
    status = combine({c.vector_alignment, c.dma_alignment, c.tensor_alignment, c.allocator_alignment}, &table->required[COMMON_TREE_DRAFT_BUFFER_KV_PAGE]);
    if (status != COMMON_TREE_DRAFT_ALIGNMENT_OK) return status;
    status = combine({c.vector_alignment, c.tensor_alignment, c.allocator_alignment}, &table->required[COMMON_TREE_DRAFT_BUFFER_TENSOR_TILE]);
    if (status != COMMON_TREE_DRAFT_ALIGNMENT_OK) return status;
    return combine({c.vector_alignment, c.allocator_alignment}, &table->required[COMMON_TREE_DRAFT_BUFFER_METADATA]);
}

common_tree_draft_alignment_status common_tree_draft_alignment_up(
        uint64_t value,
        uint64_t alignment,
        uint64_t * aligned) {
    if (aligned == nullptr || !valid_alignment(alignment)) return COMMON_TREE_DRAFT_ALIGNMENT_INVALID;
    const uint64_t mask = alignment - 1;
    if (value > std::numeric_limits<uint64_t>::max() - mask) return COMMON_TREE_DRAFT_ALIGNMENT_OVERFLOW;
    *aligned = (value + mask) & ~mask;
    return COMMON_TREE_DRAFT_ALIGNMENT_OK;
}

bool common_tree_draft_alignment_offset_valid(
        const common_tree_draft_alignment_table & table,
        common_tree_draft_buffer_class buffer_class,
        uint64_t offset) {
    if (buffer_class >= COMMON_TREE_DRAFT_BUFFER_CLASS_COUNT) return false;
    const uint64_t alignment = table.required[buffer_class];
    return valid_alignment(alignment) && offset % alignment == 0;
}

