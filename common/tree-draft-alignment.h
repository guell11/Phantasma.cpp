#pragma once

#include <array>
#include <cstdint>

enum common_tree_draft_buffer_class : uint32_t {
    COMMON_TREE_DRAFT_BUFFER_HOST_STAGING = 0,
    COMMON_TREE_DRAFT_BUFFER_DEVICE_SCRATCH,
    COMMON_TREE_DRAFT_BUFFER_KV_PAGE,
    COMMON_TREE_DRAFT_BUFFER_TENSOR_TILE,
    COMMON_TREE_DRAFT_BUFFER_METADATA,
    COMMON_TREE_DRAFT_BUFFER_CLASS_COUNT,
};

struct common_tree_draft_alignment_constraints {
    uint64_t vector_alignment = 1;
    uint64_t dma_alignment = 1;
    uint64_t tensor_alignment = 1;
    uint64_t allocator_alignment = 1;
};

struct common_tree_draft_alignment_table {
    std::array<uint64_t, COMMON_TREE_DRAFT_BUFFER_CLASS_COUNT> required = {};
};

enum common_tree_draft_alignment_status : uint32_t {
    COMMON_TREE_DRAFT_ALIGNMENT_OK = 0,
    COMMON_TREE_DRAFT_ALIGNMENT_INVALID,
    COMMON_TREE_DRAFT_ALIGNMENT_OVERFLOW,
};

common_tree_draft_alignment_status common_tree_draft_alignment_build(
        const common_tree_draft_alignment_constraints & constraints,
        common_tree_draft_alignment_table * table);

common_tree_draft_alignment_status common_tree_draft_alignment_up(
        uint64_t value,
        uint64_t alignment,
        uint64_t * aligned);

bool common_tree_draft_alignment_offset_valid(
        const common_tree_draft_alignment_table & table,
        common_tree_draft_buffer_class buffer_class,
        uint64_t offset);

