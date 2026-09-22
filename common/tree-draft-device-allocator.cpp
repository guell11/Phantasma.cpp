#include "tree-draft-device-allocator.h"

#include <cstdint>

static bool valid_alignment(uint64_t alignment) {
    return alignment != 0 && (alignment & (alignment - 1)) == 0;
}

common_tree_draft_device_allocator_status common_tree_draft_device_alloc(
        const common_tree_draft_device_allocator_ops & ops,
        uint64_t size,
        uint64_t alignment,
        common_tree_draft_device_stream stream,
        common_tree_draft_device_allocation * allocation) {
    if (allocation == nullptr || ops.alloc == nullptr || ops.free == nullptr || size == 0) {
        return COMMON_TREE_DRAFT_DEVICE_ALLOCATOR_INVALID;
    }
    if (!valid_alignment(alignment)) return COMMON_TREE_DRAFT_DEVICE_ALLOCATOR_ALIGNMENT;
    void * ptr = nullptr;
    if (!ops.alloc(ops.context, size, alignment, stream, &ptr) || ptr == nullptr) {
        return COMMON_TREE_DRAFT_DEVICE_ALLOCATOR_FAILED;
    }
    if ((reinterpret_cast<uintptr_t>(ptr) % alignment) != 0) {
        ops.free(ops.context, ptr, stream);
        return COMMON_TREE_DRAFT_DEVICE_ALLOCATOR_ALIGNMENT;
    }
    *allocation = { ptr, size, alignment, stream };
    return COMMON_TREE_DRAFT_DEVICE_ALLOCATOR_OK;
}

common_tree_draft_device_allocator_status common_tree_draft_device_free(
        const common_tree_draft_device_allocator_ops & ops,
        common_tree_draft_device_allocation * allocation) {
    if (allocation == nullptr || ops.free == nullptr) return COMMON_TREE_DRAFT_DEVICE_ALLOCATOR_INVALID;
    if (allocation->ptr != nullptr) ops.free(ops.context, allocation->ptr, allocation->stream);
    *allocation = {};
    return COMMON_TREE_DRAFT_DEVICE_ALLOCATOR_OK;
}

common_tree_draft_device_allocator_status common_tree_draft_device_reserve(
        const common_tree_draft_device_allocator_ops & ops,
        uint64_t bytes) {
    if (ops.reserve == nullptr) return COMMON_TREE_DRAFT_DEVICE_ALLOCATOR_INVALID;
    return ops.reserve(ops.context, bytes) ? COMMON_TREE_DRAFT_DEVICE_ALLOCATOR_OK : COMMON_TREE_DRAFT_DEVICE_ALLOCATOR_FAILED;
}

void common_tree_draft_device_trim(const common_tree_draft_device_allocator_ops & ops) {
    if (ops.trim != nullptr) ops.trim(ops.context);
}

