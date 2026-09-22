#pragma once

#include <cstddef>
#include <cstdint>

using common_tree_draft_device_stream = uintptr_t;

struct common_tree_draft_device_allocation {
    void * ptr = nullptr;
    uint64_t size = 0;
    uint64_t alignment = 0;
    common_tree_draft_device_stream stream = 0;
};

struct common_tree_draft_device_allocator_ops {
    void * context = nullptr;
    bool (*alloc)(void * context, uint64_t size, uint64_t alignment, common_tree_draft_device_stream stream, void ** ptr) = nullptr;
    void (*free)(void * context, void * ptr, common_tree_draft_device_stream stream) = nullptr;
    bool (*reserve)(void * context, uint64_t bytes) = nullptr;
    void (*trim)(void * context) = nullptr;
};

enum common_tree_draft_device_allocator_status : uint32_t {
    COMMON_TREE_DRAFT_DEVICE_ALLOCATOR_OK = 0,
    COMMON_TREE_DRAFT_DEVICE_ALLOCATOR_INVALID,
    COMMON_TREE_DRAFT_DEVICE_ALLOCATOR_ALIGNMENT,
    COMMON_TREE_DRAFT_DEVICE_ALLOCATOR_FAILED,
};

common_tree_draft_device_allocator_status common_tree_draft_device_alloc(
        const common_tree_draft_device_allocator_ops & ops,
        uint64_t size,
        uint64_t alignment,
        common_tree_draft_device_stream stream,
        common_tree_draft_device_allocation * allocation);

common_tree_draft_device_allocator_status common_tree_draft_device_free(
        const common_tree_draft_device_allocator_ops & ops,
        common_tree_draft_device_allocation * allocation);

common_tree_draft_device_allocator_status common_tree_draft_device_reserve(
        const common_tree_draft_device_allocator_ops & ops,
        uint64_t bytes);

void common_tree_draft_device_trim(const common_tree_draft_device_allocator_ops & ops);

