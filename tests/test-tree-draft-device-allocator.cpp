#include "tree-draft-device-allocator.h"

#include <cassert>
#include <cstdlib>

struct fixture { uint64_t reserved = 0; uint32_t trims = 0; };

static bool do_alloc(void *, uint64_t size, uint64_t alignment, common_tree_draft_device_stream, void ** ptr) {
#ifdef _WIN32
    *ptr = _aligned_malloc(static_cast<size_t>(size), static_cast<size_t>(alignment));
#else
    *ptr = std::aligned_alloc(static_cast<size_t>(alignment), static_cast<size_t>((size + alignment - 1) / alignment * alignment));
#endif
    return *ptr != nullptr;
}

static void do_free(void *, void * ptr, common_tree_draft_device_stream) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

static bool do_reserve(void * ctx, uint64_t bytes) {
    static_cast<fixture *>(ctx)->reserved = bytes;
    return bytes <= 4096;
}
static void do_trim(void * ctx) { ++static_cast<fixture *>(ctx)->trims; }

int main() {
    fixture f;
    common_tree_draft_device_allocator_ops ops = { &f, do_alloc, do_free, do_reserve, do_trim };
    common_tree_draft_device_allocation allocation = {};
    assert(common_tree_draft_device_alloc(ops, 1024, 256, 7, &allocation) == COMMON_TREE_DRAFT_DEVICE_ALLOCATOR_OK);
    assert(reinterpret_cast<uintptr_t>(allocation.ptr) % 256 == 0);
    assert(allocation.stream == 7);
    assert(common_tree_draft_device_free(ops, &allocation) == COMMON_TREE_DRAFT_DEVICE_ALLOCATOR_OK);
    assert(common_tree_draft_device_reserve(ops, 2048) == COMMON_TREE_DRAFT_DEVICE_ALLOCATOR_OK);
    assert(common_tree_draft_device_reserve(ops, 8192) == COMMON_TREE_DRAFT_DEVICE_ALLOCATOR_FAILED);
    common_tree_draft_device_trim(ops);
    assert(f.trims == 1);
    assert(common_tree_draft_device_alloc(ops, 1, 24, 0, &allocation) == COMMON_TREE_DRAFT_DEVICE_ALLOCATOR_ALIGNMENT);
    return 0;
}

