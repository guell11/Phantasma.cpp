#include "tree-draft-c-api.h"

common_tree_draft_c_buffer_status common_tree_draft_c_buffer_allocate(
        size_t size,
        const common_tree_draft_c_allocator * allocator,
        common_tree_draft_c_buffer * out) {
    if (size == 0 || allocator == nullptr || allocator->allocate == nullptr || allocator->release == nullptr || out == nullptr) {
        return COMMON_TREE_DRAFT_C_BUFFER_STATUS_INVALID_ARGUMENT;
    }

    void * data = allocator->allocate(allocator->context, size);
    if (data == nullptr) {
        return COMMON_TREE_DRAFT_C_BUFFER_STATUS_ALLOCATION_FAILED;
    }

    out->data = data;
    out->size = size;
    out->owner = COMMON_TREE_DRAFT_C_BUFFER_OWNER_LIBRARY;
    out->release = allocator->release;
    out->release_context = allocator->context;

    return COMMON_TREE_DRAFT_C_BUFFER_STATUS_OK;
}

common_tree_draft_c_buffer_status common_tree_draft_c_buffer_release(
        common_tree_draft_c_buffer * buffer) {
    if (buffer == nullptr) {
        return COMMON_TREE_DRAFT_C_BUFFER_STATUS_OK;
    }
    if (buffer->data == nullptr) {
        return COMMON_TREE_DRAFT_C_BUFFER_STATUS_ALREADY_RELEASED;
    }
    if (buffer->owner == COMMON_TREE_DRAFT_C_BUFFER_OWNER_CALLER) {
        return COMMON_TREE_DRAFT_C_BUFFER_STATUS_CALLER_OWNED;
    }
    if (buffer->owner != COMMON_TREE_DRAFT_C_BUFFER_OWNER_LIBRARY &&
        buffer->owner != COMMON_TREE_DRAFT_C_BUFFER_OWNER_SHARED) {
        return COMMON_TREE_DRAFT_C_BUFFER_STATUS_INVALID_OWNER;
    }
    if (buffer->release == nullptr) {
        return COMMON_TREE_DRAFT_C_BUFFER_STATUS_MISSING_RELEASE;
    }

    buffer->release(buffer->release_context, buffer->data);
    buffer->data = nullptr;
    buffer->size = 0;
    buffer->release = nullptr;
    buffer->release_context = nullptr;

    return COMMON_TREE_DRAFT_C_BUFFER_STATUS_OK;
}
