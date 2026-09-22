#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct common_tree_draft_c_byte_span {
    const uint8_t * data;
    size_t size;
} common_tree_draft_c_byte_span;

typedef struct common_tree_draft_c_token_span {
    const int32_t * data;
    size_t size;
} common_tree_draft_c_token_span;

typedef enum common_tree_draft_c_buffer_owner {
    COMMON_TREE_DRAFT_C_BUFFER_OWNER_CALLER = 0,
    COMMON_TREE_DRAFT_C_BUFFER_OWNER_LIBRARY,
    COMMON_TREE_DRAFT_C_BUFFER_OWNER_SHARED,
} common_tree_draft_c_buffer_owner;

typedef void * (*common_tree_draft_c_allocate_fn)(void * context, size_t size);
typedef void (*common_tree_draft_c_release_fn)(void * context, void * data);

typedef struct common_tree_draft_c_allocator {
    common_tree_draft_c_allocate_fn allocate;
    common_tree_draft_c_release_fn release;
    void * context;
} common_tree_draft_c_allocator;

typedef struct common_tree_draft_c_buffer {
    void * data;
    size_t size;
    common_tree_draft_c_buffer_owner owner;
    common_tree_draft_c_release_fn release;
    void * release_context;
} common_tree_draft_c_buffer;

typedef enum common_tree_draft_c_buffer_status {
    COMMON_TREE_DRAFT_C_BUFFER_STATUS_OK = 0,
    COMMON_TREE_DRAFT_C_BUFFER_STATUS_INVALID_ARGUMENT,
    COMMON_TREE_DRAFT_C_BUFFER_STATUS_ALLOCATION_FAILED,
    COMMON_TREE_DRAFT_C_BUFFER_STATUS_CALLER_OWNED,
    COMMON_TREE_DRAFT_C_BUFFER_STATUS_ALREADY_RELEASED,
    COMMON_TREE_DRAFT_C_BUFFER_STATUS_INVALID_OWNER,
    COMMON_TREE_DRAFT_C_BUFFER_STATUS_MISSING_RELEASE,
} common_tree_draft_c_buffer_status;

common_tree_draft_c_buffer_status common_tree_draft_c_buffer_allocate(
        size_t size,
        const common_tree_draft_c_allocator * allocator,
        common_tree_draft_c_buffer * out);

common_tree_draft_c_buffer_status common_tree_draft_c_buffer_release(
        common_tree_draft_c_buffer * buffer);

#ifdef __cplusplus
}
#endif
