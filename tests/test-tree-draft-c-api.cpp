#include "tree-draft-c-api.h"

#include <cassert>
#include <cstdlib>
#include <type_traits>

struct allocation_log {
    size_t allocated_size = 0;
    void * released_data = nullptr;
    int release_calls = 0;
};

static void * allocate(void * context, size_t size) {
    auto * log = static_cast<allocation_log *>(context);
    log->allocated_size = size;
    return std::malloc(size);
}

static void release(void * context, void * data) {
    auto * log = static_cast<allocation_log *>(context);
    log->released_data = data;
    log->release_calls++;
    std::free(data);
}

static void test_library_allocation_uses_explicit_allocator() {
    static_assert(std::is_standard_layout<common_tree_draft_c_byte_span>::value, "byte span must be C-compatible");
    static_assert(std::is_standard_layout<common_tree_draft_c_token_span>::value, "token span must be C-compatible");
    static_assert(std::is_standard_layout<common_tree_draft_c_buffer>::value, "buffer must be C-compatible");

    allocation_log log;
    const common_tree_draft_c_allocator allocator = { allocate, release, &log };
    common_tree_draft_c_buffer buffer = {};

    assert(common_tree_draft_c_buffer_allocate(64, &allocator, &buffer) == COMMON_TREE_DRAFT_C_BUFFER_STATUS_OK);
    assert(log.allocated_size == 64);
    assert(buffer.data != nullptr);
    assert(buffer.size == 64);
    assert(buffer.owner == COMMON_TREE_DRAFT_C_BUFFER_OWNER_LIBRARY);
    assert(buffer.release == release);
    assert(buffer.release_context == &log);

    void * const data = buffer.data;
    assert(common_tree_draft_c_buffer_release(&buffer) == COMMON_TREE_DRAFT_C_BUFFER_STATUS_OK);
    assert(log.released_data == data);
    assert(log.release_calls == 1);
    assert(buffer.data == nullptr);
    assert(buffer.size == 0);
    assert(common_tree_draft_c_buffer_release(&buffer) == COMMON_TREE_DRAFT_C_BUFFER_STATUS_ALREADY_RELEASED);
    assert(log.release_calls == 1);
}

static void test_release_rules() {
    assert(common_tree_draft_c_buffer_release(nullptr) == COMMON_TREE_DRAFT_C_BUFFER_STATUS_OK);

    int caller_data = 7;
    common_tree_draft_c_buffer caller = {
        &caller_data,
        sizeof(caller_data),
        COMMON_TREE_DRAFT_C_BUFFER_OWNER_CALLER,
        nullptr,
        nullptr,
    };
    assert(common_tree_draft_c_buffer_release(&caller) == COMMON_TREE_DRAFT_C_BUFFER_STATUS_CALLER_OWNED);
    assert(caller.data == &caller_data);

    common_tree_draft_c_buffer invalid = {
        &caller_data,
        sizeof(caller_data),
        (common_tree_draft_c_buffer_owner) -1,
        nullptr,
        nullptr,
    };
    assert(common_tree_draft_c_buffer_release(&invalid) == COMMON_TREE_DRAFT_C_BUFFER_STATUS_INVALID_OWNER);

    common_tree_draft_c_buffer missing_release = {
        &caller_data,
        sizeof(caller_data),
        COMMON_TREE_DRAFT_C_BUFFER_OWNER_SHARED,
        nullptr,
        nullptr,
    };
    assert(common_tree_draft_c_buffer_release(&missing_release) == COMMON_TREE_DRAFT_C_BUFFER_STATUS_MISSING_RELEASE);
}

static void test_allocation_rejects_hidden_ownership() {
    allocation_log log;
    const common_tree_draft_c_allocator full_allocator = { allocate, release, &log };
    const common_tree_draft_c_allocator missing_release = { allocate, nullptr, &log };
    common_tree_draft_c_buffer buffer = {};

    assert(common_tree_draft_c_buffer_allocate(0, &full_allocator, &buffer) == COMMON_TREE_DRAFT_C_BUFFER_STATUS_INVALID_ARGUMENT);
    assert(common_tree_draft_c_buffer_allocate(1, nullptr, &buffer) == COMMON_TREE_DRAFT_C_BUFFER_STATUS_INVALID_ARGUMENT);
    assert(common_tree_draft_c_buffer_allocate(1, &missing_release, &buffer) == COMMON_TREE_DRAFT_C_BUFFER_STATUS_INVALID_ARGUMENT);
    assert(common_tree_draft_c_buffer_allocate(1, &full_allocator, nullptr) == COMMON_TREE_DRAFT_C_BUFFER_STATUS_INVALID_ARGUMENT);
}

int main() {
    test_library_allocation_uses_explicit_allocator();
    test_release_rules();
    test_allocation_rejects_hidden_ownership();
    return 0;
}
