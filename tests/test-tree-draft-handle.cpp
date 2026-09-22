#include "tree-draft-handle.h"

#include <cassert>
#include <thread>
#include <vector>

static void test_runtime_retain_release_and_generation() {
    common_tree_draft_c_runtime_handle first = 0;
    assert(common_tree_draft_c_runtime_create(&first) == COMMON_TREE_DRAFT_C_HANDLE_STATUS_OK);
    assert(first != 0);

    assert(common_tree_draft_c_runtime_retain(first) == COMMON_TREE_DRAFT_C_HANDLE_STATUS_OK);
    assert(common_tree_draft_c_runtime_release(first) == COMMON_TREE_DRAFT_C_HANDLE_STATUS_OK);
    assert(common_tree_draft_c_runtime_release(first) == COMMON_TREE_DRAFT_C_HANDLE_STATUS_OK);
    assert(common_tree_draft_c_runtime_retain(first) == COMMON_TREE_DRAFT_C_HANDLE_STATUS_INVALID_HANDLE);
    assert(common_tree_draft_c_runtime_release(first) == COMMON_TREE_DRAFT_C_HANDLE_STATUS_INVALID_HANDLE);

    common_tree_draft_c_runtime_handle second = 0;
    assert(common_tree_draft_c_runtime_create(&second) == COMMON_TREE_DRAFT_C_HANDLE_STATUS_OK);
    assert(second != 0);
    assert(second != first);
    assert(common_tree_draft_c_runtime_retain(first) == COMMON_TREE_DRAFT_C_HANDLE_STATUS_INVALID_HANDLE);
    assert(common_tree_draft_c_runtime_release(second) == COMMON_TREE_DRAFT_C_HANDLE_STATUS_OK);
}

static void test_handle_kinds_are_isolated() {
    common_tree_draft_c_runtime_handle runtime = 0;
    common_tree_draft_c_session_handle session = 0;
    common_tree_draft_c_request_handle request = 0;

    assert(common_tree_draft_c_runtime_create(&runtime) == COMMON_TREE_DRAFT_C_HANDLE_STATUS_OK);
    assert(common_tree_draft_c_session_create(&session) == COMMON_TREE_DRAFT_C_HANDLE_STATUS_OK);
    assert(common_tree_draft_c_request_create(&request) == COMMON_TREE_DRAFT_C_HANDLE_STATUS_OK);

    assert(common_tree_draft_c_session_retain(runtime) == COMMON_TREE_DRAFT_C_HANDLE_STATUS_INVALID_HANDLE);
    assert(common_tree_draft_c_request_retain(session) == COMMON_TREE_DRAFT_C_HANDLE_STATUS_INVALID_HANDLE);
    assert(common_tree_draft_c_runtime_retain(request) == COMMON_TREE_DRAFT_C_HANDLE_STATUS_INVALID_HANDLE);

    assert(common_tree_draft_c_runtime_release(runtime) == COMMON_TREE_DRAFT_C_HANDLE_STATUS_OK);
    assert(common_tree_draft_c_session_release(session) == COMMON_TREE_DRAFT_C_HANDLE_STATUS_OK);
    assert(common_tree_draft_c_request_release(request) == COMMON_TREE_DRAFT_C_HANDLE_STATUS_OK);
}

static void test_invalid_arguments() {
    common_tree_draft_c_runtime_handle runtime = 123;

    assert(common_tree_draft_c_runtime_create(nullptr) == COMMON_TREE_DRAFT_C_HANDLE_STATUS_INVALID_ARGUMENT);
    assert(common_tree_draft_c_session_create(nullptr) == COMMON_TREE_DRAFT_C_HANDLE_STATUS_INVALID_ARGUMENT);
    assert(common_tree_draft_c_request_create(nullptr) == COMMON_TREE_DRAFT_C_HANDLE_STATUS_INVALID_ARGUMENT);

    assert(common_tree_draft_c_runtime_create(&runtime) == COMMON_TREE_DRAFT_C_HANDLE_STATUS_OK);
    assert(runtime != 0);
    assert(common_tree_draft_c_runtime_release(runtime) == COMMON_TREE_DRAFT_C_HANDLE_STATUS_OK);

    assert(common_tree_draft_c_runtime_retain(0) == COMMON_TREE_DRAFT_C_HANDLE_STATUS_INVALID_HANDLE);
    assert(common_tree_draft_c_session_release(0) == COMMON_TREE_DRAFT_C_HANDLE_STATUS_INVALID_HANDLE);
}

static void test_thread_safe_retain_release() {
    common_tree_draft_c_request_handle request = 0;
    assert(common_tree_draft_c_request_create(&request) == COMMON_TREE_DRAFT_C_HANDLE_STATUS_OK);

    constexpr int thread_count = 8;
    constexpr int iterations = 1000;
    std::vector<std::thread> workers;
    workers.reserve(thread_count);

    for (int i = 0; i < thread_count; ++i) {
        workers.emplace_back([request, iterations]() {
            for (int j = 0; j < iterations; ++j) {
                assert(common_tree_draft_c_request_retain(request) == COMMON_TREE_DRAFT_C_HANDLE_STATUS_OK);
                assert(common_tree_draft_c_request_release(request) == COMMON_TREE_DRAFT_C_HANDLE_STATUS_OK);
            }
        });
    }

    for (std::thread & worker : workers) {
        worker.join();
    }

    assert(common_tree_draft_c_request_release(request) == COMMON_TREE_DRAFT_C_HANDLE_STATUS_OK);
    assert(common_tree_draft_c_request_retain(request) == COMMON_TREE_DRAFT_C_HANDLE_STATUS_INVALID_HANDLE);
}

int main() {
    test_runtime_retain_release_and_generation();
    test_handle_kinds_are_isolated();
    test_invalid_arguments();
    test_thread_safe_retain_release();
    return 0;
}
