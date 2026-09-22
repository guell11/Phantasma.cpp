#include "tree-draft-handle.h"

#include <cstdint>
#include <limits>
#include <mutex>
#include <new>
#include <vector>

namespace {

constexpr uint64_t handle_kind_shift = 56;
constexpr uint64_t handle_generation_shift = 24;
constexpr uint64_t handle_slot_mask = (UINT64_C(1) << handle_generation_shift) - 1;
constexpr uint64_t handle_generation_mask = UINT64_C(0xffffffff);
constexpr uint32_t handle_max_slots = static_cast<uint32_t>(handle_slot_mask);

enum handle_kind : uint8_t {
    HANDLE_KIND_RUNTIME = 1,
    HANDLE_KIND_SESSION = 2,
    HANDLE_KIND_REQUEST = 3,
};

struct handle_entry {
    uint32_t generation = 1;
    uint32_t refcount = 0;
    bool allocated = false;
};

struct handle_table {
    std::mutex mutex;
    std::vector<handle_entry> entries;
};

handle_table runtime_handles;
handle_table session_handles;
handle_table request_handles;

uint64_t encode_handle(handle_kind kind, uint32_t generation, uint32_t slot) {
    return (static_cast<uint64_t>(kind) << handle_kind_shift) |
           (static_cast<uint64_t>(generation) << handle_generation_shift) |
           static_cast<uint64_t>(slot + 1);
}

bool decode_handle(uint64_t handle, handle_kind expected_kind, uint32_t & generation, uint32_t & slot) {
    if (handle == 0 || static_cast<uint8_t>(handle >> handle_kind_shift) != expected_kind) {
        return false;
    }

    const uint64_t encoded_slot = handle & handle_slot_mask;
    const uint64_t encoded_generation = (handle >> handle_generation_shift) & handle_generation_mask;
    if (encoded_slot == 0 || encoded_generation == 0) {
        return false;
    }

    generation = static_cast<uint32_t>(encoded_generation);
    slot = static_cast<uint32_t>(encoded_slot - 1);
    return true;
}

common_tree_draft_c_handle_status create_handle(handle_table & table, handle_kind kind, uint64_t * out) {
    if (out == nullptr) {
        return COMMON_TREE_DRAFT_C_HANDLE_STATUS_INVALID_ARGUMENT;
    }
    *out = 0;

    std::lock_guard<std::mutex> lock(table.mutex);
    for (uint32_t slot = 0; slot < table.entries.size(); ++slot) {
        handle_entry & entry = table.entries[slot];
        if (!entry.allocated && entry.generation != std::numeric_limits<uint32_t>::max()) {
            entry.allocated = true;
            entry.refcount = 1;
            *out = encode_handle(kind, entry.generation, slot);
            return COMMON_TREE_DRAFT_C_HANDLE_STATUS_OK;
        }
    }

    if (table.entries.size() >= handle_max_slots) {
        return COMMON_TREE_DRAFT_C_HANDLE_STATUS_ALLOCATION_FAILED;
    }

    try {
        table.entries.push_back({ 1, 1, true });
    } catch (const std::bad_alloc &) {
        return COMMON_TREE_DRAFT_C_HANDLE_STATUS_ALLOCATION_FAILED;
    }

    *out = encode_handle(kind, 1, static_cast<uint32_t>(table.entries.size() - 1));
    return COMMON_TREE_DRAFT_C_HANDLE_STATUS_OK;
}

common_tree_draft_c_handle_status retain_handle(handle_table & table, handle_kind kind, uint64_t handle) {
    uint32_t generation = 0;
    uint32_t slot = 0;
    if (!decode_handle(handle, kind, generation, slot)) {
        return COMMON_TREE_DRAFT_C_HANDLE_STATUS_INVALID_HANDLE;
    }

    std::lock_guard<std::mutex> lock(table.mutex);
    if (slot >= table.entries.size()) {
        return COMMON_TREE_DRAFT_C_HANDLE_STATUS_INVALID_HANDLE;
    }

    handle_entry & entry = table.entries[slot];
    if (!entry.allocated || entry.generation != generation || entry.refcount == 0) {
        return COMMON_TREE_DRAFT_C_HANDLE_STATUS_INVALID_HANDLE;
    }
    if (entry.refcount == std::numeric_limits<uint32_t>::max()) {
        return COMMON_TREE_DRAFT_C_HANDLE_STATUS_REFCOUNT_OVERFLOW;
    }

    ++entry.refcount;
    return COMMON_TREE_DRAFT_C_HANDLE_STATUS_OK;
}

common_tree_draft_c_handle_status release_handle(handle_table & table, handle_kind kind, uint64_t handle) {
    uint32_t generation = 0;
    uint32_t slot = 0;
    if (!decode_handle(handle, kind, generation, slot)) {
        return COMMON_TREE_DRAFT_C_HANDLE_STATUS_INVALID_HANDLE;
    }

    std::lock_guard<std::mutex> lock(table.mutex);
    if (slot >= table.entries.size()) {
        return COMMON_TREE_DRAFT_C_HANDLE_STATUS_INVALID_HANDLE;
    }

    handle_entry & entry = table.entries[slot];
    if (!entry.allocated || entry.generation != generation || entry.refcount == 0) {
        return COMMON_TREE_DRAFT_C_HANDLE_STATUS_INVALID_HANDLE;
    }

    --entry.refcount;
    if (entry.refcount == 0) {
        entry.allocated = false;
        if (entry.generation != std::numeric_limits<uint32_t>::max()) {
            ++entry.generation;
        }
    }

    return COMMON_TREE_DRAFT_C_HANDLE_STATUS_OK;
}

}

common_tree_draft_c_handle_status common_tree_draft_c_runtime_create(common_tree_draft_c_runtime_handle * out) {
    return create_handle(runtime_handles, HANDLE_KIND_RUNTIME, out);
}

common_tree_draft_c_handle_status common_tree_draft_c_runtime_retain(common_tree_draft_c_runtime_handle handle) {
    return retain_handle(runtime_handles, HANDLE_KIND_RUNTIME, handle);
}

common_tree_draft_c_handle_status common_tree_draft_c_runtime_release(common_tree_draft_c_runtime_handle handle) {
    return release_handle(runtime_handles, HANDLE_KIND_RUNTIME, handle);
}

common_tree_draft_c_handle_status common_tree_draft_c_session_create(common_tree_draft_c_session_handle * out) {
    return create_handle(session_handles, HANDLE_KIND_SESSION, out);
}

common_tree_draft_c_handle_status common_tree_draft_c_session_retain(common_tree_draft_c_session_handle handle) {
    return retain_handle(session_handles, HANDLE_KIND_SESSION, handle);
}

common_tree_draft_c_handle_status common_tree_draft_c_session_release(common_tree_draft_c_session_handle handle) {
    return release_handle(session_handles, HANDLE_KIND_SESSION, handle);
}

common_tree_draft_c_handle_status common_tree_draft_c_request_create(common_tree_draft_c_request_handle * out) {
    return create_handle(request_handles, HANDLE_KIND_REQUEST, out);
}

common_tree_draft_c_handle_status common_tree_draft_c_request_retain(common_tree_draft_c_request_handle handle) {
    return retain_handle(request_handles, HANDLE_KIND_REQUEST, handle);
}

common_tree_draft_c_handle_status common_tree_draft_c_request_release(common_tree_draft_c_request_handle handle) {
    return release_handle(request_handles, HANDLE_KIND_REQUEST, handle);
}
