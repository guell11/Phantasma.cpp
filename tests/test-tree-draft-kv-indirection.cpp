#include "tree-draft-kv-indirection.h"

#include <cassert>
#include <cstdint>
#include <limits>
#include <string>

int main() {
    const int32_t parent[]  = { -1, 0, 0, 1, -1, 4 };
    const int32_t depth[]   = {  0, 1, 1, 2,  0, 1 };
    const int32_t tree_id[] = {  7, 7, 7, 7, 42, 42 };
    const common_tree_draft_topology topology = { parent, depth, tree_id, 6 };

    const uint32_t slots[] = { 9, 12, 12, COMMON_TREE_DRAFT_KV_SLOT_INVALID, 3, 9 };
    const common_tree_draft_kv_indirection indirection = { slots, 6, 16 };
    assert(common_tree_draft_kv_indirection_validate(topology, indirection) ==
        COMMON_TREE_DRAFT_KV_INDIRECTION_OK);

    uint32_t slot = 0;
    assert(common_tree_draft_kv_slot_for_logical(topology, indirection, 0, &slot) ==
        COMMON_TREE_DRAFT_KV_INDIRECTION_OK);
    assert(slot == 9);
    assert(common_tree_draft_kv_slot_for_logical(topology, indirection, 1, &slot) ==
        COMMON_TREE_DRAFT_KV_INDIRECTION_OK);
    assert(slot == 12);
    assert(common_tree_draft_kv_slot_for_logical(topology, indirection, 2, &slot) ==
        COMMON_TREE_DRAFT_KV_INDIRECTION_OK);
    assert(slot == 12);
    assert(common_tree_draft_kv_slot_for_logical(topology, indirection, 5, &slot) ==
        COMMON_TREE_DRAFT_KV_INDIRECTION_OK);
    assert(slot == 9);
    assert(common_tree_draft_kv_slot_for_logical(topology, indirection, 3, &slot) ==
        COMMON_TREE_DRAFT_KV_INDIRECTION_INVALID_SLOT);
    assert(common_tree_draft_kv_slot_for_logical(topology, indirection, 6, &slot) ==
        COMMON_TREE_DRAFT_KV_INDIRECTION_LOGICAL_RANGE);
    assert(common_tree_draft_kv_slot_for_logical(topology, indirection, 0, nullptr) ==
        COMMON_TREE_DRAFT_KV_INDIRECTION_NULL_OUTPUT);

    const uint8_t storage[4096] = {};
    const common_tree_draft_kv_layout layout = {
        storage,
        16,
        2,
        4,
        128,
        32,
        4,
        sizeof(float),
    };
    size_t byte_offset = 0;
    assert(common_tree_draft_kv_address(layout, 12, 1, 3, &byte_offset) ==
        COMMON_TREE_DRAFT_KV_INDIRECTION_OK);
    assert(byte_offset == 12 * 128 + 1 * 32 + 3 * 4);
    assert(common_tree_draft_kv_address(layout, COMMON_TREE_DRAFT_KV_SLOT_INVALID, 0, 0, &byte_offset) ==
        COMMON_TREE_DRAFT_KV_INDIRECTION_INVALID_SLOT);
    assert(common_tree_draft_kv_address(layout, 16, 0, 0, &byte_offset) ==
        COMMON_TREE_DRAFT_KV_INDIRECTION_SLOT_RANGE);

    {
        const uint32_t bad_slots[] = { 9, 16, 12, 1, 3, 9 };
        assert(common_tree_draft_kv_indirection_validate(topology, { bad_slots, 6, 16 }) ==
            COMMON_TREE_DRAFT_KV_INDIRECTION_SLOT_RANGE);
    }
    assert(common_tree_draft_kv_indirection_validate(topology, { nullptr, 6, 16 }) ==
        COMMON_TREE_DRAFT_KV_INDIRECTION_NULL_SLOTS);
    assert(common_tree_draft_kv_indirection_validate(topology, { slots, 5, 16 }) ==
        COMMON_TREE_DRAFT_KV_INDIRECTION_SLOT_COUNT_MISMATCH);

    {
        const common_tree_draft_topology empty = { nullptr, nullptr, nullptr, 0 };
        assert(common_tree_draft_kv_indirection_validate(empty, { nullptr, 0, 0 }) ==
            COMMON_TREE_DRAFT_KV_INDIRECTION_OK);
    }
    {
        const int32_t bad_parent[] = { -1, 1 };
        const int32_t bad_depth[] = { 0, 1 };
        const int32_t bad_tree[] = { 0, 0 };
        const common_tree_draft_topology invalid = { bad_parent, bad_depth, bad_tree, 2 };
        const uint32_t invalid_slots[] = { 0, 1 };
        assert(common_tree_draft_kv_indirection_validate(invalid, { invalid_slots, 2, 2 }) ==
            COMMON_TREE_DRAFT_KV_INDIRECTION_INVALID_TOPOLOGY);
    }
    {
        auto overflow_layout = layout;
        overflow_layout.stride_slot = std::numeric_limits<size_t>::max();
        assert(common_tree_draft_kv_address(overflow_layout, 2, 0, 0, &byte_offset) ==
            COMMON_TREE_DRAFT_KV_INDIRECTION_ADDRESS_OVERFLOW);
    }
    {
        auto invalid_layout = layout;
        invalid_layout.element_size = 0;
        assert(common_tree_draft_kv_address(invalid_layout, 0, 0, 0, &byte_offset) ==
            COMMON_TREE_DRAFT_KV_INDIRECTION_INVALID_LAYOUT);
    }

    assert(std::string(common_tree_draft_kv_indirection_error_name(COMMON_TREE_DRAFT_KV_INDIRECTION_INVALID_SLOT)) ==
        "invalid_slot");
    return 0;
}
