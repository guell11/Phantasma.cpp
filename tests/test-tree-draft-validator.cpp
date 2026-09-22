#include "tree-draft-validator.h"

#include <cassert>
#include <cstdint>
#include <cstring>

static void expect_error(
        const common_tree_draft_validation_result & result,
        common_tree_draft_validator_error error,
        common_tree_draft_topology_error topology_error = COMMON_TREE_DRAFT_TOPOLOGY_OK) {
    assert(result.error == error);
    assert(result.topology_error == topology_error);
}

int main() {
    const int32_t parent[]  = { -1, 0, -1, 2, 2 };
    const int32_t depth[]   = {  0, 1,  0, 1, 1 };
    const int32_t tree_id[] = { 11, 11, 27, 27, 27 };
    const common_tree_draft_topology topology = { parent, depth, tree_id, 5 };

    const int32_t offsets[] = { 0, 2, 2, 5 };
    expect_error(common_tree_draft_validate(topology, { offsets, 3 }), COMMON_TREE_DRAFT_VALIDATOR_OK);
    assert(std::strcmp(common_tree_draft_validator_error_name(COMMON_TREE_DRAFT_VALIDATOR_OK), "ok") == 0);

    {
        const int32_t bad_parent[] = { -1, 1, -1, 2, 2 };
        const common_tree_draft_topology bad = { bad_parent, depth, tree_id, 5 };
        expect_error(
                common_tree_draft_validate(bad, { offsets, 3 }),
                COMMON_TREE_DRAFT_VALIDATOR_INVALID_TOPOLOGY,
                COMMON_TREE_DRAFT_TOPOLOGY_PARENT_RANGE);
    }
    {
        const int32_t bad_depth[] = { 0, 2, 0, 1, 1 };
        const common_tree_draft_topology bad = { parent, bad_depth, tree_id, 5 };
        expect_error(
                common_tree_draft_validate(bad, { offsets, 3 }),
                COMMON_TREE_DRAFT_VALIDATOR_INVALID_TOPOLOGY,
                COMMON_TREE_DRAFT_TOPOLOGY_DEPTH_MISMATCH);
    }
    {
        const int32_t bad_tree[] = { 11, 27, 27, 27, 27 };
        const common_tree_draft_topology bad = { parent, depth, bad_tree, 5 };
        expect_error(
                common_tree_draft_validate(bad, { offsets, 3 }),
                COMMON_TREE_DRAFT_VALIDATOR_INVALID_TOPOLOGY,
                COMMON_TREE_DRAFT_TOPOLOGY_PARENT_TREE_MISMATCH);
    }

    expect_error(common_tree_draft_validate(topology, { offsets, -1 }), COMMON_TREE_DRAFT_VALIDATOR_NEGATIVE_ENTRY_COUNT);
    expect_error(common_tree_draft_validate(topology, { nullptr, 3 }), COMMON_TREE_DRAFT_VALIDATOR_NULL_OFFSETS);

    {
        const int32_t bad[] = { 1, 2, 5 };
        expect_error(common_tree_draft_validate(topology, { bad, 2 }), COMMON_TREE_DRAFT_VALIDATOR_OFFSET_START);
    }
    {
        const int32_t bad[] = { 0, -1, 5 };
        expect_error(common_tree_draft_validate(topology, { bad, 2 }), COMMON_TREE_DRAFT_VALIDATOR_NEGATIVE_OFFSET);
    }
    {
        const int32_t bad[] = { 0, 3, 2, 5 };
        expect_error(common_tree_draft_validate(topology, { bad, 3 }), COMMON_TREE_DRAFT_VALIDATOR_OFFSET_ORDER);
    }
    {
        const int32_t bad[] = { 0, 6 };
        expect_error(common_tree_draft_validate(topology, { bad, 1 }), COMMON_TREE_DRAFT_VALIDATOR_OFFSET_RANGE);
    }
    {
        const int32_t bad[] = { 0, 4 };
        expect_error(common_tree_draft_validate(topology, { bad, 1 }), COMMON_TREE_DRAFT_VALIDATOR_OFFSET_END);
    }

    {
        const common_tree_draft_topology empty = { nullptr, nullptr, nullptr, 0 };
        const int32_t empty_offsets[] = { 0, 0, 0 };
        expect_error(common_tree_draft_validate(empty, { empty_offsets, 2 }), COMMON_TREE_DRAFT_VALIDATOR_OK);
    }

    return 0;
}
