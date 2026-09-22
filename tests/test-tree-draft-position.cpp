#include "tree-draft-position.h"

#include <cassert>
#include <cstdint>
#include <limits>
#include <string>

int main() {
    const int32_t parent[]  = { -1, 0, 0, -1, 3, 3 };
    const int32_t depth[]   = {  0, 1, 1,  0, 1, 1 };
    const int32_t tree_id[] = { 10,10,10, 20,20,20 };
    const common_tree_draft_topology topology = { parent, depth, tree_id, 6 };

    const int32_t counts[] = { 3, 0, 3 };
    int32_t offsets[4] = {};
    assert(common_tree_draft_forest_offsets_build(topology, counts, 3, offsets, 4) == COMMON_TREE_DRAFT_FOREST_OK);
    const common_tree_draft_forest_offsets forest = { offsets, 3 };

    const int32_t prefixes[] = { 100, 777, 2000 };
    int32_t positions[8] = { -1, -1, -1, -1, -1, -1, 12345, 12345 };
    assert(common_tree_draft_positions_build(topology, forest, prefixes, { positions, 8 }) ==
        COMMON_TREE_DRAFT_POSITION_OK);

    const int32_t expected[] = { 100, 101, 101, 2000, 2001, 2001 };
    for (int32_t i = 0; i < topology.n_nodes; ++i) {
        assert(positions[i] == expected[i]);
    }
    assert(positions[6] == 12345 && positions[7] == 12345);
    assert(positions[1] == positions[2]);
    assert(positions[4] == positions[5]);

    int32_t global = -1;
    int32_t position = -1;
    assert(common_tree_draft_position_for_local(topology, forest, prefixes, 0, 2, &global, &position) ==
        COMMON_TREE_DRAFT_POSITION_OK);
    assert(global == 2 && position == 101);
    assert(common_tree_draft_position_for_local(topology, forest, prefixes, 2, 1, &global, &position) ==
        COMMON_TREE_DRAFT_POSITION_OK);
    assert(global == 4 && position == 2001);
    for (int32_t expected_global = 0; expected_global < topology.n_nodes; ++expected_global) {
        int32_t entry = -1;
        int32_t local = -1;
        assert(common_tree_draft_forest_global_to_local(forest, expected_global, &entry, &local) ==
            COMMON_TREE_DRAFT_FOREST_OK);
        global = -1;
        position = -1;
        assert(common_tree_draft_position_for_local(topology, forest, prefixes, entry, local, &global, &position) ==
            COMMON_TREE_DRAFT_POSITION_OK);
        assert(global == expected_global);
        assert(position == positions[expected_global]);
    }
    assert(common_tree_draft_position_for_local(topology, forest, prefixes, 1, 0, &global, &position) ==
        COMMON_TREE_DRAFT_POSITION_LOCAL_RANGE);
    assert(common_tree_draft_position_for_local(topology, forest, prefixes, 3, 0, &global, &position) ==
        COMMON_TREE_DRAFT_POSITION_ENTRY_RANGE);

    {
        const common_tree_draft_topology empty = { nullptr, nullptr, nullptr, 0 };
        const int32_t empty_offsets[] = { 0, 0, 0 };
        const int32_t empty_prefixes[] = { 4, 9 };
        const common_tree_draft_forest_offsets empty_forest = { empty_offsets, 2 };
        assert(common_tree_draft_positions_build(empty, empty_forest, empty_prefixes, { nullptr, 0 }) ==
            COMMON_TREE_DRAFT_POSITION_OK);
    }

    assert(common_tree_draft_positions_build(topology, forest, nullptr, { positions, 8 }) ==
        COMMON_TREE_DRAFT_POSITION_NULL_PREFIX_LENGTHS);
    {
        const int32_t bad_prefixes[] = { 100, -1, 2000 };
        assert(common_tree_draft_positions_build(topology, forest, bad_prefixes, { positions, 8 }) ==
            COMMON_TREE_DRAFT_POSITION_NEGATIVE_PREFIX_LENGTH);
    }
    assert(common_tree_draft_positions_build(topology, forest, prefixes, { nullptr, 0 }) ==
        COMMON_TREE_DRAFT_POSITION_NULL_OUTPUT);
    {
        int32_t too_small[5] = { 9, 9, 9, 9, 9 };
        assert(common_tree_draft_positions_build(topology, forest, prefixes, { too_small, 5 }) ==
            COMMON_TREE_DRAFT_POSITION_OUTPUT_TOO_SMALL);
        for (int32_t value : too_small) {
            assert(value == 9);
        }
    }
    {
        const int32_t overflow_prefixes[] = { std::numeric_limits<int32_t>::max(), 0, 0 };
        int32_t untouched[6] = { 8, 8, 8, 8, 8, 8 };
        assert(common_tree_draft_positions_build(topology, forest, overflow_prefixes, { untouched, 6 }) ==
            COMMON_TREE_DRAFT_POSITION_OVERFLOW);
        for (int32_t value : untouched) {
            assert(value == 8);
        }
    }
    {
        const int32_t bad_offsets[] = { 0, 4, 3, 6 };
        const common_tree_draft_forest_offsets bad_forest = { bad_offsets, 3 };
        assert(common_tree_draft_positions_build(topology, bad_forest, prefixes, { positions, 8 }) ==
            COMMON_TREE_DRAFT_POSITION_INVALID_FOREST);
    }
    {
        const int32_t bad_offsets[] = { 0, 3, 3, 5 };
        const common_tree_draft_forest_offsets bad_forest = { bad_offsets, 3 };
        assert(common_tree_draft_positions_build(topology, bad_forest, prefixes, { positions, 8 }) ==
            COMMON_TREE_DRAFT_POSITION_INVALID_FOREST);
    }
    {
        const int32_t split_tree_offsets[] = { 0, 1, 3, 6 };
        const common_tree_draft_forest_offsets split_tree_forest = { split_tree_offsets, 3 };
        assert(common_tree_draft_positions_build(topology, split_tree_forest, prefixes, { positions, 8 }) ==
            COMMON_TREE_DRAFT_POSITION_INVALID_FOREST);
    }
    {
        const int32_t bad_parent[] = { -1, 1 };
        const int32_t bad_depth[] = { 0, 1 };
        const int32_t bad_tree[] = { 0, 0 };
        const common_tree_draft_topology invalid = { bad_parent, bad_depth, bad_tree, 2 };
        const int32_t invalid_offsets[] = { 0, 2 };
        const int32_t invalid_prefixes[] = { 0 };
        assert(common_tree_draft_positions_build(invalid, { invalid_offsets, 1 }, invalid_prefixes, { positions, 8 }) ==
            COMMON_TREE_DRAFT_POSITION_INVALID_TOPOLOGY);
    }

    assert(common_tree_draft_position_for_local(topology, forest, prefixes, 0, 0, nullptr, &position) ==
        COMMON_TREE_DRAFT_POSITION_NULL_MAPPING_OUTPUT);
    assert(std::string(common_tree_draft_position_error_name(COMMON_TREE_DRAFT_POSITION_OVERFLOW)) == "overflow");

    return 0;
}
