#include "tree-draft-forest.h"

#include <cassert>
#include <cstdint>
#include <limits>

int main() {
    const int32_t parent[]  = { -1, 0, -1, 2, 2 };
    const int32_t depth[]   = {  0, 1,  0, 1, 1 };
    const int32_t tree_id[] = { 11, 11, 27, 27, 27 };
    const common_tree_draft_topology topology = { parent, depth, tree_id, 5 };

    const int32_t counts[] = { 2, 0, 3 };
    int32_t offsets[4] = { -1, -1, -1, -1 };
    assert(common_tree_draft_forest_offsets_build(topology, counts, 3, offsets, 4) == COMMON_TREE_DRAFT_FOREST_OK);
    assert(offsets[0] == 0);
    assert(offsets[1] == 2);
    assert(offsets[2] == 2);
    assert(offsets[3] == 5);

    const common_tree_draft_forest_offsets forest = { offsets, 3 };
    int32_t global = -1;
    assert(common_tree_draft_forest_local_to_global(forest, 0, 1, &global) == COMMON_TREE_DRAFT_FOREST_OK);
    assert(global == 1);
    assert(common_tree_draft_forest_local_to_global(forest, 2, 0, &global) == COMMON_TREE_DRAFT_FOREST_OK);
    assert(global == 2);
    assert(common_tree_draft_forest_local_to_global(forest, 1, 0, &global) == COMMON_TREE_DRAFT_FOREST_LOCAL_RANGE);

    int32_t entry = -1;
    int32_t local = -1;
    assert(common_tree_draft_forest_global_to_local(forest, 0, &entry, &local) == COMMON_TREE_DRAFT_FOREST_OK);
    assert(entry == 0 && local == 0);
    assert(common_tree_draft_forest_global_to_local(forest, 2, &entry, &local) == COMMON_TREE_DRAFT_FOREST_OK);
    assert(entry == 2 && local == 0);
    assert(common_tree_draft_forest_global_to_local(forest, 4, &entry, &local) == COMMON_TREE_DRAFT_FOREST_OK);
    assert(entry == 2 && local == 2);
    assert(common_tree_draft_forest_global_to_local(forest, 5, &entry, &local) == COMMON_TREE_DRAFT_FOREST_GLOBAL_RANGE);
    assert(common_tree_draft_forest_local_to_global(forest, 0, 0, nullptr) == COMMON_TREE_DRAFT_FOREST_NULL_OUTPUT);
    assert(common_tree_draft_forest_global_to_local(forest, 0, nullptr, &local) == COMMON_TREE_DRAFT_FOREST_NULL_OUTPUT);

    {
        const common_tree_draft_topology empty = { nullptr, nullptr, nullptr, 0 };
        int32_t empty_offsets[3] = { -1, -1, -1 };
        const int32_t empty_counts[] = { 0, 0 };
        assert(common_tree_draft_forest_offsets_build(empty, empty_counts, 2, empty_offsets, 3) == COMMON_TREE_DRAFT_FOREST_OK);
        assert(empty_offsets[0] == 0 && empty_offsets[1] == 0 && empty_offsets[2] == 0);
        const common_tree_draft_forest_offsets empty_forest = { empty_offsets, 2 };
        assert(common_tree_draft_forest_global_to_local(empty_forest, 0, &entry, &local) == COMMON_TREE_DRAFT_FOREST_GLOBAL_RANGE);
    }

    {
        int32_t bad_offsets[3] = { -1, -1, -1 };
        const int32_t mismatch[] = { 2, 2 };
        assert(common_tree_draft_forest_offsets_build(topology, mismatch, 2, bad_offsets, 3) == COMMON_TREE_DRAFT_FOREST_NODE_COUNT_MISMATCH);
    }

    {
        int32_t bad_offsets[3] = { -1, -1, -1 };
        const int32_t negative[] = { 5, -1 };
        assert(common_tree_draft_forest_offsets_build(topology, negative, 2, bad_offsets, 3) == COMMON_TREE_DRAFT_FOREST_NEGATIVE_NODE_COUNT);
    }

    {
        int32_t bad_offsets[3] = { -1, -1, -1 };
        const int32_t overflow[] = { std::numeric_limits<int32_t>::max(), 1 };
        assert(common_tree_draft_forest_offsets_build(topology, overflow, 2, bad_offsets, 3) == COMMON_TREE_DRAFT_FOREST_NODE_COUNT_OVERFLOW);
    }

    return 0;
}
