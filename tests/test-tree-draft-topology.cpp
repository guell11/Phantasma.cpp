#include "tree-draft-topology.h"

#include <cassert>
#include <cstdint>
#include <cstring>

static common_tree_draft_topology make_topology(
        const int32_t * parent,
        const int32_t * depth,
        const int32_t * tree_id,
        int32_t n_nodes) {
    return { parent, depth, tree_id, n_nodes };
}

int main() {
    {
        const common_tree_draft_topology empty = { nullptr, nullptr, nullptr, 0 };
        assert(common_tree_draft_topology_validate(empty) == COMMON_TREE_DRAFT_TOPOLOGY_OK);
    }

    // Two packed trees: 0 -> {1,2}, 1 -> 3 and 4 -> 5.
    const int32_t parent[]  = { -1, 0, 0, 1, -1, 4 };
    const int32_t depth[]   = {  0, 1, 1, 2,  0, 1 };
    const int32_t tree_id[] = {  0, 0, 0, 0,  1, 1 };
    assert(common_tree_draft_topology_validate(make_topology(parent, depth, tree_id, 6)) == COMMON_TREE_DRAFT_TOPOLOGY_OK);
    assert(std::strcmp(common_tree_draft_topology_error_name(COMMON_TREE_DRAFT_TOPOLOGY_OK), "ok") == 0);

    {
        const int32_t bad_parent[] = { -1, 2, 0, 1, -1, 4 };
        assert(common_tree_draft_topology_validate(make_topology(bad_parent, depth, tree_id, 6)) == COMMON_TREE_DRAFT_TOPOLOGY_PARENT_RANGE);
    }
    {
        const int32_t bad_depth[] = { 0, 1, 1, 3, 0, 1 };
        assert(common_tree_draft_topology_validate(make_topology(parent, bad_depth, tree_id, 6)) == COMMON_TREE_DRAFT_TOPOLOGY_DEPTH_MISMATCH);
    }
    {
        const int32_t bad_tree[] = { 0, 0, 1, 0, 1, 1 };
        assert(common_tree_draft_topology_validate(make_topology(parent, depth, bad_tree, 6)) == COMMON_TREE_DRAFT_TOPOLOGY_PARENT_TREE_MISMATCH);
    }
    {
        const int32_t duplicate_root_parent[] = { -1, 0, -1 };
        const int32_t duplicate_root_depth[] = { 0, 1, 0 };
        const int32_t duplicate_root_tree[] = { 0, 0, 0 };
        assert(common_tree_draft_topology_validate(make_topology(duplicate_root_parent, duplicate_root_depth, duplicate_root_tree, 3)) == COMMON_TREE_DRAFT_TOPOLOGY_MULTIPLE_ROOTS);
    }
    {
        const int32_t sparse_tree_parent[] = { -1, -1, 0, 1 };
        const int32_t sparse_tree_depth[] = { 0, 0, 1, 1 };
        const int32_t sparse_tree_id[] = { 7, 42, 7, 42 };
        assert(common_tree_draft_topology_validate(make_topology(sparse_tree_parent, sparse_tree_depth, sparse_tree_id, 4)) == COMMON_TREE_DRAFT_TOPOLOGY_OK);
    }
    {
        const int32_t bad_root_depth[] = { 1 };
        const int32_t one_root_parent[] = { -1 };
        const int32_t one_root_tree[] = { 0 };
        assert(common_tree_draft_topology_validate(make_topology(one_root_parent, bad_root_depth, one_root_tree, 1)) == COMMON_TREE_DRAFT_TOPOLOGY_ROOT_DEPTH);
    }

    return 0;
}
