#include "tree-draft-path.h"

#include <cassert>
#include <cstdint>

int main() {
    const uint64_t root = common_tree_draft_path_root(123);
    assert(root == UINT64_C(0x8512d07c94746efa));

    uint64_t child0 = 0;
    uint64_t child1 = 0;
    assert(common_tree_draft_path_child(root, 42, 0, 1, &child0) == COMMON_TREE_DRAFT_PATH_OK);
    assert(common_tree_draft_path_child(root, 43, 1, 1, &child1) == COMMON_TREE_DRAFT_PATH_OK);
    assert(child0 == UINT64_C(0x8214b078536ae62c));
    assert(child1 == UINT64_C(0xda47aedd9173a3b0));

    uint64_t reordered = 0;
    assert(common_tree_draft_path_child(root, 42, 0, 1, &reordered) == COMMON_TREE_DRAFT_PATH_OK);
    assert(reordered == child0);
    assert(child0 != child1);

    uint64_t changed = 0;
    assert(common_tree_draft_path_child(root, 42, 1, 1, &changed) == COMMON_TREE_DRAFT_PATH_OK);
    assert(changed != child0);
    assert(common_tree_draft_path_child(root, 42, 0, 2, &changed) == COMMON_TREE_DRAFT_PATH_OK);
    assert(changed != child0);

    assert(common_tree_draft_path_child(root, -1, 0, 1, &changed) == COMMON_TREE_DRAFT_PATH_INVALID_TOKEN);
    assert(common_tree_draft_path_child(root, 42, 0, 0, &changed) == COMMON_TREE_DRAFT_PATH_INVALID_DEPTH);
    assert(common_tree_draft_path_child(0, 42, 0, 1, &changed) == COMMON_TREE_DRAFT_PATH_INVALID_DEPTH);

    const uint64_t existing[] = { root, child0, child1 };
    assert(common_tree_draft_path_check_unique(changed, existing, 3) == COMMON_TREE_DRAFT_PATH_OK);
    assert(common_tree_draft_path_check_unique(child0, existing, 3) == COMMON_TREE_DRAFT_PATH_COLLISION);
    return 0;
}

