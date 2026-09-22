#include "tree-draft-kv-branch.h"

#include <cassert>

int main() {
    const common_tree_draft_kv_layer_geometry layer = { 64, 64, COMMON_TREE_DRAFT_KV_LAYOUT_STANDARD };
    common_tree_draft_kv_page_geometry geometry = {};
    assert(common_tree_draft_kv_page_geometry_build(4, &layer, 1, 64, &geometry) == COMMON_TREE_DRAFT_KV_PAGE_OK);
    const common_tree_draft_kv_branch_span spans[] = {
        { {0,1}, 0, 4 },
        { {1,3}, 1, 3 },
    };
    const common_tree_draft_kv_branch_descriptor branch = {
        {7,1}, {3,2}, 3, 5, 9, spans, 2
    };
    assert(common_tree_draft_kv_branch_validate(branch, geometry) == COMMON_TREE_DRAFT_KV_BRANCH_OK);
    common_tree_draft_page_address address = {};
    assert(common_tree_draft_kv_branch_lookup(branch, 4, &address) == COMMON_TREE_DRAFT_KV_BRANCH_OK);
    assert(address.page.id == 1 && address.page.generation == 3 && address.offset == 1);
    assert(common_tree_draft_kv_branch_lookup(branch, 6, &address) == COMMON_TREE_DRAFT_KV_BRANCH_NOT_FOUND);
    return 0;
}

