#include "tree-draft-append.h"

#include <cassert>

int main() {
    common_tree_draft_node nodes[8] = {};
    int32_t frontier[8] = {};
    common_tree_draft_arena arena = { nodes, frontier, 8, 2, 0 };
    nodes[0] = { -1, -1, 0, 0.0f, 1, 0 };
    nodes[1] = { 0, 10, 1, -0.1f, 2, COMMON_TREE_DRAFT_NODE_FLAG_FRONTIER };
    common_tree_draft_budget_state budget = common_tree_draft_budget_make({ 8, 4, 8, 4 });

    const uint32_t counts[] = { 2, 1 };
    uint32_t offsets[3] = {};
    uint8_t written[3] = {};
    common_tree_draft_append_transaction tx = {};
    assert(common_tree_draft_append_reserve(&arena, &budget, counts, 2, offsets, 3, written, 3, &tx) == COMMON_TREE_DRAFT_APPEND_OK);
    assert(offsets[0] == 0 && offsets[1] == 2 && offsets[2] == 3);
    assert(arena.node_count == 2 && budget.reserved.nodes == 3);
    assert(common_tree_draft_append_write(&arena, &tx, 0, {1,20,2,-0.2f,3,0}) == COMMON_TREE_DRAFT_APPEND_OK);
    assert(common_tree_draft_append_write(&arena, &tx, 1, {1,21,2,-0.3f,4,0}) == COMMON_TREE_DRAFT_APPEND_OK);
    assert(common_tree_draft_append_write(&arena, &tx, 2, {1,22,2,-0.4f,5,0}) == COMMON_TREE_DRAFT_APPEND_OK);
    assert(arena.node_count == 2);
    assert(common_tree_draft_append_commit(&arena, &budget, &tx) == COMMON_TREE_DRAFT_APPEND_OK);
    assert(arena.node_count == 5 && budget.committed.nodes == 3 && budget.reserved.nodes == 0);
    assert(common_tree_draft_append_commit(&arena, &budget, &tx) == COMMON_TREE_DRAFT_APPEND_PHASE);

    common_tree_draft_node nodes2[4] = {};
    int32_t frontier2[4] = {};
    common_tree_draft_arena arena2 = { nodes2, frontier2, 4, 1, 0 };
    nodes2[0] = { -1,-1,0,0.0f,1,0 };
    common_tree_draft_budget_state budget2 = common_tree_draft_budget_make({4,4,4,4});
    const uint32_t counts2[] = { 2 };
    uint32_t offsets2[2] = {};
    uint8_t written2[2] = {};
    common_tree_draft_append_transaction tx2 = {};
    assert(common_tree_draft_append_reserve(&arena2, &budget2, counts2, 1, offsets2, 2, written2, 2, &tx2) == COMMON_TREE_DRAFT_APPEND_OK);
    assert(common_tree_draft_append_write(&arena2, &tx2, 0, {0,10,1,-0.1f,2,0}) == COMMON_TREE_DRAFT_APPEND_OK);
    assert(common_tree_draft_append_commit(&arena2, &budget2, &tx2) == COMMON_TREE_DRAFT_APPEND_INCOMPLETE);
    assert(arena2.node_count == 1);
    assert(common_tree_draft_append_abort(&budget2, &tx2) == COMMON_TREE_DRAFT_APPEND_OK);
    assert(arena2.node_count == 1 && budget2.reserved.nodes == 0);
    return 0;
}

