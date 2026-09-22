#include "tree-draft-next-frontier.h"

#include <cassert>

int main() {
    common_tree_draft_node nodes[8] = {
        { -1,-1,0,0.0f,1,0 },
        { 0,10,1,-0.1f,10,0 },
        { 0,11,1,-0.2f,20,0 },
        { 1,12,2,-0.1f,30,0 },
        { 1,13,2,-0.3f,40,COMMON_TREE_DRAFT_NODE_FLAG_TERMINAL },
    };
    int32_t frontier_storage[8] = {};
    common_tree_draft_arena arena = { nodes, frontier_storage, 8, 5, 0 };
    const int32_t children[] = { 3, 4 };
    const int32_t carry[] = { 2 };
    const common_tree_draft_rank_entry ranks[] = {
        { -0.2,-0.2,-0.2,1,20,2 },
        { -0.15,-0.15,-0.15,2,30,3 },
    };
    assert(common_tree_draft_build_next_frontier(&arena, children, 2, carry, 1, ranks, 2, 2,
                COMMON_TREE_DRAFT_SCORE_CUMULATIVE, COMMON_TREE_DRAFT_DEPTH_NEUTRAL) == COMMON_TREE_DRAFT_NEXT_FRONTIER_OK);
    assert(arena.frontier_count == 2);
    assert(arena.frontier[0] == 3 && arena.frontier[1] == 2);
    assert((arena.nodes[4].flags & COMMON_TREE_DRAFT_NODE_FLAG_FRONTIER) == 0);

    const int32_t duplicate_carry[] = { 3 };
    assert(common_tree_draft_build_next_frontier(&arena, children, 1, duplicate_carry, 1, ranks, 2, 2,
                COMMON_TREE_DRAFT_SCORE_CUMULATIVE, COMMON_TREE_DRAFT_DEPTH_NEUTRAL) == COMMON_TREE_DRAFT_NEXT_FRONTIER_DUPLICATE);
    return 0;
}

