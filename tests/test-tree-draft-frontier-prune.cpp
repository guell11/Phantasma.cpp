#include "tree-draft-frontier-prune.h"

#include <cassert>

static common_tree_draft_config config() {
    const common_tree_draft_request_params params = { 8, 4, 4, 1.0f, 1 };
    const common_tree_draft_limits limits = { 8, 4, 4 };
    auto admitted = common_tree_draft_config::admit(params, limits);
    assert(admitted.has_value());
    return *admitted;
}

int main() {
    common_tree_draft_node nodes[8] = {};
    int32_t frontier_storage[8] = {};
    common_tree_draft_arena arena = {};
    const auto cfg = config();
    assert(common_tree_draft_arena_init(&arena, nodes, frontier_storage, 8, cfg) == COMMON_TREE_DRAFT_ARENA_OK);

    const common_tree_draft_node seed[] = {
        { -1, -1, 0, 0.0f, 1, 0 },
        { 0, 10, 1, -0.1f, 10, COMMON_TREE_DRAFT_NODE_FLAG_FRONTIER },
        { 0, 11, 1, -0.2f, 20, COMMON_TREE_DRAFT_NODE_FLAG_FRONTIER },
        { 0, 12, 1, -0.3f, 30, COMMON_TREE_DRAFT_NODE_FLAG_FRONTIER },
    };
    for (const auto & node : seed) {
        assert(common_tree_draft_arena_append(&arena, node, cfg) == COMMON_TREE_DRAFT_ARENA_OK);
    }
    const int32_t frontier[] = { 1, 2, 3 };
    assert(common_tree_draft_arena_set_frontier(&arena, frontier, 3) == COMMON_TREE_DRAFT_ARENA_OK);

    const common_tree_draft_rank_entry ranks[] = {
        { -0.1, -0.1, -0.1, 1, 10, 1 },
        { -0.2, -0.2, -0.2, 1, 20, 2 },
        { -0.3, -0.3, -0.3, 1, 30, 3 },
    };
    assert(common_tree_draft_frontier_prune(&arena, ranks, 3, 2, COMMON_TREE_DRAFT_SCORE_CUMULATIVE,
                COMMON_TREE_DRAFT_DEPTH_NEUTRAL) == COMMON_TREE_DRAFT_FRONTIER_PRUNE_OK);
    assert(arena.frontier_count == 2);
    assert(arena.frontier[0] == 1 && arena.frontier[1] == 2);
    assert((arena.nodes[1].flags & COMMON_TREE_DRAFT_NODE_FLAG_FRONTIER) != 0);
    assert((arena.nodes[2].flags & COMMON_TREE_DRAFT_NODE_FLAG_FRONTIER) != 0);
    assert((arena.nodes[3].flags & COMMON_TREE_DRAFT_NODE_FLAG_PRUNED) != 0);
    assert(arena.node_count == 4);

    common_tree_draft_arena arena2 = {};
    common_tree_draft_node nodes2[8] = {};
    int32_t frontier2[8] = {};
    assert(common_tree_draft_arena_init(&arena2, nodes2, frontier2, 8, cfg) == COMMON_TREE_DRAFT_ARENA_OK);
    for (const auto & node : seed) assert(common_tree_draft_arena_append(&arena2, node, cfg) == COMMON_TREE_DRAFT_ARENA_OK);
    assert(common_tree_draft_arena_set_frontier(&arena2, frontier, 3) == COMMON_TREE_DRAFT_ARENA_OK);
    assert(common_tree_draft_frontier_prune(&arena2, ranks, 3, 0, COMMON_TREE_DRAFT_SCORE_CUMULATIVE,
                COMMON_TREE_DRAFT_DEPTH_NEUTRAL) == COMMON_TREE_DRAFT_FRONTIER_PRUNE_OK);
    assert(arena2.frontier_count == 0);
    assert((arena2.nodes[1].flags & COMMON_TREE_DRAFT_NODE_FLAG_PRUNED) != 0);
    assert((arena2.nodes[2].flags & COMMON_TREE_DRAFT_NODE_FLAG_PRUNED) != 0);
    assert((arena2.nodes[3].flags & COMMON_TREE_DRAFT_NODE_FLAG_PRUNED) != 0);
    return 0;
}

