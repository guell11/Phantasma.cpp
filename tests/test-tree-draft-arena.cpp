#include "tree-draft-arena.h"

#include <cassert>
#include <cstdint>

static common_tree_draft_config make_config(uint32_t budget = 4) {
    const common_tree_draft_request_params params = { budget, 4, 4, 1.0f, 9 };
    const common_tree_draft_limits limits = { budget, 4, 4 };
    auto config = common_tree_draft_config::admit(params, limits);
    assert(config.has_value());
    return *config;
}

static common_tree_draft_node root() {
    return { -1, -1, 0, 0.0f, 1, COMMON_TREE_DRAFT_NODE_FLAG_FRONTIER };
}

static void test_preallocated_layout_and_append() {
    const auto config = make_config();
    common_tree_draft_node nodes[4] = {};
    int32_t frontier_storage[4] = {};
    common_tree_draft_arena arena = {};

    assert(common_tree_draft_arena_init(&arena, nodes, frontier_storage, 4, config) == COMMON_TREE_DRAFT_ARENA_OK);
    assert(arena.capacity == 4 && arena.node_count == 0 && arena.frontier_count == 0);

    uint32_t index = UINT32_MAX;
    assert(common_tree_draft_arena_append(&arena, root(), config, &index) == COMMON_TREE_DRAFT_ARENA_OK);
    assert(index == 0 && arena.node_count == 1);

    const common_tree_draft_node child = { 0, 42, 1, -0.2f, 2, COMMON_TREE_DRAFT_NODE_FLAG_FRONTIER };
    assert(common_tree_draft_arena_append(&arena, child, config, &index) == COMMON_TREE_DRAFT_ARENA_OK);
    assert(index == 1 && arena.node_count == 2);
    assert(arena.nodes[1].token == 42);

    const int32_t next_frontier[] = { 1 };
    assert(common_tree_draft_arena_set_frontier(&arena, next_frontier, 1) == COMMON_TREE_DRAFT_ARENA_OK);
    assert(arena.frontier_count == 1 && arena.frontier[0] == 1);
}

static void test_explicit_capacity_and_commit_boundary() {
    const auto config = make_config(2);
    common_tree_draft_node nodes[2] = {};
    int32_t frontier_storage[2] = {};
    common_tree_draft_arena arena = {};
    assert(common_tree_draft_arena_init(&arena, nodes, frontier_storage, 2, config) == COMMON_TREE_DRAFT_ARENA_OK);
    assert(common_tree_draft_arena_append(&arena, root(), config) == COMMON_TREE_DRAFT_ARENA_OK);

    const common_tree_draft_node invalid = { 1, 42, 1, -0.2f, 2, 0 };
    assert(common_tree_draft_arena_append(&arena, invalid, config) == COMMON_TREE_DRAFT_ARENA_NODE_INVALID);
    assert(arena.node_count == 1);

    const common_tree_draft_node child = { 0, 42, 1, -0.2f, 2, 0 };
    assert(common_tree_draft_arena_append(&arena, child, config) == COMMON_TREE_DRAFT_ARENA_OK);
    assert(arena.node_count == 2);
    assert(common_tree_draft_arena_append(&arena, child, config) == COMMON_TREE_DRAFT_ARENA_CAPACITY);
    assert(arena.node_count == 2);
}

static void test_frontier_validation_is_atomic() {
    const auto config = make_config();
    common_tree_draft_node nodes[4] = {};
    int32_t frontier_storage[4] = { -1, -1, -1, -1 };
    common_tree_draft_arena arena = {};
    assert(common_tree_draft_arena_init(&arena, nodes, frontier_storage, 4, config) == COMMON_TREE_DRAFT_ARENA_OK);
    assert(common_tree_draft_arena_append(&arena, root(), config) == COMMON_TREE_DRAFT_ARENA_OK);
    const common_tree_draft_node child = { 0, 7, 1, -0.3f, 2, 0 };
    assert(common_tree_draft_arena_append(&arena, child, config) == COMMON_TREE_DRAFT_ARENA_OK);

    const int32_t valid[] = { 0, 1 };
    assert(common_tree_draft_arena_set_frontier(&arena, valid, 2) == COMMON_TREE_DRAFT_ARENA_OK);

    const int32_t duplicate[] = { 1, 1 };
    assert(common_tree_draft_arena_set_frontier(&arena, duplicate, 2) == COMMON_TREE_DRAFT_ARENA_FRONTIER_DUPLICATE);
    assert(arena.frontier_count == 2 && arena.frontier[0] == 0 && arena.frontier[1] == 1);

    const int32_t out_of_range[] = { 2 };
    assert(common_tree_draft_arena_set_frontier(&arena, out_of_range, 1) == COMMON_TREE_DRAFT_ARENA_FRONTIER_RANGE);
    assert(arena.frontier_count == 2);
}

static void test_init_rejects_hidden_growth() {
    const auto config = make_config(4);
    common_tree_draft_node nodes[5] = {};
    int32_t frontier[5] = {};
    common_tree_draft_arena arena = {};
    assert(common_tree_draft_arena_init(&arena, nodes, frontier, 5, config) == COMMON_TREE_DRAFT_ARENA_CAPACITY);
}

int main() {
    test_preallocated_layout_and_append();
    test_explicit_capacity_and_commit_boundary();
    test_frontier_validation_is_atomic();
    test_init_rejects_hidden_growth();
    return 0;
}

