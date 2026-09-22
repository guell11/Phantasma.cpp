#include "tree-draft-child-write.h"

#include <cassert>
#include <cmath>

int main() {
    common_tree_draft_node nodes[6] = {};
    int32_t frontier[6] = {};
    common_tree_draft_arena arena = { nodes, frontier, 6, 2, 0 };
    nodes[0] = { -1,-1,0,0.0f,100,0 };
    nodes[1] = { 0,10,1,std::log(0.5f),200,COMMON_TREE_DRAFT_NODE_FLAG_FRONTIER };
    common_tree_draft_budget_state budget = common_tree_draft_budget_make({6,4,6,4});
    const uint32_t counts[] = { 1 };
    uint32_t offsets[2] = {};
    uint8_t written[1] = {};
    common_tree_draft_append_transaction tx = {};
    assert(common_tree_draft_append_reserve(&arena, &budget, counts, 1, offsets, 2, written, 1, &tx) == COMMON_TREE_DRAFT_APPEND_OK);

    common_tree_draft_child_sample sample = { 42, 0.25f, std::log(0.25f), 0, {} };
    common_tree_draft_node child = {};
    assert(common_tree_draft_fill_child_metadata(&arena, &tx, 1, 0, sample, &child) == COMMON_TREE_DRAFT_CHILD_WRITE_OK);
    assert(child.parent == 1 && child.token == 42 && child.depth == 2 && child.path_id != 0 && child.flags == 0);
    assert(common_tree_draft_append_write(&arena, &tx, 0, child) == COMMON_TREE_DRAFT_APPEND_OK);

    double cumulative_before[6] = { 0.0, std::log(0.5), 0,0,0,0 };
    float local_p[6] = {};
    double cumulative[6] = {};
    assert(common_tree_draft_write_child_probability(&arena, &tx, 0, sample, cumulative_before, 2, local_p, 6, cumulative, 6) == COMMON_TREE_DRAFT_CHILD_WRITE_OK);
    assert(std::fabs(local_p[2] - 0.25f) < 1e-7f);
    assert(std::fabs(cumulative[2] - std::log(0.125)) < 1e-5);
    assert(std::fabs(nodes[2].logp - std::log(0.25f)) < 1e-6f);
    assert(common_tree_draft_append_commit(&arena, &budget, &tx) == COMMON_TREE_DRAFT_APPEND_OK);
    assert(arena.node_count == 3);
    return 0;
}

