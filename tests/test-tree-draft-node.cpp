#include "tree-draft-node.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

static common_tree_draft_config make_config(uint32_t budget = 8, uint32_t max_depth = 4) {
    const common_tree_draft_request_params params = {
        budget,
        max_depth,
        4,
        1.0f,
        7,
    };
    const common_tree_draft_limits limits = {
        budget,
        max_depth,
        4,
    };
    auto config = common_tree_draft_config::admit(params, limits);
    assert(config.has_value());
    return *config;
}

static common_tree_draft_node nodes[] = {
    { -1, -1, 0,  0.0f, 11, COMMON_TREE_DRAFT_NODE_FLAG_FRONTIER },
    {  0, 42, 1, -0.2f, 12, COMMON_TREE_DRAFT_NODE_FLAG_NONE },
    {  0, 43, 1, -0.5f, 13, COMMON_TREE_DRAFT_NODE_FLAG_FRONTIER },
    {  1, 44, 2, -0.1f, 14, COMMON_TREE_DRAFT_NODE_FLAG_TERMINAL },
};

static void test_abi_and_valid_nodes() {
    static_assert(std::is_standard_layout<common_tree_draft_node>::value, "POD ABI required");
    static_assert(std::is_trivially_copyable<common_tree_draft_node>::value, "GPU-copyable ABI required");
    static_assert(sizeof(common_tree_draft_node) == 32, "stable node size required");

    const auto config = make_config();
    assert(common_tree_draft_nodes_validate(nullptr, 0, config) == COMMON_TREE_DRAFT_NODE_OK);
    assert(common_tree_draft_nodes_validate(nodes, 4, config) == COMMON_TREE_DRAFT_NODE_OK);
}

static void test_parent_depth_and_budget_invariants() {
    auto config = make_config();

    auto copy = nodes;
    (void) copy;

    common_tree_draft_node bad_parent[4] = { nodes[0], nodes[1], nodes[2], nodes[3] };
    bad_parent[2].parent = 2;
    assert(common_tree_draft_nodes_validate(bad_parent, 4, config) == COMMON_TREE_DRAFT_NODE_PARENT_RANGE);

    common_tree_draft_node bad_depth[4] = { nodes[0], nodes[1], nodes[2], nodes[3] };
    bad_depth[3].depth = 3;
    assert(common_tree_draft_nodes_validate(bad_depth, 4, config) == COMMON_TREE_DRAFT_NODE_DEPTH_MISMATCH);

    const auto shallow = make_config(8, 1);
    assert(common_tree_draft_nodes_validate(nodes, 4, shallow) == COMMON_TREE_DRAFT_NODE_DEPTH_LIMIT);

    const auto tiny = make_config(3, 4);
    assert(common_tree_draft_nodes_validate(nodes, 4, tiny) == COMMON_TREE_DRAFT_NODE_COUNT_RANGE);
}

static void test_probability_identity_and_flags() {
    const auto config = make_config();

    common_tree_draft_node bad_logp[4] = { nodes[0], nodes[1], nodes[2], nodes[3] };
    bad_logp[1].logp = 0.1f;
    assert(common_tree_draft_nodes_validate(bad_logp, 4, config) == COMMON_TREE_DRAFT_NODE_LOGP);
    bad_logp[1].logp = -std::numeric_limits<float>::infinity();
    assert(common_tree_draft_nodes_validate(bad_logp, 4, config) == COMMON_TREE_DRAFT_NODE_LOGP);

    common_tree_draft_node bad_path[4] = { nodes[0], nodes[1], nodes[2], nodes[3] };
    bad_path[1].path_id = 0;
    assert(common_tree_draft_nodes_validate(bad_path, 4, config) == COMMON_TREE_DRAFT_NODE_PATH_ID);

    common_tree_draft_node bad_flags[4] = { nodes[0], nodes[1], nodes[2], nodes[3] };
    bad_flags[2].flags = COMMON_TREE_DRAFT_NODE_FLAG_FRONTIER | COMMON_TREE_DRAFT_NODE_FLAG_PRUNED;
    assert(common_tree_draft_nodes_validate(bad_flags, 4, config) == COMMON_TREE_DRAFT_NODE_FLAGS);

    common_tree_draft_node bad_token[4] = { nodes[0], nodes[1], nodes[2], nodes[3] };
    bad_token[1].token = -1;
    assert(common_tree_draft_nodes_validate(bad_token, 4, config) == COMMON_TREE_DRAFT_NODE_TOKEN_RANGE);
}

static void test_root_contract() {
    const auto config = make_config();

    common_tree_draft_node bad_root[4] = { nodes[0], nodes[1], nodes[2], nodes[3] };
    bad_root[0].parent = 0;
    assert(common_tree_draft_nodes_validate(bad_root, 4, config) == COMMON_TREE_DRAFT_NODE_ROOT_PARENT);

    bad_root[0] = nodes[0];
    bad_root[0].depth = 1;
    assert(common_tree_draft_nodes_validate(bad_root, 4, config) == COMMON_TREE_DRAFT_NODE_ROOT_DEPTH);

    bad_root[0] = nodes[0];
    bad_root[0].logp = -0.1f;
    assert(common_tree_draft_nodes_validate(bad_root, 4, config) == COMMON_TREE_DRAFT_NODE_ROOT_LOGP);
}

int main() {
    test_abi_and_valid_nodes();
    test_parent_depth_and_budget_invariants();
    test_probability_identity_and_flags();
    test_root_contract();
    return 0;
}

