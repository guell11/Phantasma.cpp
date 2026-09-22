#include "tree-draft-ancestor.h"

#include <cassert>
#include <cstdint>
#include <vector>

static bool reference_contains(const int32_t * parent, int32_t node, int32_t candidate) {
    for (int32_t current = node; current >= 0; current = parent[current]) {
        if (current == candidate) {
            return true;
        }
    }
    return false;
}

static void check_against_reference(
        const int32_t * parent,
        const int32_t * depth,
        const int32_t * tree_id,
        int32_t n_nodes) {
    const common_tree_draft_topology topology = { parent, depth, tree_id, n_nodes };
    const size_t words_per_row = common_tree_draft_ancestor_words_per_row(n_nodes);
    std::vector<uint32_t> words(static_cast<size_t>(n_nodes) * words_per_row, 0xffffffffu);
    common_tree_draft_ancestor_bitset bitset = { words.data(), words.size() };

    assert(common_tree_draft_ancestor_build(topology, bitset) == COMMON_TREE_DRAFT_ANCESTOR_OK);
    for (int32_t i = 0; i < n_nodes; ++i) {
        for (int32_t j = 0; j < n_nodes; ++j) {
            assert(common_tree_draft_ancestor_contains(bitset, n_nodes, i, j) == reference_contains(parent, i, j));
        }
    }
}

int main() {
    {
        const common_tree_draft_topology empty = { nullptr, nullptr, nullptr, 0 };
        const common_tree_draft_ancestor_bitset no_output = { nullptr, 0 };
        assert(common_tree_draft_ancestor_words_per_row(0) == 0);
        assert(common_tree_draft_ancestor_build(empty, no_output) == COMMON_TREE_DRAFT_ANCESTOR_OK);
    }

    {
        const int32_t parent[]  = { -1, 0, 0, 1, -1, 4 };
        const int32_t depth[]   = {  0, 1, 1, 2,  0, 1 };
        const int32_t tree_id[] = {  7, 7, 7, 7, 42, 42 };
        check_against_reference(parent, depth, tree_id, 6);
    }

    {
        const int32_t n_nodes = 65;
        std::vector<int32_t> parent(n_nodes);
        std::vector<int32_t> depth(n_nodes);
        std::vector<int32_t> tree_id(n_nodes, 3);
        parent[0] = -1;
        depth[0] = 0;
        for (int32_t i = 1; i < n_nodes; ++i) {
            parent[i] = i - 1;
            depth[i] = i;
        }
        assert(common_tree_draft_ancestor_words_per_row(n_nodes) == 3);
        check_against_reference(parent.data(), depth.data(), tree_id.data(), n_nodes);
    }

    {
        const int32_t parent[]  = { -1, 0 };
        const int32_t depth[]   = {  0, 1 };
        const int32_t tree_id[] = {  0, 0 };
        const common_tree_draft_topology topology = { parent, depth, tree_id, 2 };
        uint32_t word = 0;
        assert(common_tree_draft_ancestor_build(topology, { nullptr, 0 }) == COMMON_TREE_DRAFT_ANCESTOR_NULL_OUTPUT);
        assert(common_tree_draft_ancestor_build(topology, { &word, 1 }) == COMMON_TREE_DRAFT_ANCESTOR_OUTPUT_TOO_SMALL);
    }

    {
        const int32_t bad_parent[] = { -1, 1 };
        const int32_t depth[]      = {  0, 1 };
        const int32_t tree_id[]    = {  0, 0 };
        uint32_t words[2] = {};
        const common_tree_draft_topology topology = { bad_parent, depth, tree_id, 2 };
        assert(common_tree_draft_ancestor_build(topology, { words, 2 }) == COMMON_TREE_DRAFT_ANCESTOR_INVALID_TOPOLOGY);
    }

    return 0;
}
