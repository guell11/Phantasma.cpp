#include "tree-draft-dense-mask.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

static bool is_negative_infinity(float value) {
    return std::isinf(value) && value < 0.0f;
}

int main() {
    {
        const common_tree_draft_topology empty = { nullptr, nullptr, nullptr, 0 };
        assert(common_tree_draft_dense_mask_build(empty, { nullptr, 0 }, { nullptr, 0 }) ==
            COMMON_TREE_DRAFT_DENSE_MASK_OK);
    }

    const int32_t parent[]  = { -1, 0, 0, 1, -1, 4 };
    const int32_t depth[]   = {  0, 1, 1, 2,  0, 1 };
    const int32_t tree_id[] = {  7, 7, 7, 7, 42, 42 };
    const common_tree_draft_topology topology = { parent, depth, tree_id, 6 };

    const size_t words_per_row = common_tree_draft_ancestor_words_per_row(topology.n_nodes);
    std::vector<uint32_t> ancestor_words(static_cast<size_t>(topology.n_nodes) * words_per_row);
    const common_tree_draft_ancestor_bitset ancestors = { ancestor_words.data(), ancestor_words.size() };
    assert(common_tree_draft_ancestor_build(topology, ancestors) == COMMON_TREE_DRAFT_ANCESTOR_OK);

    const size_t required_values = static_cast<size_t>(topology.n_nodes) * topology.n_nodes;
    std::vector<float> values(required_values + 2, 123.0f);
    assert(common_tree_draft_dense_mask_build(topology, ancestors, { values.data(), values.size() }) ==
        COMMON_TREE_DRAFT_DENSE_MASK_OK);

    const auto at = [&](int32_t query, int32_t key) {
        return values[static_cast<size_t>(query) * topology.n_nodes + static_cast<size_t>(key)];
    };

    const bool expected_visible[6][6] = {
        { true,  false, false, false, false, false },
        { true,  true,  false, false, false, false },
        { true,  false, true,  false, false, false },
        { true,  true,  false, true,  false, false },
        { false, false, false, false, true,  false },
        { false, false, false, false, true,  true  },
    };
    for (int32_t i = 0; i < topology.n_nodes; ++i) {
        for (int32_t j = 0; j < topology.n_nodes; ++j) {
            if (expected_visible[i][j]) {
                assert(at(i, j) == 0.0f);
            } else {
                assert(is_negative_infinity(at(i, j)));
            }
        }
    }
    assert(values[required_values] == 123.0f);
    assert(values[required_values + 1] == 123.0f);

    {
        auto corrupted_words = ancestor_words;
        corrupted_words[5 * words_per_row] |= uint32_t(1) << 0;
        const common_tree_draft_ancestor_bitset corrupted = { corrupted_words.data(), corrupted_words.size() };
        std::vector<float> corrupted_values(required_values, 1.0f);
        assert(common_tree_draft_dense_mask_build(topology, corrupted, { corrupted_values.data(), corrupted_values.size() }) ==
            COMMON_TREE_DRAFT_DENSE_MASK_OK);
        assert(is_negative_infinity(corrupted_values[5 * topology.n_nodes]));
    }

    assert(common_tree_draft_dense_mask_build(topology, { nullptr, 0 }, { values.data(), values.size() }) ==
        COMMON_TREE_DRAFT_DENSE_MASK_NULL_ANCESTOR);
    assert(common_tree_draft_dense_mask_build(topology, { ancestor_words.data(), ancestor_words.size() - 1 }, { values.data(), values.size() }) ==
        COMMON_TREE_DRAFT_DENSE_MASK_ANCESTOR_TOO_SMALL);
    assert(common_tree_draft_dense_mask_build(topology, ancestors, { nullptr, 0 }) ==
        COMMON_TREE_DRAFT_DENSE_MASK_NULL_OUTPUT);
    {
        std::vector<float> too_small(required_values - 1, 77.0f);
        assert(common_tree_draft_dense_mask_build(topology, ancestors, { too_small.data(), too_small.size() }) ==
            COMMON_TREE_DRAFT_DENSE_MASK_OUTPUT_TOO_SMALL);
        for (float value : too_small) {
            assert(value == 77.0f);
        }
    }

    {
        const int32_t bad_parent[] = { -1, 1 };
        const int32_t bad_depth[] = { 0, 1 };
        const int32_t bad_tree_id[] = { 0, 0 };
        const common_tree_draft_topology invalid = { bad_parent, bad_depth, bad_tree_id, 2 };
        assert(common_tree_draft_dense_mask_build(invalid, ancestors, { values.data(), values.size() }) ==
            COMMON_TREE_DRAFT_DENSE_MASK_INVALID_TOPOLOGY);
    }

    assert(std::string(common_tree_draft_dense_mask_error_name(COMMON_TREE_DRAFT_DENSE_MASK_OUTPUT_TOO_SMALL)) ==
        "output_too_small");

    return 0;
}
