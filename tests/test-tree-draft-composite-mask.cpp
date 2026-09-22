#include "tree-draft-composite-mask.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

static bool visible(
        const common_tree_draft_topology & topology,
        const common_tree_draft_ancestor_bitset & ancestors,
        const common_tree_draft_forest_offsets & forest,
        const int32_t * prefix_lengths,
        int32_t entry,
        int32_t query,
        common_tree_draft_composite_key_kind kind,
        int32_t key) {
    bool result = false;
    assert(common_tree_draft_composite_mask_visible(
        topology, ancestors, forest, prefix_lengths, entry, query, kind, key, &result) ==
        COMMON_TREE_DRAFT_COMPOSITE_MASK_OK);
    return result;
}

int main() {
    const int32_t parent[]  = { -1, 0, 0, -1, -1, 4, 4 };
    const int32_t depth[]   = {  0, 1, 1,  0,  0, 1, 1 };
    const int32_t tree_id[] = { 10,10,10, 11, 20,20,20 };
    const common_tree_draft_topology topology = { parent, depth, tree_id, 7 };

    const size_t words_per_row = common_tree_draft_ancestor_words_per_row(topology.n_nodes);
    std::vector<uint32_t> words(static_cast<size_t>(topology.n_nodes) * words_per_row);
    const common_tree_draft_ancestor_bitset ancestors = { words.data(), words.size() };
    assert(common_tree_draft_ancestor_build(topology, ancestors) == COMMON_TREE_DRAFT_ANCESTOR_OK);

    const int32_t counts[] = { 4, 3 };
    int32_t offsets[3] = {};
    assert(common_tree_draft_forest_offsets_build(topology, counts, 2, offsets, 3) == COMMON_TREE_DRAFT_FOREST_OK);
    const common_tree_draft_forest_offsets forest = { offsets, 2 };
    const int32_t prefixes[] = { 3, 0 };

    // Every proposal query sees every committed prefix key in its own entry.
    for (int32_t query = 0; query < 4; ++query) {
        for (int32_t key = 0; key < prefixes[0]; ++key) {
            assert(visible(topology, ancestors, forest, prefixes, 0, query, COMMON_TREE_DRAFT_COMPOSITE_KEY_PREFIX, key));
        }
    }

    // Root, ancestor, sibling, descendant and cross-tree proposal visibility.
    assert(visible(topology, ancestors, forest, prefixes, 0, 0, COMMON_TREE_DRAFT_COMPOSITE_KEY_PROPOSAL, 0));
    assert(visible(topology, ancestors, forest, prefixes, 0, 1, COMMON_TREE_DRAFT_COMPOSITE_KEY_PROPOSAL, 0));
    assert(visible(topology, ancestors, forest, prefixes, 0, 1, COMMON_TREE_DRAFT_COMPOSITE_KEY_PROPOSAL, 1));
    assert(!visible(topology, ancestors, forest, prefixes, 0, 1, COMMON_TREE_DRAFT_COMPOSITE_KEY_PROPOSAL, 2));
    assert(!visible(topology, ancestors, forest, prefixes, 0, 0, COMMON_TREE_DRAFT_COMPOSITE_KEY_PROPOSAL, 1));
    assert(!visible(topology, ancestors, forest, prefixes, 0, 3, COMMON_TREE_DRAFT_COMPOSITE_KEY_PROPOSAL, 0));

    // Zero-length prefix has no valid prefix key, while tree ancestry remains exact.
    bool out = true;
    assert(common_tree_draft_composite_mask_visible(
        topology, ancestors, forest, prefixes, 1, 1, COMMON_TREE_DRAFT_COMPOSITE_KEY_PREFIX, 0, &out) ==
        COMMON_TREE_DRAFT_COMPOSITE_MASK_KEY_RANGE);
    assert(visible(topology, ancestors, forest, prefixes, 1, 1, COMMON_TREE_DRAFT_COMPOSITE_KEY_PROPOSAL, 0));
    assert(!visible(topology, ancestors, forest, prefixes, 1, 1, COMMON_TREE_DRAFT_COMPOSITE_KEY_PROPOSAL, 2));

    assert(common_tree_draft_composite_mask_value(true) == 0.0f);
    assert(std::isinf(common_tree_draft_composite_mask_value(false)) && common_tree_draft_composite_mask_value(false) < 0.0f);

    assert(common_tree_draft_composite_mask_visible(
        topology, ancestors, forest, prefixes, 2, 0, COMMON_TREE_DRAFT_COMPOSITE_KEY_PROPOSAL, 0, &out) ==
        COMMON_TREE_DRAFT_COMPOSITE_MASK_ENTRY_RANGE);
    assert(common_tree_draft_composite_mask_visible(
        topology, ancestors, forest, prefixes, 0, 4, COMMON_TREE_DRAFT_COMPOSITE_KEY_PROPOSAL, 0, &out) ==
        COMMON_TREE_DRAFT_COMPOSITE_MASK_QUERY_RANGE);
    assert(common_tree_draft_composite_mask_visible(
        topology, ancestors, forest, prefixes, 0, 0, COMMON_TREE_DRAFT_COMPOSITE_KEY_PROPOSAL, 4, &out) ==
        COMMON_TREE_DRAFT_COMPOSITE_MASK_KEY_RANGE);
    assert(common_tree_draft_composite_mask_visible(
        topology, ancestors, forest, prefixes, 0, 0, static_cast<common_tree_draft_composite_key_kind>(99), 0, &out) ==
        COMMON_TREE_DRAFT_COMPOSITE_MASK_INVALID_KEY_KIND);
    assert(common_tree_draft_composite_mask_visible(
        topology, ancestors, forest, prefixes, 0, 0, COMMON_TREE_DRAFT_COMPOSITE_KEY_PREFIX, 0, nullptr) ==
        COMMON_TREE_DRAFT_COMPOSITE_MASK_NULL_OUTPUT);
    assert(common_tree_draft_composite_mask_visible(
        topology, { nullptr, 0 }, forest, prefixes, 0, 0, COMMON_TREE_DRAFT_COMPOSITE_KEY_PREFIX, 0, &out) ==
        COMMON_TREE_DRAFT_COMPOSITE_MASK_NULL_ANCESTOR);
    const int32_t negative_prefixes[] = { -1, 0 };
    assert(common_tree_draft_composite_mask_visible(
        topology, ancestors, forest, negative_prefixes, 0, 0, COMMON_TREE_DRAFT_COMPOSITE_KEY_PROPOSAL, 0, &out) ==
        COMMON_TREE_DRAFT_COMPOSITE_MASK_INVALID_PREFIX_LENGTH);
    assert(std::string(common_tree_draft_composite_mask_error_name(COMMON_TREE_DRAFT_COMPOSITE_MASK_KEY_RANGE)) == "key_range");

    return 0;
}
