#include "tree-draft-tile-visibility.h"

#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

int main() {
    const int32_t parent[]  = { -1, 0, 0, -1, 3, 3, 4 };
    const int32_t depth[]   = {  0, 1, 1,  0, 1, 1, 2 };
    const int32_t tree_id[] = {  1, 1, 1,  9, 9, 9, 9 };
    const common_tree_draft_topology topology = { parent, depth, tree_id, 7 };

    const size_t words_per_row = common_tree_draft_ancestor_words_per_row(topology.n_nodes);
    std::vector<uint32_t> words(static_cast<size_t>(topology.n_nodes) * words_per_row, 0);
    common_tree_draft_ancestor_bitset ancestors = { words.data(), words.size() };
    assert(common_tree_draft_ancestor_build(topology, ancestors) == COMMON_TREE_DRAFT_ANCESTOR_OK);

    const int32_t counts[] = { 3, 0, 4 };
    int32_t offsets[4] = {};
    assert(common_tree_draft_forest_offsets_build(topology, counts, 3, offsets, 4) == COMMON_TREE_DRAFT_FOREST_OK);
    const common_tree_draft_forest_offsets forest = { offsets, 3 };

    uint8_t active[16] = {};
    common_tree_draft_tile_visibility tiles = { active, 16, 0, 0, 0, 0 };
    assert(common_tree_draft_tile_visibility_build(forest, ancestors, 2, 2, tiles) ==
        COMMON_TREE_DRAFT_TILE_VISIBILITY_OK);
    assert(tiles.n_query_tiles == 4);
    assert(tiles.n_key_tiles == 4);

    const uint8_t expected[16] = {
        1, 0, 0, 0,
        1, 1, 0, 0,
        0, 1, 1, 0,
        0, 1, 1, 1,
    };
    for (size_t i = 0; i < 16; ++i) {
        assert(active[i] == expected[i]);
    }

    assert(common_tree_draft_element_visible(forest, ancestors, 1, 0));
    assert(!common_tree_draft_element_visible(forest, ancestors, 1, 2));
    assert(!common_tree_draft_element_visible(forest, ancestors, 3, 2));
    assert(common_tree_draft_element_visible(forest, ancestors, 6, 3));
    assert(common_tree_draft_element_visible(forest, ancestors, 6, 4));
    assert(!common_tree_draft_element_visible(forest, ancestors, 6, 5));

    // Exact element masking remains authoritative inside an active tile. Tile (q=0,k=0)
    // is active, but sibling pair (query 1, key 2) is still invisible.
    assert(active[0] == 1);
    assert(!common_tree_draft_element_visible(forest, ancestors, 1, 2));

    // Conservative metadata may be widened to a false positive without changing element visibility.
    active[1] = 1;
    assert(active[1] == 1);
    assert(!common_tree_draft_element_visible(forest, ancestors, 0, 2));

    {
        uint8_t tail_active[6] = {};
        common_tree_draft_tile_visibility tail_tiles = { tail_active, 6, 0, 0, 0, 0 };
        assert(common_tree_draft_tile_visibility_build(forest, ancestors, 3, 4, tail_tiles) ==
            COMMON_TREE_DRAFT_TILE_VISIBILITY_OK);
        assert(tail_tiles.n_query_tiles == 3);
        assert(tail_tiles.n_key_tiles == 2);
        // Last query tile contains only node 6; last key tile contains nodes 4..6.
        assert(tail_active[2 * 2 + 1] == 1);
    }

    {
        const int32_t bad_offsets[] = { 0, 4, 3, 7 };
        assert(common_tree_draft_tile_visibility_build({ bad_offsets, 3 }, ancestors, 2, 2, tiles) ==
            COMMON_TREE_DRAFT_TILE_VISIBILITY_INVALID_FOREST);
    }
    assert(common_tree_draft_tile_visibility_build(forest, ancestors, 0, 2, tiles) ==
        COMMON_TREE_DRAFT_TILE_VISIBILITY_INVALID_TILE_SIZE);
    assert(common_tree_draft_tile_visibility_build(forest, { nullptr, words.size() }, 2, 2, tiles) ==
        COMMON_TREE_DRAFT_TILE_VISIBILITY_NULL_ANCESTOR);
    assert(common_tree_draft_tile_visibility_build(forest, { words.data(), words.size() - 1 }, 2, 2, tiles) ==
        COMMON_TREE_DRAFT_TILE_VISIBILITY_ANCESTOR_TOO_SMALL);
    {
        uint8_t too_small[15] = {};
        common_tree_draft_tile_visibility short_tiles = { too_small, 15, 0, 0, 0, 0 };
        assert(common_tree_draft_tile_visibility_build(forest, ancestors, 2, 2, short_tiles) ==
            COMMON_TREE_DRAFT_TILE_VISIBILITY_OUTPUT_TOO_SMALL);
    }
    {
        const common_tree_draft_forest_offsets empty_forest = { offsets, 0 };
        const common_tree_draft_ancestor_bitset empty_ancestors = { nullptr, 0 };
        common_tree_draft_tile_visibility empty_tiles = { nullptr, 0, 0, 0, -1, -1 };
        assert(common_tree_draft_tile_visibility_build(empty_forest, empty_ancestors, 2, 2, empty_tiles) ==
            COMMON_TREE_DRAFT_TILE_VISIBILITY_OK);
        assert(empty_tiles.n_query_tiles == 0 && empty_tiles.n_key_tiles == 0);
    }

    assert(std::string(common_tree_draft_tile_visibility_error_name(
        COMMON_TREE_DRAFT_TILE_VISIBILITY_OUTPUT_TOO_SMALL)) == "output_too_small");
    return 0;
}
