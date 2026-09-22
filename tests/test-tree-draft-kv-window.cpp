#include "tree-draft-kv-window.h"

#include <cassert>

int main() {
    const uint32_t row_ptr[] = {0, 3, 5};
    const common_tree_draft_kv_gather_segment segments[] = {
        {{10, 1}, 2, 5},
        {{11, 3}, 0, 4},
        {{12, 2}, 4, 6},
        {{20, 1}, 1, 4},
        {{21, 7}, 5, 8},
    };
    const common_tree_draft_kv_gather_metadata source = {
        const_cast<uint32_t *>(row_ptr), 3,
        const_cast<common_tree_draft_kv_gather_segment *>(segments), 5,
        2, 5,
    };

    const common_tree_draft_kv_window_query queries[] = {
        {8, 4},
        {5, -3},
    };
    uint32_t clipped_rows[3] = {};
    common_tree_draft_kv_gather_segment clipped_segments[5] = {};
    common_tree_draft_kv_gather_metadata clipped = {
        clipped_rows, 3, clipped_segments, 5, 0, 0,
    };

    assert(common_tree_draft_kv_window_clip(source, queries, 2, &clipped) == COMMON_TREE_DRAFT_KV_WINDOW_OK);
    assert(clipped.query_count == 2);
    assert(clipped.segment_count == 4);
    assert(clipped.row_ptr[0] == 0 && clipped.row_ptr[1] == 2 && clipped.row_ptr[2] == 4);

    assert(clipped.segments[0].page.id == 11 && clipped.segments[0].page.generation == 3);
    assert(clipped.segments[0].lo == 1 && clipped.segments[0].hi == 4);
    assert(clipped.segments[1].page.id == 12 && clipped.segments[1].page.generation == 2);
    assert(clipped.segments[1].lo == 4 && clipped.segments[1].hi == 6);
    assert(clipped.segments[2].page.id == 20 && clipped.segments[2].lo == 1 && clipped.segments[2].hi == 4);
    assert(clipped.segments[3].page.id == 21 && clipped.segments[3].lo == 5 && clipped.segments[3].hi == 8);

    const common_tree_draft_kv_window_query layer_wide[] = {
        {8, 0},
        {5, 2},
    };
    assert(common_tree_draft_kv_window_clip(source, layer_wide, 2, &clipped) == COMMON_TREE_DRAFT_KV_WINDOW_OK);
    assert(clipped.segment_count == 5);
    assert(clipped.segments[0].lo == 2 && clipped.segments[0].hi == 5);
    assert(clipped.segments[3].lo == 3 && clipped.segments[3].hi == 4);

    const common_tree_draft_kv_window_query bad_query[] = {
        {7, 0},
        {5, 0},
    };
    assert(common_tree_draft_kv_window_clip(source, bad_query, 2, &clipped) ==
           COMMON_TREE_DRAFT_KV_WINDOW_QUERY_RANGE);

    auto small = clipped;
    small.segment_capacity = 1;
    assert(common_tree_draft_kv_window_clip(source, queries, 2, &small) ==
           COMMON_TREE_DRAFT_KV_WINDOW_OUTPUT_TOO_SMALL);

    return 0;
}
