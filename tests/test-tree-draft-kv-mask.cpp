#include "tree-draft-kv-mask.h"

#include <cassert>
#include <cmath>

static bool visible(
        const common_tree_draft_kv_gather_metadata & gather,
        uint32_t query,
        uint32_t segment,
        uint32_t offset,
        int64_t expected_position) {
    bool result = false;
    int64_t position = -1;
    assert(common_tree_draft_kv_mask_visible(gather, query, segment, offset, &result, &position) ==
           COMMON_TREE_DRAFT_KV_MASK_OK);
    if (result) assert(position == expected_position);
    return result;
}

int main() {
    const uint32_t row_ptr[] = {0, 2, 4};
    const common_tree_draft_kv_gather_segment segments[] = {
        {{7, 2}, 0, 2},
        {{8, 1}, 1, 2},
        {{7, 2}, 0, 1},
        {{9, 5}, 1, 3},
    };
    common_tree_draft_kv_gather_metadata gather = {
        const_cast<uint32_t *>(row_ptr), 3,
        const_cast<common_tree_draft_kv_gather_segment *>(segments), 4,
        2, 4,
    };

    assert(visible(gather, 0, 0, 0, 0));
    assert(visible(gather, 0, 0, 1, 1));
    assert(visible(gather, 0, 1, 1, 2));
    assert(!visible(gather, 0, 2, 0, -1));
    assert(!visible(gather, 1, 0, 0, -1));
    assert(visible(gather, 1, 2, 0, 0));
    assert(visible(gather, 1, 3, 2, 2));

    bool result = true;
    int64_t position = 99;
    assert(common_tree_draft_kv_mask_visible(gather, 0, 1, 0, &result, &position) ==
           COMMON_TREE_DRAFT_KV_MASK_KEY_RANGE);
    assert(!result && position == -1);
    assert(common_tree_draft_kv_mask_visible(gather, 2, 0, 0, &result, &position) ==
           COMMON_TREE_DRAFT_KV_MASK_QUERY_RANGE);

    uint32_t bad_rows[] = {1, 2, 4};
    auto malformed = gather;
    malformed.row_ptr = bad_rows;
    assert(common_tree_draft_kv_mask_visible(malformed, 0, 0, 0, &result, &position) ==
           COMMON_TREE_DRAFT_KV_MASK_METADATA);

    assert(common_tree_draft_kv_mask_value(true) == 0.0f);
    const float masked = common_tree_draft_kv_mask_value(false);
    assert(std::isinf(masked) && masked < 0.0f);
    return 0;
}
