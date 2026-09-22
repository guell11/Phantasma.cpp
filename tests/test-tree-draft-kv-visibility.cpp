#include "tree-draft-kv-visibility.h"

#include <cassert>
#include <vector>

static common_tree_draft_kv_branch_descriptor branch(
        uint32_t id,
        uint32_t generation,
        uint32_t parent_id,
        uint32_t parent_generation,
        int64_t fork_position,
        int64_t tip_position) {
    return { { id, generation }, { parent_id, parent_generation }, fork_position, tip_position, 0, nullptr, 0 };
}

int main() {
    const common_tree_draft_kv_branch_descriptor branches[] = {
        branch(10, 1, COMMON_TREE_DRAFT_KV_ID_INVALID, 0, -1, 7),
        branch(11, 1, 10, 1, 3, 6),
        branch(12, 4, 11, 1, 5, 8),
        branch(13, 2, 10, 1, 2, 5),
    };

    bool overflow = true;
    const size_t cutoff_count = common_tree_draft_kv_visibility_cutoff_count(4, &overflow);
    assert(!overflow && cutoff_count == 16);
    std::vector<int64_t> cutoffs(cutoff_count, 99);
    common_tree_draft_kv_visibility_table table = { cutoffs.data(), cutoffs.size(), 0 };
    assert(common_tree_draft_kv_visibility_build(branches, 4, table) == COMMON_TREE_DRAFT_KV_VISIBILITY_OK);
    assert(table.branch_count == 4);

    assert(common_tree_draft_kv_visible(table, 2, 8, 0, 3));
    assert(!common_tree_draft_kv_visible(table, 2, 8, 0, 4));
    assert(common_tree_draft_kv_visible(table, 2, 8, 1, 5));
    assert(!common_tree_draft_kv_visible(table, 2, 8, 1, 6));
    assert(common_tree_draft_kv_visible(table, 2, 7, 2, 7));
    assert(!common_tree_draft_kv_visible(table, 2, 7, 2, 8));
    assert(!common_tree_draft_kv_visible(table, 3, 5, 1, 2));
    assert(!common_tree_draft_kv_visible(table, 1, 6, 2, 4));

    {
        auto bad = branches[1];
        bad.parent_branch.generation = 9;
        const common_tree_draft_kv_branch_descriptor input[] = { branches[0], bad };
        assert(common_tree_draft_kv_visibility_build(input, 2, table) == COMMON_TREE_DRAFT_KV_VISIBILITY_PARENT_ORDER);
    }
    {
        auto bad = branches[1];
        bad.fork_position = 8;
        const common_tree_draft_kv_branch_descriptor input[] = { branches[0], bad };
        assert(common_tree_draft_kv_visibility_build(input, 2, table) == COMMON_TREE_DRAFT_KV_VISIBILITY_FORK_RANGE);
    }
    {
        auto duplicate = branches[0];
        const common_tree_draft_kv_branch_descriptor input[] = { branches[0], duplicate };
        assert(common_tree_draft_kv_visibility_build(input, 2, table) == COMMON_TREE_DRAFT_KV_VISIBILITY_DUPLICATE_BRANCH);
    }
    common_tree_draft_kv_visibility_table null_table = { nullptr, cutoff_count, 0 };
    assert(common_tree_draft_kv_visibility_build(branches, 4, null_table) == COMMON_TREE_DRAFT_KV_VISIBILITY_NULL_BUFFER);
    common_tree_draft_kv_visibility_table short_table = { cutoffs.data(), cutoff_count - 1, 0 };
    assert(common_tree_draft_kv_visibility_build(branches, 4, short_table) ==
           COMMON_TREE_DRAFT_KV_VISIBILITY_OUTPUT_TOO_SMALL);

    return 0;
}
