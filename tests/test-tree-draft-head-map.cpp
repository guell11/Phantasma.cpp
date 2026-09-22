#include "tree-draft-head-map.h"

#include <cassert>

int main() {
    common_tree_draft_head_map_plan plan;

    assert(common_tree_draft_head_map_build(32, 8, {1,2,4,8}, &plan) == COMMON_TREE_DRAFT_HEAD_MAP_OK);
    assert(plan.mode == COMMON_TREE_DRAFT_HEAD_MAP_GQA);
    assert(plan.query_heads_per_kv == 4 && plan.native_ratio_supported && !plan.requires_fallback);
    assert(plan.groups.size() == 8);
    assert(plan.groups[3].query_head_begin == 12 && plan.groups[3].query_head_end == 16);
    assert(common_tree_draft_head_map_kv_head(plan, 0) == 0);
    assert(common_tree_draft_head_map_kv_head(plan, 15) == 3);
    assert(common_tree_draft_head_map_kv_head(plan, 31) == 7);
    assert(common_tree_draft_head_map_kv_head(plan, 32) == UINT32_MAX);

    assert(common_tree_draft_head_map_build(8, 8, {1}, &plan) == COMMON_TREE_DRAFT_HEAD_MAP_OK);
    assert(plan.mode == COMMON_TREE_DRAFT_HEAD_MAP_MHA && plan.query_heads_per_kv == 1);

    assert(common_tree_draft_head_map_build(32, 1, {1,2,4,8}, &plan) == COMMON_TREE_DRAFT_HEAD_MAP_OK);
    assert(plan.mode == COMMON_TREE_DRAFT_HEAD_MAP_MQA && plan.query_heads_per_kv == 32);
    assert(!plan.native_ratio_supported && plan.requires_fallback);
    assert(common_tree_draft_head_map_kv_head(plan, 31) == 0);

    const auto before = plan;
    assert(common_tree_draft_head_map_build(30, 8, {1,2,4}, &plan) == COMMON_TREE_DRAFT_HEAD_MAP_NON_DIVISIBLE);
    assert(plan.query_heads == before.query_heads && plan.kv_heads == before.kv_heads);
    assert(common_tree_draft_head_map_build(0, 8, {1}, &plan) == COMMON_TREE_DRAFT_HEAD_MAP_ZERO_HEADS);
    assert(common_tree_draft_head_map_build(8, 0, {1}, &plan) == COMMON_TREE_DRAFT_HEAD_MAP_ZERO_HEADS);
    return 0;
}
