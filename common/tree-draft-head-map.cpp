#include "tree-draft-head-map.h"

#include <algorithm>
#include <limits>

common_tree_draft_head_map_status common_tree_draft_head_map_build(
        uint32_t query_heads,
        uint32_t kv_heads,
        const std::vector<uint32_t> & supported_ratios,
        common_tree_draft_head_map_plan * plan) {
    if (plan == nullptr || query_heads == 0 || kv_heads == 0) {
        return COMMON_TREE_DRAFT_HEAD_MAP_ZERO_HEADS;
    }
    if (query_heads % kv_heads != 0) {
        return COMMON_TREE_DRAFT_HEAD_MAP_NON_DIVISIBLE;
    }

    common_tree_draft_head_map_plan out;
    out.query_heads = query_heads;
    out.kv_heads = kv_heads;
    out.query_heads_per_kv = query_heads / kv_heads;
    out.mode = kv_heads == query_heads ? COMMON_TREE_DRAFT_HEAD_MAP_MHA :
               kv_heads == 1 ? COMMON_TREE_DRAFT_HEAD_MAP_MQA : COMMON_TREE_DRAFT_HEAD_MAP_GQA;
    out.native_ratio_supported = std::find(supported_ratios.begin(), supported_ratios.end(), out.query_heads_per_kv) != supported_ratios.end();
    out.requires_fallback = !out.native_ratio_supported;
    out.groups.reserve(kv_heads);
    for (uint32_t kv = 0; kv < kv_heads; ++kv) {
        const uint64_t begin64 = static_cast<uint64_t>(kv) * out.query_heads_per_kv;
        const uint64_t end64 = begin64 + out.query_heads_per_kv;
        if (end64 > query_heads || end64 > std::numeric_limits<uint32_t>::max()) {
            return COMMON_TREE_DRAFT_HEAD_MAP_NON_DIVISIBLE;
        }
        out.groups.push_back({kv, static_cast<uint32_t>(begin64), static_cast<uint32_t>(end64)});
    }
    *plan = std::move(out);
    return COMMON_TREE_DRAFT_HEAD_MAP_OK;
}

uint32_t common_tree_draft_head_map_kv_head(
        const common_tree_draft_head_map_plan & plan,
        uint32_t query_head) {
    if (plan.query_heads_per_kv == 0 || query_head >= plan.query_heads) return UINT32_MAX;
    const uint32_t kv = query_head / plan.query_heads_per_kv;
    return kv < plan.kv_heads ? kv : UINT32_MAX;
}
