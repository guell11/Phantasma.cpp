#pragma once

#include <cstdint>
#include <vector>

enum common_tree_draft_head_map_mode : uint32_t {
    COMMON_TREE_DRAFT_HEAD_MAP_MHA = 0,
    COMMON_TREE_DRAFT_HEAD_MAP_GQA,
    COMMON_TREE_DRAFT_HEAD_MAP_MQA,
};

struct common_tree_draft_head_group {
    uint32_t kv_head = 0;
    uint32_t query_head_begin = 0;
    uint32_t query_head_end = 0;
};

struct common_tree_draft_head_map_plan {
    uint32_t query_heads = 0;
    uint32_t kv_heads = 0;
    uint32_t query_heads_per_kv = 0;
    common_tree_draft_head_map_mode mode = COMMON_TREE_DRAFT_HEAD_MAP_MHA;
    bool native_ratio_supported = false;
    bool requires_fallback = false;
    std::vector<common_tree_draft_head_group> groups;
};

enum common_tree_draft_head_map_status : uint32_t {
    COMMON_TREE_DRAFT_HEAD_MAP_OK = 0,
    COMMON_TREE_DRAFT_HEAD_MAP_ZERO_HEADS,
    COMMON_TREE_DRAFT_HEAD_MAP_NON_DIVISIBLE,
};

// supported_ratios is backend capability input, not a semantic restriction.
// A divisible mapping remains valid even when the optimized backend does not
// advertise its ratio; that case is marked requires_fallback=true.
common_tree_draft_head_map_status common_tree_draft_head_map_build(
        uint32_t query_heads,
        uint32_t kv_heads,
        const std::vector<uint32_t> & supported_ratios,
        common_tree_draft_head_map_plan * plan);

uint32_t common_tree_draft_head_map_kv_head(
        const common_tree_draft_head_map_plan & plan,
        uint32_t query_head);
