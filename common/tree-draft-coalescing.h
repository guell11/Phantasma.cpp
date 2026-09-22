#pragma once

#include <cstddef>
#include <cstdint>

struct common_tree_draft_lane_access {
    uint64_t address = 0;
    uint32_t useful_bytes = 0;
};

struct common_tree_draft_coalescing_result {
    uint32_t sectors_touched = 0;
    uint64_t useful_bytes = 0;
    float efficiency = 0.0f;
    bool adjacent_lanes_contiguous = false;
};

enum common_tree_draft_coalescing_status : uint32_t {
    COMMON_TREE_DRAFT_COALESCING_OK = 0,
    COMMON_TREE_DRAFT_COALESCING_NULL_INPUT,
    COMMON_TREE_DRAFT_COALESCING_EMPTY,
    COMMON_TREE_DRAFT_COALESCING_OVERFLOW,
};

common_tree_draft_coalescing_status common_tree_draft_coalescing_measure(
        const common_tree_draft_lane_access * lanes,
        size_t lane_count,
        common_tree_draft_coalescing_result * result);
