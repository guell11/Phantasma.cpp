#pragma once

#include "tree-draft-sm89.h"

#include <array>
#include <cstddef>
#include <cstdint>

struct common_tree_draft_smem_carveout_candidate {
    uint8_t carveout_percent = 0;
    uint32_t max_shared_bytes = 0;
    uint32_t implied_l1_bytes = 0;
    bool supported = false;
};

struct common_tree_draft_smem_carveout_request {
    uint32_t required_shared_bytes = 0;
    std::array<common_tree_draft_smem_carveout_candidate, 5> candidates = {};
};

struct common_tree_draft_smem_carveout_plan {
    bool available = false;
    uint8_t selected_index = 0;
    uint8_t candidate_count = 0;
    std::array<uint8_t, 5> valid_indices = {};
    bool has_lower_neighbor = false;
    bool has_upper_neighbor = false;
    uint8_t lower_neighbor_index = 0;
    uint8_t upper_neighbor_index = 0;
};

enum common_tree_draft_smem_carveout_status : uint32_t {
    COMMON_TREE_DRAFT_SMEM_CARVEOUT_OK = 0,
    COMMON_TREE_DRAFT_SMEM_CARVEOUT_NOT_SM89,
    COMMON_TREE_DRAFT_SMEM_CARVEOUT_INVALID_TABLE,
    COMMON_TREE_DRAFT_SMEM_CARVEOUT_UNSATISFIED,
    COMMON_TREE_DRAFT_SMEM_CARVEOUT_NULL_OUTPUT,
};

common_tree_draft_smem_carveout_status common_tree_draft_smem_carveout_plan_build(
        const common_tree_draft_device_capabilities & device,
        const common_tree_draft_smem_carveout_request & request,
        common_tree_draft_smem_carveout_plan * plan);
