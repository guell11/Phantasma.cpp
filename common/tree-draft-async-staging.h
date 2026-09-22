#pragma once

#include "tree-draft-coalescing.h"
#include "tree-draft-sm89.h"
#include "tree-draft-smem-layout.h"

#include <cstdint>

struct common_tree_draft_async_staging_request {
    uint64_t global_address = 0;
    uint32_t tile_bytes = 0;
    uint32_t staging_offset = 0;
    uint32_t staging_bytes = 0;
    uint8_t requested_pipeline_depth = 2;
    float transfer_time_us = 0.0f;
    float compute_time_us = 0.0f;
    float min_coalescing_efficiency = 0.5f;
    common_tree_draft_coalescing_result coalescing = {};
};

struct common_tree_draft_async_staging_plan {
    bool async_enabled = false;
    uint8_t pipeline_depth = 1;
    uint8_t commit_group_tiles = 1;
    uint8_t wait_group_distance = 0;
    uint8_t required_alignment = 16;
    uint32_t tile_bytes = 0;
    uint32_t stage_stride_bytes = 0;
    float estimated_tile_time_us = 0.0f;
};

enum common_tree_draft_async_staging_status : uint32_t {
    COMMON_TREE_DRAFT_ASYNC_STAGING_OK = 0,
    COMMON_TREE_DRAFT_ASYNC_STAGING_INVALID_REQUEST,
    COMMON_TREE_DRAFT_ASYNC_STAGING_LAYOUT_RANGE,
};

common_tree_draft_async_staging_status common_tree_draft_async_staging_plan_build(
        const common_tree_draft_device_capabilities & device,
        const common_tree_draft_smem_layout & layout,
        const common_tree_draft_async_staging_request & request,
        common_tree_draft_async_staging_plan * plan);
