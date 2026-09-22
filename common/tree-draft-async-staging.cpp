#include "tree-draft-async-staging.h"

#include <algorithm>
#include <cmath>
#include <limits>

common_tree_draft_async_staging_status common_tree_draft_async_staging_plan_build(
        const common_tree_draft_device_capabilities & device,
        const common_tree_draft_smem_layout & layout,
        const common_tree_draft_async_staging_request & request,
        common_tree_draft_async_staging_plan * plan) {
    if (plan == nullptr || request.tile_bytes == 0 ||
        !std::isfinite(request.transfer_time_us) || request.transfer_time_us < 0.0f ||
        !std::isfinite(request.compute_time_us) || request.compute_time_us < 0.0f ||
        !std::isfinite(request.min_coalescing_efficiency) ||
        request.min_coalescing_efficiency < 0.0f || request.min_coalescing_efficiency > 1.0f) {
        return COMMON_TREE_DRAFT_ASYNC_STAGING_INVALID_REQUEST;
    }
    if (request.staging_offset > layout.dynamic_shared_bytes ||
        request.staging_bytes > layout.dynamic_shared_bytes - request.staging_offset) {
        return COMMON_TREE_DRAFT_ASYNC_STAGING_LAYOUT_RANGE;
    }

    common_tree_draft_async_staging_plan out;
    out.tile_bytes = request.tile_bytes;
    out.required_alignment = 16;

    const bool enough_for_double =
        request.staging_bytes >= static_cast<uint64_t>(request.tile_bytes) * 2u;
    const bool aligned =
        request.global_address % out.required_alignment == 0 &&
        request.staging_offset % out.required_alignment == 0 &&
        request.tile_bytes % out.required_alignment == 0;
    const bool coalesced =
        request.coalescing.sectors_touched > 0 &&
        request.coalescing.efficiency >= request.min_coalescing_efficiency;
    const bool sm89 = common_tree_draft_device_is_sm89(device);
    const bool pipeline_requested = request.requested_pipeline_depth >= 2;

    out.async_enabled = sm89 && aligned && coalesced && enough_for_double && pipeline_requested;
    if (out.async_enabled) {
        const uint8_t max_depth = static_cast<uint8_t>(
            std::min<uint64_t>(request.requested_pipeline_depth,
                request.staging_bytes / request.tile_bytes));
        out.pipeline_depth = std::max<uint8_t>(2, max_depth);
        out.stage_stride_bytes = request.tile_bytes;
        out.commit_group_tiles = 1;
        out.wait_group_distance = static_cast<uint8_t>(out.pipeline_depth - 1);
        out.estimated_tile_time_us = std::max(request.transfer_time_us, request.compute_time_us);
    } else {
        out.pipeline_depth = 1;
        out.stage_stride_bytes = request.tile_bytes;
        out.commit_group_tiles = 1;
        out.wait_group_distance = 0;
        out.estimated_tile_time_us = request.transfer_time_us + request.compute_time_us;
    }
    *plan = out;
    return COMMON_TREE_DRAFT_ASYNC_STAGING_OK;
}
