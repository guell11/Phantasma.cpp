#include "tree-draft-async-staging.h"

#include <cassert>
#include <cmath>

static common_tree_draft_device_capabilities make_sm89() {
    common_tree_draft_device_capabilities d{};
    d.available=true; d.cc_major=8; d.cc_minor=9;
    return d;
}

int main() {
    common_tree_draft_smem_layout layout{};
    layout.dynamic_shared_bytes = 4096;
    layout.total_shared_bytes = 4096;

    common_tree_draft_async_staging_request r{};
    r.global_address = 0x1000;
    r.tile_bytes = 1024;
    r.staging_offset = 0;
    r.staging_bytes = 2048;
    r.requested_pipeline_depth = 2;
    r.transfer_time_us = 5.0f;
    r.compute_time_us = 7.0f;
    r.min_coalescing_efficiency = 0.5f;
    r.coalescing = {4,128,1.0f,true};

    common_tree_draft_async_staging_plan p{};
    assert(common_tree_draft_async_staging_plan_build(make_sm89(),layout,r,&p) ==
           COMMON_TREE_DRAFT_ASYNC_STAGING_OK);
    assert(p.async_enabled && p.pipeline_depth==2);
    assert(p.wait_group_distance==1);
    assert(std::fabs(p.estimated_tile_time_us-7.0f)<1e-6f);

    r.global_address += 4;
    assert(common_tree_draft_async_staging_plan_build(make_sm89(),layout,r,&p) ==
           COMMON_TREE_DRAFT_ASYNC_STAGING_OK);
    assert(!p.async_enabled && p.pipeline_depth==1);
    assert(std::fabs(p.estimated_tile_time_us-12.0f)<1e-6f);

    r.global_address = 0x1000;
    r.staging_bytes = 1024;
    assert(common_tree_draft_async_staging_plan_build(make_sm89(),layout,r,&p) ==
           COMMON_TREE_DRAFT_ASYNC_STAGING_OK);
    assert(!p.async_enabled);

    r.staging_offset = 3900;
    r.staging_bytes = 512;
    assert(common_tree_draft_async_staging_plan_build(make_sm89(),layout,r,&p) ==
           COMMON_TREE_DRAFT_ASYNC_STAGING_LAYOUT_RANGE);
    return 0;
}
