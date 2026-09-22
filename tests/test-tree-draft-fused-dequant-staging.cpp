#include "tree-draft-fused-dequant-staging.h"

#include <cassert>

int main() {
    common_tree_draft_quant_type q{};
    q.id=7; q.name="test-q4"; q.block_elems=32; q.block_bytes=18; q.fixed_block_size=true;

    common_tree_draft_tensor_core_policy tc{};
    tc.has_selection=true;
    tc.selected_mode=COMMON_TREE_DRAFT_TENSOR_CORE_F16;
    tc.eligible_mask=COMMON_TREE_DRAFT_TENSOR_CORE_F16;

    common_tree_draft_fused_dequant_request r{};
    r.quant=&q;
    r.format_supported=true;
    r.logical_elements=128;
    r.encoded_bytes=72;
    r.tensor_core=tc;
    r.staging.async_enabled=true;
    r.staging.pipeline_depth=2;
    r.cost={1.0f,2.0f,6.0f,2.0f,3.0f};

    common_tree_draft_fused_dequant_plan p{};
    assert(common_tree_draft_fused_dequant_staging_plan(r,&p)==COMMON_TREE_DRAFT_FUSED_DEQUANT_OK);
    assert(p.enabled);
    assert(p.fragment_dtype==COMMON_TREE_DRAFT_FRAGMENT_F16);
    assert(p.fused_cost_us==6.0f && p.separate_cost_us==11.0f);
    assert(!p.writes_global_intermediate);

    r.cost.dequant_shared_us=10.0f;
    assert(common_tree_draft_fused_dequant_staging_plan(r,&p)==COMMON_TREE_DRAFT_FUSED_DEQUANT_OK);
    assert(!p.enabled);

    r.cost.dequant_shared_us=2.0f;
    r.format_supported=false;
    assert(common_tree_draft_fused_dequant_staging_plan(r,&p)==COMMON_TREE_DRAFT_FUSED_DEQUANT_OK);
    assert(!p.enabled);

    r.format_supported=true;
    r.encoded_bytes=71;
    assert(common_tree_draft_fused_dequant_staging_plan(r,&p)==COMMON_TREE_DRAFT_FUSED_DEQUANT_BLOCK_MISMATCH);
    return 0;
}
