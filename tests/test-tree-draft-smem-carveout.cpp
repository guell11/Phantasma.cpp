#include "tree-draft-smem-carveout.h"

#include <cassert>

int main() {
    common_tree_draft_device_capabilities d{};
    d.available=true; d.cc_major=8; d.cc_minor=9; d.smem_sm=102400;
    common_tree_draft_smem_carveout_request r{};
    r.required_shared_bytes=40000;
    r.candidates = {{
        {0,0,102400,true},
        {25,25600,76800,true},
        {50,51200,51200,true},
        {75,76800,25600,true},
        {100,102400,0,true},
    }};
    common_tree_draft_smem_carveout_plan p{};
    assert(common_tree_draft_smem_carveout_plan_build(d,r,&p)==COMMON_TREE_DRAFT_SMEM_CARVEOUT_OK);
    assert(p.available && p.selected_index==2);
    assert(p.has_lower_neighbor && p.lower_neighbor_index==1);
    assert(p.has_upper_neighbor && p.upper_neighbor_index==3);

    r.required_shared_bytes=90000;
    assert(common_tree_draft_smem_carveout_plan_build(d,r,&p)==COMMON_TREE_DRAFT_SMEM_CARVEOUT_OK);
    assert(p.selected_index==4 && !p.has_upper_neighbor);
    r.required_shared_bytes=120000;
    assert(common_tree_draft_smem_carveout_plan_build(d,r,&p)==COMMON_TREE_DRAFT_SMEM_CARVEOUT_UNSATISFIED);
    return 0;
}
