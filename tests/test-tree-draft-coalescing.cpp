#include "tree-draft-coalescing.h"

#include <cassert>
#include <cmath>

int main() {
    common_tree_draft_lane_access lanes[32] = {};
    for (int i=0;i<32;++i) lanes[i]={0x1000u + static_cast<uint64_t>(i)*4u,4};
    common_tree_draft_coalescing_result r{};
    assert(common_tree_draft_coalescing_measure(lanes,32,&r)==COMMON_TREE_DRAFT_COALESCING_OK);
    assert(r.sectors_touched==4 && r.useful_bytes==128);
    assert(std::fabs(r.efficiency-1.0f)<1e-6f);
    assert(r.adjacent_lanes_contiguous);

    for (int i=0;i<32;++i) lanes[i]={0x1000u + static_cast<uint64_t>(i)*32u,4};
    assert(common_tree_draft_coalescing_measure(lanes,32,&r)==COMMON_TREE_DRAFT_COALESCING_OK);
    assert(r.sectors_touched==32);
    assert(std::fabs(r.efficiency-0.125f)<1e-6f);
    assert(!r.adjacent_lanes_contiguous);

    common_tree_draft_lane_access crossing[]={{31,4}};
    assert(common_tree_draft_coalescing_measure(crossing,1,&r)==COMMON_TREE_DRAFT_COALESCING_OK);
    assert(r.sectors_touched==2);
    return 0;
}
