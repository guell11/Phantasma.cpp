#include "tree-draft-attention-baseline.h"

#include <cassert>
#include <cmath>
#include <vector>

int main() {
    const int32_t parent[]  = {-1,0,0,-1,3};
    const int32_t depth[]   = { 0,1,1, 0,1};
    const int32_t tree_id[] = { 4,4,4, 9,9};
    const common_tree_draft_topology topology{parent,depth,tree_id,5};
    const size_t wpr = common_tree_draft_ancestor_words_per_row(5);
    std::vector<uint32_t> words(5*wpr,0);
    common_tree_draft_ancestor_bitset ancestors{words.data(),words.size()};
    assert(common_tree_draft_ancestor_build(topology,ancestors) == COMMON_TREE_DRAFT_ANCESTOR_OK);

    const int32_t offsets[] = {0,3,5};
    const common_tree_draft_forest_offsets forest{offsets,2};
    const int32_t prefix[] = {2,1};
    const int32_t kv_base[] = {0,16};
    common_tree_draft_ragged_attention_entry entries[2] = {};
    common_tree_draft_ragged_attention_batch batch{entries,2,0,0,0};
    assert(common_tree_draft_ragged_attention_build(forest,prefix,kv_base,batch) ==
           COMMON_TREE_DRAFT_RAGGED_ATTENTION_OK);

    // Hq=4,Hkv=2 exercises GQA ratio 2. Dk=2,Dv=2.
    std::vector<float> q(5*4*2);
    std::vector<float> pk(3*2*2), pv(3*2*2);
    std::vector<float> tk(5*2*2), tv(5*2*2);
    for (size_t i=0;i<q.size();++i) q[i]=0.01f*static_cast<float>(i+1);
    for (size_t i=0;i<pk.size();++i) pk[i]=0.02f*static_cast<float>(i+1);
    for (size_t i=0;i<pv.size();++i) pv[i]=0.03f*static_cast<float>(i+1);
    for (size_t i=0;i<tk.size();++i) tk[i]=0.04f*static_cast<float>(i+1);
    for (size_t i=0;i<tv.size();++i) tv[i]=0.05f*static_cast<float>(i+1);

    std::vector<float> reference(5*4*2,0), tiled(reference.size(),0);
    common_tree_draft_attention_oracle_view ref{
        q.data(),pk.data(),pv.data(),tk.data(),tv.data(),reference.data(),
        4,2,2,2,0.70710678f,reference.size()
    };
    assert(common_tree_draft_attention_oracle(topology,ancestors,forest,prefix,ref) ==
           COMMON_TREE_DRAFT_ATTENTION_ORACLE_OK);
    auto got=ref; got.output=tiled.data();
    assert(common_tree_draft_attention_baseline_forward(
        topology,ancestors,batch,got,{2,4}) == COMMON_TREE_DRAFT_ATTENTION_BASELINE_OK);
    for (size_t i=0;i<reference.size();++i) {
        assert(std::fabs(reference[i]-tiled[i]) < 2e-5f);
    }

    assert(common_tree_draft_attention_baseline_forward(
        topology,ancestors,batch,got,{0,4}) == COMMON_TREE_DRAFT_ATTENTION_BASELINE_INVALID_LAUNCH);
    return 0;
}
