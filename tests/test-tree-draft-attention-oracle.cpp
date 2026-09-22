#include "tree-draft-attention-oracle.h"

#include <cassert>
#include <cmath>
#include <vector>

static bool near(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) <= eps;
}

int main() {
    // Entry 0: prefix length 1, tree root 0 with children 1 and 2.
    // Entry 1: zero prefix, independent root 3.
    const int32_t parent[]  = {-1,0,0,-1};
    const int32_t depth[]   = { 0,1,1, 0};
    const int32_t tree_id[] = { 7,7,7, 9};
    const common_tree_draft_topology topology{parent,depth,tree_id,4};
    std::vector<uint32_t> words(4,0);
    common_tree_draft_ancestor_bitset ancestors{words.data(),words.size()};
    assert(common_tree_draft_ancestor_build(topology,ancestors) == COMMON_TREE_DRAFT_ANCESTOR_OK);
    const int32_t offsets[] = {0,3,4};
    const common_tree_draft_forest_offsets forest{offsets,2};
    const int32_t prefixes[] = {1,0};

    // Hq=2, Hkv=1 exercises MQA mapping; Dk=1, Dv=1 keeps expected values transparent.
    const float q[] = {
        1,2,  // q0 heads
        1,2,  // q1
        1,2,  // q2
        1,2,  // q3
    };
    const float prefix_k[] = {1};
    const float prefix_v[] = {10};
    const float tree_k[] = {0,1,2,3};
    const float tree_v[] = {20,30,40,50};
    float out[8] = {};
    common_tree_draft_attention_oracle_view view{
        q,prefix_k,prefix_v,tree_k,tree_v,out,
        2,1,1,1,1.0f,8
    };
    assert(common_tree_draft_attention_oracle(topology,ancestors,forest,prefixes,view) ==
           COMMON_TREE_DRAFT_ATTENTION_ORACLE_OK);

    // q0 sees prefix + itself(tree k=0): softmax([1,0]).
    const float e1 = std::exp(1.0f);
    assert(near(out[0], (e1*10.0f + 20.0f)/(e1+1.0f)));
    const float e2 = std::exp(2.0f);
    assert(near(out[1], (e2*10.0f + 20.0f)/(e2+1.0f)));

    // q1 sees prefix + root + itself, but NOT sibling q2.
    const float d10 = std::exp(1.0f) + std::exp(0.0f) + std::exp(1.0f);
    assert(near(out[2], (std::exp(1.0f)*10 + 20 + std::exp(1.0f)*30)/d10));
    const float d11 = std::exp(2.0f) + std::exp(0.0f) + std::exp(2.0f);
    assert(near(out[3], (std::exp(2.0f)*10 + 20 + std::exp(2.0f)*30)/d11));

    // q3 is isolated in entry/tree 1 and therefore sees only itself.
    assert(near(out[6],50.0f) && near(out[7],50.0f));

    auto bad_heads = view; bad_heads.n_query_heads = 3; bad_heads.n_kv_heads = 2;
    assert(common_tree_draft_attention_oracle(topology,ancestors,forest,prefixes,bad_heads) ==
           COMMON_TREE_DRAFT_ATTENTION_ORACLE_INVALID_SHAPE);
    auto short_out = view; short_out.output_count = 7;
    assert(common_tree_draft_attention_oracle(topology,ancestors,forest,prefixes,short_out) ==
           COMMON_TREE_DRAFT_ATTENTION_ORACLE_OUTPUT_TOO_SMALL);
    const int32_t bad_prefix[] = {-1,0};
    assert(common_tree_draft_attention_oracle(topology,ancestors,forest,bad_prefix,view) ==
           COMMON_TREE_DRAFT_ATTENTION_ORACLE_INVALID_PREFIX);
    return 0;
}
