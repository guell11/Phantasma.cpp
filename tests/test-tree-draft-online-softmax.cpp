#include "tree-draft-online-softmax.h"

#include <cassert>
#include <cmath>

static bool near(float a, float b, float eps=1e-5f) {
    return std::fabs(a-b) <= eps;
}

int main() {
    float acc[2] = {};
    common_tree_draft_online_softmax_state s{};
    assert(common_tree_draft_online_softmax_init(&s,acc,2) == COMMON_TREE_DRAFT_ONLINE_SOFTMAX_OK);

    const float score1[] = {1.0f,2.0f};
    const float value1[] = {10,100,20,200};
    assert(common_tree_draft_online_softmax_update(&s,score1,value1,nullptr,2) == COMMON_TREE_DRAFT_ONLINE_SOFTMAX_OK);

    const float score2[] = {3.0f,100.0f};
    const float value2[] = {30,300,999,999};
    const uint8_t masked2[] = {0,1};
    assert(common_tree_draft_online_softmax_update(&s,score2,value2,masked2,2) == COMMON_TREE_DRAFT_ONLINE_SOFTMAX_OK);

    float out[2] = {};
    assert(common_tree_draft_online_softmax_finalize(s,out,2) == COMMON_TREE_DRAFT_ONLINE_SOFTMAX_OK);
    const float w1=std::exp(1.0f-3.0f), w2=std::exp(2.0f-3.0f), w3=1.0f;
    const float z=w1+w2+w3;
    assert(near(out[0],(w1*10+w2*20+w3*30)/z));
    assert(near(out[1],(w1*100+w2*200+w3*300)/z));

    // Empty and fully masked blocks are exact no-ops.
    const float before0=out[0], before1=out[1];
    assert(common_tree_draft_online_softmax_update(&s,nullptr,nullptr,nullptr,0) == COMMON_TREE_DRAFT_ONLINE_SOFTMAX_OK);
    const float score3[]={7,8};
    const float value3[]={1,2,3,4};
    const uint8_t all_masked[]={1,1};
    assert(common_tree_draft_online_softmax_update(&s,score3,value3,all_masked,2) == COMMON_TREE_DRAFT_ONLINE_SOFTMAX_OK);
    assert(common_tree_draft_online_softmax_finalize(s,out,2) == COMMON_TREE_DRAFT_ONLINE_SOFTMAX_OK);
    assert(near(out[0],before0) && near(out[1],before1));

    float acc2[1]={};
    common_tree_draft_online_softmax_state empty{};
    assert(common_tree_draft_online_softmax_init(&empty,acc2,1) == COMMON_TREE_DRAFT_ONLINE_SOFTMAX_OK);
    float o2=0;
    assert(common_tree_draft_online_softmax_finalize(empty,&o2,1) == COMMON_TREE_DRAFT_ONLINE_SOFTMAX_EMPTY);

    const float bad_score[]={NAN};
    const float good_value[]={1};
    assert(common_tree_draft_online_softmax_update(&empty,bad_score,good_value,nullptr,1) == COMMON_TREE_DRAFT_ONLINE_SOFTMAX_INVALID_SCORE);
    return 0;
}
