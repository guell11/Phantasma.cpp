#include "tree-draft-repack-plan.h"

#include <cassert>

static common_tree_draft_tensor_layout dense() {
    return {COMMON_TREE_DRAFT_LAYOUT_DENSE,
        {COMMON_TREE_DRAFT_AXIS_INPUT,COMMON_TREE_DRAFT_AXIS_OUTPUT},
        {COMMON_TREE_DRAFT_AXIS_INPUT,COMMON_TREE_DRAFT_AXIS_OUTPUT},
        {0,1},{64,128},{64,128},{},{128,1},false};
}

static common_tree_draft_tensor_layout blocked() {
    return {COMMON_TREE_DRAFT_LAYOUT_BLOCKED_QUANT,
        {COMMON_TREE_DRAFT_AXIS_INPUT,COMMON_TREE_DRAFT_AXIS_OUTPUT},
        {COMMON_TREE_DRAFT_AXIS_INPUT,COMMON_TREE_DRAFT_AXIS_OUTPUT},
        {0,1},{64,128},{64,128},{32,1},{128,1},false};
}

int main() {
    common_tree_draft_quant_type q{};
    q.id=9; q.block_elems=32; q.block_bytes=18; q.fixed_block_size=true;
    common_tree_draft_repack_request r{};
    r.source={"weights.gguf","sha256:abc",4096,8192};
    r.quant=&q;
    r.source_layout=dense();
    r.target_layout=blocked();
    r.target_kernel="sm89-q4-mma";
    r.kernel_abi_version=3;
    r.tolerance=0.5;

    common_tree_draft_repack_plan a{},b{};
    assert(common_tree_draft_repack_plan_build(r,&a)==COMMON_TREE_DRAFT_REPACK_OK);
    assert(common_tree_draft_repack_plan_build(r,&b)==COMMON_TREE_DRAFT_REPACK_OK);
    assert(a.cache_key==b.cache_key && !a.cache_key.empty());

    r.kernel_abi_version=4;
    assert(common_tree_draft_repack_plan_build(r,&b)==COMMON_TREE_DRAFT_REPACK_OK);
    assert(a.cache_key!=b.cache_key);

    r.kernel_abi_version=3;
    r.source.tensor_offset++;
    assert(common_tree_draft_repack_plan_build(r,&b)==COMMON_TREE_DRAFT_REPACK_OK);
    assert(a.cache_key!=b.cache_key);

    const float src[]={1,2,3};
    const float good[]={1.01f,1.99f,3.02f};
    const float bad[]={1,2,4};
    double err=0;
    assert(common_tree_draft_repack_equivalent(src,good,3,0.03,&err));
    assert(err>=0.019 && err<=0.021);
    assert(!common_tree_draft_repack_equivalent(src,bad,3,0.5,&err));
    return 0;
}
