#include "tree-draft-quant-matmul-dispatch.h"

#include <cassert>

int main() {
    common_tree_draft_device_capabilities d{};
    d.available=true; d.cc_major=8; d.cc_minor=9;
    common_tree_draft_quant_type q{};
    q.id=7; q.source=COMMON_TREE_DRAFT_QUANT_GGML; q.block_elems=32; q.block_bytes=18; q.fixed_block_size=true;
    const common_tree_draft_kernel_capability_entry entries[] = {{
        89,COMMON_TREE_DRAFT_QUANT_GGML,7,COMMON_TREE_DRAFT_KERNEL_LINEAR,
        COMMON_TREE_DRAFT_KERNEL_LAYOUT_BLOCKED,COMMON_TREE_DRAFT_DTYPE_F16,
        0,16,32,8,false,true
    }};
    common_tree_draft_quant_matmul_request req{};
    req.capability_query.quant=&q;
    req.capability_query.op=COMMON_TREE_DRAFT_KERNEL_LINEAR;
    req.capability_query.layout=COMMON_TREE_DRAFT_KERNEL_LAYOUT_BLOCKED;
    req.capability_query.compute_dtype=COMMON_TREE_DRAFT_DTYPE_F16;
    req.capability_query.address=0x1000;
    req.m=1; req.n=64; req.k=128;
    req.workspace_budget=4096;
    req.dequant_output_element_bytes=2;

    common_tree_draft_quant_matmul_plan p{};
    assert(common_tree_draft_quant_matmul_plan_build(d,entries,1,req,&p)==COMMON_TREE_DRAFT_QUANT_MATMUL_OK);
    assert(p.path==COMMON_TREE_DRAFT_QUANT_MATMUL_NATIVE);

    req.capability_query.address=0x1004;
    assert(common_tree_draft_quant_matmul_plan_build(d,entries,1,req,&p)==COMMON_TREE_DRAFT_QUANT_MATMUL_OK);
    assert(p.path==COMMON_TREE_DRAFT_QUANT_MATMUL_DEQUANT_ON_THE_FLY);
    assert(!p.dequant.tiles.empty());

    req.workspace_budget=8;
    assert(common_tree_draft_quant_matmul_plan_build(d,entries,1,req,&p)==COMMON_TREE_DRAFT_QUANT_MATMUL_OK);
    assert(p.path==COMMON_TREE_DRAFT_QUANT_MATMUL_UNSUPPORTED);
    return 0;
}
