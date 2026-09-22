#include "tree-draft-kernel-capability.h"

#include <cassert>

int main() {
    common_tree_draft_device_capabilities d{};
    d.available=true; d.cc_major=8; d.cc_minor=9;

    common_tree_draft_quant_type q{};
    q.id=42; q.source=COMMON_TREE_DRAFT_QUANT_GGML; q.block_elems=32; q.block_bytes=18;

    const common_tree_draft_kernel_capability_entry entries[] = {
        {89,COMMON_TREE_DRAFT_QUANT_GGML,42,COMMON_TREE_DRAFT_KERNEL_LINEAR,
         COMMON_TREE_DRAFT_KERNEL_LAYOUT_BLOCKED,COMMON_TREE_DRAFT_DTYPE_F16,
         32,16,32,8,false,true},
        {89,COMMON_TREE_DRAFT_QUANT_GGML,42,COMMON_TREE_DRAFT_KERNEL_ATTENTION_KV,
         COMMON_TREE_DRAFT_KERNEL_LAYOUT_PAGED_KV,COMMON_TREE_DRAFT_DTYPE_F16,
         0,16,32,1,true,false},
    };
    common_tree_draft_kernel_capability_query query{};
    query.quant=&q;
    query.op=COMMON_TREE_DRAFT_KERNEL_LINEAR;
    query.layout=COMMON_TREE_DRAFT_KERNEL_LAYOUT_BLOCKED;
    query.compute_dtype=COMMON_TREE_DRAFT_DTYPE_F16;
    query.group_size=32;
    query.address=0x1000;
    query.k=128;
    query.n=64;

    common_tree_draft_kernel_capability_result r{};
    assert(common_tree_draft_kernel_capability_lookup(d,entries,2,query,&r)==COMMON_TREE_DRAFT_KERNEL_CAPABILITY_OK);
    assert(r.supported && r.entry_index==0 && r.fused_dequant);

    query.address += 4;
    assert(common_tree_draft_kernel_capability_lookup(d,entries,2,query,&r)==COMMON_TREE_DRAFT_KERNEL_CAPABILITY_OK);
    assert(!r.supported);

    query.address=0x1000; query.k=130;
    assert(common_tree_draft_kernel_capability_lookup(d,entries,2,query,&r)==COMMON_TREE_DRAFT_KERNEL_CAPABILITY_OK);
    assert(!r.supported);

    query.k=128; query.group_size=64;
    assert(common_tree_draft_kernel_capability_lookup(d,entries,2,query,&r)==COMMON_TREE_DRAFT_KERNEL_CAPABILITY_OK);
    assert(!r.supported);

    // Unknown combinations are unsupported, never guessed.
    query.group_size=32;
    query.compute_dtype=COMMON_TREE_DRAFT_DTYPE_BF16;
    assert(common_tree_draft_kernel_capability_lookup(d,entries,2,query,&r)==COMMON_TREE_DRAFT_KERNEL_CAPABILITY_OK);
    assert(!r.supported);
    return 0;
}
