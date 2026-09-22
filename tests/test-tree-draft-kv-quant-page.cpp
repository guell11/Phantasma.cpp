#include "tree-draft-kv-quant-page.h"

#include <cassert>

int main() {
    common_tree_draft_kv_layer_geometry layers[]={{64,64,COMMON_TREE_DRAFT_KV_LAYOUT_QUANTIZED}};
    common_tree_draft_kv_page_geometry g{};
    assert(common_tree_draft_kv_page_geometry_build(16,layers,1,16,&g)==COMMON_TREE_DRAFT_KV_PAGE_OK);

    common_tree_draft_kv_quant_page_request r{};
    r.type_k=GGML_TYPE_Q8_0;
    r.type_v=GGML_TYPE_Q8_0;
    r.head_dim_k=128;
    r.head_dim_v=128;
    r.capabilities={true,true,true,true};
    common_tree_draft_kv_quant_page_plan p{};
    assert(common_tree_draft_kv_quant_page_plan_build(g,r,&p)==COMMON_TREE_DRAFT_KV_QUANT_PAGE_OK);
    assert(p.page_aligned_k && p.page_aligned_v);
    assert(p.token_aligned_k && p.token_aligned_v);
    assert(p.compatibility.selected_path==COMMON_TREE_DRAFT_KV_PATH_DIRECT);

    common_tree_draft_kv_page_descriptor page{COMMON_TREE_DRAFT_KV_PAGE_LIVE,1,1,8,0};
    assert(common_tree_draft_kv_quant_valid_range(p,page,0,8));
    assert(!common_tree_draft_kv_quant_valid_range(p,page,0,9));

    // Page alignment may hold while one token is not a complete quant block;
    // direct is then disabled and exact staging/backend fallback is selected.
    r.head_dim_k=24;
    r.head_dim_v=24;
    r.capabilities={true,true,false,true};
    assert(common_tree_draft_kv_quant_page_plan_build(g,r,&p)==COMMON_TREE_DRAFT_KV_QUANT_PAGE_OK);
    assert(p.page_aligned_k && !p.token_aligned_k);
    assert(p.compatibility.selected_path==COMMON_TREE_DRAFT_KV_PATH_STAGING);

    r.head_dim_k=11;
    r.head_dim_v=11;
    assert(common_tree_draft_kv_quant_page_plan_build(g,r,&p)==COMMON_TREE_DRAFT_KV_QUANT_PAGE_BLOCK_ALIGNMENT);
    return 0;
}
