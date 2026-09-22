#include "tree-draft-packed-linear.h"

#include <cassert>

static common_tree_draft_safetensors_table fixture() {
    common_tree_draft_safetensors_table t;
    t.tensors.push_back({"layer.qweight","I32",{16,64},0,4096});
    t.tensors.push_back({"layer.qzeros","I32",{2,8},4096,4160});
    t.tensors.push_back({"layer.scales","F16",{2,64},4160,4416});
    t.tensors.push_back({"layer.g_idx","I32",{128},4416,4928});
    return t;
}

int main() {
    auto t = fixture();
    common_tree_draft_gptq_descriptor g = {4,64,true,true,std::nullopt,"v1"};
    common_tree_draft_packed_linear_view view;
    assert(common_tree_draft_gptq_bind(t,"layer",g,64,128,&view) == COMMON_TREE_DRAFT_PACKED_LINEAR_OK);
    assert(view.unpack_factor == 8 && view.group_count == 2 && view.g_idx.has_value());
    common_tree_draft_awq_descriptor a = {4,64,true,"gemm"};
    assert(common_tree_draft_awq_bind(t,"layer",a,64,128,&view) == COMMON_TREE_DRAFT_PACKED_LINEAR_OK);
    assert(!view.g_idx.has_value() && view.zero_point);
    auto bad = t;
    bad.tensors[2].shape = {3,64};
    assert(common_tree_draft_gptq_bind(bad,"layer",g,64,128,&view) == COMMON_TREE_DRAFT_PACKED_LINEAR_SHAPE);
    return 0;
}

