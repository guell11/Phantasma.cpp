#include "tree-draft-quant-groups.h"

#include <cassert>

int main() {
    common_tree_draft_packed_linear_view gptq;
    gptq.format = COMMON_TREE_DRAFT_PACKED_LINEAR_GPTQ;
    gptq.in_features = 128; gptq.out_features = 64; gptq.group_size = 64; gptq.group_count = 2; gptq.zero_point = true;
    auto gd = common_tree_draft_quant_groups_from_packed(gptq);
    bool decodable = false, eligible = false;
    assert(common_tree_draft_quant_group_validate(gd,{32,64,128},false,&decodable,&eligible) == COMMON_TREE_DRAFT_QUANT_GROUP_OK);
    assert(decodable && eligible);
    assert(common_tree_draft_quant_group_validate(gd,{32,128},false,&decodable,&eligible) == COMMON_TREE_DRAFT_QUANT_GROUP_KERNEL_GROUP_UNSUPPORTED);
    assert(decodable && !eligible);

    common_tree_draft_bnb4_descriptor b4;
    b4.block_size = 64; b4.logical_elements = 130;
    auto bd = common_tree_draft_quant_groups_from_bnb4(b4);
    assert(common_tree_draft_quant_group_validate(bd,{64},false,&decodable,&eligible) == COMMON_TREE_DRAFT_QUANT_GROUP_KERNEL_GROUP_UNSUPPORTED);
    assert(decodable && !eligible);
    assert(common_tree_draft_quant_group_validate(bd,{64},true,&decodable,&eligible) == COMMON_TREE_DRAFT_QUANT_GROUP_OK);
    gd.axis_length = 130; gd.group_count = 3; gd.tail_group_allowed = false;
    assert(common_tree_draft_quant_group_validate(gd,{64},true,&decodable,&eligible) == COMMON_TREE_DRAFT_QUANT_GROUP_TAIL_REQUIRED);
    return 0;
}

