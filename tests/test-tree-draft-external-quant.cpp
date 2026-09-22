#include "tree-draft-external-quant.h"

#include <cassert>

int main() {
    common_tree_draft_safetensors_table gptq;
    gptq.metadata = {{"quant_method","gptq"},{"bits","4"},{"group_size","128"},{"sym","true"},{"desc_act","false"},{"damp_percent","0.1"},{"packing_version","v1"}};
    common_tree_draft_gptq_descriptor gd;
    assert(common_tree_draft_gptq_parse(gptq, &gd) == COMMON_TREE_DRAFT_EXTERNAL_QUANT_OK);
    assert(gd.bits == 4 && gd.group_size == 128 && gd.sym && !gd.desc_act && gd.damp_percent.has_value());
    auto gptq2 = gd;
    assert(common_tree_draft_gptq_equal(gd, gptq2));

    common_tree_draft_safetensors_table awq;
    awq.metadata = {{"quant_method","awq"},{"bits","4"},{"group_size","64"},{"zero_point","true"},{"packing_version","gemm"}};
    common_tree_draft_awq_descriptor ad;
    assert(common_tree_draft_awq_parse(awq, &ad) == COMMON_TREE_DRAFT_EXTERNAL_QUANT_OK);
    assert(ad.bits == 4 && ad.group_size == 64 && ad.zero_point);

    common_tree_draft_safetensors_table b8;
    b8.tensors.push_back({"layer.weight","I8",{16,32},0,512});
    b8.tensors.push_back({"layer.SCB","F32",{16},512,576});
    common_tree_draft_bnb8_descriptor b8d;
    assert(common_tree_draft_bnb8_parse(b8,"layer",&b8d) == COMMON_TREE_DRAFT_EXTERNAL_QUANT_OK);
    assert(b8d.rows == 16 && b8d.cols == 32);

    common_tree_draft_safetensors_table b4;
    b4.metadata = {{"bnb_4bit_quant_type","nf4"},{"bnb_4bit_block_size","64"},{"bnb_4bit_nested_depth","1"}};
    b4.tensors.push_back({"layer.weight","U8",{128},0,64});
    b4.tensors.push_back({"layer.absmax","F32",{2},64,72});
    b4.tensors.push_back({"layer.nested_absmax","F32",{1},72,76});
    common_tree_draft_bnb4_descriptor b4d;
    assert(common_tree_draft_bnb4_parse(b4,"layer",&b4d) == COMMON_TREE_DRAFT_EXTERNAL_QUANT_OK);
    assert(b4d.kind == COMMON_TREE_DRAFT_BNB4_NF4 && b4d.block_size == 64 && b4d.nested_depth == 1);

    gptq.metadata["bits"] = "5";
    assert(common_tree_draft_gptq_parse(gptq,&gd) == COMMON_TREE_DRAFT_EXTERNAL_QUANT_UNSUPPORTED);
    return 0;
}

