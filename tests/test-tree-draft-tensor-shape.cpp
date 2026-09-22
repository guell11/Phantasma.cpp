#include "tree-draft-tensor-shape.h"

#include <cassert>

static common_tree_draft_architecture_config cfg(const char * arch) {
    common_tree_draft_model_metadata m;
    m.architecture = common_tree_draft_model_metadata_text{arch, {}};
    m.n_layers = common_tree_draft_model_metadata_u64{32,{}};
    m.n_embd = common_tree_draft_model_metadata_u64{4096,{}};
    m.n_heads = common_tree_draft_model_metadata_u64{32,{}};
    m.n_heads_kv = common_tree_draft_model_metadata_u64{8,{}};
    m.n_ff = common_tree_draft_model_metadata_u64{11008,{}};
    m.vocab_size = common_tree_draft_model_metadata_u64{32000,{}};
    auto out = common_tree_draft_architecture_config::normalize(m);
    assert(out.has_value());
    return *out;
}

int main() {
    const auto llama = cfg("llama");
    assert(common_tree_draft_tensor_shape_validate(llama, {COMMON_TREE_DRAFT_TENSOR_TOKEN_EMBEDDING,-1,-1,COMMON_TREE_DRAFT_TENSOR_SUBROLE_WEIGHT}, {4096,32000}).status == COMMON_TREE_DRAFT_TENSOR_SHAPE_OK);
    assert(common_tree_draft_tensor_shape_validate(llama, {COMMON_TREE_DRAFT_TENSOR_ATTN_K,0,-1,COMMON_TREE_DRAFT_TENSOR_SUBROLE_WEIGHT}, {4096,1024}).status == COMMON_TREE_DRAFT_TENSOR_SHAPE_OK);
    assert(common_tree_draft_tensor_shape_validate(llama, {COMMON_TREE_DRAFT_TENSOR_FFN_UP,3,-1,COMMON_TREE_DRAFT_TENSOR_SUBROLE_WEIGHT}, {4096,11008}).status == COMMON_TREE_DRAFT_TENSOR_SHAPE_OK);
    auto bad = common_tree_draft_tensor_shape_validate(llama, {COMMON_TREE_DRAFT_TENSOR_FFN_DOWN,3,-1,COMMON_TREE_DRAFT_TENSOR_SUBROLE_WEIGHT}, {4096,4096});
    assert(bad.status == COMMON_TREE_DRAFT_TENSOR_SHAPE_MISMATCH && !bad.accepted.empty());
    assert(common_tree_draft_tensor_shape_validate(llama, {COMMON_TREE_DRAFT_TENSOR_ATTN_Q,32,-1,COMMON_TREE_DRAFT_TENSOR_SUBROLE_WEIGHT}, {4096,4096}).status == COMMON_TREE_DRAFT_TENSOR_SHAPE_LAYER_RANGE);

    const auto gemma4 = cfg("gemma4");
    assert(common_tree_draft_tensor_shape_validate(gemma4, {COMMON_TREE_DRAFT_TENSOR_FFN_GATE,1,-1,COMMON_TREE_DRAFT_TENSOR_SUBROLE_WEIGHT}, {4096,8192}).status == COMMON_TREE_DRAFT_TENSOR_SHAPE_UNRESOLVED);
    assert(common_tree_draft_tensor_shape_validate(gemma4, {COMMON_TREE_DRAFT_TENSOR_ATTN_Q_NORM,1,-1,COMMON_TREE_DRAFT_TENSOR_SUBROLE_WEIGHT}, {128}).status == COMMON_TREE_DRAFT_TENSOR_SHAPE_OK);
    return 0;
}

