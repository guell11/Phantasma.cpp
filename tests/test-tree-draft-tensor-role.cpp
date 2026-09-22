#include "tree-draft-tensor-role.h"

#include <cassert>

static common_tree_draft_tensor_role_result canon(const char * arch, const char * name) {
    common_tree_draft_tensor_role_result result = {};
    assert(common_tree_draft_tensor_role_canonicalize(arch, name, &result) == COMMON_TREE_DRAFT_TENSOR_ROLE_OK);
    return result;
}

int main() {
    auto token = canon("gemma4", "token_embd.weight");
    assert(token.key.role == COMMON_TREE_DRAFT_TENSOR_TOKEN_EMBEDDING && token.key.layer == -1 && token.key.subrole == COMMON_TREE_DRAFT_TENSOR_SUBROLE_WEIGHT);

    auto q = canon("gemma4", "blk.12.attn_q.weight");
    assert(q.key.role == COMMON_TREE_DRAFT_TENSOR_ATTN_Q && q.key.layer == 12 && q.key.expert == -1);
    auto expert = canon("qwen3", "blk.7.ffn_down.3.weight");
    assert(expert.key.role == COMMON_TREE_DRAFT_TENSOR_FFN_DOWN_EXPERT && expert.key.layer == 7 && expert.key.expert == 3);
    auto packed = canon("qwen3", "blk.7.ffn_gate_up_exps.weight");
    assert(packed.key.role == COMMON_TREE_DRAFT_TENSOR_FFN_GATE_UP_EXPERTS && packed.key.layer == 7);

    auto hf_q = canon("llama", "model.layers.2.self_attn.q_proj.weight");
    assert(hf_q.key.role == COMMON_TREE_DRAFT_TENSOR_ATTN_Q && hf_q.key.layer == 2);
    auto hf_embed = canon("gemma4", "model.embed_tokens.weight");
    assert(hf_embed.key.role == COMMON_TREE_DRAFT_TENSOR_TOKEN_EMBEDDING);

    auto unknown = canon("gemma4", "model.layers.0.weird_new_tensor.weight");
    assert(unknown.key.role == COMMON_TREE_DRAFT_TENSOR_UNKNOWN);
    assert(unknown.source_name == "model.layers.0.weird_new_tensor.weight");

    common_tree_draft_tensor_role_result aliases[] = { token, hf_embed };
    size_t first = SIZE_MAX;
    size_t second = SIZE_MAX;
    assert(common_tree_draft_tensor_role_validate_unique(aliases, 2, &first, &second) == COMMON_TREE_DRAFT_TENSOR_ROLE_COLLISION);
    assert(first == 0 && second == 1);

    common_tree_draft_tensor_role_result unique[] = { q, expert, packed, unknown };
    assert(common_tree_draft_tensor_role_validate_unique(unique, 4) == COMMON_TREE_DRAFT_TENSOR_ROLE_OK);

    common_tree_draft_tensor_role_result unsupported = {};
    assert(common_tree_draft_tensor_role_canonicalize("unknown_arch", "token_embd.weight", &unsupported) == COMMON_TREE_DRAFT_TENSOR_ROLE_UNSUPPORTED_ARCH);
    return 0;
}

