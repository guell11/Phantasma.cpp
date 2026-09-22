#include "tree-draft-tensor-shape.h"

#include <algorithm>

static bool shape_eq(const std::vector<uint64_t> & a, std::initializer_list<uint64_t> b) {
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
}

static common_tree_draft_tensor_shape_check resolved(
        const std::vector<uint64_t> & observed,
        std::initializer_list<std::vector<uint64_t>> accepted,
        const char * symbolic) {
    common_tree_draft_tensor_shape_check out;
    out.observed = observed;
    out.accepted.assign(accepted.begin(), accepted.end());
    out.symbolic_expectation = symbolic;
    out.status = COMMON_TREE_DRAFT_TENSOR_SHAPE_MISMATCH;
    for (const auto & candidate : out.accepted) {
        if (candidate == observed) {
            out.status = COMMON_TREE_DRAFT_TENSOR_SHAPE_OK;
            break;
        }
    }
    return out;
}

common_tree_draft_tensor_shape_check common_tree_draft_tensor_shape_validate(
        const common_tree_draft_architecture_config & config,
        const common_tree_draft_tensor_role_key & key,
        const std::vector<uint64_t> & shape) {
    common_tree_draft_tensor_shape_check out;
    out.observed = shape;
    if (key.role == COMMON_TREE_DRAFT_TENSOR_UNKNOWN) {
        return out;
    }
    if (key.layer >= 0 && static_cast<uint64_t>(key.layer) >= config.n_layers()) {
        out.status = COMMON_TREE_DRAFT_TENSOR_SHAPE_LAYER_RANGE;
        return out;
    }

    const uint64_t D = config.n_embd();
    const uint64_t V = config.vocab_size();
    const uint64_t FF = config.n_ff();
    const uint64_t H = config.n_heads();
    const uint64_t HKV = config.n_heads_kv();
    const uint64_t DH = D / H;
    const uint64_t KV = DH * HKV;
    const bool gemma4 = config.architecture() == "gemma4";

    switch (key.role) {
        case COMMON_TREE_DRAFT_TENSOR_TOKEN_EMBEDDING:
        case COMMON_TREE_DRAFT_TENSOR_OUTPUT:
            if (key.subrole == COMMON_TREE_DRAFT_TENSOR_SUBROLE_BIAS) {
                return resolved(shape, {{V}}, "[vocab]");
            }
            return resolved(shape, {{D, V}, {V, D}}, "[hidden,vocab] or documented transpose [vocab,hidden]");
        case COMMON_TREE_DRAFT_TENSOR_OUTPUT_NORM:
        case COMMON_TREE_DRAFT_TENSOR_ATTN_NORM:
        case COMMON_TREE_DRAFT_TENSOR_FFN_NORM:
            return resolved(shape, {{D}}, "[hidden]");
        case COMMON_TREE_DRAFT_TENSOR_ATTN_Q_NORM:
        case COMMON_TREE_DRAFT_TENSOR_ATTN_K_NORM:
            return resolved(shape, {{DH}}, "[head_dim]");
        case COMMON_TREE_DRAFT_TENSOR_ATTN_Q:
            if (key.subrole == COMMON_TREE_DRAFT_TENSOR_SUBROLE_BIAS) {
                return resolved(shape, {{D}}, "[n_heads*head_dim]");
            }
            return resolved(shape, {{D, D}, {D, D}}, "[hidden,n_heads*head_dim]");
        case COMMON_TREE_DRAFT_TENSOR_ATTN_K:
        case COMMON_TREE_DRAFT_TENSOR_ATTN_V:
            if (key.subrole == COMMON_TREE_DRAFT_TENSOR_SUBROLE_BIAS) {
                return resolved(shape, {{KV}}, "[n_kv_heads*head_dim]");
            }
            return resolved(shape, {{D, KV}, {KV, D}}, "[hidden,n_kv_heads*head_dim] or documented transpose");
        case COMMON_TREE_DRAFT_TENSOR_ATTN_OUT:
            if (key.subrole == COMMON_TREE_DRAFT_TENSOR_SUBROLE_BIAS) {
                return resolved(shape, {{D}}, "[hidden]");
            }
            return resolved(shape, {{D, D}}, "[n_heads*head_dim,hidden]");
        case COMMON_TREE_DRAFT_TENSOR_FFN_GATE:
        case COMMON_TREE_DRAFT_TENSOR_FFN_UP:
            if (gemma4) {
                out.status = COMMON_TREE_DRAFT_TENSOR_SHAPE_UNRESOLVED;
                out.symbolic_expectation = "[hidden,ff_layer] where ff_layer is per-layer Gemma4 metadata";
                return out;
            }
            if (key.subrole == COMMON_TREE_DRAFT_TENSOR_SUBROLE_BIAS) return resolved(shape, {{FF}}, "[ff]");
            return resolved(shape, {{D, FF}, {FF, D}}, "[hidden,ff] or documented transpose");
        case COMMON_TREE_DRAFT_TENSOR_FFN_DOWN:
            if (gemma4) {
                out.status = COMMON_TREE_DRAFT_TENSOR_SHAPE_UNRESOLVED;
                out.symbolic_expectation = "[ff_layer,hidden] where ff_layer is per-layer Gemma4 metadata";
                return out;
            }
            if (key.subrole == COMMON_TREE_DRAFT_TENSOR_SUBROLE_BIAS) return resolved(shape, {{D}}, "[hidden]");
            return resolved(shape, {{FF, D}, {D, FF}}, "[ff,hidden] or documented transpose");
        case COMMON_TREE_DRAFT_TENSOR_FFN_ROUTER:
        case COMMON_TREE_DRAFT_TENSOR_FFN_GATE_EXPERT:
        case COMMON_TREE_DRAFT_TENSOR_FFN_UP_EXPERT:
        case COMMON_TREE_DRAFT_TENSOR_FFN_DOWN_EXPERT:
        case COMMON_TREE_DRAFT_TENSOR_FFN_GATE_EXPERTS:
        case COMMON_TREE_DRAFT_TENSOR_FFN_GATE_UP_EXPERTS:
        case COMMON_TREE_DRAFT_TENSOR_FFN_UP_EXPERTS:
        case COMMON_TREE_DRAFT_TENSOR_FFN_DOWN_EXPERTS:
            out.status = COMMON_TREE_DRAFT_TENSOR_SHAPE_UNRESOLVED;
            out.symbolic_expectation = "expert shape requires canonical expert-count/per-layer FF metadata";
            return out;
        default:
            return out;
    }
}

