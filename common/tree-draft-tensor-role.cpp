#include "tree-draft-tensor-role.h"

#include <cctype>
#include <limits>
#include <regex>

static bool common_tree_draft_tensor_arch_supported(const std::string & arch) {
    return arch == "llama" || arch == "mistral" || arch == "gemma" || arch == "gemma2" ||
           arch == "gemma3" || arch == "gemma4" || arch == "qwen2" || arch == "qwen3" || arch == "phi3";
}

static common_tree_draft_tensor_subrole common_tree_draft_tensor_suffix(const std::string & suffix) {
    if (suffix.empty()) return COMMON_TREE_DRAFT_TENSOR_SUBROLE_NONE;
    if (suffix == "weight") return COMMON_TREE_DRAFT_TENSOR_SUBROLE_WEIGHT;
    if (suffix == "bias") return COMMON_TREE_DRAFT_TENSOR_SUBROLE_BIAS;
    return COMMON_TREE_DRAFT_TENSOR_SUBROLE_NONE;
}

static bool common_tree_draft_parse_index(const std::string & text, int32_t * value) {
    if (text.empty() || value == nullptr) return false;
    uint64_t parsed = 0;
    for (char c : text) {
        if (!std::isdigit(static_cast<unsigned char>(c))) return false;
        parsed = parsed * 10u + static_cast<uint32_t>(c - '0');
        if (parsed > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) return false;
    }
    *value = static_cast<int32_t>(parsed);
    return true;
}

static common_tree_draft_tensor_role_status common_tree_draft_set(
        common_tree_draft_tensor_role_result * result,
        const std::string & source,
        common_tree_draft_tensor_role role,
        int32_t layer,
        int32_t expert,
        common_tree_draft_tensor_subrole subrole) {
    result->key = { role, layer, expert, subrole };
    result->source_name = source;
    return COMMON_TREE_DRAFT_TENSOR_ROLE_OK;
}

static bool common_tree_draft_hf_alias_arch(const std::string & arch) {
    return arch == "llama" || arch == "mistral" || arch == "gemma" || arch == "gemma2" ||
           arch == "gemma3" || arch == "gemma4" || arch == "qwen2" || arch == "qwen3";
}

common_tree_draft_tensor_role_status common_tree_draft_tensor_role_canonicalize(
        const std::string & architecture,
        const std::string & name,
        common_tree_draft_tensor_role_result * result) {
    if (result == nullptr || name.empty()) return COMMON_TREE_DRAFT_TENSOR_ROLE_INVALID_NAME;
    if (!common_tree_draft_tensor_arch_supported(architecture)) return COMMON_TREE_DRAFT_TENSOR_ROLE_UNSUPPORTED_ARCH;
    result->key = {};
    result->source_name = name;

    std::smatch m;
    static const std::regex root_re(R"(^(token_embd|output_norm|output)(?:\.(weight|bias))?$)");
    if (std::regex_match(name, m, root_re)) {
        common_tree_draft_tensor_role role = COMMON_TREE_DRAFT_TENSOR_UNKNOWN;
        if (m[1] == "token_embd") role = COMMON_TREE_DRAFT_TENSOR_TOKEN_EMBEDDING;
        if (m[1] == "output_norm") role = COMMON_TREE_DRAFT_TENSOR_OUTPUT_NORM;
        if (m[1] == "output") role = COMMON_TREE_DRAFT_TENSOR_OUTPUT;
        return common_tree_draft_set(result, name, role, -1, -1, common_tree_draft_tensor_suffix(m[2].str()));
    }

    static const std::regex layer_re(
        R"(^blk\.([0-9]+)\.(attn_norm|attn_q|attn_k|attn_v|attn_output|attn_q_norm|attn_k_norm|ffn_norm|ffn_gate_inp|ffn_gate|ffn_up|ffn_down|ffn_gate_exps|ffn_gate_up_exps|ffn_up_exps|ffn_down_exps)(?:\.(weight|bias))?$)");
    if (std::regex_match(name, m, layer_re)) {
        int32_t layer = -1;
        if (!common_tree_draft_parse_index(m[1].str(), &layer)) return COMMON_TREE_DRAFT_TENSOR_ROLE_INDEX_RANGE;
        const std::string kind = m[2].str();
        common_tree_draft_tensor_role role = COMMON_TREE_DRAFT_TENSOR_UNKNOWN;
        if (kind == "attn_norm") role = COMMON_TREE_DRAFT_TENSOR_ATTN_NORM;
        else if (kind == "attn_q") role = COMMON_TREE_DRAFT_TENSOR_ATTN_Q;
        else if (kind == "attn_k") role = COMMON_TREE_DRAFT_TENSOR_ATTN_K;
        else if (kind == "attn_v") role = COMMON_TREE_DRAFT_TENSOR_ATTN_V;
        else if (kind == "attn_output") role = COMMON_TREE_DRAFT_TENSOR_ATTN_OUT;
        else if (kind == "attn_q_norm") role = COMMON_TREE_DRAFT_TENSOR_ATTN_Q_NORM;
        else if (kind == "attn_k_norm") role = COMMON_TREE_DRAFT_TENSOR_ATTN_K_NORM;
        else if (kind == "ffn_norm") role = COMMON_TREE_DRAFT_TENSOR_FFN_NORM;
        else if (kind == "ffn_gate_inp") role = COMMON_TREE_DRAFT_TENSOR_FFN_ROUTER;
        else if (kind == "ffn_gate") role = COMMON_TREE_DRAFT_TENSOR_FFN_GATE;
        else if (kind == "ffn_up") role = COMMON_TREE_DRAFT_TENSOR_FFN_UP;
        else if (kind == "ffn_down") role = COMMON_TREE_DRAFT_TENSOR_FFN_DOWN;
        else if (kind == "ffn_gate_exps") role = COMMON_TREE_DRAFT_TENSOR_FFN_GATE_EXPERTS;
        else if (kind == "ffn_gate_up_exps") role = COMMON_TREE_DRAFT_TENSOR_FFN_GATE_UP_EXPERTS;
        else if (kind == "ffn_up_exps") role = COMMON_TREE_DRAFT_TENSOR_FFN_UP_EXPERTS;
        else if (kind == "ffn_down_exps") role = COMMON_TREE_DRAFT_TENSOR_FFN_DOWN_EXPERTS;
        return common_tree_draft_set(result, name, role, layer, -1, common_tree_draft_tensor_suffix(m[3].str()));
    }

    static const std::regex expert_re(R"(^blk\.([0-9]+)\.(ffn_gate|ffn_up|ffn_down)\.([0-9]+)(?:\.(weight|bias))?$)");
    if (std::regex_match(name, m, expert_re)) {
        int32_t layer = -1;
        int32_t expert = -1;
        if (!common_tree_draft_parse_index(m[1].str(), &layer) || !common_tree_draft_parse_index(m[3].str(), &expert)) {
            return COMMON_TREE_DRAFT_TENSOR_ROLE_INDEX_RANGE;
        }
        common_tree_draft_tensor_role role = COMMON_TREE_DRAFT_TENSOR_UNKNOWN;
        if (m[2] == "ffn_gate") role = COMMON_TREE_DRAFT_TENSOR_FFN_GATE_EXPERT;
        else if (m[2] == "ffn_up") role = COMMON_TREE_DRAFT_TENSOR_FFN_UP_EXPERT;
        else if (m[2] == "ffn_down") role = COMMON_TREE_DRAFT_TENSOR_FFN_DOWN_EXPERT;
        return common_tree_draft_set(result, name, role, layer, expert, common_tree_draft_tensor_suffix(m[4].str()));
    }

    if (common_tree_draft_hf_alias_arch(architecture)) {
        if (name == "model.embed_tokens.weight") return common_tree_draft_set(result, name, COMMON_TREE_DRAFT_TENSOR_TOKEN_EMBEDDING, -1, -1, COMMON_TREE_DRAFT_TENSOR_SUBROLE_WEIGHT);
        if (name == "model.norm.weight") return common_tree_draft_set(result, name, COMMON_TREE_DRAFT_TENSOR_OUTPUT_NORM, -1, -1, COMMON_TREE_DRAFT_TENSOR_SUBROLE_WEIGHT);
        if (name == "lm_head.weight") return common_tree_draft_set(result, name, COMMON_TREE_DRAFT_TENSOR_OUTPUT, -1, -1, COMMON_TREE_DRAFT_TENSOR_SUBROLE_WEIGHT);

        static const std::regex hf_layer_re(
            R"(^model\.layers\.([0-9]+)\.(input_layernorm|post_attention_layernorm|self_attn\.q_proj|self_attn\.k_proj|self_attn\.v_proj|self_attn\.o_proj|self_attn\.q_norm|self_attn\.k_norm|mlp\.gate_proj|mlp\.up_proj|mlp\.down_proj|mlp\.gate)(?:\.(weight|bias))$)");
        if (std::regex_match(name, m, hf_layer_re)) {
            int32_t layer = -1;
            if (!common_tree_draft_parse_index(m[1].str(), &layer)) return COMMON_TREE_DRAFT_TENSOR_ROLE_INDEX_RANGE;
            const std::string kind = m[2].str();
            common_tree_draft_tensor_role role = COMMON_TREE_DRAFT_TENSOR_UNKNOWN;
            if (kind == "input_layernorm") role = COMMON_TREE_DRAFT_TENSOR_ATTN_NORM;
            else if (kind == "post_attention_layernorm") role = COMMON_TREE_DRAFT_TENSOR_FFN_NORM;
            else if (kind == "self_attn.q_proj") role = COMMON_TREE_DRAFT_TENSOR_ATTN_Q;
            else if (kind == "self_attn.k_proj") role = COMMON_TREE_DRAFT_TENSOR_ATTN_K;
            else if (kind == "self_attn.v_proj") role = COMMON_TREE_DRAFT_TENSOR_ATTN_V;
            else if (kind == "self_attn.o_proj") role = COMMON_TREE_DRAFT_TENSOR_ATTN_OUT;
            else if (kind == "self_attn.q_norm") role = COMMON_TREE_DRAFT_TENSOR_ATTN_Q_NORM;
            else if (kind == "self_attn.k_norm") role = COMMON_TREE_DRAFT_TENSOR_ATTN_K_NORM;
            else if (kind == "mlp.gate_proj") role = COMMON_TREE_DRAFT_TENSOR_FFN_GATE;
            else if (kind == "mlp.up_proj") role = COMMON_TREE_DRAFT_TENSOR_FFN_UP;
            else if (kind == "mlp.down_proj") role = COMMON_TREE_DRAFT_TENSOR_FFN_DOWN;
            else if (kind == "mlp.gate") role = COMMON_TREE_DRAFT_TENSOR_FFN_ROUTER;
            return common_tree_draft_set(result, name, role, layer, -1, common_tree_draft_tensor_suffix(m[3].str()));
        }
    }

    return COMMON_TREE_DRAFT_TENSOR_ROLE_OK;
}

bool common_tree_draft_tensor_role_key_equal(
        const common_tree_draft_tensor_role_key & a,
        const common_tree_draft_tensor_role_key & b) {
    return a.role == b.role && a.layer == b.layer && a.expert == b.expert && a.subrole == b.subrole;
}

common_tree_draft_tensor_role_status common_tree_draft_tensor_role_validate_unique(
        const common_tree_draft_tensor_role_result * results,
        size_t count,
        size_t * first_collision,
        size_t * second_collision) {
    if (count > 0 && results == nullptr) return COMMON_TREE_DRAFT_TENSOR_ROLE_INVALID_NAME;
    for (size_t i = 0; i < count; ++i) {
        if (results[i].key.role == COMMON_TREE_DRAFT_TENSOR_UNKNOWN) continue;
        for (size_t j = 0; j < i; ++j) {
            if (results[j].key.role != COMMON_TREE_DRAFT_TENSOR_UNKNOWN &&
                common_tree_draft_tensor_role_key_equal(results[i].key, results[j].key)) {
                if (first_collision != nullptr) *first_collision = j;
                if (second_collision != nullptr) *second_collision = i;
                return COMMON_TREE_DRAFT_TENSOR_ROLE_COLLISION;
            }
        }
    }
    return COMMON_TREE_DRAFT_TENSOR_ROLE_OK;
}

