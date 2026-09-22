#include "tree-draft-quant-registry.h"

#include <cmath>

static common_tree_draft_quant_type ggml_entry(enum ggml_type type) {
    const int64_t block = ggml_blck_size(type);
    const size_t bytes = ggml_type_size(type);
    const bool dense = block == 1;
    return {
        static_cast<uint32_t>(type),
        ggml_type_name(type),
        COMMON_TREE_DRAFT_QUANT_GGML,
        static_cast<uint32_t>(block),
        static_cast<uint32_t>(bytes),
        dense ? "none" : "format-defined",
        dense ? "none" : "format-defined",
        dense ? COMMON_TREE_DRAFT_QUANT_PACKING_DENSE : COMMON_TREE_DRAFT_QUANT_PACKING_BLOCK,
        (type == GGML_TYPE_F64 ? "f64" : type == GGML_TYPE_F32 ? "f32" : type == GGML_TYPE_F16 ? "f16" : type == GGML_TYPE_BF16 ? "bf16" : "f32"),
        true,
        static_cast<int32_t>(type),
    };
}

static std::vector<common_tree_draft_quant_type> build_registry() {
    std::vector<common_tree_draft_quant_type> r;
    const enum ggml_type native[] = {
        GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1,
        GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0, GGML_TYPE_Q8_1,
        GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K,
        GGML_TYPE_Q6_K, GGML_TYPE_Q8_K, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS,
        GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ1_S, GGML_TYPE_IQ4_NL, GGML_TYPE_IQ3_S,
        GGML_TYPE_IQ2_S, GGML_TYPE_IQ4_XS, GGML_TYPE_I8, GGML_TYPE_I16,
        GGML_TYPE_I32, GGML_TYPE_I64, GGML_TYPE_F64, GGML_TYPE_IQ1_M,
        GGML_TYPE_BF16, GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0, GGML_TYPE_MXFP4,
        GGML_TYPE_NVFP4, GGML_TYPE_Q1_0, GGML_TYPE_Q2_0,
    };
    for (enum ggml_type type : native) r.push_back(ggml_entry(type));

    uint32_t id = 0x10000u;
    r.push_back({id++, "gptq", COMMON_TREE_DRAFT_QUANT_GPTQ, 1, 0, "f16/f32", "packed-int", COMMON_TREE_DRAFT_QUANT_PACKING_GROUPED, "f16/f32", false, -1});
    r.push_back({id++, "awq", COMMON_TREE_DRAFT_QUANT_AWQ, 1, 0, "f16/f32", "packed-int", COMMON_TREE_DRAFT_QUANT_PACKING_GROUPED, "f16/f32", false, -1});
    r.push_back({id++, "bnb.int8", COMMON_TREE_DRAFT_QUANT_BNB8, 1, 1, "format-defined", "none", COMMON_TREE_DRAFT_QUANT_PACKING_GROUPED, "f16/f32", false, -1});
    r.push_back({id++, "bnb.nf4", COMMON_TREE_DRAFT_QUANT_BNB4, 1, 0, "absmax", "codebook", COMMON_TREE_DRAFT_QUANT_PACKING_CODEBOOK, "f16/f32", false, -1});
    r.push_back({id++, "bnb.fp4", COMMON_TREE_DRAFT_QUANT_BNB4, 1, 0, "absmax", "codebook", COMMON_TREE_DRAFT_QUANT_PACKING_CODEBOOK, "f16/f32", false, -1});
    return r;
}

const std::vector<common_tree_draft_quant_type> & common_tree_draft_quant_registry() {
    static const std::vector<common_tree_draft_quant_type> registry = build_registry();
    return registry;
}

common_tree_draft_quant_registry_status common_tree_draft_quant_registry_validate(
        const std::vector<common_tree_draft_quant_type> & registry,
        size_t * error_index) {
    for (size_t i = 0; i < registry.size(); ++i) {
        const auto & q = registry[i];
        if (q.name.empty() || q.block_elems == 0 || (q.fixed_block_size && q.block_bytes == 0)) {
            if (error_index) *error_index = i;
            return COMMON_TREE_DRAFT_QUANT_REGISTRY_INVALID_BLOCK;
        }
        for (size_t j = 0; j < i; ++j) {
            if (registry[j].id == q.id) {
                if (error_index) *error_index = i;
                return COMMON_TREE_DRAFT_QUANT_REGISTRY_DUPLICATE_ID;
            }
            if (registry[j].name == q.name) {
                if (error_index) *error_index = i;
                return COMMON_TREE_DRAFT_QUANT_REGISTRY_DUPLICATE_NAME;
            }
        }
    }
    return COMMON_TREE_DRAFT_QUANT_REGISTRY_OK;
}

const common_tree_draft_quant_type * common_tree_draft_quant_find_name(const std::string & name) {
    for (const auto & q : common_tree_draft_quant_registry()) if (q.name == name) return &q;
    return nullptr;
}

const common_tree_draft_quant_type * common_tree_draft_quant_find_ggml(enum ggml_type type) {
    for (const auto & q : common_tree_draft_quant_registry()) if (q.ggml_type_id == static_cast<int32_t>(type)) return &q;
    return nullptr;
}

double common_tree_draft_quant_bits_per_weight(const common_tree_draft_quant_type & type) {
    if (!type.fixed_block_size || type.block_elems == 0 || type.block_bytes == 0) return 0.0;
    return 8.0 * static_cast<double>(type.block_bytes) / static_cast<double>(type.block_elems);
}

