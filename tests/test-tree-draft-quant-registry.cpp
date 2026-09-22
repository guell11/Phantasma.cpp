#include "tree-draft-quant-registry.h"

#include <cassert>
#include <cmath>

int main() {
    const auto & registry = common_tree_draft_quant_registry();
    assert(!registry.empty());
    assert(common_tree_draft_quant_registry_validate(registry) == COMMON_TREE_DRAFT_QUANT_REGISTRY_OK);

    const auto * q4 = common_tree_draft_quant_find_ggml(GGML_TYPE_Q4_0);
    assert(q4 != nullptr);
    assert(q4->block_elems == static_cast<uint32_t>(ggml_blck_size(GGML_TYPE_Q4_0)));
    assert(q4->block_bytes == static_cast<uint32_t>(ggml_type_size(GGML_TYPE_Q4_0)));
    assert(std::fabs(common_tree_draft_quant_bits_per_weight(*q4) - 8.0 * ggml_type_size(GGML_TYPE_Q4_0) / ggml_blck_size(GGML_TYPE_Q4_0)) < 1e-12);

    const auto * f16 = common_tree_draft_quant_find_ggml(GGML_TYPE_F16);
    assert(f16 != nullptr && f16->block_elems == 1 && f16->logical_dtype == "f16");
    assert(common_tree_draft_quant_find_name("gptq") != nullptr);
    assert(common_tree_draft_quant_find_name("awq") != nullptr);
    assert(common_tree_draft_quant_find_name("bnb.nf4") != nullptr);

    auto duplicate = registry;
    duplicate.push_back(registry.front());
    assert(common_tree_draft_quant_registry_validate(duplicate) == COMMON_TREE_DRAFT_QUANT_REGISTRY_DUPLICATE_ID);
    return 0;
}

