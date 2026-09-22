#include "tree-draft-dequant.h"

#include <cassert>
#include <cmath>
#include <vector>

int main() {
    const auto * q = common_tree_draft_quant_find_ggml(GGML_TYPE_Q4_0);
    assert(q != nullptr);
    const int64_t n = ggml_blck_size(GGML_TYPE_Q4_0);
    std::vector<float> reference(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) reference[static_cast<size_t>(i)] = std::sin(static_cast<float>(i) * 0.2f);
    const ggml_type_traits * traits = ggml_get_type_traits(GGML_TYPE_Q4_0);
    assert(traits != nullptr && traits->from_float_ref != nullptr && traits->to_float != nullptr);
    std::vector<uint8_t> encoded(ggml_type_size(GGML_TYPE_Q4_0));
    traits->from_float_ref(reference.data(), encoded.data(), n);
    std::vector<float> actual(static_cast<size_t>(n));
    assert(common_tree_draft_dequant_cpu(*q, encoded.data(), encoded.size(), static_cast<uint64_t>(n), COMMON_TREE_DRAFT_DEQUANT_F32,
                actual.data(), actual.size() * sizeof(float)) == COMMON_TREE_DRAFT_DEQUANT_OK);
    const double error = common_tree_draft_dequant_max_abs_error(actual.data(), reference.data(), actual.size());
    assert(std::isfinite(error) && error <= common_tree_draft_dequant_default_tolerance(*q));

    const auto * f32 = common_tree_draft_quant_find_ggml(GGML_TYPE_F32);
    assert(f32 != nullptr);
    float input[] = {1.0f, -2.0f, 3.5f};
    float output[3] = {};
    assert(common_tree_draft_dequant_cpu(*f32, input, sizeof(input), 3, COMMON_TREE_DRAFT_DEQUANT_F32, output, sizeof(output)) == COMMON_TREE_DRAFT_DEQUANT_OK);
    assert(output[0] == input[0] && output[1] == input[1] && output[2] == input[2]);

    const auto * gptq = common_tree_draft_quant_find_name("gptq");
    assert(gptq != nullptr);
    assert(common_tree_draft_dequant_cpu(*gptq, encoded.data(), encoded.size(), static_cast<uint64_t>(n), COMMON_TREE_DRAFT_DEQUANT_F32,
                actual.data(), actual.size() * sizeof(float)) == COMMON_TREE_DRAFT_DEQUANT_UNSUPPORTED);
    return 0;
}

