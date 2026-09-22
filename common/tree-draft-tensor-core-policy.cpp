#include "tree-draft-tensor-core-policy.h"

#include <cmath>

static bool valid_mode(common_tree_draft_tensor_core_mode mode) {
    return mode == COMMON_TREE_DRAFT_TENSOR_CORE_F16 ||
           mode == COMMON_TREE_DRAFT_TENSOR_CORE_BF16 ||
           mode == COMMON_TREE_DRAFT_TENSOR_CORE_TF32 ||
           mode == COMMON_TREE_DRAFT_TENSOR_CORE_FP8;
}

static bool native_mode(common_tree_draft_float_dtype dtype, common_tree_draft_tensor_core_mode mode) {
    switch (dtype) {
        case COMMON_TREE_DRAFT_DTYPE_F16:  return mode == COMMON_TREE_DRAFT_TENSOR_CORE_F16;
        case COMMON_TREE_DRAFT_DTYPE_BF16: return mode == COMMON_TREE_DRAFT_TENSOR_CORE_BF16;
        case COMMON_TREE_DRAFT_DTYPE_F32:  return false;
    }
    return false;
}

common_tree_draft_tensor_core_policy_status common_tree_draft_tensor_core_policy_resolve(
        const common_tree_draft_device_capabilities & device,
        const common_tree_draft_tensor_core_request & request,
        common_tree_draft_tensor_core_policy * policy) {
    if (policy == nullptr) return COMMON_TREE_DRAFT_TENSOR_CORE_POLICY_NULL_OUTPUT;
    if (!common_tree_draft_device_is_sm89(device)) return COMMON_TREE_DRAFT_TENSOR_CORE_POLICY_NOT_SM89;

    common_tree_draft_tensor_core_policy out;
    float best_score = -1.0f;
    for (size_t i = 0; i < request.candidates.size(); ++i) {
        const auto & in = request.candidates[i];
        if (!valid_mode(in.mode) || !std::isfinite(in.effective_tflops) || in.effective_tflops < 0.0f ||
            !std::isfinite(in.conversion_overhead) || in.conversion_overhead < 1.0f) {
            return COMMON_TREE_DRAFT_TENSOR_CORE_POLICY_INVALID_CANDIDATE;
        }
        auto & c = out.candidates[i];
        c.mode = in.mode;
        c.native_precision = native_mode(request.model_dtype, in.mode);
        const bool hw = (device.tensor_core_modes & static_cast<uint32_t>(in.mode)) != 0;
        const bool precision_allowed = c.native_precision || request.allow_precision_conversion;
        c.eligible = !request.deterministic_compare && hw && in.library_supported &&
                     in.shape_supported && in.numerical_tolerance_ok && precision_allowed;
        c.score = c.eligible ? in.effective_tflops / in.conversion_overhead : 0.0f;
        if (c.eligible) {
            out.eligible_mask |= static_cast<uint32_t>(in.mode);
            if (!out.has_selection || c.score > best_score) {
                out.has_selection = true;
                out.selected_mode = in.mode;
                best_score = c.score;
            }
        }
    }
    *policy = out;
    return COMMON_TREE_DRAFT_TENSOR_CORE_POLICY_OK;
}
