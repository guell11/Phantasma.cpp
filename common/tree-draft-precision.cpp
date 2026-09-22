#include "tree-draft-precision.h"

static bool is_input_dtype(common_tree_draft_float_dtype dtype) {
    return dtype == COMMON_TREE_DRAFT_DTYPE_F16 || dtype == COMMON_TREE_DRAFT_DTYPE_BF16;
}

static bool is_output_dtype(common_tree_draft_float_dtype dtype) {
    return is_input_dtype(dtype) || dtype == COMMON_TREE_DRAFT_DTYPE_F32;
}

common_tree_draft_precision_status common_tree_draft_precision_resolve(
        const common_tree_draft_precision_request & request,
        common_tree_draft_precision_policy * policy) {
    if (!is_input_dtype(request.q_dtype) || !is_input_dtype(request.k_dtype) ||
        !is_input_dtype(request.v_dtype)) {
        return COMMON_TREE_DRAFT_PRECISION_UNSUPPORTED_INPUT;
    }
    if (request.q_dtype != request.k_dtype) {
        return COMMON_TREE_DRAFT_PRECISION_QK_MISMATCH;
    }
    if (!is_output_dtype(request.output_dtype)) {
        return COMMON_TREE_DRAFT_PRECISION_UNSUPPORTED_OUTPUT;
    }
    if (policy == nullptr) {
        return COMMON_TREE_DRAFT_PRECISION_UNSUPPORTED_OUTPUT;
    }

    common_tree_draft_precision_policy out;
    out.q_dtype = request.q_dtype;
    out.k_dtype = request.k_dtype;
    out.v_dtype = request.v_dtype;
    out.output_dtype = request.output_dtype;
    out.softmax_dtype = COMMON_TREE_DRAFT_DTYPE_F32;
    out.output_accumulator_dtype = COMMON_TREE_DRAFT_DTYPE_F32;
    out.product_mode = request.deterministic_compare ?
        COMMON_TREE_DRAFT_PRODUCT_FP32_REFERENCE :
        COMMON_TREE_DRAFT_PRODUCT_INPUT_PRECISION;
    out.reject_visible_nan_inf = true;
    out.masked_nan_inf_ignored = true;
    *policy = out;
    return COMMON_TREE_DRAFT_PRECISION_OK;
}
