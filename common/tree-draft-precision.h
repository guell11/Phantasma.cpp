#pragma once

#include <cstdint>

enum common_tree_draft_float_dtype : uint32_t {
    COMMON_TREE_DRAFT_DTYPE_F16 = 0,
    COMMON_TREE_DRAFT_DTYPE_BF16,
    COMMON_TREE_DRAFT_DTYPE_F32,
};

enum common_tree_draft_product_mode : uint32_t {
    COMMON_TREE_DRAFT_PRODUCT_INPUT_PRECISION = 0,
    COMMON_TREE_DRAFT_PRODUCT_FP32_REFERENCE,
};

struct common_tree_draft_precision_request {
    common_tree_draft_float_dtype q_dtype = COMMON_TREE_DRAFT_DTYPE_F16;
    common_tree_draft_float_dtype k_dtype = COMMON_TREE_DRAFT_DTYPE_F16;
    common_tree_draft_float_dtype v_dtype = COMMON_TREE_DRAFT_DTYPE_F16;
    common_tree_draft_float_dtype output_dtype = COMMON_TREE_DRAFT_DTYPE_F16;
    bool deterministic_compare = false;
};

struct common_tree_draft_precision_policy {
    common_tree_draft_float_dtype q_dtype = COMMON_TREE_DRAFT_DTYPE_F16;
    common_tree_draft_float_dtype k_dtype = COMMON_TREE_DRAFT_DTYPE_F16;
    common_tree_draft_float_dtype v_dtype = COMMON_TREE_DRAFT_DTYPE_F16;
    common_tree_draft_float_dtype output_dtype = COMMON_TREE_DRAFT_DTYPE_F16;
    common_tree_draft_float_dtype softmax_dtype = COMMON_TREE_DRAFT_DTYPE_F32;
    common_tree_draft_float_dtype output_accumulator_dtype = COMMON_TREE_DRAFT_DTYPE_F32;
    common_tree_draft_product_mode product_mode = COMMON_TREE_DRAFT_PRODUCT_INPUT_PRECISION;
    bool reject_visible_nan_inf = true;
    bool masked_nan_inf_ignored = true;
};

enum common_tree_draft_precision_status : uint32_t {
    COMMON_TREE_DRAFT_PRECISION_OK = 0,
    COMMON_TREE_DRAFT_PRECISION_UNSUPPORTED_INPUT,
    COMMON_TREE_DRAFT_PRECISION_QK_MISMATCH,
    COMMON_TREE_DRAFT_PRECISION_UNSUPPORTED_OUTPUT,
};

common_tree_draft_precision_status common_tree_draft_precision_resolve(
        const common_tree_draft_precision_request & request,
        common_tree_draft_precision_policy * policy);
