#include "tree-draft-precision.h"

#include <cassert>

int main() {
    common_tree_draft_precision_policy p{};
    common_tree_draft_precision_request r{};
    r.q_dtype = COMMON_TREE_DRAFT_DTYPE_F16;
    r.k_dtype = COMMON_TREE_DRAFT_DTYPE_F16;
    r.v_dtype = COMMON_TREE_DRAFT_DTYPE_BF16;
    r.output_dtype = COMMON_TREE_DRAFT_DTYPE_F16;
    assert(common_tree_draft_precision_resolve(r,&p) == COMMON_TREE_DRAFT_PRECISION_OK);
    assert(p.product_mode == COMMON_TREE_DRAFT_PRODUCT_INPUT_PRECISION);
    assert(p.softmax_dtype == COMMON_TREE_DRAFT_DTYPE_F32);
    assert(p.output_accumulator_dtype == COMMON_TREE_DRAFT_DTYPE_F32);
    assert(p.output_dtype == COMMON_TREE_DRAFT_DTYPE_F16);
    assert(p.reject_visible_nan_inf && p.masked_nan_inf_ignored);

    r.deterministic_compare = true;
    r.output_dtype = COMMON_TREE_DRAFT_DTYPE_F32;
    assert(common_tree_draft_precision_resolve(r,&p) == COMMON_TREE_DRAFT_PRECISION_OK);
    assert(p.product_mode == COMMON_TREE_DRAFT_PRODUCT_FP32_REFERENCE);
    assert(p.output_dtype == COMMON_TREE_DRAFT_DTYPE_F32);

    r.q_dtype = COMMON_TREE_DRAFT_DTYPE_BF16;
    r.k_dtype = COMMON_TREE_DRAFT_DTYPE_F16;
    assert(common_tree_draft_precision_resolve(r,&p) == COMMON_TREE_DRAFT_PRECISION_QK_MISMATCH);

    r.q_dtype = COMMON_TREE_DRAFT_DTYPE_F32;
    r.k_dtype = COMMON_TREE_DRAFT_DTYPE_F32;
    assert(common_tree_draft_precision_resolve(r,&p) == COMMON_TREE_DRAFT_PRECISION_UNSUPPORTED_INPUT);
    return 0;
}
