#include "tree-draft-tensor-core-policy.h"

#include <cassert>

static common_tree_draft_device_capabilities make_sm89() {
    common_tree_draft_device_capabilities d{};
    d.available = true;
    d.cc_major = 8;
    d.cc_minor = 9;
    d.tensor_core_modes =
        COMMON_TREE_DRAFT_TENSOR_CORE_F16 |
        COMMON_TREE_DRAFT_TENSOR_CORE_BF16 |
        COMMON_TREE_DRAFT_TENSOR_CORE_TF32 |
        COMMON_TREE_DRAFT_TENSOR_CORE_FP8;
    return d;
}

int main() {
    common_tree_draft_tensor_core_request r{};
    r.model_dtype = COMMON_TREE_DRAFT_DTYPE_F16;
    r.candidates = {{
        {COMMON_TREE_DRAFT_TENSOR_CORE_F16,  true,true,true, 80.0f,1.0f},
        {COMMON_TREE_DRAFT_TENSOR_CORE_BF16, true,true,true, 78.0f,1.0f},
        {COMMON_TREE_DRAFT_TENSOR_CORE_TF32, true,true,true, 45.0f,1.0f},
        {COMMON_TREE_DRAFT_TENSOR_CORE_FP8,  true,true,true,120.0f,1.2f},
    }};
    common_tree_draft_tensor_core_policy p{};
    assert(common_tree_draft_tensor_core_policy_resolve(make_sm89(),r,&p) == COMMON_TREE_DRAFT_TENSOR_CORE_POLICY_OK);
    assert(p.has_selection && p.selected_mode == COMMON_TREE_DRAFT_TENSOR_CORE_F16);
    assert(p.eligible_mask == COMMON_TREE_DRAFT_TENSOR_CORE_F16);

    r.allow_precision_conversion = true;
    assert(common_tree_draft_tensor_core_policy_resolve(make_sm89(),r,&p) == COMMON_TREE_DRAFT_TENSOR_CORE_POLICY_OK);
    assert(p.has_selection && p.selected_mode == COMMON_TREE_DRAFT_TENSOR_CORE_FP8);
    assert((p.eligible_mask & COMMON_TREE_DRAFT_TENSOR_CORE_FP8) != 0);

    r.candidates[3].numerical_tolerance_ok = false;
    assert(common_tree_draft_tensor_core_policy_resolve(make_sm89(),r,&p) == COMMON_TREE_DRAFT_TENSOR_CORE_POLICY_OK);
    assert(p.selected_mode == COMMON_TREE_DRAFT_TENSOR_CORE_F16);

    r.deterministic_compare = true;
    assert(common_tree_draft_tensor_core_policy_resolve(make_sm89(),r,&p) == COMMON_TREE_DRAFT_TENSOR_CORE_POLICY_OK);
    assert(!p.has_selection && p.eligible_mask == 0);

    auto bad = make_sm89(); bad.cc_minor = 6;
    assert(common_tree_draft_tensor_core_policy_resolve(bad,r,&p) == COMMON_TREE_DRAFT_TENSOR_CORE_POLICY_NOT_SM89);
    return 0;
}
