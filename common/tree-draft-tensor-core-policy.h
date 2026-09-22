#pragma once

#include "tree-draft-precision.h"
#include "tree-draft-sm89.h"

#include <array>
#include <cstdint>

struct common_tree_draft_tensor_core_candidate_input {
    common_tree_draft_tensor_core_mode mode = COMMON_TREE_DRAFT_TENSOR_CORE_F16;
    bool library_supported = false;
    bool shape_supported = false;
    bool numerical_tolerance_ok = false;
    float effective_tflops = 0.0f;
    float conversion_overhead = 1.0f;
};

struct common_tree_draft_tensor_core_candidate {
    common_tree_draft_tensor_core_mode mode = COMMON_TREE_DRAFT_TENSOR_CORE_F16;
    bool eligible = false;
    bool native_precision = false;
    float score = 0.0f;
};

struct common_tree_draft_tensor_core_request {
    common_tree_draft_float_dtype model_dtype = COMMON_TREE_DRAFT_DTYPE_F16;
    bool deterministic_compare = false;
    bool allow_precision_conversion = false;
    std::array<common_tree_draft_tensor_core_candidate_input, 4> candidates = {};
};

struct common_tree_draft_tensor_core_policy {
    std::array<common_tree_draft_tensor_core_candidate, 4> candidates = {};
    uint32_t eligible_mask = 0;
    common_tree_draft_tensor_core_mode selected_mode = COMMON_TREE_DRAFT_TENSOR_CORE_F16;
    bool has_selection = false;
};

enum common_tree_draft_tensor_core_policy_status : uint32_t {
    COMMON_TREE_DRAFT_TENSOR_CORE_POLICY_OK = 0,
    COMMON_TREE_DRAFT_TENSOR_CORE_POLICY_NOT_SM89,
    COMMON_TREE_DRAFT_TENSOR_CORE_POLICY_INVALID_CANDIDATE,
    COMMON_TREE_DRAFT_TENSOR_CORE_POLICY_NULL_OUTPUT,
};

common_tree_draft_tensor_core_policy_status common_tree_draft_tensor_core_policy_resolve(
        const common_tree_draft_device_capabilities & device,
        const common_tree_draft_tensor_core_request & request,
        common_tree_draft_tensor_core_policy * policy);
