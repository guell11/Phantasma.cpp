#pragma once

#include "tree-draft-async-staging.h"
#include "tree-draft-quant-registry.h"
#include "tree-draft-tensor-core-policy.h"

#include <cstdint>

enum common_tree_draft_fragment_dtype : uint32_t {
    COMMON_TREE_DRAFT_FRAGMENT_F16 = 0,
    COMMON_TREE_DRAFT_FRAGMENT_BF16,
    COMMON_TREE_DRAFT_FRAGMENT_TF32,
    COMMON_TREE_DRAFT_FRAGMENT_FP8,
};

struct common_tree_draft_fused_dequant_cost {
    float load_quant_us = 0.0f;
    float dequant_shared_us = 0.0f;
    float dequant_global_us = 0.0f;
    float reload_us = 0.0f;
    float mma_us = 0.0f;
};

struct common_tree_draft_fused_dequant_request {
    const common_tree_draft_quant_type * quant = nullptr;
    bool format_supported = false;
    uint64_t logical_elements = 0;
    uint64_t encoded_bytes = 0;
    common_tree_draft_async_staging_plan staging = {};
    common_tree_draft_tensor_core_policy tensor_core = {};
    common_tree_draft_fused_dequant_cost cost = {};
};

struct common_tree_draft_fused_dequant_plan {
    bool enabled = false;
    common_tree_draft_fragment_dtype fragment_dtype = COMMON_TREE_DRAFT_FRAGMENT_F16;
    uint32_t quant_block_elems = 0;
    uint32_t quant_block_bytes = 0;
    uint64_t logical_elements = 0;
    uint64_t encoded_bytes = 0;
    float fused_cost_us = 0.0f;
    float separate_cost_us = 0.0f;
    bool writes_global_intermediate = false;
};

enum common_tree_draft_fused_dequant_status : uint32_t {
    COMMON_TREE_DRAFT_FUSED_DEQUANT_OK = 0,
    COMMON_TREE_DRAFT_FUSED_DEQUANT_INVALID,
    COMMON_TREE_DRAFT_FUSED_DEQUANT_BLOCK_MISMATCH,
    COMMON_TREE_DRAFT_FUSED_DEQUANT_TENSOR_CORE,
    COMMON_TREE_DRAFT_FUSED_DEQUANT_COST,
};

common_tree_draft_fused_dequant_status common_tree_draft_fused_dequant_staging_plan(
        const common_tree_draft_fused_dequant_request & request,
        common_tree_draft_fused_dequant_plan * plan);
