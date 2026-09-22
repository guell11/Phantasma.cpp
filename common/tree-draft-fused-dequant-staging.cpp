#include "tree-draft-fused-dequant-staging.h"

#include <cmath>
#include <limits>

static bool fragment_dtype_for(
        common_tree_draft_tensor_core_mode mode,
        common_tree_draft_fragment_dtype * dtype) {
    if (dtype == nullptr) return false;
    switch (mode) {
        case COMMON_TREE_DRAFT_TENSOR_CORE_F16:  *dtype = COMMON_TREE_DRAFT_FRAGMENT_F16; return true;
        case COMMON_TREE_DRAFT_TENSOR_CORE_BF16: *dtype = COMMON_TREE_DRAFT_FRAGMENT_BF16; return true;
        case COMMON_TREE_DRAFT_TENSOR_CORE_TF32: *dtype = COMMON_TREE_DRAFT_FRAGMENT_TF32; return true;
        case COMMON_TREE_DRAFT_TENSOR_CORE_FP8:  *dtype = COMMON_TREE_DRAFT_FRAGMENT_FP8; return true;
    }
    return false;
}

static bool finite_nonnegative(float v) {
    return std::isfinite(v) && v >= 0.0f;
}

common_tree_draft_fused_dequant_status common_tree_draft_fused_dequant_staging_plan(
        const common_tree_draft_fused_dequant_request & request,
        common_tree_draft_fused_dequant_plan * plan) {
    if (plan == nullptr || request.quant == nullptr || request.logical_elements == 0 ||
        !request.quant->fixed_block_size || request.quant->block_elems == 0 ||
        request.quant->block_bytes == 0) {
        return COMMON_TREE_DRAFT_FUSED_DEQUANT_INVALID;
    }
    if (request.logical_elements % request.quant->block_elems != 0) {
        return COMMON_TREE_DRAFT_FUSED_DEQUANT_BLOCK_MISMATCH;
    }
    const uint64_t blocks = request.logical_elements / request.quant->block_elems;
    if (blocks > std::numeric_limits<uint64_t>::max() / request.quant->block_bytes ||
        request.encoded_bytes != blocks * request.quant->block_bytes) {
        return COMMON_TREE_DRAFT_FUSED_DEQUANT_BLOCK_MISMATCH;
    }
    if (!request.tensor_core.has_selection ||
        (request.tensor_core.eligible_mask & static_cast<uint32_t>(request.tensor_core.selected_mode)) == 0) {
        return COMMON_TREE_DRAFT_FUSED_DEQUANT_TENSOR_CORE;
    }
    if (!finite_nonnegative(request.cost.load_quant_us) ||
        !finite_nonnegative(request.cost.dequant_shared_us) ||
        !finite_nonnegative(request.cost.dequant_global_us) ||
        !finite_nonnegative(request.cost.reload_us) ||
        !finite_nonnegative(request.cost.mma_us)) {
        return COMMON_TREE_DRAFT_FUSED_DEQUANT_COST;
    }

    common_tree_draft_fused_dequant_plan out;
    if (!fragment_dtype_for(request.tensor_core.selected_mode, &out.fragment_dtype)) {
        return COMMON_TREE_DRAFT_FUSED_DEQUANT_TENSOR_CORE;
    }
    out.quant_block_elems = request.quant->block_elems;
    out.quant_block_bytes = request.quant->block_bytes;
    out.logical_elements = request.logical_elements;
    out.encoded_bytes = request.encoded_bytes;
    out.fused_cost_us =
        request.cost.load_quant_us + request.cost.dequant_shared_us + request.cost.mma_us;
    out.separate_cost_us =
        request.cost.dequant_global_us + request.cost.reload_us + request.cost.mma_us;
    out.writes_global_intermediate = false;
    out.enabled = request.format_supported && out.fused_cost_us < out.separate_cost_us;
    *plan = out;
    return COMMON_TREE_DRAFT_FUSED_DEQUANT_OK;
}
