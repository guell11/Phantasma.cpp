#include "tree-draft-quant-matmul-dispatch.h"

#include <limits>

common_tree_draft_quant_matmul_status common_tree_draft_quant_matmul_plan_build(
        const common_tree_draft_device_capabilities & device,
        const common_tree_draft_kernel_capability_entry * entries,
        size_t entry_count,
        const common_tree_draft_quant_matmul_request & request,
        common_tree_draft_quant_matmul_plan * plan) {
    if (plan == nullptr || request.capability_query.quant == nullptr ||
        request.m == 0 || request.n == 0 || request.k == 0 ||
        request.n > std::numeric_limits<uint64_t>::max() / request.k) {
        return COMMON_TREE_DRAFT_QUANT_MATMUL_INVALID;
    }

    common_tree_draft_quant_matmul_plan out;
    auto query = request.capability_query;
    query.n = request.n;
    query.k = request.k;
    const auto cap_status = common_tree_draft_kernel_capability_lookup(
        device, entries, entry_count, query, &out.native);
    if (cap_status != COMMON_TREE_DRAFT_KERNEL_CAPABILITY_OK) {
        return COMMON_TREE_DRAFT_QUANT_MATMUL_CAPABILITY;
    }
    if (out.native.supported) {
        out.path = COMMON_TREE_DRAFT_QUANT_MATMUL_NATIVE;
        *plan = std::move(out);
        return COMMON_TREE_DRAFT_QUANT_MATMUL_OK;
    }

    const uint64_t logical_elements = request.n * request.k;
    const auto dstatus = common_tree_draft_dequant_plan_build(
        *request.capability_query.quant,
        logical_elements,
        request.workspace_budget,
        request.dequant_output_element_bytes,
        request.dequant_mode,
        &out.dequant);
    out.path = dstatus == COMMON_TREE_DRAFT_DEQUANT_PLAN_OK
        ? COMMON_TREE_DRAFT_QUANT_MATMUL_DEQUANT_ON_THE_FLY
        : COMMON_TREE_DRAFT_QUANT_MATMUL_UNSUPPORTED;
    *plan = std::move(out);
    return COMMON_TREE_DRAFT_QUANT_MATMUL_OK;
}
