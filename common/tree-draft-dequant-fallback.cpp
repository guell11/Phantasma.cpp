#include "tree-draft-dequant-fallback.h"

#include <limits>

static uint32_t dtype_bytes(common_tree_draft_float_dtype dtype) {
    switch (dtype) {
        case COMMON_TREE_DRAFT_DTYPE_F16:
        case COMMON_TREE_DRAFT_DTYPE_BF16: return 2;
        case COMMON_TREE_DRAFT_DTYPE_F32:  return 4;
    }
    return 0;
}

static bool budget_valid(const common_tree_draft_fallback_budget & budget) {
    return budget.committed_bytes <= budget.limit_bytes &&
           budget.reserved_bytes <= budget.limit_bytes - budget.committed_bytes;
}

common_tree_draft_dequant_fallback_status common_tree_draft_dequant_fallback_plan_build(
        const common_tree_draft_dequant_fallback_request & request,
        const common_tree_draft_fallback_budget & budget,
        common_tree_draft_dequant_fallback_plan * plan) {
    if (plan == nullptr || request.quant == nullptr || request.logical_elements == 0 ||
        !budget_valid(budget)) {
        return COMMON_TREE_DRAFT_DEQUANT_FALLBACK_INVALID;
    }
    if (!request.policy_permits) return COMMON_TREE_DRAFT_DEQUANT_FALLBACK_POLICY;
    const uint32_t bytes_per_element = dtype_bytes(request.output_dtype);
    if (bytes_per_element == 0 ||
        request.logical_elements > std::numeric_limits<uint64_t>::max() / bytes_per_element) {
        return COMMON_TREE_DRAFT_DEQUANT_FALLBACK_INVALID;
    }
    const uint64_t bytes = request.logical_elements * bytes_per_element;
    const uint64_t available = budget.limit_bytes - budget.committed_bytes - budget.reserved_bytes;
    if (bytes > available) return COMMON_TREE_DRAFT_DEQUANT_FALLBACK_RESOURCE;

    *plan = {true, bytes, request.output_dtype, request.quant->id};
    return COMMON_TREE_DRAFT_DEQUANT_FALLBACK_OK;
}

common_tree_draft_dequant_fallback_status common_tree_draft_dequant_fallback_reserve(
        common_tree_draft_fallback_budget * budget,
        const common_tree_draft_dequant_fallback_plan & plan,
        common_tree_draft_dequant_fallback_reservation * reservation) {
    if (budget == nullptr || reservation == nullptr || !budget_valid(*budget) ||
        !plan.materialize || plan.output_bytes == 0) {
        return COMMON_TREE_DRAFT_DEQUANT_FALLBACK_INVALID;
    }
    if (reservation->active) return COMMON_TREE_DRAFT_DEQUANT_FALLBACK_STATE;
    const uint64_t available = budget->limit_bytes - budget->committed_bytes - budget->reserved_bytes;
    if (plan.output_bytes > available) return COMMON_TREE_DRAFT_DEQUANT_FALLBACK_RESOURCE;
    budget->reserved_bytes += plan.output_bytes;
    reservation->bytes = plan.output_bytes;
    reservation->active = true;
    return COMMON_TREE_DRAFT_DEQUANT_FALLBACK_OK;
}

common_tree_draft_dequant_fallback_status common_tree_draft_dequant_fallback_commit(
        common_tree_draft_fallback_budget * budget,
        common_tree_draft_dequant_fallback_reservation * reservation) {
    if (budget == nullptr || reservation == nullptr || !reservation->active ||
        reservation->bytes > budget->reserved_bytes) {
        return COMMON_TREE_DRAFT_DEQUANT_FALLBACK_STATE;
    }
    budget->reserved_bytes -= reservation->bytes;
    budget->committed_bytes += reservation->bytes;
    reservation->active = false;
    reservation->bytes = 0;
    return COMMON_TREE_DRAFT_DEQUANT_FALLBACK_OK;
}

common_tree_draft_dequant_fallback_status common_tree_draft_dequant_fallback_abort(
        common_tree_draft_fallback_budget * budget,
        common_tree_draft_dequant_fallback_reservation * reservation) {
    if (budget == nullptr || reservation == nullptr || !reservation->active ||
        reservation->bytes > budget->reserved_bytes) {
        return COMMON_TREE_DRAFT_DEQUANT_FALLBACK_STATE;
    }
    budget->reserved_bytes -= reservation->bytes;
    reservation->active = false;
    reservation->bytes = 0;
    return COMMON_TREE_DRAFT_DEQUANT_FALLBACK_OK;
}
