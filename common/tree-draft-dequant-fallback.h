#pragma once

#include "tree-draft-precision.h"
#include "tree-draft-quant-registry.h"

#include <cstdint>

struct common_tree_draft_fallback_budget {
    uint64_t limit_bytes = 0;
    uint64_t committed_bytes = 0;
    uint64_t reserved_bytes = 0;
};

struct common_tree_draft_dequant_fallback_request {
    const common_tree_draft_quant_type * quant = nullptr;
    uint64_t logical_elements = 0;
    common_tree_draft_float_dtype output_dtype = COMMON_TREE_DRAFT_DTYPE_F16;
    bool policy_permits = false;
};

struct common_tree_draft_dequant_fallback_plan {
    bool materialize = false;
    uint64_t output_bytes = 0;
    common_tree_draft_float_dtype output_dtype = COMMON_TREE_DRAFT_DTYPE_F16;
    uint32_t quant_id = 0;
};

struct common_tree_draft_dequant_fallback_reservation {
    uint64_t bytes = 0;
    bool active = false;
};

enum common_tree_draft_dequant_fallback_status : uint32_t {
    COMMON_TREE_DRAFT_DEQUANT_FALLBACK_OK = 0,
    COMMON_TREE_DRAFT_DEQUANT_FALLBACK_INVALID,
    COMMON_TREE_DRAFT_DEQUANT_FALLBACK_POLICY,
    COMMON_TREE_DRAFT_DEQUANT_FALLBACK_RESOURCE,
    COMMON_TREE_DRAFT_DEQUANT_FALLBACK_STATE,
};

common_tree_draft_dequant_fallback_status common_tree_draft_dequant_fallback_plan_build(
        const common_tree_draft_dequant_fallback_request & request,
        const common_tree_draft_fallback_budget & budget,
        common_tree_draft_dequant_fallback_plan * plan);

common_tree_draft_dequant_fallback_status common_tree_draft_dequant_fallback_reserve(
        common_tree_draft_fallback_budget * budget,
        const common_tree_draft_dequant_fallback_plan & plan,
        common_tree_draft_dequant_fallback_reservation * reservation);

common_tree_draft_dequant_fallback_status common_tree_draft_dequant_fallback_commit(
        common_tree_draft_fallback_budget * budget,
        common_tree_draft_dequant_fallback_reservation * reservation);

common_tree_draft_dequant_fallback_status common_tree_draft_dequant_fallback_abort(
        common_tree_draft_fallback_budget * budget,
        common_tree_draft_dequant_fallback_reservation * reservation);
