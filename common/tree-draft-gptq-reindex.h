#pragma once

#include "tree-draft-external-quant.h"
#include "tree-draft-packed-linear.h"

#include <cstdint>
#include <vector>

struct common_tree_draft_gptq_reindex_descriptor {
    uint64_t column_count = 0;
    uint64_t group_count = 0;
    bool activation_order = false;

    // Exact quantization group used by each logical input column.
    std::vector<uint32_t> group_for_logical_column;

    // Stable metadata-only permutation useful to kernels that prefer columns
    // grouped by quantization group. No weight bytes are physically moved.
    std::vector<uint32_t> grouped_to_logical;
    std::vector<uint32_t> logical_to_grouped;
};

enum common_tree_draft_gptq_reindex_status : uint32_t {
    COMMON_TREE_DRAFT_GPTQ_REINDEX_OK = 0,
    COMMON_TREE_DRAFT_GPTQ_REINDEX_INVALID,
    COMMON_TREE_DRAFT_GPTQ_REINDEX_MISSING_GIDX,
    COMMON_TREE_DRAFT_GPTQ_REINDEX_LENGTH,
    COMMON_TREE_DRAFT_GPTQ_REINDEX_GROUP_RANGE,
};

// g_idx_values contains decoded integer values from the optional serialized
// GPTQ g_idx tensor. It is required exactly when quant.desc_act is true.
// The output is transactional: descriptor is not modified on failure.
common_tree_draft_gptq_reindex_status common_tree_draft_gptq_build_reindex(
        const common_tree_draft_gptq_descriptor & quant,
        const common_tree_draft_packed_linear_view & view,
        const std::vector<int64_t> * g_idx_values,
        common_tree_draft_gptq_reindex_descriptor * descriptor);
