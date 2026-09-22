#include "tree-draft-gptq-reindex.h"

#include <algorithm>
#include <limits>
#include <numeric>

common_tree_draft_gptq_reindex_status common_tree_draft_gptq_build_reindex(
        const common_tree_draft_gptq_descriptor & quant,
        const common_tree_draft_packed_linear_view & view,
        const std::vector<int64_t> * g_idx_values,
        common_tree_draft_gptq_reindex_descriptor * descriptor) {
    if (descriptor == nullptr || view.format != COMMON_TREE_DRAFT_PACKED_LINEAR_GPTQ ||
        view.in_features == 0 || view.group_size == 0 || view.group_count == 0 ||
        view.in_features > std::numeric_limits<uint32_t>::max()) {
        return COMMON_TREE_DRAFT_GPTQ_REINDEX_INVALID;
    }

    const uint64_t expected_groups = (view.in_features + view.group_size - 1) / view.group_size;
    if (expected_groups != view.group_count) {
        return COMMON_TREE_DRAFT_GPTQ_REINDEX_INVALID;
    }

    common_tree_draft_gptq_reindex_descriptor out;
    out.column_count = view.in_features;
    out.group_count = view.group_count;
    out.activation_order = quant.desc_act;
    out.group_for_logical_column.resize(static_cast<size_t>(view.in_features));
    out.grouped_to_logical.resize(static_cast<size_t>(view.in_features));
    out.logical_to_grouped.resize(static_cast<size_t>(view.in_features));

    if (quant.desc_act) {
        if (!view.g_idx.has_value() || g_idx_values == nullptr) {
            return COMMON_TREE_DRAFT_GPTQ_REINDEX_MISSING_GIDX;
        }
        if (g_idx_values->size() != static_cast<size_t>(view.in_features)) {
            return COMMON_TREE_DRAFT_GPTQ_REINDEX_LENGTH;
        }
        for (size_t k = 0; k < g_idx_values->size(); ++k) {
            const int64_t group = (*g_idx_values)[k];
            if (group < 0 || static_cast<uint64_t>(group) >= view.group_count) {
                return COMMON_TREE_DRAFT_GPTQ_REINDEX_GROUP_RANGE;
            }
            out.group_for_logical_column[k] = static_cast<uint32_t>(group);
        }
    } else {
        for (uint64_t k = 0; k < view.in_features; ++k) {
            out.group_for_logical_column[static_cast<size_t>(k)] = static_cast<uint32_t>(k / view.group_size);
        }
    }

    std::iota(out.grouped_to_logical.begin(), out.grouped_to_logical.end(), uint32_t{0});
    std::stable_sort(out.grouped_to_logical.begin(), out.grouped_to_logical.end(), [&](uint32_t a, uint32_t b) {
        const uint32_t ga = out.group_for_logical_column[a];
        const uint32_t gb = out.group_for_logical_column[b];
        return ga < gb || (ga == gb && a < b);
    });
    for (uint32_t grouped = 0; grouped < out.grouped_to_logical.size(); ++grouped) {
        out.logical_to_grouped[out.grouped_to_logical[grouped]] = grouped;
    }

    *descriptor = std::move(out);
    return COMMON_TREE_DRAFT_GPTQ_REINDEX_OK;
}
