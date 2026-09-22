#include "tree-draft-kv-gather.h"

#include <limits>

common_tree_draft_kv_gather_status common_tree_draft_kv_gather_build(
        const common_tree_draft_kv_query * queries,
        size_t query_count,
        const common_tree_draft_kv_branch_descriptor * branches,
        uint32_t branch_count,
        const common_tree_draft_kv_visibility_table & visibility,
        const common_tree_draft_kv_page_geometry & geometry,
        common_tree_draft_kv_gather_metadata * output) {
    if (output == nullptr || (query_count > 0 && queries == nullptr) ||
        (branch_count > 0 && branches == nullptr)) {
        return COMMON_TREE_DRAFT_KV_GATHER_NULL_BUFFER;
    }
    if (output->row_ptr == nullptr || output->row_ptr_capacity < query_count + 1) {
        return COMMON_TREE_DRAFT_KV_GATHER_OUTPUT_TOO_SMALL;
    }
    if (visibility.branch_count != branch_count) {
        return COMMON_TREE_DRAFT_KV_GATHER_VISIBILITY;
    }

    size_t needed_segments = 0;
    for (size_t qi = 0; qi < query_count; ++qi) {
        const auto & q = queries[qi];
        if (q.branch_index >= branch_count) return COMMON_TREE_DRAFT_KV_GATHER_QUERY_RANGE;
        const auto & b = branches[q.branch_index];
        if (common_tree_draft_kv_branch_validate(b, geometry) != COMMON_TREE_DRAFT_KV_BRANCH_OK) {
            return COMMON_TREE_DRAFT_KV_GATHER_BRANCH;
        }
        if (q.position < 0 || q.position > b.tip_position) {
            return COMMON_TREE_DRAFT_KV_GATHER_QUERY_RANGE;
        }
        if (!common_tree_draft_kv_visible(
                visibility, q.branch_index, q.position, q.branch_index, q.position)) {
            return COMMON_TREE_DRAFT_KV_GATHER_VISIBILITY;
        }

        int64_t remaining = q.position + 1;
        for (uint32_t si = 0; si < b.span_count && remaining > 0; ++si) {
            const auto & span = b.spans[si];
            const uint32_t len = span.hi - span.lo;
            if (len == 0) continue;
            ++needed_segments;
            remaining -= static_cast<int64_t>(len);
        }
        if (remaining > 0) return COMMON_TREE_DRAFT_KV_GATHER_BRANCH;
    }
    if (needed_segments > UINT32_MAX) return COMMON_TREE_DRAFT_KV_GATHER_OVERFLOW;
    if (needed_segments > output->segment_capacity ||
        (needed_segments > 0 && output->segments == nullptr)) {
        return COMMON_TREE_DRAFT_KV_GATHER_OUTPUT_TOO_SMALL;
    }

    uint32_t cursor = 0;
    output->row_ptr[0] = 0;
    for (size_t qi = 0; qi < query_count; ++qi) {
        const auto & q = queries[qi];
        const auto & b = branches[q.branch_index];
        int64_t remaining = q.position + 1;
        for (uint32_t si = 0; si < b.span_count && remaining > 0; ++si) {
            const auto & span = b.spans[si];
            const uint32_t len = span.hi - span.lo;
            const uint32_t take = static_cast<uint32_t>(
                remaining < static_cast<int64_t>(len) ? remaining : len);
            output->segments[cursor++] = {span.page, span.lo, static_cast<uint32_t>(span.lo + take)};
            remaining -= take;
        }
        output->row_ptr[qi + 1] = cursor;
    }
    output->query_count = static_cast<uint32_t>(query_count);
    output->segment_count = cursor;
    return COMMON_TREE_DRAFT_KV_GATHER_OK;
}
