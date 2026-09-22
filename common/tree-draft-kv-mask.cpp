#include "tree-draft-kv-mask.h"

#include <limits>

static common_tree_draft_kv_mask_status common_tree_draft_kv_mask_validate(
        const common_tree_draft_kv_gather_metadata & gather) {
    if (gather.row_ptr == nullptr || (gather.segment_count > 0 && gather.segments == nullptr)) {
        return COMMON_TREE_DRAFT_KV_MASK_METADATA;
    }
    if (gather.row_ptr_capacity < static_cast<size_t>(gather.query_count) + 1 ||
        gather.segment_capacity < gather.segment_count || gather.row_ptr[0] != 0 ||
        gather.row_ptr[gather.query_count] != gather.segment_count) {
        return COMMON_TREE_DRAFT_KV_MASK_METADATA;
    }
    for (uint32_t q = 0; q < gather.query_count; ++q) {
        if (gather.row_ptr[q] > gather.row_ptr[q + 1] || gather.row_ptr[q + 1] > gather.segment_count) {
            return COMMON_TREE_DRAFT_KV_MASK_METADATA;
        }
    }
    for (uint32_t i = 0; i < gather.segment_count; ++i) {
        const auto & segment = gather.segments[i];
        if (segment.page.id == COMMON_TREE_DRAFT_KV_ID_INVALID || segment.lo >= segment.hi) {
            return COMMON_TREE_DRAFT_KV_MASK_METADATA;
        }
    }
    return COMMON_TREE_DRAFT_KV_MASK_OK;
}

common_tree_draft_kv_mask_status common_tree_draft_kv_mask_visible(
        const common_tree_draft_kv_gather_metadata & gather,
        uint32_t query_index,
        uint32_t segment_index,
        uint32_t page_offset,
        bool * visible,
        int64_t * logical_position) {
    if (visible == nullptr || logical_position == nullptr) {
        return COMMON_TREE_DRAFT_KV_MASK_NULL_OUTPUT;
    }
    *visible = false;
    *logical_position = -1;

    const auto metadata_status = common_tree_draft_kv_mask_validate(gather);
    if (metadata_status != COMMON_TREE_DRAFT_KV_MASK_OK) return metadata_status;
    if (query_index >= gather.query_count) return COMMON_TREE_DRAFT_KV_MASK_QUERY_RANGE;
    if (segment_index >= gather.segment_count) return COMMON_TREE_DRAFT_KV_MASK_KEY_RANGE;

    const auto & key_segment = gather.segments[segment_index];
    if (page_offset < key_segment.lo || page_offset >= key_segment.hi) {
        return COMMON_TREE_DRAFT_KV_MASK_KEY_RANGE;
    }

    const uint32_t row_begin = gather.row_ptr[query_index];
    const uint32_t row_end = gather.row_ptr[query_index + 1];
    if (segment_index < row_begin || segment_index >= row_end) {
        return COMMON_TREE_DRAFT_KV_MASK_OK;
    }

    uint64_t logical = 0;
    for (uint32_t i = row_begin; i < segment_index; ++i) {
        logical += static_cast<uint64_t>(gather.segments[i].hi - gather.segments[i].lo);
        if (logical > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            return COMMON_TREE_DRAFT_KV_MASK_OVERFLOW;
        }
    }
    logical += static_cast<uint64_t>(page_offset - key_segment.lo);
    if (logical > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        return COMMON_TREE_DRAFT_KV_MASK_OVERFLOW;
    }

    *visible = true;
    *logical_position = static_cast<int64_t>(logical);
    return COMMON_TREE_DRAFT_KV_MASK_OK;
}

float common_tree_draft_kv_mask_value(bool visible) {
    return common_tree_draft_composite_mask_value(visible);
}
