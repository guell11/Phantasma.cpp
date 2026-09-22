#include "tree-draft-kv-block-table.h"

common_tree_draft_kv_block_table_status common_tree_draft_kv_block_table_pack(
        const common_tree_draft_kv_gather_metadata & gather,
        common_tree_draft_kv_block_table * output) {
    if (output == nullptr || gather.row_ptr == nullptr ||
        (gather.segment_count > 0 && gather.segments == nullptr)) {
        return COMMON_TREE_DRAFT_KV_BLOCK_TABLE_NULL_BUFFER;
    }
    if (gather.row_ptr_capacity < static_cast<size_t>(gather.query_count) + 1) {
        return COMMON_TREE_DRAFT_KV_BLOCK_TABLE_RANGE;
    }
    if (output->row_ptr == nullptr || output->row_ptr_capacity < static_cast<size_t>(gather.query_count) + 1 ||
        output->segment_capacity < gather.segment_count ||
        (gather.segment_count > 0 && output->segments == nullptr)) {
        return COMMON_TREE_DRAFT_KV_BLOCK_TABLE_OUTPUT_TOO_SMALL;
    }
    if (gather.row_ptr[0] != 0 || gather.row_ptr[gather.query_count] != gather.segment_count) {
        return COMMON_TREE_DRAFT_KV_BLOCK_TABLE_RANGE;
    }
    for (uint32_t q = 0; q < gather.query_count; ++q) {
        if (gather.row_ptr[q] > gather.row_ptr[q + 1] ||
            gather.row_ptr[q + 1] > gather.segment_count) {
            return COMMON_TREE_DRAFT_KV_BLOCK_TABLE_RANGE;
        }
    }
    for (uint32_t i = 0; i < gather.segment_count; ++i) {
        const auto & s = gather.segments[i];
        if (s.page.id == COMMON_TREE_DRAFT_KV_ID_INVALID || s.lo >= s.hi) {
            return COMMON_TREE_DRAFT_KV_BLOCK_TABLE_RANGE;
        }
    }

    for (uint32_t q = 0; q <= gather.query_count; ++q) output->row_ptr[q] = gather.row_ptr[q];
    for (uint32_t i = 0; i < gather.segment_count; ++i) {
        const auto & s = gather.segments[i];
        output->segments[i] = {s.page.id, s.page.generation, s.lo, s.hi};
    }
    output->query_count = gather.query_count;
    output->segment_count = gather.segment_count;
    return COMMON_TREE_DRAFT_KV_BLOCK_TABLE_OK;
}
