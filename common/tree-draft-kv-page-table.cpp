#include "tree-draft-kv-page-table.h"

#include <limits>

common_tree_draft_kv_page_table_status common_tree_draft_kv_page_table_validate(
        const common_tree_draft_kv_page_table_entry * entries,
        size_t entry_count,
        const common_tree_draft_kv_page_descriptor * pages,
        size_t page_count,
        const common_tree_draft_kv_page_geometry & geometry) {
    if ((entry_count > 0 && entries == nullptr) || (page_count > 0 && pages == nullptr)) {
        return COMMON_TREE_DRAFT_KV_PAGE_TABLE_NULL_BUFFER;
    }
    for (size_t i = 0; i < entry_count; ++i) {
        const auto & e = entries[i];
        if (e.page_id >= page_count || e.base_position < 0 || e.valid_len == 0 || e.valid_len > geometry.tokens_per_page) {
            return COMMON_TREE_DRAFT_KV_PAGE_TABLE_RANGE;
        }
        if (pages[e.page_id].generation != e.generation) return COMMON_TREE_DRAFT_KV_PAGE_TABLE_STALE;
        if (pages[e.page_id].state != COMMON_TREE_DRAFT_KV_PAGE_LIVE || pages[e.page_id].used < e.valid_len) {
            return COMMON_TREE_DRAFT_KV_PAGE_TABLE_RANGE;
        }
        if (e.base_position > std::numeric_limits<int64_t>::max() - static_cast<int64_t>(e.valid_len)) {
            return COMMON_TREE_DRAFT_KV_PAGE_TABLE_RANGE;
        }
        const int64_t end = e.base_position + static_cast<int64_t>(e.valid_len);
        for (size_t j = 0; j < i; ++j) {
            const auto & prev = entries[j];
            const int64_t prev_end = prev.base_position + static_cast<int64_t>(prev.valid_len);
            if (e.base_position < prev_end && prev.base_position < end) return COMMON_TREE_DRAFT_KV_PAGE_TABLE_OVERLAP;
        }
    }
    return COMMON_TREE_DRAFT_KV_PAGE_TABLE_OK;
}

common_tree_draft_kv_page_table_status common_tree_draft_kv_page_table_resolve(
        const common_tree_draft_kv_page_table_entry * entries,
        size_t entry_count,
        const common_tree_draft_kv_page_descriptor * pages,
        size_t page_count,
        int64_t logical_position,
        common_tree_draft_page_address * address) {
    if (address == nullptr) return COMMON_TREE_DRAFT_KV_PAGE_TABLE_NULL_BUFFER;
    if (logical_position < 0) return COMMON_TREE_DRAFT_KV_PAGE_TABLE_NOT_FOUND;
    for (size_t i = 0; i < entry_count; ++i) {
        const auto & e = entries[i];
        const int64_t end = e.base_position + static_cast<int64_t>(e.valid_len);
        if (logical_position >= e.base_position && logical_position < end) {
            if (e.page_id >= page_count) return COMMON_TREE_DRAFT_KV_PAGE_TABLE_RANGE;
            if (pages[e.page_id].generation != e.generation) return COMMON_TREE_DRAFT_KV_PAGE_TABLE_STALE;
            *address = { { e.page_id, e.generation }, static_cast<uint32_t>(logical_position - e.base_position) };
            return COMMON_TREE_DRAFT_KV_PAGE_TABLE_OK;
        }
    }
    return COMMON_TREE_DRAFT_KV_PAGE_TABLE_NOT_FOUND;
}

