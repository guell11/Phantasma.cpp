#pragma once

#include "tree-draft-kv-branch.h"

#include <cstddef>
#include <cstdint>

struct common_tree_draft_kv_page_table_entry {
    uint32_t page_id;
    uint32_t generation;
    int64_t base_position;
    uint32_t valid_len;
};

enum common_tree_draft_kv_page_table_status : uint32_t {
    COMMON_TREE_DRAFT_KV_PAGE_TABLE_OK = 0,
    COMMON_TREE_DRAFT_KV_PAGE_TABLE_NULL_BUFFER,
    COMMON_TREE_DRAFT_KV_PAGE_TABLE_RANGE,
    COMMON_TREE_DRAFT_KV_PAGE_TABLE_OVERLAP,
    COMMON_TREE_DRAFT_KV_PAGE_TABLE_STALE,
    COMMON_TREE_DRAFT_KV_PAGE_TABLE_NOT_FOUND,
};

common_tree_draft_kv_page_table_status common_tree_draft_kv_page_table_validate(
        const common_tree_draft_kv_page_table_entry * entries,
        size_t entry_count,
        const common_tree_draft_kv_page_descriptor * pages,
        size_t page_count,
        const common_tree_draft_kv_page_geometry & geometry);

common_tree_draft_kv_page_table_status common_tree_draft_kv_page_table_resolve(
        const common_tree_draft_kv_page_table_entry * entries,
        size_t entry_count,
        const common_tree_draft_kv_page_descriptor * pages,
        size_t page_count,
        int64_t logical_position,
        common_tree_draft_page_address * address);

