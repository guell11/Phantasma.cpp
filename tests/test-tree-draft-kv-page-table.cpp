#include "tree-draft-kv-page-table.h"

#include <cassert>
#include <type_traits>

int main() {
    static_assert(std::is_standard_layout<common_tree_draft_kv_page_table_entry>::value, "GPU upload layout must be standard");
    static_assert(std::is_trivially_copyable<common_tree_draft_kv_page_table_entry>::value, "GPU upload layout must be copyable");
    const common_tree_draft_kv_layer_geometry layer = { 64,64,COMMON_TREE_DRAFT_KV_LAYOUT_STANDARD };
    common_tree_draft_kv_page_geometry geometry = {};
    assert(common_tree_draft_kv_page_geometry_build(4, &layer, 1, 64, &geometry) == COMMON_TREE_DRAFT_KV_PAGE_OK);
    const common_tree_draft_kv_page_descriptor pages[] = {
        { COMMON_TREE_DRAFT_KV_PAGE_LIVE, 2, 1, 4, 1 },
        { COMMON_TREE_DRAFT_KV_PAGE_LIVE, 5, 1, 2, 1 },
    };
    const common_tree_draft_kv_page_table_entry entries[] = {
        {0,2,0,4},
        {1,5,4,2},
    };
    assert(common_tree_draft_kv_page_table_validate(entries, 2, pages, 2, geometry) == COMMON_TREE_DRAFT_KV_PAGE_TABLE_OK);
    common_tree_draft_page_address address = {};
    assert(common_tree_draft_kv_page_table_resolve(entries, 2, pages, 2, 5, &address) == COMMON_TREE_DRAFT_KV_PAGE_TABLE_OK);
    assert(address.page.id == 1 && address.page.generation == 5 && address.offset == 1);
    common_tree_draft_kv_page_table_entry stale[] = { entries[0], entries[1] };
    stale[1].generation = 4;
    assert(common_tree_draft_kv_page_table_validate(stale, 2, pages, 2, geometry) == COMMON_TREE_DRAFT_KV_PAGE_TABLE_STALE);
    const common_tree_draft_kv_page_table_entry overlap[] = { {0,2,0,4}, {1,5,3,2} };
    assert(common_tree_draft_kv_page_table_validate(overlap, 2, pages, 2, geometry) == COMMON_TREE_DRAFT_KV_PAGE_TABLE_OVERLAP);
    return 0;
}

