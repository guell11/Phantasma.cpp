#include "tree-draft-kv-page.h"

#include <cassert>

int main() {
    const common_tree_draft_kv_layer_geometry layers[] = {
        { 128, 128, COMMON_TREE_DRAFT_KV_LAYOUT_STANDARD },
        { 64, 64, COMMON_TREE_DRAFT_KV_LAYOUT_QUANTIZED },
        { 96, 0, COMMON_TREE_DRAFT_KV_LAYOUT_K_ONLY },
    };
    common_tree_draft_kv_page_geometry geometry = {};
    assert(common_tree_draft_kv_page_geometry_build(16, layers, 3, 256, &geometry) == COMMON_TREE_DRAFT_KV_PAGE_OK);
    assert(geometry.page_bytes == 16u * (256u + 128u + 96u));
    assert(common_tree_draft_kv_page_base_aligned(4096, geometry));
    assert(!common_tree_draft_kv_page_base_aligned(4097, geometry));

    common_tree_draft_kv_page_descriptor page = { COMMON_TREE_DRAFT_KV_PAGE_FREE, 1, 0, 0, 0 };
    assert(common_tree_draft_kv_page_descriptor_validate(page, geometry) == COMMON_TREE_DRAFT_KV_PAGE_OK);
    assert(common_tree_draft_kv_page_transition(&page, COMMON_TREE_DRAFT_KV_PAGE_RESERVED, geometry) == COMMON_TREE_DRAFT_KV_PAGE_OK);
    page.owner_epoch = 9;
    page.used = 10;
    assert(common_tree_draft_kv_page_transition(&page, COMMON_TREE_DRAFT_KV_PAGE_LIVE, geometry) == COMMON_TREE_DRAFT_KV_PAGE_OK);
    page.refcount = 1;
    assert(common_tree_draft_kv_page_transition(&page, COMMON_TREE_DRAFT_KV_PAGE_RETIRING, geometry) == COMMON_TREE_DRAFT_KV_PAGE_OK);
    assert(common_tree_draft_kv_page_transition(&page, COMMON_TREE_DRAFT_KV_PAGE_FREE, geometry) == COMMON_TREE_DRAFT_KV_PAGE_REFCOUNT);
    page.refcount = 0;
    assert(common_tree_draft_kv_page_transition(&page, COMMON_TREE_DRAFT_KV_PAGE_FREE, geometry) == COMMON_TREE_DRAFT_KV_PAGE_OK);
    assert(page.generation == 2 && page.used == 0 && page.owner_epoch == 0);

    assert(common_tree_draft_kv_page_geometry_build(16, layers, 3, 24, &geometry) == COMMON_TREE_DRAFT_KV_PAGE_GEOMETRY);
    return 0;
}

