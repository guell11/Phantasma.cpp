#include "tree-draft-kv-head-map.h"

#include <cassert>
#include <cstdint>

int main() {
    common_tree_draft_head_map_plan heads;
    assert(common_tree_draft_head_map_build(32, 8, {4}, &heads) == COMMON_TREE_DRAFT_HEAD_MAP_OK);

    const common_tree_draft_kv_layer_geometry layers[] = {
        { 8 * 64 * 2, 8 * 64 * 4, COMMON_TREE_DRAFT_KV_LAYOUT_STANDARD },
    };
    common_tree_draft_kv_page_geometry pages = {};
    assert(common_tree_draft_kv_page_geometry_build(16, layers, 1, 256, &pages) == COMMON_TREE_DRAFT_KV_PAGE_OK);

    common_tree_draft_paged_kv_head_layout layout = {};
    assert(common_tree_draft_paged_kv_head_layout_build(heads, pages, 0, 64, 2, 4, &layout) ==
        COMMON_TREE_DRAFT_PAGED_KV_HEAD_OK);
    assert(layout.query_heads_per_kv == 4);
    assert(layout.k_stride_dim == 2 && layout.k_stride_head == 128 && layout.k_stride_token == 1024);
    assert(layout.v_stride_dim == 4 && layout.v_stride_head == 256 && layout.v_stride_token == 2048);

    const common_tree_draft_kv_gpu_segment segment = { 17, 3, 5, 9 };
    uint64_t offset = UINT64_MAX;
    assert(common_tree_draft_paged_kv_head_offset(layout, segment, 2, 15, 7, false, &offset) ==
        COMMON_TREE_DRAFT_PAGED_KV_HEAD_OK);
    assert(offset == 7 * 1024 + 3 * 128 + 7 * 2);
    uint64_t same_group_offset = UINT64_MAX;
    assert(common_tree_draft_paged_kv_head_offset(layout, segment, 2, 12, 7, false, &same_group_offset) ==
        COMMON_TREE_DRAFT_PAGED_KV_HEAD_OK);
    assert(same_group_offset == offset);
    assert(common_tree_draft_paged_kv_head_offset(layout, segment, 2, 15, 7, true, &offset) ==
        COMMON_TREE_DRAFT_PAGED_KV_HEAD_OK);
    assert(offset == 7 * 2048 + 3 * 256 + 7 * 4);

    assert(common_tree_draft_paged_kv_head_offset(layout, segment, 0, 0, 0, false, &offset) ==
        COMMON_TREE_DRAFT_PAGED_KV_HEAD_OK);
    assert(offset == 5 * 1024);
    assert(common_tree_draft_paged_kv_head_offset(layout, segment, 3, 31, 63, false, &offset) ==
        COMMON_TREE_DRAFT_PAGED_KV_HEAD_OK);
    assert(offset == 8 * 1024 + 7 * 128 + 63 * 2);

    assert(common_tree_draft_paged_kv_head_offset(layout, segment, 4, 0, 0, false, &offset) ==
        COMMON_TREE_DRAFT_PAGED_KV_HEAD_RANGE);
    assert(common_tree_draft_paged_kv_head_offset(layout, segment, 0, 32, 0, false, &offset) ==
        COMMON_TREE_DRAFT_PAGED_KV_HEAD_RANGE);
    assert(common_tree_draft_paged_kv_head_offset(layout, segment, 0, 0, 64, false, &offset) ==
        COMMON_TREE_DRAFT_PAGED_KV_HEAD_RANGE);
    assert(common_tree_draft_paged_kv_head_offset(layout, { COMMON_TREE_DRAFT_KV_ID_INVALID, 0, 0, 1 }, 0, 0, 0, false, &offset) ==
        COMMON_TREE_DRAFT_PAGED_KV_HEAD_RANGE);

    const common_tree_draft_kv_layer_geometry wrong_bytes[] = {
        { 1023, 2048, COMMON_TREE_DRAFT_KV_LAYOUT_STANDARD },
    };
    common_tree_draft_kv_page_geometry wrong_pages = {};
    assert(common_tree_draft_kv_page_geometry_build(16, wrong_bytes, 1, 256, &wrong_pages) == COMMON_TREE_DRAFT_KV_PAGE_OK);
    const auto before = layout;
    assert(common_tree_draft_paged_kv_head_layout_build(heads, wrong_pages, 0, 64, 2, 4, &layout) ==
        COMMON_TREE_DRAFT_PAGED_KV_HEAD_LAYOUT);
    assert(layout.k_stride_token == before.k_stride_token && layout.v_stride_token == before.v_stride_token);

    const common_tree_draft_kv_layer_geometry quantized[] = {
        { 1024, 2048, COMMON_TREE_DRAFT_KV_LAYOUT_QUANTIZED },
    };
    common_tree_draft_kv_page_geometry quant_pages = {};
    assert(common_tree_draft_kv_page_geometry_build(16, quantized, 1, 256, &quant_pages) == COMMON_TREE_DRAFT_KV_PAGE_OK);
    assert(common_tree_draft_paged_kv_head_layout_build(heads, quant_pages, 0, 64, 2, 4, &layout) ==
        COMMON_TREE_DRAFT_PAGED_KV_HEAD_LAYOUT);

    common_tree_draft_head_map_plan mqa;
    assert(common_tree_draft_head_map_build(32, 1, {32}, &mqa) == COMMON_TREE_DRAFT_HEAD_MAP_OK);
    const common_tree_draft_kv_layer_geometry mqa_layer[] = {
        { 64 * 2, 64 * 2, COMMON_TREE_DRAFT_KV_LAYOUT_STANDARD },
    };
    common_tree_draft_kv_page_geometry mqa_pages = {};
    assert(common_tree_draft_kv_page_geometry_build(8, mqa_layer, 1, 64, &mqa_pages) == COMMON_TREE_DRAFT_KV_PAGE_OK);
    assert(common_tree_draft_paged_kv_head_layout_build(mqa, mqa_pages, 0, 64, 2, 2, &layout) ==
        COMMON_TREE_DRAFT_PAGED_KV_HEAD_OK);
    assert(common_tree_draft_paged_kv_head_offset(layout, { 4, 1, 0, 2 }, 1, 31, 63, false, &offset) ==
        COMMON_TREE_DRAFT_PAGED_KV_HEAD_OK);
    assert(offset == 1 * 128 + 63 * 2);

    common_tree_draft_head_map_plan mha;
    assert(common_tree_draft_head_map_build(8, 8, {1}, &mha) == COMMON_TREE_DRAFT_HEAD_MAP_OK);
    const common_tree_draft_kv_layer_geometry mha_layer[] = {
        { 8 * 16 * 2, 8 * 16 * 2, COMMON_TREE_DRAFT_KV_LAYOUT_STANDARD },
    };
    common_tree_draft_kv_page_geometry mha_pages = {};
    assert(common_tree_draft_kv_page_geometry_build(4, mha_layer, 1, 64, &mha_pages) == COMMON_TREE_DRAFT_KV_PAGE_OK);
    assert(common_tree_draft_paged_kv_head_layout_build(mha, mha_pages, 0, 16, 2, 2, &layout) ==
        COMMON_TREE_DRAFT_PAGED_KV_HEAD_OK);
    assert(common_tree_draft_paged_kv_head_offset(layout, { 2, 1, 0, 1 }, 0, 7, 15, false, &offset) ==
        COMMON_TREE_DRAFT_PAGED_KV_HEAD_OK);
    assert(offset == 7 * 32 + 15 * 2);

    return 0;
}
