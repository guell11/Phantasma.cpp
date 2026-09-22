#include "tree-draft-kv-quant-page.h"

#include <limits>

static bool mul_u64_checked(uint64_t a, uint64_t b, uint64_t * out) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) return false;
    *out = a * b;
    return true;
}

common_tree_draft_kv_quant_page_status common_tree_draft_kv_quant_page_plan_build(
        const common_tree_draft_kv_page_geometry & geometry,
        const common_tree_draft_kv_quant_page_request & request,
        common_tree_draft_kv_quant_page_plan * plan) {
    if (plan == nullptr || geometry.tokens_per_page == 0 ||
        request.head_dim_k == 0 || request.head_dim_v == 0) {
        return COMMON_TREE_DRAFT_KV_QUANT_PAGE_INVALID_GEOMETRY;
    }
    const int64_t bk_raw = ggml_blck_size(request.type_k);
    const int64_t bv_raw = ggml_blck_size(request.type_v);
    if (bk_raw <= 0 || bv_raw <= 0 || bk_raw > UINT32_MAX || bv_raw > UINT32_MAX) {
        return COMMON_TREE_DRAFT_KV_QUANT_PAGE_UNSUPPORTED_TYPE;
    }

    common_tree_draft_kv_quant_page_plan out;
    out.type_k = request.type_k;
    out.type_v = request.type_v;
    out.block_elems_k = static_cast<uint32_t>(bk_raw);
    out.block_elems_v = static_cast<uint32_t>(bv_raw);
    if (!mul_u64_checked(geometry.tokens_per_page, request.head_dim_k, &out.page_elements_k) ||
        !mul_u64_checked(geometry.tokens_per_page, request.head_dim_v, &out.page_elements_v)) {
        return COMMON_TREE_DRAFT_KV_QUANT_PAGE_INVALID_GEOMETRY;
    }
    out.page_aligned_k = out.page_elements_k % out.block_elems_k == 0;
    out.page_aligned_v = out.page_elements_v % out.block_elems_v == 0;
    if (!out.page_aligned_k || !out.page_aligned_v) {
        return COMMON_TREE_DRAFT_KV_QUANT_PAGE_BLOCK_ALIGNMENT;
    }
    out.token_aligned_k = request.head_dim_k % out.block_elems_k == 0;
    out.token_aligned_v = request.head_dim_v % out.block_elems_v == 0;

    auto caps = request.capabilities;
    caps.direct_k = caps.direct_k && out.token_aligned_k;
    caps.direct_v = caps.direct_v && out.token_aligned_v;
    out.compatibility = common_tree_draft_kv_compatibility_report(request.type_k, request.type_v, caps);
    *plan = out;
    return COMMON_TREE_DRAFT_KV_QUANT_PAGE_OK;
}

bool common_tree_draft_kv_quant_valid_range(
        const common_tree_draft_kv_quant_page_plan & plan,
        const common_tree_draft_kv_page_descriptor & page,
        uint32_t lo,
        uint32_t hi) {
    if (page.state != COMMON_TREE_DRAFT_KV_PAGE_LIVE || lo >= hi || hi > page.used) return false;
    const uint64_t k_begin = static_cast<uint64_t>(lo);
    const uint64_t k_end = static_cast<uint64_t>(hi);
    // Direct kernels require token boundaries to be quant-block boundaries.
    if (plan.compatibility.selected_path == COMMON_TREE_DRAFT_KV_PATH_DIRECT) {
        return plan.token_aligned_k && plan.token_aligned_v;
    }
    // Existing backend/staging paths may decode block-aligned page storage as
    // long as they never read beyond initialized tokens.
    return k_begin < k_end;
}
