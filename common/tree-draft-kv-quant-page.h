#pragma once

#include "tree-draft-kv-compat.h"
#include "tree-draft-kv-page.h"

#include <cstdint>

struct common_tree_draft_kv_quant_page_request {
    ggml_type type_k = GGML_TYPE_F16;
    ggml_type type_v = GGML_TYPE_F16;
    uint32_t head_dim_k = 0;
    uint32_t head_dim_v = 0;
    common_tree_draft_kv_capabilities capabilities = {};
};

struct common_tree_draft_kv_quant_page_plan {
    ggml_type type_k = GGML_TYPE_F16;
    ggml_type type_v = GGML_TYPE_F16;
    uint32_t block_elems_k = 1;
    uint32_t block_elems_v = 1;
    uint64_t page_elements_k = 0;
    uint64_t page_elements_v = 0;
    bool page_aligned_k = false;
    bool page_aligned_v = false;
    bool token_aligned_k = false;
    bool token_aligned_v = false;
    common_tree_draft_kv_compatibility compatibility = {};
};

enum common_tree_draft_kv_quant_page_status : uint32_t {
    COMMON_TREE_DRAFT_KV_QUANT_PAGE_OK = 0,
    COMMON_TREE_DRAFT_KV_QUANT_PAGE_INVALID_GEOMETRY,
    COMMON_TREE_DRAFT_KV_QUANT_PAGE_UNSUPPORTED_TYPE,
    COMMON_TREE_DRAFT_KV_QUANT_PAGE_BLOCK_ALIGNMENT,
};

common_tree_draft_kv_quant_page_status common_tree_draft_kv_quant_page_plan_build(
        const common_tree_draft_kv_page_geometry & geometry,
        const common_tree_draft_kv_quant_page_request & request,
        common_tree_draft_kv_quant_page_plan * plan);

bool common_tree_draft_kv_quant_valid_range(
        const common_tree_draft_kv_quant_page_plan & plan,
        const common_tree_draft_kv_page_descriptor & page,
        uint32_t lo,
        uint32_t hi);
