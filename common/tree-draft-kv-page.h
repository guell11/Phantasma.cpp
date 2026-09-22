#pragma once

#include "tree-draft-kv-identity.h"

#include <cstddef>
#include <cstdint>

enum common_tree_draft_kv_page_layout : uint32_t {
    COMMON_TREE_DRAFT_KV_LAYOUT_STANDARD = 0,
    COMMON_TREE_DRAFT_KV_LAYOUT_QUANTIZED,
    COMMON_TREE_DRAFT_KV_LAYOUT_K_ONLY,
};

struct common_tree_draft_kv_layer_geometry {
    uint64_t k_bytes_per_token;
    uint64_t v_bytes_per_token;
    common_tree_draft_kv_page_layout layout;
};

struct common_tree_draft_kv_page_geometry {
    uint32_t tokens_per_page;
    uint32_t layer_count;
    uint64_t alignment;
    uint64_t page_bytes;
    const common_tree_draft_kv_layer_geometry * layers;
};

enum common_tree_draft_kv_page_state : uint32_t {
    COMMON_TREE_DRAFT_KV_PAGE_FREE = 0,
    COMMON_TREE_DRAFT_KV_PAGE_RESERVED,
    COMMON_TREE_DRAFT_KV_PAGE_LIVE,
    COMMON_TREE_DRAFT_KV_PAGE_RETIRING,
};

struct common_tree_draft_kv_page_descriptor {
    common_tree_draft_kv_page_state state;
    uint32_t generation;
    uint32_t refcount;
    uint32_t used;
    uint64_t owner_epoch;
};

enum common_tree_draft_kv_page_status : uint32_t {
    COMMON_TREE_DRAFT_KV_PAGE_OK = 0,
    COMMON_TREE_DRAFT_KV_PAGE_NULL_BUFFER,
    COMMON_TREE_DRAFT_KV_PAGE_GEOMETRY,
    COMMON_TREE_DRAFT_KV_PAGE_ALIGNMENT,
    COMMON_TREE_DRAFT_KV_PAGE_OVERFLOW,
    COMMON_TREE_DRAFT_KV_PAGE_STATE,
    COMMON_TREE_DRAFT_KV_PAGE_USED_RANGE,
    COMMON_TREE_DRAFT_KV_PAGE_REFCOUNT,
    COMMON_TREE_DRAFT_KV_PAGE_GENERATION,
};

common_tree_draft_kv_page_status common_tree_draft_kv_page_geometry_build(
        uint32_t tokens_per_page,
        const common_tree_draft_kv_layer_geometry * layers,
        uint32_t layer_count,
        uint64_t alignment,
        common_tree_draft_kv_page_geometry * out);

bool common_tree_draft_kv_page_base_aligned(uintptr_t address, const common_tree_draft_kv_page_geometry & geometry);

common_tree_draft_kv_page_status common_tree_draft_kv_page_descriptor_validate(
        const common_tree_draft_kv_page_descriptor & descriptor,
        const common_tree_draft_kv_page_geometry & geometry);

common_tree_draft_kv_page_status common_tree_draft_kv_page_transition(
        common_tree_draft_kv_page_descriptor * descriptor,
        common_tree_draft_kv_page_state next_state,
        const common_tree_draft_kv_page_geometry & geometry);
