#include "tree-draft-kv-page.h"

#include <limits>

static bool common_tree_draft_u64_add(uint64_t a, uint64_t b, uint64_t * out) {
    if (b > std::numeric_limits<uint64_t>::max() - a) return false;
    *out = a + b;
    return true;
}

static bool common_tree_draft_u64_mul(uint64_t a, uint64_t b, uint64_t * out) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) return false;
    *out = a * b;
    return true;
}

common_tree_draft_kv_page_status common_tree_draft_kv_page_geometry_build(
        uint32_t tokens_per_page,
        const common_tree_draft_kv_layer_geometry * layers,
        uint32_t layer_count,
        uint64_t alignment,
        common_tree_draft_kv_page_geometry * out) {
    if (out == nullptr || (layer_count > 0 && layers == nullptr)) return COMMON_TREE_DRAFT_KV_PAGE_NULL_BUFFER;
    if (tokens_per_page == 0 || layer_count == 0 || alignment == 0 || (alignment & (alignment - 1)) != 0) {
        return COMMON_TREE_DRAFT_KV_PAGE_GEOMETRY;
    }
    uint64_t total = 0;
    for (uint32_t i = 0; i < layer_count; ++i) {
        if (layers[i].k_bytes_per_token == 0 ||
            (layers[i].layout != COMMON_TREE_DRAFT_KV_LAYOUT_K_ONLY && layers[i].v_bytes_per_token == 0) ||
            (layers[i].layout == COMMON_TREE_DRAFT_KV_LAYOUT_K_ONLY && layers[i].v_bytes_per_token != 0)) {
            return COMMON_TREE_DRAFT_KV_PAGE_GEOMETRY;
        }
        uint64_t per_token = 0;
        uint64_t layer_bytes = 0;
        if (!common_tree_draft_u64_add(layers[i].k_bytes_per_token, layers[i].v_bytes_per_token, &per_token) ||
            !common_tree_draft_u64_mul(per_token, tokens_per_page, &layer_bytes) ||
            !common_tree_draft_u64_add(total, layer_bytes, &total)) {
            return COMMON_TREE_DRAFT_KV_PAGE_OVERFLOW;
        }
    }
    *out = { tokens_per_page, layer_count, alignment, total, layers };
    return COMMON_TREE_DRAFT_KV_PAGE_OK;
}

bool common_tree_draft_kv_page_base_aligned(uintptr_t address, const common_tree_draft_kv_page_geometry & geometry) {
    return geometry.alignment != 0 && (address % geometry.alignment) == 0;
}

common_tree_draft_kv_page_status common_tree_draft_kv_page_descriptor_validate(
        const common_tree_draft_kv_page_descriptor & descriptor,
        const common_tree_draft_kv_page_geometry & geometry) {
    if (descriptor.used > geometry.tokens_per_page) return COMMON_TREE_DRAFT_KV_PAGE_USED_RANGE;
    switch (descriptor.state) {
        case COMMON_TREE_DRAFT_KV_PAGE_FREE:
            if (descriptor.refcount != 0 || descriptor.used != 0) return COMMON_TREE_DRAFT_KV_PAGE_REFCOUNT;
            break;
        case COMMON_TREE_DRAFT_KV_PAGE_RESERVED:
        case COMMON_TREE_DRAFT_KV_PAGE_LIVE:
        case COMMON_TREE_DRAFT_KV_PAGE_RETIRING:
            break;
        default:
            return COMMON_TREE_DRAFT_KV_PAGE_STATE;
    }
    return COMMON_TREE_DRAFT_KV_PAGE_OK;
}

common_tree_draft_kv_page_status common_tree_draft_kv_page_transition(
        common_tree_draft_kv_page_descriptor * descriptor,
        common_tree_draft_kv_page_state next_state,
        const common_tree_draft_kv_page_geometry & geometry) {
    if (descriptor == nullptr) return COMMON_TREE_DRAFT_KV_PAGE_NULL_BUFFER;
    const auto valid = common_tree_draft_kv_page_descriptor_validate(*descriptor, geometry);
    if (valid != COMMON_TREE_DRAFT_KV_PAGE_OK) return valid;
    const auto current = descriptor->state;
    bool allowed = (current == COMMON_TREE_DRAFT_KV_PAGE_FREE && next_state == COMMON_TREE_DRAFT_KV_PAGE_RESERVED) ||
                   (current == COMMON_TREE_DRAFT_KV_PAGE_RESERVED && next_state == COMMON_TREE_DRAFT_KV_PAGE_LIVE) ||
                   (current == COMMON_TREE_DRAFT_KV_PAGE_LIVE && next_state == COMMON_TREE_DRAFT_KV_PAGE_RETIRING) ||
                   (current == COMMON_TREE_DRAFT_KV_PAGE_RETIRING && next_state == COMMON_TREE_DRAFT_KV_PAGE_FREE);
    if (!allowed) return COMMON_TREE_DRAFT_KV_PAGE_STATE;
    if (next_state == COMMON_TREE_DRAFT_KV_PAGE_FREE) {
        if (descriptor->refcount != 0) return COMMON_TREE_DRAFT_KV_PAGE_REFCOUNT;
        if (descriptor->generation == UINT32_MAX) return COMMON_TREE_DRAFT_KV_PAGE_GENERATION;
        ++descriptor->generation;
        descriptor->used = 0;
        descriptor->owner_epoch = 0;
    }
    descriptor->state = next_state;
    return COMMON_TREE_DRAFT_KV_PAGE_OK;
}

