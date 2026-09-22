#include "tree-draft-smem-layout.h"

#include <limits>

static bool is_pow2(uint32_t x) {
    return x != 0 && (x & (x - 1)) == 0;
}

static bool align_up(uint32_t value, uint32_t alignment, uint32_t * out) {
    const uint64_t v = (static_cast<uint64_t>(value) + alignment - 1) & ~static_cast<uint64_t>(alignment - 1);
    if (v > std::numeric_limits<uint32_t>::max()) return false;
    *out = static_cast<uint32_t>(v);
    return true;
}

common_tree_draft_smem_layout_status common_tree_draft_smem_layout_build(
        const common_tree_draft_smem_layout_request & request,
        common_tree_draft_smem_layout * layout) {
    if (layout == nullptr || (request.component_count > 0 && request.components == nullptr)) {
        return COMMON_TREE_DRAFT_SMEM_LAYOUT_NULL_INPUT;
    }
    if (request.component_count > layout->components.size()) {
        return COMMON_TREE_DRAFT_SMEM_LAYOUT_TOO_MANY_COMPONENTS;
    }

    common_tree_draft_smem_layout out;
    uint32_t cursor = 0;
    for (size_t i = 0; i < request.component_count; ++i) {
        const auto & c = request.components[i];
        if (!is_pow2(c.alignment)) return COMMON_TREE_DRAFT_SMEM_LAYOUT_BAD_ALIGNMENT;
        uint32_t aligned = 0;
        if (!align_up(cursor, c.alignment, &aligned)) return COMMON_TREE_DRAFT_SMEM_LAYOUT_OVERFLOW;
        if (c.size_bytes > std::numeric_limits<uint32_t>::max() - aligned) {
            return COMMON_TREE_DRAFT_SMEM_LAYOUT_OVERFLOW;
        }
        out.components[i] = {c.kind, aligned, c.size_bytes};
        cursor = aligned + c.size_bytes;
    }
    out.component_count = static_cast<uint8_t>(request.component_count);
    out.dynamic_shared_bytes = cursor;
    if (request.static_shared_bytes > std::numeric_limits<uint32_t>::max() - cursor) {
        return COMMON_TREE_DRAFT_SMEM_LAYOUT_OVERFLOW;
    }
    out.total_shared_bytes = request.static_shared_bytes + cursor;
    if (out.total_shared_bytes > request.selected_shared_limit) {
        return COMMON_TREE_DRAFT_SMEM_LAYOUT_EXCEEDS_LIMIT;
    }
    *layout = out;
    return COMMON_TREE_DRAFT_SMEM_LAYOUT_OK;
}
