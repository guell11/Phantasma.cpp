#include "tree-draft-storage-view.h"

#include <numeric>

static bool common_tree_draft_storage_view_valid_alignment(uint64_t alignment) {
    return alignment != 0 && (alignment & (alignment - 1)) == 0;
}

static uint64_t common_tree_draft_storage_view_subalignment(uint64_t base_alignment, uint64_t offset) {
    if (offset == 0) return base_alignment;
    return std::gcd(base_alignment, offset);
}

static bool common_tree_draft_storage_view_range(uint64_t total, uint64_t offset, uint64_t length) {
    return offset <= total && length <= total - offset;
}

common_tree_draft_storage_view_status common_tree_draft_storage_view_make(
        const common_tree_draft_storage_owner & owner,
        uint64_t offset,
        uint64_t length,
        bool readonly,
        common_tree_draft_storage_view * view) {
    if (view == nullptr || owner.lifetime == nullptr || (owner.length > 0 && owner.base == nullptr)) {
        return COMMON_TREE_DRAFT_STORAGE_VIEW_NULL_OWNER;
    }
    if (!common_tree_draft_storage_view_valid_alignment(owner.alignment)) {
        return COMMON_TREE_DRAFT_STORAGE_VIEW_ALIGNMENT;
    }
    if (!common_tree_draft_storage_view_range(owner.length, offset, length)) {
        return COMMON_TREE_DRAFT_STORAGE_VIEW_RANGE;
    }
    if (owner.readonly && !readonly) {
        return COMMON_TREE_DRAFT_STORAGE_VIEW_MUTABILITY;
    }
    *view = {
        owner.lifetime,
        owner.base + offset,
        offset,
        length,
        common_tree_draft_storage_view_subalignment(owner.alignment, offset),
        owner.readonly || readonly,
    };
    return COMMON_TREE_DRAFT_STORAGE_VIEW_OK;
}

common_tree_draft_storage_view_status common_tree_draft_storage_view_subview(
        const common_tree_draft_storage_view & parent,
        uint64_t offset,
        uint64_t length,
        bool readonly,
        common_tree_draft_storage_view * view) {
    if (view == nullptr || parent.owner == nullptr || (parent.length > 0 && parent.data == nullptr)) {
        return COMMON_TREE_DRAFT_STORAGE_VIEW_NULL_OWNER;
    }
    if (!common_tree_draft_storage_view_range(parent.length, offset, length)) {
        return COMMON_TREE_DRAFT_STORAGE_VIEW_RANGE;
    }
    if (parent.readonly && !readonly) {
        return COMMON_TREE_DRAFT_STORAGE_VIEW_MUTABILITY;
    }
    *view = {
        parent.owner,
        parent.data + offset,
        parent.owner_offset + offset,
        length,
        common_tree_draft_storage_view_subalignment(parent.alignment, offset),
        parent.readonly || readonly,
    };
    return COMMON_TREE_DRAFT_STORAGE_VIEW_OK;
}

const uint8_t * common_tree_draft_storage_view_data(const common_tree_draft_storage_view & view) {
    return view.data;
}

uint8_t * common_tree_draft_storage_view_mutable_data(common_tree_draft_storage_view & view) {
    return view.readonly ? nullptr : view.data;
}

