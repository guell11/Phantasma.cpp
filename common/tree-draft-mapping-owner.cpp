#include "tree-draft-mapping-owner.h"

common_tree_draft_mapping_owner_status common_tree_draft_mapping_owner_make_view(
        const common_tree_draft_mapping_owner & mapping,
        uint64_t offset,
        uint64_t length,
        common_tree_draft_storage_view * view) {
    if (mapping.lifetime == nullptr || (mapping.length > 0 && mapping.base == nullptr) || mapping.alignment == 0) {
        return COMMON_TREE_DRAFT_MAPPING_OWNER_INVALID;
    }
    common_tree_draft_storage_owner owner = {
        mapping.lifetime,
        mapping.base,
        mapping.length,
        mapping.alignment,
        mapping.readonly,
    };
    const auto status = common_tree_draft_storage_view_make(owner, offset, length, true, view);
    if (status == COMMON_TREE_DRAFT_STORAGE_VIEW_RANGE) return COMMON_TREE_DRAFT_MAPPING_OWNER_RANGE;
    if (status != COMMON_TREE_DRAFT_STORAGE_VIEW_OK) return COMMON_TREE_DRAFT_MAPPING_OWNER_INVALID;
    return COMMON_TREE_DRAFT_MAPPING_OWNER_OK;
}

common_tree_draft_mapping_owner common_tree_draft_mapping_owner_retain(
        const common_tree_draft_mapping_owner & mapping) {
    return mapping;
}

void common_tree_draft_mapping_owner_release(common_tree_draft_mapping_owner * mapping) {
    if (mapping == nullptr) return;
    mapping->lifetime.reset();
    mapping->base = nullptr;
    mapping->length = 0;
    mapping->alignment = 1;
    mapping->provenance.clear();
}

