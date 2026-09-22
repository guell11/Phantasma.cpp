#pragma once

#include "tree-draft-storage-view.h"

#include <cstdint>
#include <memory>
#include <string>

struct common_tree_draft_mapping_owner {
    std::shared_ptr<void> lifetime;
    uint8_t * base = nullptr;
    uint64_t length = 0;
    uint64_t alignment = 1;
    bool readonly = true;
    std::string provenance;
};

enum common_tree_draft_mapping_owner_status : uint32_t {
    COMMON_TREE_DRAFT_MAPPING_OWNER_OK = 0,
    COMMON_TREE_DRAFT_MAPPING_OWNER_INVALID,
    COMMON_TREE_DRAFT_MAPPING_OWNER_RANGE,
};

common_tree_draft_mapping_owner_status common_tree_draft_mapping_owner_make_view(
        const common_tree_draft_mapping_owner & mapping,
        uint64_t offset,
        uint64_t length,
        common_tree_draft_storage_view * view);

common_tree_draft_mapping_owner common_tree_draft_mapping_owner_retain(
        const common_tree_draft_mapping_owner & mapping);

void common_tree_draft_mapping_owner_release(common_tree_draft_mapping_owner * mapping);

