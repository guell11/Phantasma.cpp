#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

struct common_tree_draft_storage_owner {
    std::shared_ptr<void> lifetime;
    uint8_t * base;
    uint64_t length;
    uint64_t alignment;
    bool readonly;
};

struct common_tree_draft_storage_view {
    std::shared_ptr<void> owner;
    uint8_t * data;
    uint64_t owner_offset;
    uint64_t length;
    uint64_t alignment;
    bool readonly;
};

enum common_tree_draft_storage_view_status : uint32_t {
    COMMON_TREE_DRAFT_STORAGE_VIEW_OK = 0,
    COMMON_TREE_DRAFT_STORAGE_VIEW_NULL_OWNER,
    COMMON_TREE_DRAFT_STORAGE_VIEW_RANGE,
    COMMON_TREE_DRAFT_STORAGE_VIEW_ALIGNMENT,
    COMMON_TREE_DRAFT_STORAGE_VIEW_MUTABILITY,
};

common_tree_draft_storage_view_status common_tree_draft_storage_view_make(
        const common_tree_draft_storage_owner & owner,
        uint64_t offset,
        uint64_t length,
        bool readonly,
        common_tree_draft_storage_view * view);

common_tree_draft_storage_view_status common_tree_draft_storage_view_subview(
        const common_tree_draft_storage_view & parent,
        uint64_t offset,
        uint64_t length,
        bool readonly,
        common_tree_draft_storage_view * view);

const uint8_t * common_tree_draft_storage_view_data(const common_tree_draft_storage_view & view);
uint8_t * common_tree_draft_storage_view_mutable_data(common_tree_draft_storage_view & view);

