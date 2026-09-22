#include "tree-draft-storage-view.h"

#include <cassert>
#include <memory>
#include <vector>

int main() {
    auto bytes = std::make_shared<std::vector<uint8_t>>(64);
    for (size_t i = 0; i < bytes->size(); ++i) (*bytes)[i] = static_cast<uint8_t>(i);
    std::shared_ptr<void> lifetime(bytes, bytes->data());
    common_tree_draft_storage_owner owner = { lifetime, bytes->data(), 64, 16, true };
    common_tree_draft_storage_view view = {};
    assert(common_tree_draft_storage_view_make(owner, 16, 32, true, &view) == COMMON_TREE_DRAFT_STORAGE_VIEW_OK);
    assert(view.owner_offset == 16 && view.length == 32 && view.alignment == 16 && view.readonly);
    assert(common_tree_draft_storage_view_data(view)[0] == 16);
    assert(common_tree_draft_storage_view_mutable_data(view) == nullptr);

    common_tree_draft_storage_view sub = {};
    assert(common_tree_draft_storage_view_subview(view, 8, 8, true, &sub) == COMMON_TREE_DRAFT_STORAGE_VIEW_OK);
    assert(sub.owner == view.owner && sub.owner_offset == 24 && sub.alignment == 8);
    bytes.reset();
    assert(common_tree_draft_storage_view_data(sub)[0] == 24);

    assert(common_tree_draft_storage_view_make(owner, 60, 8, true, &view) == COMMON_TREE_DRAFT_STORAGE_VIEW_RANGE);
    assert(common_tree_draft_storage_view_make(owner, 0, 8, false, &view) == COMMON_TREE_DRAFT_STORAGE_VIEW_MUTABILITY);

    auto writable_bytes = std::make_shared<std::vector<uint8_t>>(32);
    std::shared_ptr<void> writable_lifetime(writable_bytes, writable_bytes->data());
    common_tree_draft_storage_owner writable = { writable_lifetime, writable_bytes->data(), 32, 8, false };
    assert(common_tree_draft_storage_view_make(writable, 0, 32, false, &view) == COMMON_TREE_DRAFT_STORAGE_VIEW_OK);
    assert(common_tree_draft_storage_view_mutable_data(view) != nullptr);
    return 0;
}

