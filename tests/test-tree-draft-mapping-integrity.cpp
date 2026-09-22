#include "tree-draft-integrity.h"
#include "tree-draft-mapping-owner.h"

#include <cassert>
#include <memory>
#include <vector>

int main() {
    auto bytes = std::make_shared<std::vector<uint8_t>>(32);
    for (size_t i = 0; i < bytes->size(); ++i) (*bytes)[i] = static_cast<uint8_t>(i + 1);
    std::weak_ptr<std::vector<uint8_t>> weak = bytes;
    std::shared_ptr<void> lifetime(bytes, bytes->data());
    common_tree_draft_mapping_owner mapping = { lifetime, bytes->data(), 32, 16, true, "fixture.bin" };
    common_tree_draft_storage_view view = {};
    assert(common_tree_draft_mapping_owner_make_view(mapping, 8, 16, &view) == COMMON_TREE_DRAFT_MAPPING_OWNER_OK);

    auto retained = common_tree_draft_mapping_owner_retain(mapping);
    bytes.reset();
    lifetime.reset();
    common_tree_draft_mapping_owner_release(&mapping);
    assert(!weak.expired());
    assert(view.data[0] == 9);
    common_tree_draft_mapping_owner_release(&retained);
    assert(!weak.expired()); // view still pins mapping bytes

    common_tree_draft_integrity_digest digest;
    assert(common_tree_draft_integrity_hash(view, "fnv1a64", &digest) == COMMON_TREE_DRAFT_INTEGRITY_OK);
    assert(digest.hex.size() == 16);
    common_tree_draft_integrity_check check;
    common_tree_draft_integrity_digest expected = digest;
    expected.provenance = "manifest";
    assert(common_tree_draft_integrity_verify(view, &expected, &check) == COMMON_TREE_DRAFT_INTEGRITY_OK);
    assert(check.result == COMMON_TREE_DRAFT_INTEGRITY_VERIFIED);
    expected.hex[0] = expected.hex[0] == '0' ? '1' : '0';
    assert(common_tree_draft_integrity_verify(view, &expected, &check) == COMMON_TREE_DRAFT_INTEGRITY_OK);
    assert(check.result == COMMON_TREE_DRAFT_INTEGRITY_FAILED);
    assert(common_tree_draft_integrity_verify(view, nullptr, &check) == COMMON_TREE_DRAFT_INTEGRITY_OK);
    assert(check.result == COMMON_TREE_DRAFT_INTEGRITY_UNVERIFIED);
    return 0;
}

