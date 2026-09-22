#include "tree-draft-abi.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

struct common_tree_draft_c_abi_info_v1 {
    uint32_t struct_size;
    common_tree_draft_c_abi_version version;
};

static void test_version_compatibility(void) {
    const common_tree_draft_c_abi_version library = common_tree_draft_c_abi_version_get();
    const common_tree_draft_c_abi_version older_minor = { library.major, 0 };
    const common_tree_draft_c_abi_version newer_minor = { library.major, library.minor + 1 };
    const common_tree_draft_c_abi_version other_major = { library.major + 1, 0 };

    assert(library.major == 1);
    assert(library.minor == 0);
    assert(common_tree_draft_c_abi_is_compatible(library));
    assert(common_tree_draft_c_abi_is_compatible(older_minor));
    assert(!common_tree_draft_c_abi_is_compatible(newer_minor));
    assert(!common_tree_draft_c_abi_is_compatible(other_major));
}

static void test_feature_negotiation(void) {
    const uint64_t library = common_tree_draft_c_abi_features_get();
    const uint64_t client = COMMON_TREE_DRAFT_C_ABI_FEATURE_HOST_ENVELOPE |
                            COMMON_TREE_DRAFT_C_ABI_FEATURE_ERROR |
                            (UINT64_C(1) << 63);

    assert((library & COMMON_TREE_DRAFT_C_ABI_FEATURE_BUFFER_OWNERSHIP) != 0);
    assert((library & COMMON_TREE_DRAFT_C_ABI_FEATURE_OPAQUE_HANDLES) != 0);
    assert(common_tree_draft_c_abi_features_negotiate(client) ==
           (COMMON_TREE_DRAFT_C_ABI_FEATURE_HOST_ENVELOPE | COMMON_TREE_DRAFT_C_ABI_FEATURE_ERROR));
}

static void test_struct_size_forward_compatibility(void) {
    common_tree_draft_c_abi_info full = common_tree_draft_c_abi_info_default();
    struct common_tree_draft_c_abi_info_v1 old = {
        sizeof(old),
        { 99, 99 },
    };
    common_tree_draft_c_abi_info too_small = { 0, { 0, 0 }, 0 };

    assert(full.struct_size == sizeof(full));
    assert(common_tree_draft_c_abi_info_get(&full));
    assert(full.version.major == 1);
    assert(full.features == common_tree_draft_c_abi_features_get());

    assert(common_tree_draft_c_abi_info_get((common_tree_draft_c_abi_info *) &old));
    assert(old.version.major == 1);
    assert(old.version.minor == 0);

    assert(!common_tree_draft_c_abi_info_get(&too_small));
    assert(!common_tree_draft_c_abi_info_get(NULL));
}

int main(void) {
    test_version_compatibility();
    test_feature_negotiation();
    test_struct_size_forward_compatibility();
    return 0;
}
