#include "tree-draft-abi.h"

static constexpr common_tree_draft_c_abi_version common_tree_draft_c_abi_version_value = { 1, 0 };

static constexpr uint64_t common_tree_draft_c_abi_features =
        COMMON_TREE_DRAFT_C_ABI_FEATURE_HOST_ENVELOPE |
        COMMON_TREE_DRAFT_C_ABI_FEATURE_LIFECYCLE |
        COMMON_TREE_DRAFT_C_ABI_FEATURE_ERROR |
        COMMON_TREE_DRAFT_C_ABI_FEATURE_BUFFER_OWNERSHIP |
        COMMON_TREE_DRAFT_C_ABI_FEATURE_OPAQUE_HANDLES;

common_tree_draft_c_abi_version common_tree_draft_c_abi_version_get(void) {
    return common_tree_draft_c_abi_version_value;
}

int common_tree_draft_c_abi_is_compatible(common_tree_draft_c_abi_version client_version) {
    return client_version.major == common_tree_draft_c_abi_version_value.major &&
           client_version.minor <= common_tree_draft_c_abi_version_value.minor;
}

uint64_t common_tree_draft_c_abi_features_get(void) {
    return common_tree_draft_c_abi_features;
}

uint64_t common_tree_draft_c_abi_features_negotiate(uint64_t client_features) {
    return client_features & common_tree_draft_c_abi_features;
}

common_tree_draft_c_abi_info common_tree_draft_c_abi_info_default(void) {
    return {
        /* .struct_size = */ sizeof(common_tree_draft_c_abi_info),
        /* .version = */ common_tree_draft_c_abi_version_value,
        /* .features = */ common_tree_draft_c_abi_features,
    };
}

int common_tree_draft_c_abi_info_get(common_tree_draft_c_abi_info * info) {
    if (info == nullptr || info->struct_size < offsetof(common_tree_draft_c_abi_info, version) + sizeof(info->version)) {
        return 0;
    }

    info->version = common_tree_draft_c_abi_version_value;
    if (info->struct_size >= offsetof(common_tree_draft_c_abi_info, features) + sizeof(info->features)) {
        info->features = common_tree_draft_c_abi_features;
    }

    return 1;
}
