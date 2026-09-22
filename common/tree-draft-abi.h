#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct common_tree_draft_c_abi_version {
    uint32_t major;
    uint32_t minor;
} common_tree_draft_c_abi_version;

enum common_tree_draft_c_abi_feature {
    COMMON_TREE_DRAFT_C_ABI_FEATURE_HOST_ENVELOPE = 1 << 0,
    COMMON_TREE_DRAFT_C_ABI_FEATURE_LIFECYCLE = 1 << 1,
    COMMON_TREE_DRAFT_C_ABI_FEATURE_ERROR = 1 << 2,
    COMMON_TREE_DRAFT_C_ABI_FEATURE_BUFFER_OWNERSHIP = 1 << 3,
    COMMON_TREE_DRAFT_C_ABI_FEATURE_OPAQUE_HANDLES = 1 << 4,
};

typedef struct common_tree_draft_c_abi_info {
    uint32_t struct_size;
    common_tree_draft_c_abi_version version;
    uint64_t features;
} common_tree_draft_c_abi_info;

common_tree_draft_c_abi_version common_tree_draft_c_abi_version_get(void);
int common_tree_draft_c_abi_is_compatible(common_tree_draft_c_abi_version client_version);
uint64_t common_tree_draft_c_abi_features_get(void);
uint64_t common_tree_draft_c_abi_features_negotiate(uint64_t client_features);
common_tree_draft_c_abi_info common_tree_draft_c_abi_info_default(void);
int common_tree_draft_c_abi_info_get(common_tree_draft_c_abi_info * info);

#ifdef __cplusplus
}
#endif
