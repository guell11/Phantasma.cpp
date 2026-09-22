#include "tree-draft-path.h"

static uint64_t common_tree_draft_path_mix(uint64_t x) {
    x += UINT64_C(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}

static uint64_t common_tree_draft_path_nonzero(uint64_t value) {
    return value == 0 ? UINT64_C(0xd1b54a32d192ed03) : value;
}

uint64_t common_tree_draft_path_root(uint64_t seed) {
    return common_tree_draft_path_nonzero(
            common_tree_draft_path_mix(seed ^ UINT64_C(0x524f4f545f504154)));
}

common_tree_draft_path_status common_tree_draft_path_child(
        uint64_t parent_path,
        int32_t token,
        uint32_t child_ordinal,
        uint32_t depth,
        uint64_t * path_id) {
    if (path_id == nullptr || parent_path == 0) {
        return COMMON_TREE_DRAFT_PATH_INVALID_DEPTH;
    }
    if (token < 0) {
        return COMMON_TREE_DRAFT_PATH_INVALID_TOKEN;
    }
    if (depth == 0) {
        return COMMON_TREE_DRAFT_PATH_INVALID_DEPTH;
    }

    uint64_t value = common_tree_draft_path_mix(parent_path ^ UINT64_C(0x4348494c445f5041));
    value = common_tree_draft_path_mix(value ^ static_cast<uint32_t>(token));
    value = common_tree_draft_path_mix(value ^
            (static_cast<uint64_t>(child_ordinal) << 32) ^ static_cast<uint64_t>(depth));
    *path_id = common_tree_draft_path_nonzero(value);
    return COMMON_TREE_DRAFT_PATH_OK;
}

common_tree_draft_path_status common_tree_draft_path_check_unique(
        uint64_t path_id,
        const uint64_t * existing,
        size_t n_existing) {
    if (path_id == 0) {
        return COMMON_TREE_DRAFT_PATH_COLLISION;
    }
    if (n_existing > 0 && existing == nullptr) {
        return COMMON_TREE_DRAFT_PATH_COLLISION;
    }
    for (size_t i = 0; i < n_existing; ++i) {
        if (existing[i] == path_id) {
            return COMMON_TREE_DRAFT_PATH_COLLISION;
        }
    }
    return COMMON_TREE_DRAFT_PATH_OK;
}

