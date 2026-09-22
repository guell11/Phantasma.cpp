#pragma once

#include <cstddef>
#include <cstdint>

enum common_tree_draft_path_status : uint32_t {
    COMMON_TREE_DRAFT_PATH_OK = 0,
    COMMON_TREE_DRAFT_PATH_INVALID_TOKEN,
    COMMON_TREE_DRAFT_PATH_INVALID_DEPTH,
    COMMON_TREE_DRAFT_PATH_COLLISION,
};

uint64_t common_tree_draft_path_root(uint64_t seed);

common_tree_draft_path_status common_tree_draft_path_child(
        uint64_t parent_path,
        int32_t token,
        uint32_t child_ordinal,
        uint32_t depth,
        uint64_t * path_id);

common_tree_draft_path_status common_tree_draft_path_check_unique(
        uint64_t path_id,
        const uint64_t * existing,
        size_t n_existing);

