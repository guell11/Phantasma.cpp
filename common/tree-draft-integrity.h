#pragma once

#include "tree-draft-storage-view.h"

#include <cstddef>
#include <cstdint>
#include <string>

enum common_tree_draft_integrity_result : uint32_t {
    COMMON_TREE_DRAFT_INTEGRITY_UNVERIFIED = 0,
    COMMON_TREE_DRAFT_INTEGRITY_VERIFIED,
    COMMON_TREE_DRAFT_INTEGRITY_FAILED,
};

struct common_tree_draft_integrity_digest {
    std::string algorithm;
    std::string hex;
    std::string provenance;
};

struct common_tree_draft_integrity_check {
    common_tree_draft_integrity_result result = COMMON_TREE_DRAFT_INTEGRITY_UNVERIFIED;
    common_tree_draft_integrity_digest actual;
    common_tree_draft_integrity_digest expected;
};

enum common_tree_draft_integrity_status : uint32_t {
    COMMON_TREE_DRAFT_INTEGRITY_OK = 0,
    COMMON_TREE_DRAFT_INTEGRITY_NULL_BUFFER,
    COMMON_TREE_DRAFT_INTEGRITY_UNSUPPORTED_ALGORITHM,
    COMMON_TREE_DRAFT_INTEGRITY_INVALID_EXPECTED,
};

common_tree_draft_integrity_status common_tree_draft_integrity_hash(
        const common_tree_draft_storage_view & view,
        const std::string & algorithm,
        common_tree_draft_integrity_digest * digest);

common_tree_draft_integrity_status common_tree_draft_integrity_verify(
        const common_tree_draft_storage_view & view,
        const common_tree_draft_integrity_digest * expected,
        common_tree_draft_integrity_check * check);

