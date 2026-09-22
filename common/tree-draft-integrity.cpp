#include "tree-draft-integrity.h"

#include <iomanip>
#include <sstream>

static std::string fnv1a64_hex(const uint8_t * data, uint64_t length) {
    uint64_t h = 14695981039346656037ull;
    for (uint64_t i = 0; i < length; ++i) {
        h ^= data[i];
        h *= 1099511628211ull;
    }
    std::ostringstream ss;
    ss << std::hex << std::setfill('0') << std::setw(16) << h;
    return ss.str();
}

common_tree_draft_integrity_status common_tree_draft_integrity_hash(
        const common_tree_draft_storage_view & view,
        const std::string & algorithm,
        common_tree_draft_integrity_digest * digest) {
    if (digest == nullptr || view.owner == nullptr || (view.length > 0 && view.data == nullptr)) {
        return COMMON_TREE_DRAFT_INTEGRITY_NULL_BUFFER;
    }
    if (algorithm != "fnv1a64") return COMMON_TREE_DRAFT_INTEGRITY_UNSUPPORTED_ALGORITHM;
    digest->algorithm = algorithm;
    digest->hex = fnv1a64_hex(view.data, view.length);
    return COMMON_TREE_DRAFT_INTEGRITY_OK;
}

common_tree_draft_integrity_status common_tree_draft_integrity_verify(
        const common_tree_draft_storage_view & view,
        const common_tree_draft_integrity_digest * expected,
        common_tree_draft_integrity_check * check) {
    if (check == nullptr) return COMMON_TREE_DRAFT_INTEGRITY_NULL_BUFFER;
    *check = {};
    if (expected == nullptr || expected->hex.empty()) {
        check->result = COMMON_TREE_DRAFT_INTEGRITY_UNVERIFIED;
        return COMMON_TREE_DRAFT_INTEGRITY_OK;
    }
    if (expected->algorithm.empty()) return COMMON_TREE_DRAFT_INTEGRITY_INVALID_EXPECTED;
    common_tree_draft_integrity_digest actual;
    const auto status = common_tree_draft_integrity_hash(view, expected->algorithm, &actual);
    if (status != COMMON_TREE_DRAFT_INTEGRITY_OK) return status;
    check->actual = actual;
    check->expected = *expected;
    check->result = actual.hex == expected->hex ? COMMON_TREE_DRAFT_INTEGRITY_VERIFIED : COMMON_TREE_DRAFT_INTEGRITY_FAILED;
    return COMMON_TREE_DRAFT_INTEGRITY_OK;
}

