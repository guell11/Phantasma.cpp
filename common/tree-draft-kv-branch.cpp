#include "tree-draft-kv-branch.h"

#include <limits>

common_tree_draft_kv_branch_status common_tree_draft_kv_branch_validate(
        const common_tree_draft_kv_branch_descriptor & branch,
        const common_tree_draft_kv_page_geometry & geometry) {
    if (branch.branch.id == COMMON_TREE_DRAFT_KV_ID_INVALID) return COMMON_TREE_DRAFT_KV_BRANCH_HANDLE;
    if (branch.tip_position < -1 || branch.fork_position < -1 || branch.fork_position > branch.tip_position) {
        return COMMON_TREE_DRAFT_KV_BRANCH_RANGE;
    }
    if (branch.span_count > 0 && branch.spans == nullptr) return COMMON_TREE_DRAFT_KV_BRANCH_NULL_BUFFER;
    if (branch.parent_branch.id == COMMON_TREE_DRAFT_KV_ID_INVALID && branch.fork_position != -1) {
        return COMMON_TREE_DRAFT_KV_BRANCH_HANDLE;
    }
    uint64_t total_tokens = 0;
    for (uint32_t i = 0; i < branch.span_count; ++i) {
        const auto & span = branch.spans[i];
        if (span.page.id == COMMON_TREE_DRAFT_KV_ID_INVALID || span.lo >= span.hi || span.hi > geometry.tokens_per_page) {
            return COMMON_TREE_DRAFT_KV_BRANCH_PAGE;
        }
        total_tokens += static_cast<uint64_t>(span.hi - span.lo);
        if (total_tokens > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) return COMMON_TREE_DRAFT_KV_BRANCH_LENGTH;
    }
    const uint64_t expected = branch.tip_position < 0 ? 0 : static_cast<uint64_t>(branch.tip_position) + 1;
    if (total_tokens != expected) return COMMON_TREE_DRAFT_KV_BRANCH_LENGTH;
    return COMMON_TREE_DRAFT_KV_BRANCH_OK;
}

common_tree_draft_kv_branch_status common_tree_draft_kv_branch_lookup(
        const common_tree_draft_kv_branch_descriptor & branch,
        int64_t logical_position,
        common_tree_draft_page_address * address) {
    if (address == nullptr) return COMMON_TREE_DRAFT_KV_BRANCH_NULL_BUFFER;
    if (logical_position < 0 || logical_position > branch.tip_position) return COMMON_TREE_DRAFT_KV_BRANCH_NOT_FOUND;
    int64_t base = 0;
    for (uint32_t i = 0; i < branch.span_count; ++i) {
        const auto & span = branch.spans[i];
        const int64_t length = static_cast<int64_t>(span.hi - span.lo);
        if (logical_position >= base && logical_position < base + length) {
            *address = { span.page, static_cast<uint32_t>(span.lo + (logical_position - base)) };
            return COMMON_TREE_DRAFT_KV_BRANCH_OK;
        }
        base += length;
    }
    return COMMON_TREE_DRAFT_KV_BRANCH_NOT_FOUND;
}

