#include "tree-draft-kv-visibility.h"

#include <algorithm>
#include <limits>

static bool common_tree_draft_kv_same_branch(
        const common_tree_draft_branch_handle & a,
        const common_tree_draft_branch_handle & b) {
    return a.id == b.id && a.generation == b.generation;
}

size_t common_tree_draft_kv_visibility_cutoff_count(uint32_t branch_count, bool * overflow) {
    const size_t n = static_cast<size_t>(branch_count);
    const bool failed = n != 0 && n > std::numeric_limits<size_t>::max() / n;
    if (overflow != nullptr) {
        *overflow = failed;
    }
    return failed ? 0 : n * n;
}

common_tree_draft_kv_visibility_status common_tree_draft_kv_visibility_build(
        const common_tree_draft_kv_branch_descriptor * branches,
        uint32_t branch_count,
        common_tree_draft_kv_visibility_table & output) {
    bool overflow = false;
    const size_t required = common_tree_draft_kv_visibility_cutoff_count(branch_count, &overflow);
    if (overflow) {
        return COMMON_TREE_DRAFT_KV_VISIBILITY_SIZE_OVERFLOW;
    }
    if (branch_count == 0) {
        output.branch_count = 0;
        return COMMON_TREE_DRAFT_KV_VISIBILITY_OK;
    }
    if (branches == nullptr || output.cutoffs == nullptr) {
        return COMMON_TREE_DRAFT_KV_VISIBILITY_NULL_BUFFER;
    }
    if (output.cutoff_count < required) {
        return COMMON_TREE_DRAFT_KV_VISIBILITY_OUTPUT_TOO_SMALL;
    }

    std::fill(output.cutoffs, output.cutoffs + required, COMMON_TREE_DRAFT_KV_VISIBILITY_NONE);

    for (uint32_t i = 0; i < branch_count; ++i) {
        const auto & branch = branches[i];
        if (branch.branch.id == COMMON_TREE_DRAFT_KV_ID_INVALID || branch.tip_position < -1 || branch.fork_position < -1) {
            return COMMON_TREE_DRAFT_KV_VISIBILITY_BRANCH_HANDLE;
        }
        for (uint32_t j = 0; j < i; ++j) {
            if (common_tree_draft_kv_same_branch(branch.branch, branches[j].branch)) {
                return COMMON_TREE_DRAFT_KV_VISIBILITY_DUPLICATE_BRANCH;
            }
        }

        int64_t * row = output.cutoffs + static_cast<size_t>(i) * branch_count;
        row[i] = branch.tip_position;

        if (branch.parent_branch.id == COMMON_TREE_DRAFT_KV_ID_INVALID) {
            if (branch.fork_position != -1) {
                return COMMON_TREE_DRAFT_KV_VISIBILITY_FORK_RANGE;
            }
            continue;
        }

        uint32_t parent_index = branch_count;
        for (uint32_t j = 0; j < i; ++j) {
            if (common_tree_draft_kv_same_branch(branch.parent_branch, branches[j].branch)) {
                parent_index = j;
                break;
            }
        }
        if (parent_index == branch_count) {
            return COMMON_TREE_DRAFT_KV_VISIBILITY_PARENT_ORDER;
        }
        if (branch.fork_position < 0 || branch.fork_position > branches[parent_index].tip_position ||
            branch.fork_position > branch.tip_position) {
            return COMMON_TREE_DRAFT_KV_VISIBILITY_FORK_RANGE;
        }

        const int64_t * parent_row = output.cutoffs + static_cast<size_t>(parent_index) * branch_count;
        for (uint32_t j = 0; j < i; ++j) {
            if (parent_row[j] >= 0) {
                row[j] = std::min(parent_row[j], branch.fork_position);
            }
        }
    }

    output.branch_count = branch_count;
    return COMMON_TREE_DRAFT_KV_VISIBILITY_OK;
}

bool common_tree_draft_kv_visible(
        const common_tree_draft_kv_visibility_table & table,
        uint32_t query_branch,
        int64_t query_position,
        uint32_t key_branch,
        int64_t key_position) {
    if (table.cutoffs == nullptr || query_branch >= table.branch_count || key_branch >= table.branch_count ||
        query_position < 0 || key_position < 0 || key_position > query_position) {
        return false;
    }
    const size_t index = static_cast<size_t>(query_branch) * table.branch_count + key_branch;
    if (index >= table.cutoff_count) {
        return false;
    }
    return key_position <= table.cutoffs[index];
}
