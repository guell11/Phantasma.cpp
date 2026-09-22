#include "tree-draft-runtime-status.h"

#include <cassert>
#include <cstring>

int main() {
    common_tree_draft_node nodes[4] = {};
    int32_t frontier[4] = {};
    common_tree_draft_arena arena = { nodes, frontier, 4, 3, 2 };
    const auto result = common_tree_draft_runtime_snapshot(COMMON_TREE_DRAFT_RUNTIME_CAPACITY, arena);
    assert(result.status == COMMON_TREE_DRAFT_RUNTIME_CAPACITY);
    assert(result.committed_node_count == 3);
    assert(result.committed_frontier_count == 2);
    assert(common_tree_draft_runtime_can_fallback(COMMON_TREE_DRAFT_RUNTIME_CAPACITY));
    assert(common_tree_draft_runtime_can_fallback(COMMON_TREE_DRAFT_RUNTIME_UNSUPPORTED));
    assert(!common_tree_draft_runtime_can_fallback(COMMON_TREE_DRAFT_RUNTIME_OK));
    assert(!common_tree_draft_runtime_can_fallback(COMMON_TREE_DRAFT_RUNTIME_CANCELLED));
    assert(std::strcmp(common_tree_draft_runtime_status_name(COMMON_TREE_DRAFT_RUNTIME_MODEL), "model") == 0);
    return 0;
}

