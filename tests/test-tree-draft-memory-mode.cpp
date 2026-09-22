#include "tree-draft-memory-mode.h"

#include <cassert>

int main() {
    uint32_t rows[] = {0, 1};
    common_tree_draft_kv_gather_segment segments[] = {{{7, 2}, 0, 4}};
    common_tree_draft_kv_gather_metadata gather = {rows, 2, segments, 1, 1, 1};

    common_tree_draft_layer_memory_view view;
    const common_tree_draft_layer_memory_metadata attention = {COMMON_TREE_DRAFT_MEMORY_ATTENTION};
    assert(common_tree_draft_memory_mode_bind(attention, &gather, 3, 5, &view) == COMMON_TREE_DRAFT_MEMORY_MODE_OK);
    assert(view.paged_kv == &gather);
    assert(view.sequence_index == UINT32_MAX && view.branch_index == UINT32_MAX);
    assert(!view.requires_state_fork && !view.requires_state_rollback);
    assert(common_tree_draft_memory_mode_bind(attention, nullptr, 3, 5, &view) ==
           COMMON_TREE_DRAFT_MEMORY_MODE_MISSING_PAGED_KV);

    const common_tree_draft_layer_memory_metadata recurrent = {COMMON_TREE_DRAFT_MEMORY_RECURRENT};
    assert(common_tree_draft_memory_mode_bind(recurrent, nullptr, 3, 5, &view) == COMMON_TREE_DRAFT_MEMORY_MODE_OK);
    assert(view.paged_kv == nullptr);
    assert(view.sequence_index == 3 && view.branch_index == 5);
    assert(view.requires_state_fork && view.requires_state_rollback);
    assert(common_tree_draft_memory_mode_bind(recurrent, &gather, 3, 5, &view) ==
           COMMON_TREE_DRAFT_MEMORY_MODE_RECURRENT_USES_PAGED_KV);
    assert(common_tree_draft_memory_mode_bind(recurrent, nullptr, UINT32_MAX, 5, &view) ==
           COMMON_TREE_DRAFT_MEMORY_MODE_MISSING_RECURRENT_IDENTITY);

    const common_tree_draft_layer_memory_metadata invalid = {static_cast<common_tree_draft_memory_mode>(99)};
    assert(common_tree_draft_memory_mode_bind(invalid, nullptr, 0, 0, &view) ==
           COMMON_TREE_DRAFT_MEMORY_MODE_INVALID_MODE);
    assert(common_tree_draft_memory_mode_bind(attention, &gather, 0, 0, nullptr) ==
           COMMON_TREE_DRAFT_MEMORY_MODE_NULL_OUTPUT);
    return 0;
}
