#include "tree-draft-memory-mode.h"

common_tree_draft_memory_mode_status common_tree_draft_memory_mode_bind(
        const common_tree_draft_layer_memory_metadata & layer,
        const common_tree_draft_kv_gather_metadata * paged_kv,
        uint32_t sequence_index,
        uint32_t branch_index,
        common_tree_draft_layer_memory_view * output) {
    if (output == nullptr) return COMMON_TREE_DRAFT_MEMORY_MODE_NULL_OUTPUT;

    common_tree_draft_layer_memory_view view;
    switch (layer.mode) {
        case COMMON_TREE_DRAFT_MEMORY_ATTENTION:
            if (paged_kv == nullptr) return COMMON_TREE_DRAFT_MEMORY_MODE_MISSING_PAGED_KV;
            view.paged_kv = paged_kv;
            break;
        case COMMON_TREE_DRAFT_MEMORY_RECURRENT:
            if (paged_kv != nullptr) return COMMON_TREE_DRAFT_MEMORY_MODE_RECURRENT_USES_PAGED_KV;
            if (sequence_index == UINT32_MAX || branch_index == UINT32_MAX) {
                return COMMON_TREE_DRAFT_MEMORY_MODE_MISSING_RECURRENT_IDENTITY;
            }
            view.sequence_index = sequence_index;
            view.branch_index = branch_index;
            view.requires_state_fork = true;
            view.requires_state_rollback = true;
            break;
        default:
            return COMMON_TREE_DRAFT_MEMORY_MODE_INVALID_MODE;
    }

    *output = view;
    return COMMON_TREE_DRAFT_MEMORY_MODE_OK;
}
