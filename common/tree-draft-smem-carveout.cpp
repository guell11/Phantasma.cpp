#include "tree-draft-smem-carveout.h"

#include <algorithm>

common_tree_draft_smem_carveout_status common_tree_draft_smem_carveout_plan_build(
        const common_tree_draft_device_capabilities & device,
        const common_tree_draft_smem_carveout_request & request,
        common_tree_draft_smem_carveout_plan * plan) {
    if (plan == nullptr) return COMMON_TREE_DRAFT_SMEM_CARVEOUT_NULL_OUTPUT;
    if (!common_tree_draft_device_is_sm89(device)) return COMMON_TREE_DRAFT_SMEM_CARVEOUT_NOT_SM89;

    common_tree_draft_smem_carveout_plan out;
    uint8_t previous_percent = 0;
    bool have_previous = false;
    int selected_pos = -1;
    for (size_t i = 0; i < request.candidates.size(); ++i) {
        const auto & c = request.candidates[i];
        if (!c.supported) continue;
        if (c.carveout_percent > 100 || c.max_shared_bytes > device.smem_sm ||
            (have_previous && c.carveout_percent <= previous_percent)) {
            return COMMON_TREE_DRAFT_SMEM_CARVEOUT_INVALID_TABLE;
        }
        have_previous = true;
        previous_percent = c.carveout_percent;
        out.valid_indices[out.candidate_count] = static_cast<uint8_t>(i);
        if (selected_pos < 0 && request.required_shared_bytes <= c.max_shared_bytes) {
            selected_pos = out.candidate_count;
            out.selected_index = static_cast<uint8_t>(i);
        }
        ++out.candidate_count;
    }
    if (selected_pos < 0) return COMMON_TREE_DRAFT_SMEM_CARVEOUT_UNSATISFIED;

    out.available = true;
    if (selected_pos > 0) {
        out.has_lower_neighbor = true;
        out.lower_neighbor_index = out.valid_indices[static_cast<size_t>(selected_pos - 1)];
    }
    if (selected_pos + 1 < out.candidate_count) {
        out.has_upper_neighbor = true;
        out.upper_neighbor_index = out.valid_indices[static_cast<size_t>(selected_pos + 1)];
    }
    *plan = out;
    return COMMON_TREE_DRAFT_SMEM_CARVEOUT_OK;
}
