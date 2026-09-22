#include "tree-draft-moe-layout.h"

#include <algorithm>
#include <limits>

static common_tree_draft_tensor_role packed_role_for(common_tree_draft_tensor_role separate_role) {
    switch (separate_role) {
        case COMMON_TREE_DRAFT_TENSOR_FFN_GATE_EXPERT: return COMMON_TREE_DRAFT_TENSOR_FFN_GATE_EXPERTS;
        case COMMON_TREE_DRAFT_TENSOR_FFN_UP_EXPERT:   return COMMON_TREE_DRAFT_TENSOR_FFN_UP_EXPERTS;
        case COMMON_TREE_DRAFT_TENSOR_FFN_DOWN_EXPERT: return COMMON_TREE_DRAFT_TENSOR_FFN_DOWN_EXPERTS;
        default: return COMMON_TREE_DRAFT_TENSOR_UNKNOWN;
    }
}

static bool fused_gate_up_matches(common_tree_draft_tensor_role separate_role, common_tree_draft_tensor_role actual) {
    return actual == COMMON_TREE_DRAFT_TENSOR_FFN_GATE_UP_EXPERTS &&
           (separate_role == COMMON_TREE_DRAFT_TENSOR_FFN_GATE_EXPERT ||
            separate_role == COMMON_TREE_DRAFT_TENSOR_FFN_UP_EXPERT);
}

static size_t logical_expert_axis(const common_tree_draft_tensor_layout & layout) {
    const auto it = std::find(layout.logical_axes.begin(), layout.logical_axes.end(), COMMON_TREE_DRAFT_AXIS_EXPERT);
    return it == layout.logical_axes.end() ? SIZE_MAX : static_cast<size_t>(it - layout.logical_axes.begin());
}

static bool quant_crosses_expert(const common_tree_draft_moe_tensor_descriptor & tensor, size_t expert_axis) {
    if (tensor.quant == nullptr || tensor.quant_cross_expert_defined || expert_axis == SIZE_MAX) return false;

    // Explicit blocked layouts must never bundle multiple experts into one
    // storage block unless the encoding opts in to that semantic.
    if (!tensor.layout.block_shape.empty()) {
        const uint32_t storage_axis = tensor.layout.permutation[expert_axis];
        if (storage_axis < tensor.layout.block_shape.size() && tensor.layout.block_shape[storage_axis] > 1) return true;
    }

    // Registry-defined fixed quant blocks/groups advancing on the expert axis
    // would also consume more than one expert per encoded block.
    return tensor.quant_axis == expert_axis && tensor.quant->block_elems > 1;
}

common_tree_draft_moe_layout_status common_tree_draft_moe_validate_expert_family(
        common_tree_draft_tensor_role separate_role,
        uint32_t expert_count,
        const common_tree_draft_moe_tensor_descriptor * tensors,
        size_t tensor_count,
        common_tree_draft_moe_expert_axis_view * view) {
    if (view == nullptr || tensors == nullptr || tensor_count == 0 || expert_count == 0) {
        return COMMON_TREE_DRAFT_MOE_LAYOUT_INVALID;
    }
    const common_tree_draft_tensor_role packed_role = packed_role_for(separate_role);
    if (packed_role == COMMON_TREE_DRAFT_TENSOR_UNKNOWN) return COMMON_TREE_DRAFT_MOE_LAYOUT_ROLE;

    common_tree_draft_moe_expert_axis_view out;
    out.expert_count = expert_count;
    out.tensor_for_expert.assign(expert_count, UINT32_MAX);
    uint32_t packed_index = UINT32_MAX;
    size_t packed_axis = SIZE_MAX;

    for (size_t i = 0; i < tensor_count; ++i) {
        const auto & tensor = tensors[i];
        if (common_tree_draft_tensor_layout_validate(tensor.layout) != COMMON_TREE_DRAFT_TENSOR_LAYOUT_OK) {
            return COMMON_TREE_DRAFT_MOE_LAYOUT_INVALID;
        }

        if (tensor.key.role == separate_role) {
            if (tensor.key.expert < 0 || static_cast<uint32_t>(tensor.key.expert) >= expert_count) {
                return COMMON_TREE_DRAFT_MOE_LAYOUT_COVERAGE;
            }
            if (logical_expert_axis(tensor.layout) != SIZE_MAX) return COMMON_TREE_DRAFT_MOE_LAYOUT_EXPERT_AXIS;
            auto & slot = out.tensor_for_expert[static_cast<uint32_t>(tensor.key.expert)];
            if (slot != UINT32_MAX) return COMMON_TREE_DRAFT_MOE_LAYOUT_DUPLICATE;
            slot = static_cast<uint32_t>(i);
            continue;
        }

        if (tensor.key.role == packed_role || fused_gate_up_matches(separate_role, tensor.key.role)) {
            if (tensor.key.expert != -1 || packed_index != UINT32_MAX) return COMMON_TREE_DRAFT_MOE_LAYOUT_DUPLICATE;
            const size_t expert_axis = logical_expert_axis(tensor.layout);
            if (expert_axis == SIZE_MAX || tensor.layout.logical_shape[expert_axis] != expert_count) {
                return COMMON_TREE_DRAFT_MOE_LAYOUT_EXPERT_AXIS;
            }
            if (quant_crosses_expert(tensor, expert_axis)) {
                return COMMON_TREE_DRAFT_MOE_LAYOUT_QUANT_CROSSES_EXPERT;
            }
            packed_index = static_cast<uint32_t>(i);
            packed_axis = expert_axis;
        }
    }

    const bool any_separate = std::any_of(out.tensor_for_expert.begin(), out.tensor_for_expert.end(), [](uint32_t v) { return v != UINT32_MAX; });
    if (packed_index != UINT32_MAX && any_separate) return COMMON_TREE_DRAFT_MOE_LAYOUT_DUPLICATE;
    if (packed_index != UINT32_MAX) {
        out.packed = true;
        out.packed_tensor = packed_index;
        out.expert_axis = packed_axis;
        *view = std::move(out);
        return COMMON_TREE_DRAFT_MOE_LAYOUT_OK;
    }
    if (!std::all_of(out.tensor_for_expert.begin(), out.tensor_for_expert.end(), [](uint32_t v) { return v != UINT32_MAX; })) {
        return COMMON_TREE_DRAFT_MOE_LAYOUT_COVERAGE;
    }
    *view = std::move(out);
    return COMMON_TREE_DRAFT_MOE_LAYOUT_OK;
}

common_tree_draft_moe_layout_status common_tree_draft_moe_validate_router(
        uint64_t hidden_size,
        uint32_t expert_count,
        const common_tree_draft_moe_tensor_descriptor & router) {
    if (hidden_size == 0 || expert_count == 0 || router.key.role != COMMON_TREE_DRAFT_TENSOR_FFN_ROUTER ||
        router.key.expert != -1 || common_tree_draft_tensor_layout_validate(router.layout) != COMMON_TREE_DRAFT_TENSOR_LAYOUT_OK) {
        return COMMON_TREE_DRAFT_MOE_LAYOUT_ROUTER;
    }
    if (router.layout.logical_axes.size() != 2 || router.layout.logical_shape.size() != 2) {
        return COMMON_TREE_DRAFT_MOE_LAYOUT_ROUTER;
    }
    size_t hidden_axis = SIZE_MAX, expert_axis = SIZE_MAX;
    for (size_t i = 0; i < 2; ++i) {
        if (router.layout.logical_axes[i] == COMMON_TREE_DRAFT_AXIS_HIDDEN || router.layout.logical_axes[i] == COMMON_TREE_DRAFT_AXIS_INPUT) hidden_axis = i;
        if (router.layout.logical_axes[i] == COMMON_TREE_DRAFT_AXIS_EXPERT) expert_axis = i;
    }
    if (hidden_axis == SIZE_MAX || expert_axis == SIZE_MAX || hidden_axis == expert_axis ||
        router.layout.logical_shape[hidden_axis] != hidden_size || router.layout.logical_shape[expert_axis] != expert_count) {
        return COMMON_TREE_DRAFT_MOE_LAYOUT_ROUTER;
    }
    if (quant_crosses_expert(router, expert_axis)) return COMMON_TREE_DRAFT_MOE_LAYOUT_QUANT_CROSSES_EXPERT;
    return COMMON_TREE_DRAFT_MOE_LAYOUT_OK;
}
