#pragma once

#include "tree-draft-quant-registry.h"
#include "tree-draft-tensor-layout.h"
#include "tree-draft-tensor-role.h"

#include <cstddef>
#include <cstdint>
#include <vector>

struct common_tree_draft_moe_tensor_descriptor {
    common_tree_draft_tensor_role_key key;
    common_tree_draft_tensor_layout layout;
    const common_tree_draft_quant_type * quant = nullptr;

    // Logical axis on which quant blocks/groups advance. SIZE_MAX means that
    // the encoding is not represented as an axis-local block scheme here.
    size_t quant_axis = SIZE_MAX;

    // Only formats whose specification explicitly packs one quant block
    // across multiple experts may set this. Default is deliberately false.
    bool quant_cross_expert_defined = false;
};

struct common_tree_draft_moe_expert_axis_view {
    bool packed = false;
    uint32_t expert_count = 0;
    size_t expert_axis = SIZE_MAX;
    std::vector<uint32_t> tensor_for_expert;
    uint32_t packed_tensor = UINT32_MAX;
};

enum common_tree_draft_moe_layout_status : uint32_t {
    COMMON_TREE_DRAFT_MOE_LAYOUT_OK = 0,
    COMMON_TREE_DRAFT_MOE_LAYOUT_INVALID,
    COMMON_TREE_DRAFT_MOE_LAYOUT_ROLE,
    COMMON_TREE_DRAFT_MOE_LAYOUT_COVERAGE,
    COMMON_TREE_DRAFT_MOE_LAYOUT_DUPLICATE,
    COMMON_TREE_DRAFT_MOE_LAYOUT_EXPERT_AXIS,
    COMMON_TREE_DRAFT_MOE_LAYOUT_ROUTER,
    COMMON_TREE_DRAFT_MOE_LAYOUT_QUANT_CROSSES_EXPERT,
};

common_tree_draft_moe_layout_status common_tree_draft_moe_validate_expert_family(
        common_tree_draft_tensor_role separate_role,
        uint32_t expert_count,
        const common_tree_draft_moe_tensor_descriptor * tensors,
        size_t tensor_count,
        common_tree_draft_moe_expert_axis_view * view);

common_tree_draft_moe_layout_status common_tree_draft_moe_validate_router(
        uint64_t hidden_size,
        uint32_t expert_count,
        const common_tree_draft_moe_tensor_descriptor & router);
