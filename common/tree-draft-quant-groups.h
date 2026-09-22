#pragma once

#include "tree-draft-external-quant.h"
#include "tree-draft-packed-linear.h"
#include "tree-draft-quant-registry.h"

#include <cstdint>
#include <vector>

enum common_tree_draft_quant_group_axis : uint32_t {
    COMMON_TREE_DRAFT_QUANT_GROUP_INPUT = 0,
    COMMON_TREE_DRAFT_QUANT_GROUP_OUTPUT,
    COMMON_TREE_DRAFT_QUANT_GROUP_BLOCK,
};

enum common_tree_draft_quant_param_orientation : uint32_t {
    COMMON_TREE_DRAFT_QUANT_PARAMS_GROUP_OUT = 0,
    COMMON_TREE_DRAFT_QUANT_PARAMS_OUT_GROUP,
    COMMON_TREE_DRAFT_QUANT_PARAMS_PER_OUTPUT,
    COMMON_TREE_DRAFT_QUANT_PARAMS_PER_BLOCK,
};

struct common_tree_draft_quant_group_descriptor {
    common_tree_draft_quant_source source = COMMON_TREE_DRAFT_QUANT_GGML;
    common_tree_draft_quant_group_axis axis = COMMON_TREE_DRAFT_QUANT_GROUP_INPUT;
    uint64_t axis_length = 0;
    uint32_t group_size = 0;
    uint64_t group_count = 0;
    common_tree_draft_quant_param_orientation scale_orientation = COMMON_TREE_DRAFT_QUANT_PARAMS_GROUP_OUT;
    common_tree_draft_quant_param_orientation zero_orientation = COMMON_TREE_DRAFT_QUANT_PARAMS_GROUP_OUT;
    bool has_zero = false;
    bool tail_group_allowed = false;
};

enum common_tree_draft_quant_group_status : uint32_t {
    COMMON_TREE_DRAFT_QUANT_GROUP_OK = 0,
    COMMON_TREE_DRAFT_QUANT_GROUP_INVALID,
    COMMON_TREE_DRAFT_QUANT_GROUP_TAIL_REQUIRED,
    COMMON_TREE_DRAFT_QUANT_GROUP_KERNEL_GROUP_UNSUPPORTED,
};

common_tree_draft_quant_group_descriptor common_tree_draft_quant_groups_from_packed(const common_tree_draft_packed_linear_view & view);
common_tree_draft_quant_group_descriptor common_tree_draft_quant_groups_from_bnb8(const common_tree_draft_bnb8_descriptor & view);
common_tree_draft_quant_group_descriptor common_tree_draft_quant_groups_from_bnb4(const common_tree_draft_bnb4_descriptor & view);

common_tree_draft_quant_group_status common_tree_draft_quant_group_validate(
        const common_tree_draft_quant_group_descriptor & descriptor,
        const std::vector<uint32_t> & kernel_supported_group_sizes,
        bool kernel_supports_tail,
        bool * decodable,
        bool * kernel_eligible);

