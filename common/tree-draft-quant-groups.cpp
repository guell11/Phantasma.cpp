#include "tree-draft-quant-groups.h"

#include <algorithm>

common_tree_draft_quant_group_descriptor common_tree_draft_quant_groups_from_packed(const common_tree_draft_packed_linear_view & view) {
    return {
        view.format == COMMON_TREE_DRAFT_PACKED_LINEAR_GPTQ ? COMMON_TREE_DRAFT_QUANT_GPTQ : COMMON_TREE_DRAFT_QUANT_AWQ,
        COMMON_TREE_DRAFT_QUANT_GROUP_INPUT,
        view.in_features,
        view.group_size,
        view.group_count,
        COMMON_TREE_DRAFT_QUANT_PARAMS_GROUP_OUT,
        COMMON_TREE_DRAFT_QUANT_PARAMS_GROUP_OUT,
        view.zero_point,
        false,
    };
}

common_tree_draft_quant_group_descriptor common_tree_draft_quant_groups_from_bnb8(const common_tree_draft_bnb8_descriptor & view) {
    return {COMMON_TREE_DRAFT_QUANT_BNB8, COMMON_TREE_DRAFT_QUANT_GROUP_OUTPUT, view.rows, 1, view.rows,
        COMMON_TREE_DRAFT_QUANT_PARAMS_PER_OUTPUT, COMMON_TREE_DRAFT_QUANT_PARAMS_PER_OUTPUT, false, false};
}

common_tree_draft_quant_group_descriptor common_tree_draft_quant_groups_from_bnb4(const common_tree_draft_bnb4_descriptor & view) {
    const uint64_t count = (view.logical_elements + view.block_size - 1) / view.block_size;
    return {COMMON_TREE_DRAFT_QUANT_BNB4, COMMON_TREE_DRAFT_QUANT_GROUP_BLOCK, view.logical_elements, view.block_size, count,
        COMMON_TREE_DRAFT_QUANT_PARAMS_PER_BLOCK, COMMON_TREE_DRAFT_QUANT_PARAMS_PER_BLOCK, false, true};
}

common_tree_draft_quant_group_status common_tree_draft_quant_group_validate(
        const common_tree_draft_quant_group_descriptor & descriptor,
        const std::vector<uint32_t> & kernel_supported_group_sizes,
        bool kernel_supports_tail,
        bool * decodable,
        bool * kernel_eligible) {
    if (!decodable || !kernel_eligible || descriptor.axis_length == 0 || descriptor.group_size == 0 || descriptor.group_count == 0) {
        return COMMON_TREE_DRAFT_QUANT_GROUP_INVALID;
    }
    *decodable = false;
    *kernel_eligible = false;
    const uint64_t expected_groups = (descriptor.axis_length + descriptor.group_size - 1) / descriptor.group_size;
    if (expected_groups != descriptor.group_count) return COMMON_TREE_DRAFT_QUANT_GROUP_INVALID;
    const bool has_tail = descriptor.axis_length % descriptor.group_size != 0;
    if (has_tail && !descriptor.tail_group_allowed) return COMMON_TREE_DRAFT_QUANT_GROUP_TAIL_REQUIRED;
    *decodable = true;
    const bool group_supported = std::find(kernel_supported_group_sizes.begin(), kernel_supported_group_sizes.end(), descriptor.group_size) != kernel_supported_group_sizes.end();
    if (!group_supported || (has_tail && !kernel_supports_tail)) return COMMON_TREE_DRAFT_QUANT_GROUP_KERNEL_GROUP_UNSUPPORTED;
    *kernel_eligible = true;
    return COMMON_TREE_DRAFT_QUANT_GROUP_OK;
}

