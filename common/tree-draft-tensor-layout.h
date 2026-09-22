#pragma once

#include <cstdint>
#include <string>
#include <vector>

enum common_tree_draft_tensor_axis : uint32_t {
    COMMON_TREE_DRAFT_AXIS_UNKNOWN = 0,
    COMMON_TREE_DRAFT_AXIS_INPUT,
    COMMON_TREE_DRAFT_AXIS_OUTPUT,
    COMMON_TREE_DRAFT_AXIS_HIDDEN,
    COMMON_TREE_DRAFT_AXIS_VOCAB,
    COMMON_TREE_DRAFT_AXIS_HEAD,
    COMMON_TREE_DRAFT_AXIS_HEAD_DIM,
    COMMON_TREE_DRAFT_AXIS_EXPERT,
    COMMON_TREE_DRAFT_AXIS_GROUP,
    COMMON_TREE_DRAFT_AXIS_QKV,
};

enum common_tree_draft_tensor_layout_kind : uint32_t {
    COMMON_TREE_DRAFT_LAYOUT_DENSE = 0,
    COMMON_TREE_DRAFT_LAYOUT_TRANSPOSED,
    COMMON_TREE_DRAFT_LAYOUT_BLOCKED_QUANT,
    COMMON_TREE_DRAFT_LAYOUT_FUSED_QKV,
    COMMON_TREE_DRAFT_LAYOUT_EXPERT_PACKED,
};

struct common_tree_draft_tensor_layout {
    common_tree_draft_tensor_layout_kind kind = COMMON_TREE_DRAFT_LAYOUT_DENSE;
    std::vector<common_tree_draft_tensor_axis> logical_axes;
    std::vector<common_tree_draft_tensor_axis> storage_axes;
    std::vector<uint32_t> permutation;
    std::vector<uint64_t> storage_shape;
    std::vector<uint64_t> logical_shape;
    std::vector<uint64_t> block_shape;
    std::vector<uint64_t> strides;
    bool allow_tail_block = false;
};

enum common_tree_draft_tensor_layout_status : uint32_t {
    COMMON_TREE_DRAFT_TENSOR_LAYOUT_OK = 0,
    COMMON_TREE_DRAFT_TENSOR_LAYOUT_RANK,
    COMMON_TREE_DRAFT_TENSOR_LAYOUT_AXIS,
    COMMON_TREE_DRAFT_TENSOR_LAYOUT_PERMUTATION,
    COMMON_TREE_DRAFT_TENSOR_LAYOUT_SHAPE,
    COMMON_TREE_DRAFT_TENSOR_LAYOUT_BLOCK,
    COMMON_TREE_DRAFT_TENSOR_LAYOUT_STRIDE,
};

common_tree_draft_tensor_layout_status common_tree_draft_tensor_layout_validate(
        const common_tree_draft_tensor_layout & layout);

std::vector<uint64_t> common_tree_draft_tensor_layout_recover_logical_shape(
        const common_tree_draft_tensor_layout & layout);

