#include "tree-draft-tensor-layout.h"

#include <cassert>

int main() {
    common_tree_draft_tensor_layout dense = {
        COMMON_TREE_DRAFT_LAYOUT_DENSE,
        { COMMON_TREE_DRAFT_AXIS_INPUT, COMMON_TREE_DRAFT_AXIS_OUTPUT },
        { COMMON_TREE_DRAFT_AXIS_INPUT, COMMON_TREE_DRAFT_AXIS_OUTPUT },
        { 0, 1 },
        { 4096, 11008 },
        { 4096, 11008 },
        {},
        { 11008, 1 },
        false,
    };
    assert(common_tree_draft_tensor_layout_validate(dense) == COMMON_TREE_DRAFT_TENSOR_LAYOUT_OK);
    assert(common_tree_draft_tensor_layout_recover_logical_shape(dense) == dense.logical_shape);

    common_tree_draft_tensor_layout transposed = {
        COMMON_TREE_DRAFT_LAYOUT_TRANSPOSED,
        { COMMON_TREE_DRAFT_AXIS_INPUT, COMMON_TREE_DRAFT_AXIS_OUTPUT },
        { COMMON_TREE_DRAFT_AXIS_OUTPUT, COMMON_TREE_DRAFT_AXIS_INPUT },
        { 1, 0 },
        { 11008, 4096 },
        { 4096, 11008 },
        {},
        { 4096, 1 },
        false,
    };
    assert(common_tree_draft_tensor_layout_validate(transposed) == COMMON_TREE_DRAFT_TENSOR_LAYOUT_OK);

    common_tree_draft_tensor_layout blocked = {
        COMMON_TREE_DRAFT_LAYOUT_BLOCKED_QUANT,
        { COMMON_TREE_DRAFT_AXIS_OUTPUT, COMMON_TREE_DRAFT_AXIS_INPUT },
        { COMMON_TREE_DRAFT_AXIS_OUTPUT, COMMON_TREE_DRAFT_AXIS_INPUT },
        { 0, 1 },
        { 4096, 11008 },
        { 4096, 11008 },
        { 1, 32 },
        {},
        true,
    };
    assert(common_tree_draft_tensor_layout_validate(blocked) == COMMON_TREE_DRAFT_TENSOR_LAYOUT_OK);

    common_tree_draft_tensor_layout expert = {
        COMMON_TREE_DRAFT_LAYOUT_EXPERT_PACKED,
        { COMMON_TREE_DRAFT_AXIS_EXPERT, COMMON_TREE_DRAFT_AXIS_INPUT, COMMON_TREE_DRAFT_AXIS_OUTPUT },
        { COMMON_TREE_DRAFT_AXIS_INPUT, COMMON_TREE_DRAFT_AXIS_OUTPUT, COMMON_TREE_DRAFT_AXIS_EXPERT },
        { 2, 0, 1 },
        { 4096, 8192, 64 },
        { 64, 4096, 8192 },
        {}, {}, false,
    };
    assert(common_tree_draft_tensor_layout_validate(expert) == COMMON_TREE_DRAFT_TENSOR_LAYOUT_OK);

    auto bad_perm = transposed;
    bad_perm.permutation = { 0, 0 };
    assert(common_tree_draft_tensor_layout_validate(bad_perm) == COMMON_TREE_DRAFT_TENSOR_LAYOUT_PERMUTATION);
    auto bad_block = blocked;
    bad_block.allow_tail_block = false;
    bad_block.block_shape = {1, 30};
    assert(common_tree_draft_tensor_layout_validate(bad_block) == COMMON_TREE_DRAFT_TENSOR_LAYOUT_BLOCK);
    return 0;
}

