#include "tree-draft-moe-layout.h"

#include <cassert>

static common_tree_draft_tensor_layout matrix_layout(uint64_t in, uint64_t out) {
    return {COMMON_TREE_DRAFT_LAYOUT_DENSE,
        {COMMON_TREE_DRAFT_AXIS_INPUT, COMMON_TREE_DRAFT_AXIS_OUTPUT},
        {COMMON_TREE_DRAFT_AXIS_INPUT, COMMON_TREE_DRAFT_AXIS_OUTPUT},
        {0,1}, {in,out}, {in,out}, {}, {out,1}, false};
}

static common_tree_draft_tensor_layout packed_layout(uint64_t experts, uint64_t in, uint64_t out) {
    return {COMMON_TREE_DRAFT_LAYOUT_EXPERT_PACKED,
        {COMMON_TREE_DRAFT_AXIS_EXPERT, COMMON_TREE_DRAFT_AXIS_INPUT, COMMON_TREE_DRAFT_AXIS_OUTPUT},
        {COMMON_TREE_DRAFT_AXIS_INPUT, COMMON_TREE_DRAFT_AXIS_OUTPUT, COMMON_TREE_DRAFT_AXIS_EXPERT},
        {2,0,1}, {in,out,experts}, {experts,in,out}, {}, {}, false};
}

int main() {
    common_tree_draft_moe_tensor_descriptor separate[3];
    for (int i = 0; i < 3; ++i) {
        separate[i].key = {COMMON_TREE_DRAFT_TENSOR_FFN_GATE_EXPERT, 2, i, COMMON_TREE_DRAFT_TENSOR_SUBROLE_WEIGHT};
        separate[i].layout = matrix_layout(4096, 8192);
    }
    common_tree_draft_moe_expert_axis_view view;
    assert(common_tree_draft_moe_validate_expert_family(COMMON_TREE_DRAFT_TENSOR_FFN_GATE_EXPERT, 3, separate, 3, &view) == COMMON_TREE_DRAFT_MOE_LAYOUT_OK);
    assert(!view.packed && view.tensor_for_expert[0] == 0 && view.tensor_for_expert[2] == 2);

    // Packed gate_up is a valid canonical source for either gate or up.
    common_tree_draft_moe_tensor_descriptor packed;
    packed.key = {COMMON_TREE_DRAFT_TENSOR_FFN_GATE_UP_EXPERTS, 2, -1, COMMON_TREE_DRAFT_TENSOR_SUBROLE_WEIGHT};
    packed.layout = packed_layout(3,4096,16384);
    assert(common_tree_draft_moe_validate_expert_family(COMMON_TREE_DRAFT_TENSOR_FFN_UP_EXPERT, 3, &packed, 1, &view) == COMMON_TREE_DRAFT_MOE_LAYOUT_OK);
    assert(view.packed && view.expert_axis == 0 && view.packed_tensor == 0);

    // Missing/duplicate separate experts never silently form a valid family.
    assert(common_tree_draft_moe_validate_expert_family(COMMON_TREE_DRAFT_TENSOR_FFN_GATE_EXPERT, 4, separate, 3, &view) == COMMON_TREE_DRAFT_MOE_LAYOUT_COVERAGE);
    auto dup = separate[1]; dup.key.expert = 0;
    common_tree_draft_moe_tensor_descriptor with_dup[] = {separate[0], dup, separate[2]};
    assert(common_tree_draft_moe_validate_expert_family(COMMON_TREE_DRAFT_TENSOR_FFN_GATE_EXPERT, 3, with_dup, 3, &view) == COMMON_TREE_DRAFT_MOE_LAYOUT_DUPLICATE);

    // Quant blocks cannot advance across the expert axis unless the encoding
    // explicitly declares cross-expert packing.
    common_tree_draft_quant_type q;
    q.block_elems = 32;
    packed.quant = &q;
    packed.quant_axis = 0;
    assert(common_tree_draft_moe_validate_expert_family(COMMON_TREE_DRAFT_TENSOR_FFN_GATE_EXPERT, 3, &packed, 1, &view) == COMMON_TREE_DRAFT_MOE_LAYOUT_QUANT_CROSSES_EXPERT);
    packed.quant_cross_expert_defined = true;
    assert(common_tree_draft_moe_validate_expert_family(COMMON_TREE_DRAFT_TENSOR_FFN_GATE_EXPERT, 3, &packed, 1, &view) == COMMON_TREE_DRAFT_MOE_LAYOUT_OK);

    common_tree_draft_moe_tensor_descriptor router;
    router.key = {COMMON_TREE_DRAFT_TENSOR_FFN_ROUTER, 2, -1, COMMON_TREE_DRAFT_TENSOR_SUBROLE_WEIGHT};
    router.layout = {COMMON_TREE_DRAFT_LAYOUT_DENSE,
        {COMMON_TREE_DRAFT_AXIS_HIDDEN, COMMON_TREE_DRAFT_AXIS_EXPERT},
        {COMMON_TREE_DRAFT_AXIS_HIDDEN, COMMON_TREE_DRAFT_AXIS_EXPERT},
        {0,1}, {4096,3}, {4096,3}, {}, {3,1}, false};
    assert(common_tree_draft_moe_validate_router(4096,3,router) == COMMON_TREE_DRAFT_MOE_LAYOUT_OK);
    router.layout.logical_shape[1] = 4;
    router.layout.storage_shape[1] = 4;
    assert(common_tree_draft_moe_validate_router(4096,3,router) == COMMON_TREE_DRAFT_MOE_LAYOUT_ROUTER);
    return 0;
}
