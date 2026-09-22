#include "tree-draft-dequant-plan.h"

#include <cassert>

int main() {
    const auto * q4 = common_tree_draft_quant_find_ggml(GGML_TYPE_Q4_0);
    assert(q4 != nullptr);
    const uint64_t B = q4->block_elems;
    common_tree_draft_dequant_plan plan;
    assert(common_tree_draft_dequant_plan_build(*q4, B * 10, B * 3 * sizeof(float), sizeof(float),
                COMMON_TREE_DRAFT_DEQUANT_ACCUM_DETERMINISTIC, &plan) == COMMON_TREE_DRAFT_DEQUANT_PLAN_OK);
    assert(plan.tiles.size() == 4);
    assert(plan.tiles[0].logical_count == B * 3);
    assert(plan.tiles.back().logical_count == B);
    assert(plan.peak_temporary_bytes <= B * 3 * sizeof(float));
    for (size_t i = 1; i < plan.tiles.size(); ++i) {
        assert(plan.tiles[i].logical_begin == plan.tiles[i-1].logical_begin + plan.tiles[i-1].logical_count);
        assert(plan.tiles[i].encoded_offset == plan.tiles[i-1].encoded_offset + plan.tiles[i-1].encoded_bytes);
    }
    assert(common_tree_draft_dequant_plan_build(*q4, B * 2, B * sizeof(float) - 1, sizeof(float),
                COMMON_TREE_DRAFT_DEQUANT_ACCUM_THROUGHPUT, &plan) == COMMON_TREE_DRAFT_DEQUANT_PLAN_BUDGET);
    assert(common_tree_draft_dequant_plan_build(*q4, B * 2 + 1, B * 2 * sizeof(float), sizeof(float),
                COMMON_TREE_DRAFT_DEQUANT_ACCUM_THROUGHPUT, &plan) == COMMON_TREE_DRAFT_DEQUANT_PLAN_INVALID);
    return 0;
}

