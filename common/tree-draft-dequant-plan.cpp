#include "tree-draft-dequant-plan.h"

#include <algorithm>
#include <limits>

static bool mul_u64(uint64_t a, uint64_t b, uint64_t * out) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) return false;
    *out = a * b;
    return true;
}

common_tree_draft_dequant_plan_status common_tree_draft_dequant_plan_build(
        const common_tree_draft_quant_type & quant,
        uint64_t logical_elements,
        uint64_t workspace_budget,
        uint32_t output_element_bytes,
        common_tree_draft_dequant_accumulation_mode mode,
        common_tree_draft_dequant_plan * plan) {
    if (plan == nullptr || logical_elements == 0 || output_element_bytes == 0 ||
        !quant.fixed_block_size || quant.block_elems == 0 || quant.block_bytes == 0 ||
        logical_elements % quant.block_elems != 0) {
        return COMMON_TREE_DRAFT_DEQUANT_PLAN_INVALID;
    }
    if (workspace_budget < static_cast<uint64_t>(quant.block_elems) * output_element_bytes) {
        return COMMON_TREE_DRAFT_DEQUANT_PLAN_BUDGET;
    }

    uint64_t max_elements = workspace_budget / output_element_bytes;
    max_elements -= max_elements % quant.block_elems;
    if (max_elements == 0) return COMMON_TREE_DRAFT_DEQUANT_PLAN_BUDGET;

    common_tree_draft_dequant_plan temp;
    temp.mode = mode;
    uint64_t begin = 0;
    while (begin < logical_elements) {
        const uint64_t remaining = logical_elements - begin;
        const uint64_t count = std::min(remaining, max_elements);
        const uint64_t first_block = begin / quant.block_elems;
        const uint64_t block_count = count / quant.block_elems;
        uint64_t encoded_offset = 0;
        uint64_t encoded_bytes = 0;
        uint64_t temporary_bytes = 0;
        if (!mul_u64(first_block, quant.block_bytes, &encoded_offset) ||
            !mul_u64(block_count, quant.block_bytes, &encoded_bytes) ||
            !mul_u64(count, output_element_bytes, &temporary_bytes)) {
            return COMMON_TREE_DRAFT_DEQUANT_PLAN_OVERFLOW;
        }
        temp.tiles.push_back({begin, count, encoded_offset, encoded_bytes, temporary_bytes});
        temp.peak_temporary_bytes = std::max(temp.peak_temporary_bytes, temporary_bytes);
        begin += count;
    }
    *plan = std::move(temp);
    return COMMON_TREE_DRAFT_DEQUANT_PLAN_OK;
}

