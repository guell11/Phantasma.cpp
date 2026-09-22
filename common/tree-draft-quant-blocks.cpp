#include "tree-draft-quant-blocks.h"

#include <limits>

static bool mul_u64(uint64_t a, uint64_t b, uint64_t * out) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) return false;
    *out = a * b;
    return true;
}

common_tree_draft_quant_block_status common_tree_draft_quant_block_plan_build(
        const common_tree_draft_gguf_tensor_descriptor & tensor,
        common_tree_draft_quant_block_plan * plan) {
    if (plan == nullptr) return COMMON_TREE_DRAFT_QUANT_BLOCK_RANGE;
    const auto * q = common_tree_draft_quant_find_ggml(tensor.type);
    if (q == nullptr || !q->fixed_block_size || q->block_elems == 0 || q->block_bytes == 0) {
        return COMMON_TREE_DRAFT_QUANT_BLOCK_UNREGISTERED;
    }
    if (tensor.dims.empty()) return COMMON_TREE_DRAFT_QUANT_BLOCK_DIMENSION;

    uint64_t elements = 1;
    for (int64_t dim : tensor.dims) {
        if (dim <= 0) return COMMON_TREE_DRAFT_QUANT_BLOCK_DIMENSION;
        if (!mul_u64(elements, static_cast<uint64_t>(dim), &elements)) return COMMON_TREE_DRAFT_QUANT_BLOCK_OVERFLOW;
    }
    if (q->block_elems > 1 && static_cast<uint64_t>(tensor.dims[0]) % q->block_elems != 0) {
        return COMMON_TREE_DRAFT_QUANT_BLOCK_ALIGNMENT;
    }
    if (elements % q->block_elems != 0) return COMMON_TREE_DRAFT_QUANT_BLOCK_ALIGNMENT;

    const uint64_t blocks = elements / q->block_elems;
    uint64_t encoded = 0;
    if (!mul_u64(blocks, q->block_bytes, &encoded)) return COMMON_TREE_DRAFT_QUANT_BLOCK_OVERFLOW;
    if (encoded != tensor.storage_size) return COMMON_TREE_DRAFT_QUANT_BLOCK_SIZE_MISMATCH;

    *plan = { elements, blocks, q->block_elems, q->block_bytes, encoded };
    return COMMON_TREE_DRAFT_QUANT_BLOCK_OK;
}

common_tree_draft_quant_block_status common_tree_draft_quant_block_span_map(
        const common_tree_draft_quant_block_plan & plan,
        uint64_t logical_begin,
        uint64_t logical_end,
        common_tree_draft_quant_block_span * span) {
    if (span == nullptr || logical_begin > logical_end || logical_end > plan.logical_elements) {
        return COMMON_TREE_DRAFT_QUANT_BLOCK_RANGE;
    }
    if (plan.block_elems == 0 || plan.block_bytes == 0) return COMMON_TREE_DRAFT_QUANT_BLOCK_UNREGISTERED;
    if (logical_begin % plan.block_elems != 0 || logical_end % plan.block_elems != 0) {
        return COMMON_TREE_DRAFT_QUANT_BLOCK_ALIGNMENT;
    }
    const uint64_t first = logical_begin / plan.block_elems;
    const uint64_t end_block = logical_end / plan.block_elems;
    uint64_t byte_begin = 0;
    uint64_t byte_end = 0;
    if (!mul_u64(first, plan.block_bytes, &byte_begin) || !mul_u64(end_block, plan.block_bytes, &byte_end)) {
        return COMMON_TREE_DRAFT_QUANT_BLOCK_OVERFLOW;
    }
    *span = { logical_begin, logical_end, byte_begin, byte_end, first, end_block - first };
    return COMMON_TREE_DRAFT_QUANT_BLOCK_OK;
}

