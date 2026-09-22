#include "tree-draft-quant-blocks.h"

#include <cassert>

int main() {
    common_tree_draft_gguf_tensor_descriptor q4;
    q4.name = "blk.0.ffn_up.weight";
    q4.dims = { 4096, 8 };
    q4.type = GGML_TYPE_Q4_0;
    const uint64_t elements = 4096ull * 8ull;
    q4.storage_size = elements / ggml_blck_size(GGML_TYPE_Q4_0) * ggml_type_size(GGML_TYPE_Q4_0);

    common_tree_draft_quant_block_plan plan = {};
    assert(common_tree_draft_quant_block_plan_build(q4, &plan) == COMMON_TREE_DRAFT_QUANT_BLOCK_OK);
    assert(plan.logical_elements == elements);
    assert(plan.encoded_bytes == q4.storage_size);

    common_tree_draft_quant_block_span span = {};
    const uint64_t B = plan.block_elems;
    assert(common_tree_draft_quant_block_span_map(plan, B, 3 * B, &span) == COMMON_TREE_DRAFT_QUANT_BLOCK_OK);
    assert(span.first_block == 1 && span.block_count == 2);
    assert(span.byte_begin == plan.block_bytes && span.byte_end == 3ull * plan.block_bytes);
    assert(common_tree_draft_quant_block_span_map(plan, 1, B, &span) == COMMON_TREE_DRAFT_QUANT_BLOCK_ALIGNMENT);

    auto bad_size = q4;
    ++bad_size.storage_size;
    assert(common_tree_draft_quant_block_plan_build(bad_size, &plan) == COMMON_TREE_DRAFT_QUANT_BLOCK_SIZE_MISMATCH);
    auto bad_row = q4;
    bad_row.dims[0] = 4095;
    bad_row.storage_size = q4.storage_size;
    assert(common_tree_draft_quant_block_plan_build(bad_row, &plan) == COMMON_TREE_DRAFT_QUANT_BLOCK_ALIGNMENT);

    common_tree_draft_gguf_tensor_descriptor f16;
    f16.name = "token_embd.weight";
    f16.dims = { 4, 3 };
    f16.type = GGML_TYPE_F16;
    f16.storage_size = 12 * ggml_type_size(GGML_TYPE_F16);
    assert(common_tree_draft_quant_block_plan_build(f16, &plan) == COMMON_TREE_DRAFT_QUANT_BLOCK_OK);
    assert(plan.block_elems == 1);
    return 0;
}

