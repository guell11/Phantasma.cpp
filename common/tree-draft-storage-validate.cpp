#include "tree-draft-storage-validate.h"

#include <limits>

static bool mul_u64(uint64_t a, uint64_t b, uint64_t * out) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) return false;
    *out = a * b;
    return true;
}

static bool host_little_endian() {
    const uint16_t one = 1;
    return *reinterpret_cast<const uint8_t *>(&one) == 1;
}

common_tree_draft_storage_validation common_tree_draft_storage_validate(
        common_tree_draft_endian source_endian,
        const common_tree_draft_tensor_layout & layout,
        const common_tree_draft_quant_type & quant,
        const common_tree_draft_storage_view & view,
        const common_tree_draft_alignment_table & alignments,
        common_tree_draft_buffer_class buffer_class,
        bool copy_fallback_allowed) {
    common_tree_draft_storage_validation out;
    if (buffer_class >= COMMON_TREE_DRAFT_BUFFER_CLASS_COUNT || view.owner == nullptr ||
        common_tree_draft_tensor_layout_validate(layout) != COMMON_TREE_DRAFT_TENSOR_LAYOUT_OK) {
        out.result = COMMON_TREE_DRAFT_STORAGE_CORRUPT;
        return out;
    }
    const bool source_is_little = source_endian == COMMON_TREE_DRAFT_ENDIAN_LITTLE;
    if (source_is_little != host_little_endian()) {
        out.result = COMMON_TREE_DRAFT_STORAGE_UNSUPPORTED;
        return out;
    }
    if (!quant.fixed_block_size || quant.block_elems == 0 || quant.block_bytes == 0) {
        out.result = COMMON_TREE_DRAFT_STORAGE_UNSUPPORTED;
        return out;
    }
    uint64_t elements = 1;
    for (uint64_t dim : layout.logical_shape) {
        if (dim == 0 || !mul_u64(elements, dim, &elements)) {
            out.result = COMMON_TREE_DRAFT_STORAGE_CORRUPT;
            return out;
        }
    }
    if (elements % quant.block_elems != 0) {
        out.result = COMMON_TREE_DRAFT_STORAGE_CORRUPT;
        return out;
    }
    const uint64_t blocks = elements / quant.block_elems;
    if (!mul_u64(blocks, quant.block_bytes, &out.expected_span)) {
        out.result = COMMON_TREE_DRAFT_STORAGE_CORRUPT;
        return out;
    }
    if (out.expected_span != view.length) {
        out.result = COMMON_TREE_DRAFT_STORAGE_CORRUPT;
        return out;
    }
    out.required_alignment = alignments.required[buffer_class];
    if (out.required_alignment == 0) {
        out.result = COMMON_TREE_DRAFT_STORAGE_CORRUPT;
        return out;
    }
    const uintptr_t address = reinterpret_cast<uintptr_t>(view.data);
    const bool aligned = (address % out.required_alignment) == 0 && (view.owner_offset % out.required_alignment) == 0;
    if (!aligned) {
        out.result = copy_fallback_allowed ? COMMON_TREE_DRAFT_STORAGE_RECOVERABLE_COPY : COMMON_TREE_DRAFT_STORAGE_CORRUPT;
        return out;
    }
    out.result = COMMON_TREE_DRAFT_STORAGE_VALID;
    return out;
}

