#include "tree-draft-qkv.h"

#include <cstring>
#include <limits>

static bool common_tree_draft_qkv_mul_overflow(size_t a, size_t b, size_t * result) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        return true;
    }
    *result = a * b;
    return false;
}

static bool common_tree_draft_qkv_add_overflow(size_t a, size_t b, size_t * result) {
    if (b > std::numeric_limits<size_t>::max() - a) {
        return true;
    }
    *result = a + b;
    return false;
}

size_t common_tree_draft_qkv_packed_bytes(const common_tree_draft_qkv_view & view, bool * overflow) {
    bool failed = false;
    size_t total = 0;
    size_t tmp = 0;
    if (view.n_nodes < 0 || view.n_heads < 0 || view.head_dim < 0 || view.element_size == 0) {
        failed = true;
    } else if (common_tree_draft_qkv_mul_overflow(static_cast<size_t>(view.n_nodes), static_cast<size_t>(view.n_heads), &tmp) ||
               common_tree_draft_qkv_mul_overflow(tmp, static_cast<size_t>(view.head_dim), &tmp) ||
               common_tree_draft_qkv_mul_overflow(tmp, view.element_size, &total)) {
        failed = true;
    }
    if (overflow != nullptr) {
        *overflow = failed;
    }
    return failed ? 0 : total;
}

static common_tree_draft_qkv_error common_tree_draft_qkv_validate_source(
        const common_tree_draft_qkv_view & source,
        int32_t expected_nodes) {
    if (source.n_nodes != expected_nodes || source.n_nodes < 0 || source.n_heads < 0 || source.head_dim < 0) {
        return COMMON_TREE_DRAFT_QKV_INVALID_SHAPE;
    }
    if (source.element_size == 0) {
        return COMMON_TREE_DRAFT_QKV_ZERO_ELEMENT_SIZE;
    }
    if (source.n_nodes > 0 && source.n_heads > 0 && source.head_dim > 0 && source.data == nullptr) {
        return COMMON_TREE_DRAFT_QKV_NULL_SOURCE;
    }

    if (source.n_nodes == 0 || source.n_heads == 0 || source.head_dim == 0) {
        return COMMON_TREE_DRAFT_QKV_OK;
    }

    size_t node_offset = 0;
    size_t head_offset = 0;
    size_t dim_offset = 0;
    size_t max_offset = 0;
    if (common_tree_draft_qkv_mul_overflow(static_cast<size_t>(source.n_nodes - 1), source.stride_node, &node_offset) ||
        common_tree_draft_qkv_mul_overflow(static_cast<size_t>(source.n_heads - 1), source.stride_head, &head_offset) ||
        common_tree_draft_qkv_mul_overflow(static_cast<size_t>(source.head_dim - 1), source.stride_dim, &dim_offset) ||
        common_tree_draft_qkv_add_overflow(node_offset, head_offset, &max_offset) ||
        common_tree_draft_qkv_add_overflow(max_offset, dim_offset, &max_offset) ||
        common_tree_draft_qkv_add_overflow(max_offset, source.element_size, &max_offset)) {
        return COMMON_TREE_DRAFT_QKV_STRIDE_OVERFLOW;
    }
    return COMMON_TREE_DRAFT_QKV_OK;
}

common_tree_draft_qkv_error common_tree_draft_qkv_gather(
        const common_tree_draft_forest_offsets & forest,
        const int32_t * positions,
        size_t position_count,
        const common_tree_draft_qkv_view & source,
        common_tree_draft_qkv_output output) {
    if (forest.n_entries < 0 || forest.offsets == nullptr || forest.offsets[0] != 0) {
        return COMMON_TREE_DRAFT_QKV_INVALID_FOREST;
    }
    for (int32_t b = 0; b < forest.n_entries; ++b) {
        if (forest.offsets[b] < 0 || forest.offsets[b + 1] < forest.offsets[b]) {
            return COMMON_TREE_DRAFT_QKV_INVALID_FOREST;
        }
    }
    const int32_t n_nodes = forest.offsets[forest.n_entries];
    if (n_nodes < 0) {
        return COMMON_TREE_DRAFT_QKV_INVALID_FOREST;
    }
    if (position_count != static_cast<size_t>(n_nodes)) {
        return COMMON_TREE_DRAFT_QKV_POSITION_COUNT_MISMATCH;
    }
    if (n_nodes > 0 && positions == nullptr) {
        return COMMON_TREE_DRAFT_QKV_NULL_POSITIONS;
    }

    const common_tree_draft_qkv_error source_error = common_tree_draft_qkv_validate_source(source, n_nodes);
    if (source_error != COMMON_TREE_DRAFT_QKV_OK) {
        return source_error;
    }

    bool packed_overflow = false;
    const size_t required = common_tree_draft_qkv_packed_bytes(source, &packed_overflow);
    if (packed_overflow) {
        return COMMON_TREE_DRAFT_QKV_STRIDE_OVERFLOW;
    }
    if (required == 0) {
        return COMMON_TREE_DRAFT_QKV_OK;
    }
    if (output.data == nullptr) {
        return COMMON_TREE_DRAFT_QKV_NULL_OUTPUT;
    }
    if (output.byte_count < required) {
        return COMMON_TREE_DRAFT_QKV_OUTPUT_TOO_SMALL;
    }

    size_t dst_offset = 0;
    for (int32_t g = 0; g < source.n_nodes; ++g) {
        for (int32_t h = 0; h < source.n_heads; ++h) {
            for (int32_t d = 0; d < source.head_dim; ++d) {
                const size_t src_offset = static_cast<size_t>(g) * source.stride_node +
                    static_cast<size_t>(h) * source.stride_head +
                    static_cast<size_t>(d) * source.stride_dim;
                std::memcpy(output.data + dst_offset, source.data + src_offset, source.element_size);
                dst_offset += source.element_size;
            }
        }
    }
    return COMMON_TREE_DRAFT_QKV_OK;
}

const char * common_tree_draft_qkv_error_name(common_tree_draft_qkv_error error) {
    switch (error) {
        case COMMON_TREE_DRAFT_QKV_OK:                      return "ok";
        case COMMON_TREE_DRAFT_QKV_INVALID_FOREST:          return "invalid_forest";
        case COMMON_TREE_DRAFT_QKV_NULL_POSITIONS:          return "null_positions";
        case COMMON_TREE_DRAFT_QKV_POSITION_COUNT_MISMATCH: return "position_count_mismatch";
        case COMMON_TREE_DRAFT_QKV_INVALID_SHAPE:           return "invalid_shape";
        case COMMON_TREE_DRAFT_QKV_NULL_SOURCE:             return "null_source";
        case COMMON_TREE_DRAFT_QKV_ZERO_ELEMENT_SIZE:       return "zero_element_size";
        case COMMON_TREE_DRAFT_QKV_STRIDE_OVERFLOW:         return "stride_overflow";
        case COMMON_TREE_DRAFT_QKV_NULL_OUTPUT:             return "null_output";
        case COMMON_TREE_DRAFT_QKV_OUTPUT_TOO_SMALL:        return "output_too_small";
    }
    return "unknown";
}
