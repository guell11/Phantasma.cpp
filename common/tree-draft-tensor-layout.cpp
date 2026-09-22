#include "tree-draft-tensor-layout.h"

#include <algorithm>
#include <limits>

static bool valid_axis(common_tree_draft_tensor_axis axis) {
    return axis != COMMON_TREE_DRAFT_AXIS_UNKNOWN;
}

common_tree_draft_tensor_layout_status common_tree_draft_tensor_layout_validate(
        const common_tree_draft_tensor_layout & layout) {
    const size_t rank = layout.logical_axes.size();
    if (rank == 0 || layout.storage_axes.size() != rank || layout.permutation.size() != rank ||
        layout.storage_shape.size() != rank || layout.logical_shape.size() != rank) {
        return COMMON_TREE_DRAFT_TENSOR_LAYOUT_RANK;
    }

    std::vector<uint8_t> seen(rank, 0);
    for (size_t i = 0; i < rank; ++i) {
        if (!valid_axis(layout.logical_axes[i]) || !valid_axis(layout.storage_axes[i])) {
            return COMMON_TREE_DRAFT_TENSOR_LAYOUT_AXIS;
        }
        if (layout.permutation[i] >= rank || seen[layout.permutation[i]] != 0) {
            return COMMON_TREE_DRAFT_TENSOR_LAYOUT_PERMUTATION;
        }
        seen[layout.permutation[i]] = 1;
        if (layout.storage_axes[layout.permutation[i]] != layout.logical_axes[i]) {
            return COMMON_TREE_DRAFT_TENSOR_LAYOUT_PERMUTATION;
        }
        if (layout.storage_shape[layout.permutation[i]] != layout.logical_shape[i] || layout.logical_shape[i] == 0) {
            return COMMON_TREE_DRAFT_TENSOR_LAYOUT_SHAPE;
        }
    }

    if (!layout.block_shape.empty()) {
        if (layout.block_shape.size() != rank) return COMMON_TREE_DRAFT_TENSOR_LAYOUT_BLOCK;
        for (size_t i = 0; i < rank; ++i) {
            const uint64_t block = layout.block_shape[i];
            if (block == 0) return COMMON_TREE_DRAFT_TENSOR_LAYOUT_BLOCK;
            const uint64_t dim = layout.storage_shape[i];
            if (!layout.allow_tail_block && dim % block != 0) {
                return COMMON_TREE_DRAFT_TENSOR_LAYOUT_BLOCK;
            }
        }
    } else if (layout.kind == COMMON_TREE_DRAFT_LAYOUT_BLOCKED_QUANT) {
        return COMMON_TREE_DRAFT_TENSOR_LAYOUT_BLOCK;
    }

    if (!layout.strides.empty()) {
        if (layout.strides.size() != rank) return COMMON_TREE_DRAFT_TENSOR_LAYOUT_STRIDE;
        for (uint64_t stride : layout.strides) {
            if (stride == 0) return COMMON_TREE_DRAFT_TENSOR_LAYOUT_STRIDE;
        }
        if (layout.kind == COMMON_TREE_DRAFT_LAYOUT_DENSE || layout.kind == COMMON_TREE_DRAFT_LAYOUT_TRANSPOSED) {
            uint64_t expected = 1;
            for (size_t rev = rank; rev-- > 0;) {
                if (layout.strides[rev] != expected) return COMMON_TREE_DRAFT_TENSOR_LAYOUT_STRIDE;
                if (layout.storage_shape[rev] != 0 && expected > std::numeric_limits<uint64_t>::max() / layout.storage_shape[rev]) {
                    return COMMON_TREE_DRAFT_TENSOR_LAYOUT_STRIDE;
                }
                expected *= layout.storage_shape[rev];
            }
        }
    }

    return COMMON_TREE_DRAFT_TENSOR_LAYOUT_OK;
}

std::vector<uint64_t> common_tree_draft_tensor_layout_recover_logical_shape(
        const common_tree_draft_tensor_layout & layout) {
    std::vector<uint64_t> logical(layout.permutation.size(), 0);
    if (layout.storage_shape.size() != layout.permutation.size()) return {};
    for (size_t i = 0; i < layout.permutation.size(); ++i) {
        if (layout.permutation[i] >= layout.storage_shape.size()) return {};
        logical[i] = layout.storage_shape[layout.permutation[i]];
    }
    return logical;
}

