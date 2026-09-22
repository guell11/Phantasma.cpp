#pragma once

#include "tree-draft-precision.h"
#include "tree-draft-quant-registry.h"
#include "tree-draft-sm89.h"

#include <cstddef>
#include <cstdint>

enum common_tree_draft_kernel_op : uint32_t {
    COMMON_TREE_DRAFT_KERNEL_LINEAR = 0,
    COMMON_TREE_DRAFT_KERNEL_ATTENTION_KV,
    COMMON_TREE_DRAFT_KERNEL_DEQUANT_MMA,
};

enum common_tree_draft_kernel_layout : uint32_t {
    COMMON_TREE_DRAFT_KERNEL_LAYOUT_DENSE = 0,
    COMMON_TREE_DRAFT_KERNEL_LAYOUT_BLOCKED,
    COMMON_TREE_DRAFT_KERNEL_LAYOUT_GROUPED,
    COMMON_TREE_DRAFT_KERNEL_LAYOUT_PAGED_KV,
};

struct common_tree_draft_kernel_capability_entry {
    uint32_t device_cc = 0;
    common_tree_draft_quant_source quant_source = COMMON_TREE_DRAFT_QUANT_GGML;
    uint32_t quant_id = 0;
    common_tree_draft_kernel_op op = COMMON_TREE_DRAFT_KERNEL_LINEAR;
    common_tree_draft_kernel_layout layout = COMMON_TREE_DRAFT_KERNEL_LAYOUT_DENSE;
    common_tree_draft_float_dtype compute_dtype = COMMON_TREE_DRAFT_DTYPE_F16;
    uint32_t group_size = 0;
    uint32_t alignment_bytes = 1;
    uint32_t k_multiple = 1;
    uint32_t n_multiple = 1;
    bool supports_tail_group = false;
    bool fused_dequant = false;
};

struct common_tree_draft_kernel_capability_query {
    const common_tree_draft_quant_type * quant = nullptr;
    common_tree_draft_kernel_op op = COMMON_TREE_DRAFT_KERNEL_LINEAR;
    common_tree_draft_kernel_layout layout = COMMON_TREE_DRAFT_KERNEL_LAYOUT_DENSE;
    common_tree_draft_float_dtype compute_dtype = COMMON_TREE_DRAFT_DTYPE_F16;
    uint32_t group_size = 0;
    uint64_t address = 0;
    uint64_t k = 0;
    uint64_t n = 0;
    bool has_tail_group = false;
};

struct common_tree_draft_kernel_capability_result {
    bool supported = false;
    uint32_t entry_index = UINT32_MAX;
    uint32_t required_alignment = 0;
    uint32_t required_k_multiple = 0;
    uint32_t required_n_multiple = 0;
    bool fused_dequant = false;
};

enum common_tree_draft_kernel_capability_status : uint32_t {
    COMMON_TREE_DRAFT_KERNEL_CAPABILITY_OK = 0,
    COMMON_TREE_DRAFT_KERNEL_CAPABILITY_INVALID_REGISTRY,
    COMMON_TREE_DRAFT_KERNEL_CAPABILITY_INVALID_QUERY,
};

common_tree_draft_kernel_capability_status common_tree_draft_kernel_capability_lookup(
        const common_tree_draft_device_capabilities & device,
        const common_tree_draft_kernel_capability_entry * entries,
        size_t entry_count,
        const common_tree_draft_kernel_capability_query & query,
        common_tree_draft_kernel_capability_result * result);
