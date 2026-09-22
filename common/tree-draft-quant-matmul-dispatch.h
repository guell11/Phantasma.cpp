#pragma once

#include "tree-draft-dequant-plan.h"
#include "tree-draft-kernel-capability.h"

#include <cstddef>
#include <cstdint>

enum common_tree_draft_quant_matmul_path : uint32_t {
    COMMON_TREE_DRAFT_QUANT_MATMUL_NATIVE = 0,
    COMMON_TREE_DRAFT_QUANT_MATMUL_DEQUANT_ON_THE_FLY,
    COMMON_TREE_DRAFT_QUANT_MATMUL_UNSUPPORTED,
};

struct common_tree_draft_quant_matmul_request {
    common_tree_draft_kernel_capability_query capability_query;
    uint64_t m = 0;
    uint64_t n = 0;
    uint64_t k = 0;
    uint64_t workspace_budget = 0;
    uint32_t dequant_output_element_bytes = 2;
    common_tree_draft_dequant_accumulation_mode dequant_mode = COMMON_TREE_DRAFT_DEQUANT_ACCUM_THROUGHPUT;
};

struct common_tree_draft_quant_matmul_plan {
    common_tree_draft_quant_matmul_path path = COMMON_TREE_DRAFT_QUANT_MATMUL_UNSUPPORTED;
    common_tree_draft_kernel_capability_result native = {};
    common_tree_draft_dequant_plan dequant = {};
};

enum common_tree_draft_quant_matmul_status : uint32_t {
    COMMON_TREE_DRAFT_QUANT_MATMUL_OK = 0,
    COMMON_TREE_DRAFT_QUANT_MATMUL_INVALID,
    COMMON_TREE_DRAFT_QUANT_MATMUL_CAPABILITY,
};

common_tree_draft_quant_matmul_status common_tree_draft_quant_matmul_plan_build(
        const common_tree_draft_device_capabilities & device,
        const common_tree_draft_kernel_capability_entry * entries,
        size_t entry_count,
        const common_tree_draft_quant_matmul_request & request,
        common_tree_draft_quant_matmul_plan * plan);
