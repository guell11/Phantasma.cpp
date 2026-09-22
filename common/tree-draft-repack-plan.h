#pragma once

#include "tree-draft-quant-registry.h"
#include "tree-draft-tensor-layout.h"

#include <cstddef>
#include <cstdint>
#include <string>

struct common_tree_draft_repack_source_identity {
    std::string source_id;
    std::string source_hash;
    uint64_t tensor_offset = 0;
    uint64_t tensor_bytes = 0;
};

struct common_tree_draft_repack_request {
    common_tree_draft_repack_source_identity source;
    const common_tree_draft_quant_type * quant = nullptr;
    common_tree_draft_tensor_layout source_layout;
    common_tree_draft_tensor_layout target_layout;
    std::string target_kernel;
    uint32_t kernel_abi_version = 0;
    double tolerance = 0.0;
};

struct common_tree_draft_repack_plan {
    std::string cache_key;
    common_tree_draft_repack_source_identity source;
    uint32_t quant_id = 0;
    common_tree_draft_tensor_layout_kind source_layout = COMMON_TREE_DRAFT_LAYOUT_DENSE;
    common_tree_draft_tensor_layout_kind target_layout = COMMON_TREE_DRAFT_LAYOUT_DENSE;
    std::string target_kernel;
    uint32_t kernel_abi_version = 0;
    double tolerance = 0.0;
};

enum common_tree_draft_repack_status : uint32_t {
    COMMON_TREE_DRAFT_REPACK_OK = 0,
    COMMON_TREE_DRAFT_REPACK_INVALID,
    COMMON_TREE_DRAFT_REPACK_LAYOUT,
    COMMON_TREE_DRAFT_REPACK_SOURCE_RANGE,
    COMMON_TREE_DRAFT_REPACK_NUMERIC_MISMATCH,
};

common_tree_draft_repack_status common_tree_draft_repack_plan_build(
        const common_tree_draft_repack_request & request,
        common_tree_draft_repack_plan * plan);

bool common_tree_draft_repack_equivalent(
        const float * source_decoded,
        const float * repacked_decoded,
        size_t count,
        double tolerance,
        double * max_abs_error = nullptr);
