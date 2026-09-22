#pragma once

#include "tree-draft-architecture-config.h"
#include "tree-draft-tensor-role.h"

#include <cstdint>
#include <string>
#include <vector>

enum common_tree_draft_tensor_shape_status : uint32_t {
    COMMON_TREE_DRAFT_TENSOR_SHAPE_OK = 0,
    COMMON_TREE_DRAFT_TENSOR_SHAPE_UNKNOWN_ROLE,
    COMMON_TREE_DRAFT_TENSOR_SHAPE_LAYER_RANGE,
    COMMON_TREE_DRAFT_TENSOR_SHAPE_UNRESOLVED,
    COMMON_TREE_DRAFT_TENSOR_SHAPE_MISMATCH,
};

struct common_tree_draft_tensor_shape_check {
    common_tree_draft_tensor_shape_status status = COMMON_TREE_DRAFT_TENSOR_SHAPE_UNKNOWN_ROLE;
    std::vector<uint64_t> observed;
    std::vector<std::vector<uint64_t>> accepted;
    std::string symbolic_expectation;
};

common_tree_draft_tensor_shape_check common_tree_draft_tensor_shape_validate(
        const common_tree_draft_architecture_config & config,
        const common_tree_draft_tensor_role_key & key,
        const std::vector<uint64_t> & shape);

