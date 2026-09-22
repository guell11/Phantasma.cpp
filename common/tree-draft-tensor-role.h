#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

enum common_tree_draft_tensor_role : uint32_t {
    COMMON_TREE_DRAFT_TENSOR_UNKNOWN = 0,
    COMMON_TREE_DRAFT_TENSOR_TOKEN_EMBEDDING,
    COMMON_TREE_DRAFT_TENSOR_OUTPUT_NORM,
    COMMON_TREE_DRAFT_TENSOR_OUTPUT,
    COMMON_TREE_DRAFT_TENSOR_ATTN_NORM,
    COMMON_TREE_DRAFT_TENSOR_ATTN_Q,
    COMMON_TREE_DRAFT_TENSOR_ATTN_K,
    COMMON_TREE_DRAFT_TENSOR_ATTN_V,
    COMMON_TREE_DRAFT_TENSOR_ATTN_OUT,
    COMMON_TREE_DRAFT_TENSOR_ATTN_Q_NORM,
    COMMON_TREE_DRAFT_TENSOR_ATTN_K_NORM,
    COMMON_TREE_DRAFT_TENSOR_FFN_NORM,
    COMMON_TREE_DRAFT_TENSOR_FFN_ROUTER,
    COMMON_TREE_DRAFT_TENSOR_FFN_GATE,
    COMMON_TREE_DRAFT_TENSOR_FFN_UP,
    COMMON_TREE_DRAFT_TENSOR_FFN_DOWN,
    COMMON_TREE_DRAFT_TENSOR_FFN_GATE_EXPERT,
    COMMON_TREE_DRAFT_TENSOR_FFN_UP_EXPERT,
    COMMON_TREE_DRAFT_TENSOR_FFN_DOWN_EXPERT,
    COMMON_TREE_DRAFT_TENSOR_FFN_GATE_EXPERTS,
    COMMON_TREE_DRAFT_TENSOR_FFN_GATE_UP_EXPERTS,
    COMMON_TREE_DRAFT_TENSOR_FFN_UP_EXPERTS,
    COMMON_TREE_DRAFT_TENSOR_FFN_DOWN_EXPERTS,
};

enum common_tree_draft_tensor_subrole : uint32_t {
    COMMON_TREE_DRAFT_TENSOR_SUBROLE_NONE = 0,
    COMMON_TREE_DRAFT_TENSOR_SUBROLE_WEIGHT,
    COMMON_TREE_DRAFT_TENSOR_SUBROLE_BIAS,
};

struct common_tree_draft_tensor_role_key {
    common_tree_draft_tensor_role role = COMMON_TREE_DRAFT_TENSOR_UNKNOWN;
    int32_t layer = -1;
    int32_t expert = -1;
    common_tree_draft_tensor_subrole subrole = COMMON_TREE_DRAFT_TENSOR_SUBROLE_NONE;
};

struct common_tree_draft_tensor_role_result {
    common_tree_draft_tensor_role_key key;
    std::string source_name;
};

enum common_tree_draft_tensor_role_status : uint32_t {
    COMMON_TREE_DRAFT_TENSOR_ROLE_OK = 0,
    COMMON_TREE_DRAFT_TENSOR_ROLE_UNSUPPORTED_ARCH,
    COMMON_TREE_DRAFT_TENSOR_ROLE_INVALID_NAME,
    COMMON_TREE_DRAFT_TENSOR_ROLE_INDEX_RANGE,
    COMMON_TREE_DRAFT_TENSOR_ROLE_COLLISION,
};

common_tree_draft_tensor_role_status common_tree_draft_tensor_role_canonicalize(
        const std::string & architecture,
        const std::string & name,
        common_tree_draft_tensor_role_result * result);

common_tree_draft_tensor_role_status common_tree_draft_tensor_role_validate_unique(
        const common_tree_draft_tensor_role_result * results,
        size_t count,
        size_t * first_collision = nullptr,
        size_t * second_collision = nullptr);

bool common_tree_draft_tensor_role_key_equal(
        const common_tree_draft_tensor_role_key & a,
        const common_tree_draft_tensor_role_key & b);

