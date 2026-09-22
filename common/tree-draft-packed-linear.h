#pragma once

#include "tree-draft-external-quant.h"
#include "tree-draft-safetensors.h"

#include <cstdint>
#include <optional>
#include <string>

enum common_tree_draft_packed_linear_format : uint32_t {
    COMMON_TREE_DRAFT_PACKED_LINEAR_GPTQ = 0,
    COMMON_TREE_DRAFT_PACKED_LINEAR_AWQ,
};

struct common_tree_draft_packed_linear_view {
    common_tree_draft_packed_linear_format format = COMMON_TREE_DRAFT_PACKED_LINEAR_GPTQ;
    std::string qweight;
    std::string qzeros;
    std::string scales;
    std::optional<std::string> g_idx;
    uint64_t out_features = 0;
    uint64_t in_features = 0;
    uint32_t bits = 0;
    uint32_t packed_word_bits = 32;
    uint32_t unpack_factor = 0;
    uint32_t group_size = 0;
    uint64_t group_count = 0;
    bool symmetric = false;
    bool zero_point = false;
    bool storage_transposed = false;
};

enum common_tree_draft_packed_linear_status : uint32_t {
    COMMON_TREE_DRAFT_PACKED_LINEAR_OK = 0,
    COMMON_TREE_DRAFT_PACKED_LINEAR_MISSING,
    COMMON_TREE_DRAFT_PACKED_LINEAR_DTYPE,
    COMMON_TREE_DRAFT_PACKED_LINEAR_SHAPE,
    COMMON_TREE_DRAFT_PACKED_LINEAR_BITS,
    COMMON_TREE_DRAFT_PACKED_LINEAR_GROUP,
};

common_tree_draft_packed_linear_status common_tree_draft_gptq_bind(
        const common_tree_draft_safetensors_table & table,
        const std::string & prefix,
        const common_tree_draft_gptq_descriptor & quant,
        uint64_t out_features,
        uint64_t in_features,
        common_tree_draft_packed_linear_view * view);

common_tree_draft_packed_linear_status common_tree_draft_awq_bind(
        const common_tree_draft_safetensors_table & table,
        const std::string & prefix,
        const common_tree_draft_awq_descriptor & quant,
        uint64_t out_features,
        uint64_t in_features,
        common_tree_draft_packed_linear_view * view);

