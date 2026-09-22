#pragma once

#include "tree-draft-safetensors.h"

#include <cstdint>
#include <optional>
#include <string>

struct common_tree_draft_gptq_descriptor {
    uint32_t bits = 0;
    int32_t group_size = 0;
    bool sym = false;
    bool desc_act = false;
    std::optional<double> damp_percent;
    std::string packing_version;
};

struct common_tree_draft_awq_descriptor {
    uint32_t bits = 0;
    uint32_t group_size = 0;
    bool zero_point = false;
    std::string packing_version;
};

struct common_tree_draft_bnb8_descriptor {
    std::string weight_tensor;
    std::string scale_tensor;
    std::optional<std::string> outlier_tensor;
    uint64_t rows = 0;
    uint64_t cols = 0;
};

enum common_tree_draft_bnb4_kind : uint32_t {
    COMMON_TREE_DRAFT_BNB4_NF4 = 0,
    COMMON_TREE_DRAFT_BNB4_FP4,
};

struct common_tree_draft_bnb4_descriptor {
    common_tree_draft_bnb4_kind kind = COMMON_TREE_DRAFT_BNB4_NF4;
    uint32_t block_size = 0;
    uint32_t nested_depth = 0;
    std::string weight_tensor;
    std::string absmax_tensor;
    std::optional<std::string> nested_absmax_tensor;
    uint64_t logical_elements = 0;
};

enum common_tree_draft_external_quant_status : uint32_t {
    COMMON_TREE_DRAFT_EXTERNAL_QUANT_OK = 0,
    COMMON_TREE_DRAFT_EXTERNAL_QUANT_NOT_PRESENT,
    COMMON_TREE_DRAFT_EXTERNAL_QUANT_MALFORMED,
    COMMON_TREE_DRAFT_EXTERNAL_QUANT_UNSUPPORTED,
    COMMON_TREE_DRAFT_EXTERNAL_QUANT_INCONSISTENT,
    COMMON_TREE_DRAFT_EXTERNAL_QUANT_MISSING_TENSOR,
    COMMON_TREE_DRAFT_EXTERNAL_QUANT_SHAPE,
};

common_tree_draft_external_quant_status common_tree_draft_gptq_parse(
        const common_tree_draft_safetensors_table & table,
        common_tree_draft_gptq_descriptor * descriptor);

common_tree_draft_external_quant_status common_tree_draft_awq_parse(
        const common_tree_draft_safetensors_table & table,
        common_tree_draft_awq_descriptor * descriptor);

common_tree_draft_external_quant_status common_tree_draft_bnb8_parse(
        const common_tree_draft_safetensors_table & table,
        const std::string & prefix,
        common_tree_draft_bnb8_descriptor * descriptor);

common_tree_draft_external_quant_status common_tree_draft_bnb4_parse(
        const common_tree_draft_safetensors_table & table,
        const std::string & prefix,
        common_tree_draft_bnb4_descriptor * descriptor);

bool common_tree_draft_gptq_equal(const common_tree_draft_gptq_descriptor & a, const common_tree_draft_gptq_descriptor & b);
bool common_tree_draft_awq_equal(const common_tree_draft_awq_descriptor & a, const common_tree_draft_awq_descriptor & b);

