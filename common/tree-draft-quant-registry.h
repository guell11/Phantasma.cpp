#pragma once

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

enum common_tree_draft_quant_source : uint32_t {
    COMMON_TREE_DRAFT_QUANT_GGML = 0,
    COMMON_TREE_DRAFT_QUANT_GPTQ,
    COMMON_TREE_DRAFT_QUANT_AWQ,
    COMMON_TREE_DRAFT_QUANT_BNB8,
    COMMON_TREE_DRAFT_QUANT_BNB4,
};

enum common_tree_draft_quant_packing : uint32_t {
    COMMON_TREE_DRAFT_QUANT_PACKING_DENSE = 0,
    COMMON_TREE_DRAFT_QUANT_PACKING_BLOCK,
    COMMON_TREE_DRAFT_QUANT_PACKING_GROUPED,
    COMMON_TREE_DRAFT_QUANT_PACKING_CODEBOOK,
};

struct common_tree_draft_quant_type {
    uint32_t id = 0;
    std::string name;
    common_tree_draft_quant_source source = COMMON_TREE_DRAFT_QUANT_GGML;
    uint32_t block_elems = 0;
    uint32_t block_bytes = 0;
    std::string scale_type;
    std::string zero_type;
    common_tree_draft_quant_packing packing = COMMON_TREE_DRAFT_QUANT_PACKING_DENSE;
    std::string logical_dtype;
    bool fixed_block_size = true;
    int32_t ggml_type_id = -1;
};

enum common_tree_draft_quant_registry_status : uint32_t {
    COMMON_TREE_DRAFT_QUANT_REGISTRY_OK = 0,
    COMMON_TREE_DRAFT_QUANT_REGISTRY_DUPLICATE_ID,
    COMMON_TREE_DRAFT_QUANT_REGISTRY_DUPLICATE_NAME,
    COMMON_TREE_DRAFT_QUANT_REGISTRY_INVALID_BLOCK,
};

const std::vector<common_tree_draft_quant_type> & common_tree_draft_quant_registry();

common_tree_draft_quant_registry_status common_tree_draft_quant_registry_validate(
        const std::vector<common_tree_draft_quant_type> & registry,
        size_t * error_index = nullptr);

const common_tree_draft_quant_type * common_tree_draft_quant_find_name(const std::string & name);
const common_tree_draft_quant_type * common_tree_draft_quant_find_ggml(enum ggml_type type);
double common_tree_draft_quant_bits_per_weight(const common_tree_draft_quant_type & type);

