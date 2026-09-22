#pragma once

#include "tree-draft-model-source.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

enum common_tree_draft_safetensors_error : int32_t {
    COMMON_TREE_DRAFT_SAFETENSORS_OK = 0,
    COMMON_TREE_DRAFT_SAFETENSORS_INVALID_SOURCE,
    COMMON_TREE_DRAFT_SAFETENSORS_OPEN_FAILED,
    COMMON_TREE_DRAFT_SAFETENSORS_TRUNCATED,
    COMMON_TREE_DRAFT_SAFETENSORS_HEADER_TOO_LARGE,
    COMMON_TREE_DRAFT_SAFETENSORS_INVALID_JSON,
    COMMON_TREE_DRAFT_SAFETENSORS_INVALID_HEADER,
    COMMON_TREE_DRAFT_SAFETENSORS_DUPLICATE_TENSOR,
    COMMON_TREE_DRAFT_SAFETENSORS_UNSUPPORTED_DTYPE,
    COMMON_TREE_DRAFT_SAFETENSORS_INVALID_SHAPE,
    COMMON_TREE_DRAFT_SAFETENSORS_INVALID_OFFSETS,
    COMMON_TREE_DRAFT_SAFETENSORS_SIZE_MISMATCH,
    COMMON_TREE_DRAFT_SAFETENSORS_OVERLAPPING_TENSORS,
};

struct common_tree_draft_safetensors_tensor {
    std::string name;
    std::string dtype;
    std::vector<uint64_t> shape;
    uint64_t offset_begin = 0;
    uint64_t offset_end = 0;
};

struct common_tree_draft_safetensors_table {
    uint64_t header_size = 0;
    uint64_t payload_size = 0;
    std::map<std::string, std::string> metadata;
    std::vector<common_tree_draft_safetensors_tensor> tensors;
};

struct common_tree_draft_safetensors_result {
    common_tree_draft_safetensors_table table;
    common_tree_draft_safetensors_error error = COMMON_TREE_DRAFT_SAFETENSORS_OK;
    uint64_t error_offset = 0;
    std::string message;

    explicit operator bool() const {
        return error == COMMON_TREE_DRAFT_SAFETENSORS_OK;
    }
};

common_tree_draft_safetensors_result common_tree_draft_safetensors_parse(
        const common_tree_draft_model_source & source,
        size_t path_index = 0);

const char * common_tree_draft_safetensors_error_name(common_tree_draft_safetensors_error error);
