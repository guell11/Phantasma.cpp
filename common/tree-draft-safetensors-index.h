#pragma once

#include "tree-draft-safetensors.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

enum common_tree_draft_safetensors_index_error : int32_t {
    COMMON_TREE_DRAFT_SAFETENSORS_INDEX_OK = 0,
    COMMON_TREE_DRAFT_SAFETENSORS_INDEX_INVALID_SOURCE,
    COMMON_TREE_DRAFT_SAFETENSORS_INDEX_MISSING_INDEX,
    COMMON_TREE_DRAFT_SAFETENSORS_INDEX_INVALID_JSON,
    COMMON_TREE_DRAFT_SAFETENSORS_INDEX_INVALID_WEIGHT_MAP,
    COMMON_TREE_DRAFT_SAFETENSORS_INDEX_DUPLICATE_TENSOR,
    COMMON_TREE_DRAFT_SAFETENSORS_INDEX_MISSING_SHARD,
    COMMON_TREE_DRAFT_SAFETENSORS_INDEX_SHARD_PARSE_ERROR,
    COMMON_TREE_DRAFT_SAFETENSORS_INDEX_TENSOR_MISSING,
    COMMON_TREE_DRAFT_SAFETENSORS_INDEX_TENSOR_CONFLICT,
};

struct common_tree_draft_safetensors_logical_tensor {
    common_tree_draft_safetensors_tensor descriptor;
    size_t source_path_index = 0;
    std::string source_path;
};

struct common_tree_draft_safetensors_index_directory {
    std::string index_path;
    std::vector<std::string> shard_paths;
    std::vector<common_tree_draft_safetensors_logical_tensor> tensors;
};

struct common_tree_draft_safetensors_index_result {
    common_tree_draft_safetensors_index_directory directory;
    common_tree_draft_safetensors_index_error error = COMMON_TREE_DRAFT_SAFETENSORS_INDEX_OK;
    std::string message;

    explicit operator bool() const {
        return error == COMMON_TREE_DRAFT_SAFETENSORS_INDEX_OK;
    }
};

common_tree_draft_safetensors_index_result common_tree_draft_safetensors_index_resolve(
        const common_tree_draft_model_source & source);

const char * common_tree_draft_safetensors_index_error_name(common_tree_draft_safetensors_index_error error);
