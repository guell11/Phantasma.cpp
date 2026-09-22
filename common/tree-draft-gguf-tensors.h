#pragma once

#include "tree-draft-gguf-header.h"

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

enum common_tree_draft_gguf_tensor_error : int32_t {
    COMMON_TREE_DRAFT_GGUF_TENSOR_OK = 0,
    COMMON_TREE_DRAFT_GGUF_TENSOR_INVALID_SOURCE,
    COMMON_TREE_DRAFT_GGUF_TENSOR_HEADER_ERROR,
    COMMON_TREE_DRAFT_GGUF_TENSOR_OPEN_FAILED,
    COMMON_TREE_DRAFT_GGUF_TENSOR_TRUNCATED,
    COMMON_TREE_DRAFT_GGUF_TENSOR_METADATA_INVALID,
    COMMON_TREE_DRAFT_GGUF_TENSOR_EMPTY_NAME,
    COMMON_TREE_DRAFT_GGUF_TENSOR_DUPLICATE_NAME,
    COMMON_TREE_DRAFT_GGUF_TENSOR_INVALID_RANK,
    COMMON_TREE_DRAFT_GGUF_TENSOR_INVALID_DIMENSION,
    COMMON_TREE_DRAFT_GGUF_TENSOR_INVALID_TYPE,
    COMMON_TREE_DRAFT_GGUF_TENSOR_SIZE_OVERFLOW,
    COMMON_TREE_DRAFT_GGUF_TENSOR_SPAN_OUT_OF_BOUNDS,
};

struct common_tree_draft_gguf_tensor_descriptor {
    std::string name;
    std::vector<int64_t> dims;
    ggml_type type = GGML_TYPE_COUNT;
    uint64_t offset = 0;
    uint64_t storage_size = 0;
};

struct common_tree_draft_gguf_tensor_directory {
    std::vector<common_tree_draft_gguf_tensor_descriptor> tensors;
    uint64_t directory_end = 0;
    uint64_t data_base = 0;
    uint32_t alignment = 32;
};

struct common_tree_draft_gguf_tensor_result {
    common_tree_draft_gguf_tensor_directory directory;
    common_tree_draft_gguf_tensor_error error = COMMON_TREE_DRAFT_GGUF_TENSOR_OK;
    common_tree_draft_gguf_header_error header_error = COMMON_TREE_DRAFT_GGUF_HEADER_OK;
    uint64_t error_offset = 0;
    std::string message;

    explicit operator bool() const {
        return error == COMMON_TREE_DRAFT_GGUF_TENSOR_OK;
    }
};

common_tree_draft_gguf_tensor_result common_tree_draft_gguf_tensor_directory_parse(
        const common_tree_draft_model_source & source,
        size_t path_index = 0);

const char * common_tree_draft_gguf_tensor_error_name(common_tree_draft_gguf_tensor_error error);
