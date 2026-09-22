#pragma once

#include "tree-draft-gguf-metadata.h"
#include "tree-draft-gguf-tensors.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

enum common_tree_draft_gguf_shard_error : int32_t {
    COMMON_TREE_DRAFT_GGUF_SHARD_OK = 0,
    COMMON_TREE_DRAFT_GGUF_SHARD_INVALID_SOURCE,
    COMMON_TREE_DRAFT_GGUF_SHARD_METADATA_ERROR,
    COMMON_TREE_DRAFT_GGUF_SHARD_TENSOR_ERROR,
    COMMON_TREE_DRAFT_GGUF_SHARD_MISSING_SPLIT_METADATA,
    COMMON_TREE_DRAFT_GGUF_SHARD_MALFORMED_SPLIT_METADATA,
    COMMON_TREE_DRAFT_GGUF_SHARD_INCONSISTENT_SPLIT_COUNT,
    COMMON_TREE_DRAFT_GGUF_SHARD_DUPLICATE_INDEX,
    COMMON_TREE_DRAFT_GGUF_SHARD_MISSING_INDEX,
    COMMON_TREE_DRAFT_GGUF_SHARD_FOREIGN_MODEL,
    COMMON_TREE_DRAFT_GGUF_SHARD_DUPLICATE_TENSOR,
    COMMON_TREE_DRAFT_GGUF_SHARD_TENSOR_COUNT_MISMATCH,
};

struct common_tree_draft_gguf_logical_tensor {
    common_tree_draft_gguf_tensor_descriptor descriptor;
    uint16_t shard_index = 0;
    size_t source_path_index = 0;
    std::string source_path;
};

struct common_tree_draft_gguf_shard_set {
    uint16_t split_count = 0;
    uint32_t expected_tensor_count = 0;
    std::vector<size_t> source_path_indices;
    std::vector<std::string> source_paths;
    std::vector<common_tree_draft_gguf_logical_tensor> tensors;
};

struct common_tree_draft_gguf_shard_result {
    common_tree_draft_gguf_shard_set set;
    common_tree_draft_gguf_shard_error error = COMMON_TREE_DRAFT_GGUF_SHARD_OK;
    size_t source_path_index = 0;
    std::string message;

    explicit operator bool() const {
        return error == COMMON_TREE_DRAFT_GGUF_SHARD_OK;
    }
};

common_tree_draft_gguf_shard_result common_tree_draft_gguf_resolve_shards(
        const common_tree_draft_model_source & source);

const char * common_tree_draft_gguf_shard_error_name(common_tree_draft_gguf_shard_error error);
