#pragma once

#include "tree-draft-model-source.h"

#include <cstddef>
#include <cstdint>
#include <string>

enum common_tree_draft_gguf_header_error : int32_t {
    COMMON_TREE_DRAFT_GGUF_HEADER_OK = 0,
    COMMON_TREE_DRAFT_GGUF_HEADER_INVALID_SOURCE,
    COMMON_TREE_DRAFT_GGUF_HEADER_OPEN_FAILED,
    COMMON_TREE_DRAFT_GGUF_HEADER_TRUNCATED,
    COMMON_TREE_DRAFT_GGUF_HEADER_BAD_MAGIC,
    COMMON_TREE_DRAFT_GGUF_HEADER_UNSUPPORTED_VERSION,
    COMMON_TREE_DRAFT_GGUF_HEADER_IMPOSSIBLE_COUNT,
};

struct common_tree_draft_gguf_header {
    uint32_t version = 0;
    uint64_t n_tensors = 0;
    uint64_t n_kv = 0;
    uint64_t header_end = 0;
};

struct common_tree_draft_gguf_header_result {
    common_tree_draft_gguf_header header;
    common_tree_draft_gguf_header_error error = COMMON_TREE_DRAFT_GGUF_HEADER_OK;
    uint64_t error_offset = 0;
    std::string message;

    explicit operator bool() const {
        return error == COMMON_TREE_DRAFT_GGUF_HEADER_OK;
    }
};

bool common_tree_draft_gguf_version_supported(uint32_t version);

common_tree_draft_gguf_header_result common_tree_draft_gguf_header_parse(
        const common_tree_draft_model_source & source,
        size_t path_index = 0);

const char * common_tree_draft_gguf_header_error_name(common_tree_draft_gguf_header_error error);
