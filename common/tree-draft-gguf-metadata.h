#pragma once

#include "tree-draft-gguf-header.h"

#include "gguf.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

enum common_tree_draft_gguf_metadata_error : int32_t {
    COMMON_TREE_DRAFT_GGUF_METADATA_OK = 0,
    COMMON_TREE_DRAFT_GGUF_METADATA_INVALID_SOURCE,
    COMMON_TREE_DRAFT_GGUF_METADATA_HEADER_ERROR,
    COMMON_TREE_DRAFT_GGUF_METADATA_OPEN_FAILED,
    COMMON_TREE_DRAFT_GGUF_METADATA_TRUNCATED,
    COMMON_TREE_DRAFT_GGUF_METADATA_EMPTY_KEY,
    COMMON_TREE_DRAFT_GGUF_METADATA_DUPLICATE_KEY,
    COMMON_TREE_DRAFT_GGUF_METADATA_INVALID_TYPE,
    COMMON_TREE_DRAFT_GGUF_METADATA_UNSUPPORTED_TYPE,
    COMMON_TREE_DRAFT_GGUF_METADATA_LIMIT_EXCEEDED,
};

struct common_tree_draft_gguf_metadata_value {
    gguf_type type = GGUF_TYPE_COUNT;
    bool is_array = false;
    gguf_type array_type = GGUF_TYPE_COUNT;
    std::vector<unsigned char> raw;
    std::vector<std::string> strings;
};

struct common_tree_draft_gguf_metadata_entry {
    std::string key;
    common_tree_draft_gguf_metadata_value value;
};

struct common_tree_draft_gguf_metadata_table {
    std::vector<common_tree_draft_gguf_metadata_entry> entries;
    uint64_t metadata_end = 0;
};

struct common_tree_draft_gguf_metadata_result {
    common_tree_draft_gguf_metadata_table table;
    common_tree_draft_gguf_metadata_error error = COMMON_TREE_DRAFT_GGUF_METADATA_OK;
    common_tree_draft_gguf_header_error header_error = COMMON_TREE_DRAFT_GGUF_HEADER_OK;
    uint64_t error_offset = 0;
    std::string message;

    explicit operator bool() const {
        return error == COMMON_TREE_DRAFT_GGUF_METADATA_OK;
    }
};

enum common_tree_draft_gguf_metadata_lookup_status : int32_t {
    COMMON_TREE_DRAFT_GGUF_METADATA_LOOKUP_OK = 0,
    COMMON_TREE_DRAFT_GGUF_METADATA_LOOKUP_MISSING,
    COMMON_TREE_DRAFT_GGUF_METADATA_LOOKUP_MALFORMED,
    COMMON_TREE_DRAFT_GGUF_METADATA_LOOKUP_UNSUPPORTED,
};

struct common_tree_draft_gguf_metadata_lookup_result {
    common_tree_draft_gguf_metadata_lookup_status status = COMMON_TREE_DRAFT_GGUF_METADATA_LOOKUP_MISSING;
    const common_tree_draft_gguf_metadata_entry * entry = nullptr;

    explicit operator bool() const {
        return status == COMMON_TREE_DRAFT_GGUF_METADATA_LOOKUP_OK;
    }
};

common_tree_draft_gguf_metadata_result common_tree_draft_gguf_metadata_parse(
        const common_tree_draft_model_source & source,
        size_t path_index = 0);

common_tree_draft_gguf_metadata_lookup_result common_tree_draft_gguf_metadata_find(
        const common_tree_draft_gguf_metadata_table & table,
        const std::string & key);

common_tree_draft_gguf_metadata_lookup_result common_tree_draft_gguf_metadata_get_scalar(
        const common_tree_draft_gguf_metadata_table & table,
        const std::string & key,
        gguf_type expected_type);

common_tree_draft_gguf_metadata_lookup_result common_tree_draft_gguf_metadata_get_array(
        const common_tree_draft_gguf_metadata_table & table,
        const std::string & key,
        gguf_type expected_element_type);

const char * common_tree_draft_gguf_metadata_error_name(common_tree_draft_gguf_metadata_error error);
const char * common_tree_draft_gguf_metadata_lookup_status_name(common_tree_draft_gguf_metadata_lookup_status status);
