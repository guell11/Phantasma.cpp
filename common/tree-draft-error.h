#pragma once

#include "tree-draft-host.h"

#include <cstdint>
#include <string>
#include <vector>

enum common_tree_draft_error_domain : int32_t {
    COMMON_TREE_DRAFT_ERROR_DOMAIN_INVALID_ARGUMENT = 0,
    COMMON_TREE_DRAFT_ERROR_DOMAIN_RESOURCE_EXHAUSTED,
    COMMON_TREE_DRAFT_ERROR_DOMAIN_CANCELLED,
    COMMON_TREE_DRAFT_ERROR_DOMAIN_DEADLINE,
    COMMON_TREE_DRAFT_ERROR_DOMAIN_UNAVAILABLE,
    COMMON_TREE_DRAFT_ERROR_DOMAIN_INTERNAL,
    COMMON_TREE_DRAFT_ERROR_DOMAIN_PROTOCOL,
};

enum common_tree_draft_error_code : int32_t {
    COMMON_TREE_DRAFT_ERROR_CODE_INVALID_ARGUMENT = 0,
    COMMON_TREE_DRAFT_ERROR_CODE_RESOURCE_EXHAUSTED,
    COMMON_TREE_DRAFT_ERROR_CODE_CANCELLED,
    COMMON_TREE_DRAFT_ERROR_CODE_DEADLINE_EXCEEDED,
    COMMON_TREE_DRAFT_ERROR_CODE_UNAVAILABLE,
    COMMON_TREE_DRAFT_ERROR_CODE_INTERNAL,
    COMMON_TREE_DRAFT_ERROR_CODE_PROTOCOL,
};

enum common_tree_draft_engine_error : int32_t {
    COMMON_TREE_DRAFT_ENGINE_ERROR_INVALID_ARGUMENT = 0,
    COMMON_TREE_DRAFT_ENGINE_ERROR_RESOURCE_EXHAUSTED,
    COMMON_TREE_DRAFT_ENGINE_ERROR_CANCELLED,
    COMMON_TREE_DRAFT_ENGINE_ERROR_DEADLINE_EXCEEDED,
    COMMON_TREE_DRAFT_ENGINE_ERROR_UNAVAILABLE,
    COMMON_TREE_DRAFT_ENGINE_ERROR_INTERNAL,
    COMMON_TREE_DRAFT_ENGINE_ERROR_PROTOCOL,
};

struct common_tree_draft_error {
    common_tree_draft_error_domain domain = COMMON_TREE_DRAFT_ERROR_DOMAIN_INTERNAL;
    common_tree_draft_error_code code = COMMON_TREE_DRAFT_ERROR_CODE_INTERNAL;
    std::string message;
    bool retryable = false;
    std::string request_id;
    std::vector<common_tree_draft_metadata_entry> details;
};

common_tree_draft_error common_tree_draft_error_from_engine_error(
        common_tree_draft_engine_error engine_error,
        std::string message = {},
        std::string request_id = {},
        std::vector<common_tree_draft_metadata_entry> details = {});

const char * common_tree_draft_error_domain_name(common_tree_draft_error_domain domain);
const char * common_tree_draft_error_code_name(common_tree_draft_error_code code);
