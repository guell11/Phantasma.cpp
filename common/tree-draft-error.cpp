#include "tree-draft-error.h"

#include <utility>

common_tree_draft_error common_tree_draft_error_from_engine_error(
        common_tree_draft_engine_error engine_error,
        std::string message,
        std::string request_id,
        std::vector<common_tree_draft_metadata_entry> details) {
    common_tree_draft_error result = {
        /* .domain = */ COMMON_TREE_DRAFT_ERROR_DOMAIN_INTERNAL,
        /* .code = */ COMMON_TREE_DRAFT_ERROR_CODE_INTERNAL,
        /* .message = */ std::move(message),
        /* .retryable = */ false,
        /* .request_id = */ std::move(request_id),
        /* .details = */ std::move(details),
    };

    switch (engine_error) {
        case COMMON_TREE_DRAFT_ENGINE_ERROR_INVALID_ARGUMENT:
            result.domain = COMMON_TREE_DRAFT_ERROR_DOMAIN_INVALID_ARGUMENT;
            result.code = COMMON_TREE_DRAFT_ERROR_CODE_INVALID_ARGUMENT;
            break;
        case COMMON_TREE_DRAFT_ENGINE_ERROR_RESOURCE_EXHAUSTED:
            result.domain = COMMON_TREE_DRAFT_ERROR_DOMAIN_RESOURCE_EXHAUSTED;
            result.code = COMMON_TREE_DRAFT_ERROR_CODE_RESOURCE_EXHAUSTED;
            result.retryable = true;
            break;
        case COMMON_TREE_DRAFT_ENGINE_ERROR_CANCELLED:
            result.domain = COMMON_TREE_DRAFT_ERROR_DOMAIN_CANCELLED;
            result.code = COMMON_TREE_DRAFT_ERROR_CODE_CANCELLED;
            break;
        case COMMON_TREE_DRAFT_ENGINE_ERROR_DEADLINE_EXCEEDED:
            result.domain = COMMON_TREE_DRAFT_ERROR_DOMAIN_DEADLINE;
            result.code = COMMON_TREE_DRAFT_ERROR_CODE_DEADLINE_EXCEEDED;
            result.retryable = true;
            break;
        case COMMON_TREE_DRAFT_ENGINE_ERROR_UNAVAILABLE:
            result.domain = COMMON_TREE_DRAFT_ERROR_DOMAIN_UNAVAILABLE;
            result.code = COMMON_TREE_DRAFT_ERROR_CODE_UNAVAILABLE;
            result.retryable = true;
            break;
        case COMMON_TREE_DRAFT_ENGINE_ERROR_INTERNAL:
        case COMMON_TREE_DRAFT_ENGINE_ERROR_PROTOCOL:
            break;
    }

    if (engine_error == COMMON_TREE_DRAFT_ENGINE_ERROR_PROTOCOL) {
        result.domain = COMMON_TREE_DRAFT_ERROR_DOMAIN_PROTOCOL;
        result.code = COMMON_TREE_DRAFT_ERROR_CODE_PROTOCOL;
    }

    return result;
}

const char * common_tree_draft_error_domain_name(common_tree_draft_error_domain domain) {
    switch (domain) {
        case COMMON_TREE_DRAFT_ERROR_DOMAIN_INVALID_ARGUMENT:    return "invalid_argument";
        case COMMON_TREE_DRAFT_ERROR_DOMAIN_RESOURCE_EXHAUSTED:  return "resource_exhausted";
        case COMMON_TREE_DRAFT_ERROR_DOMAIN_CANCELLED:           return "cancelled";
        case COMMON_TREE_DRAFT_ERROR_DOMAIN_DEADLINE:            return "deadline";
        case COMMON_TREE_DRAFT_ERROR_DOMAIN_UNAVAILABLE:         return "unavailable";
        case COMMON_TREE_DRAFT_ERROR_DOMAIN_INTERNAL:            return "internal";
        case COMMON_TREE_DRAFT_ERROR_DOMAIN_PROTOCOL:            return "protocol";
    }

    return "internal";
}

const char * common_tree_draft_error_code_name(common_tree_draft_error_code code) {
    switch (code) {
        case COMMON_TREE_DRAFT_ERROR_CODE_INVALID_ARGUMENT:     return "invalid_argument";
        case COMMON_TREE_DRAFT_ERROR_CODE_RESOURCE_EXHAUSTED:   return "resource_exhausted";
        case COMMON_TREE_DRAFT_ERROR_CODE_CANCELLED:            return "cancelled";
        case COMMON_TREE_DRAFT_ERROR_CODE_DEADLINE_EXCEEDED:    return "deadline_exceeded";
        case COMMON_TREE_DRAFT_ERROR_CODE_UNAVAILABLE:          return "unavailable";
        case COMMON_TREE_DRAFT_ERROR_CODE_INTERNAL:             return "internal";
        case COMMON_TREE_DRAFT_ERROR_CODE_PROTOCOL:             return "protocol";
    }

    return "internal";
}
