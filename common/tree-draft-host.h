#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct common_tree_draft_token_span {
    const int32_t * data = nullptr;
    size_t size = 0;
};

struct common_tree_draft_sampling_params {
    float temperature = 1.0f;
    float top_p = 1.0f;
    uint32_t top_k = 0;
    uint64_t seed = 0;
};

struct common_tree_draft_request_limits {
    uint32_t max_tokens = 0;
    uint32_t tree_budget = 0;
    uint32_t max_depth = 0;
    uint32_t candidate_cap = 0;
};

struct common_tree_draft_metadata_entry {
    std::string key;
    std::string value;
};

enum common_tree_draft_result_status : int32_t {
    COMMON_TREE_DRAFT_RESULT_STATUS_OK = 0,
    COMMON_TREE_DRAFT_RESULT_STATUS_CANCELLED,
    COMMON_TREE_DRAFT_RESULT_STATUS_FAILED,
};

enum common_tree_draft_finish_reason : int32_t {
    COMMON_TREE_DRAFT_FINISH_REASON_NONE = 0,
    COMMON_TREE_DRAFT_FINISH_REASON_STOP,
    COMMON_TREE_DRAFT_FINISH_REASON_LENGTH,
    COMMON_TREE_DRAFT_FINISH_REASON_CANCELLED,
    COMMON_TREE_DRAFT_FINISH_REASON_ERROR,
};

struct common_tree_draft_usage {
    uint32_t input_tokens = 0;
    uint32_t output_tokens = 0;
    uint32_t accepted_tokens = 0;
};

struct common_tree_draft_request {
    std::string request_id;
    std::string session_id;
    common_tree_draft_token_span input_tokens;
    common_tree_draft_sampling_params sampling;
    common_tree_draft_request_limits limits;
    std::vector<common_tree_draft_metadata_entry> metadata;
};

struct common_tree_draft_result {
    std::string request_id;
    common_tree_draft_result_status status = COMMON_TREE_DRAFT_RESULT_STATUS_OK;
    std::vector<int32_t> accepted_tokens;
    std::vector<int32_t> output_tokens;
    common_tree_draft_usage usage;
    common_tree_draft_finish_reason finish_reason = COMMON_TREE_DRAFT_FINISH_REASON_NONE;
};
