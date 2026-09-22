#pragma once

#include "tree-draft-topology.h"

#include <cstddef>
#include <cstdint>

static constexpr uint32_t COMMON_TREE_DRAFT_KV_ID_INVALID = UINT32_MAX;
static constexpr uint32_t COMMON_TREE_DRAFT_KV_OFFSET_INVALID = UINT32_MAX;

struct common_tree_draft_branch_handle {
    uint32_t id;
    uint32_t generation;
};

struct common_tree_draft_page_handle {
    uint32_t id;
    uint32_t generation;
};

struct common_tree_draft_page_address {
    common_tree_draft_page_handle page;
    uint32_t offset;
};

struct common_tree_draft_kv_token_identity {
    int32_t logical_node;
    int64_t logical_position;
    common_tree_draft_branch_handle branch;
    common_tree_draft_page_address address;
    int32_t shared_from_node;
    bool materialized;
};

enum common_tree_draft_kv_identity_error : int32_t {
    COMMON_TREE_DRAFT_KV_IDENTITY_OK = 0,
    COMMON_TREE_DRAFT_KV_IDENTITY_INVALID_TOPOLOGY,
    COMMON_TREE_DRAFT_KV_IDENTITY_NULL_TOKENS,
    COMMON_TREE_DRAFT_KV_IDENTITY_TOKEN_COUNT_MISMATCH,
    COMMON_TREE_DRAFT_KV_IDENTITY_LOGICAL_NODE_MISMATCH,
    COMMON_TREE_DRAFT_KV_IDENTITY_POSITION_MISMATCH,
    COMMON_TREE_DRAFT_KV_IDENTITY_INVALID_BRANCH,
    COMMON_TREE_DRAFT_KV_IDENTITY_INVALID_PAGE,
    COMMON_TREE_DRAFT_KV_IDENTITY_INVALID_OFFSET,
    COMMON_TREE_DRAFT_KV_IDENTITY_UNMATERIALIZED_ADDRESS,
    COMMON_TREE_DRAFT_KV_IDENTITY_INVALID_SHARED_ANCESTOR,
    COMMON_TREE_DRAFT_KV_IDENTITY_DUPLICATE_ADDRESS,
};

common_tree_draft_kv_identity_error common_tree_draft_kv_identity_validate(
        const common_tree_draft_topology & topology,
        const common_tree_draft_kv_token_identity * tokens,
        size_t token_count);

bool common_tree_draft_branch_handle_equal(
        const common_tree_draft_branch_handle & a,
        const common_tree_draft_branch_handle & b);

bool common_tree_draft_page_handle_equal(
        const common_tree_draft_page_handle & a,
        const common_tree_draft_page_handle & b);

bool common_tree_draft_page_address_equal(
        const common_tree_draft_page_address & a,
        const common_tree_draft_page_address & b);

const char * common_tree_draft_kv_identity_error_name(common_tree_draft_kv_identity_error error);
