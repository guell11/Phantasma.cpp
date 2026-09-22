#pragma once

#include "tree-draft-config.h"

#include <cstddef>
#include <cstdint>

enum common_tree_draft_node_flag : uint32_t {
    COMMON_TREE_DRAFT_NODE_FLAG_NONE     = 0,
    COMMON_TREE_DRAFT_NODE_FLAG_FRONTIER = 1u << 0,
    COMMON_TREE_DRAFT_NODE_FLAG_PRUNED   = 1u << 1,
    COMMON_TREE_DRAFT_NODE_FLAG_TERMINAL = 1u << 2,
};

struct common_tree_draft_node {
    int32_t  parent;
    int32_t  token;
    int32_t  depth;
    float    logp;
    uint64_t path_id;
    uint32_t flags;
};

enum common_tree_draft_node_error : int32_t {
    COMMON_TREE_DRAFT_NODE_OK = 0,
    COMMON_TREE_DRAFT_NODE_NULL_NODES,
    COMMON_TREE_DRAFT_NODE_COUNT_RANGE,
    COMMON_TREE_DRAFT_NODE_ROOT_PARENT,
    COMMON_TREE_DRAFT_NODE_ROOT_DEPTH,
    COMMON_TREE_DRAFT_NODE_ROOT_LOGP,
    COMMON_TREE_DRAFT_NODE_PARENT_RANGE,
    COMMON_TREE_DRAFT_NODE_DEPTH_MISMATCH,
    COMMON_TREE_DRAFT_NODE_DEPTH_LIMIT,
    COMMON_TREE_DRAFT_NODE_TOKEN_RANGE,
    COMMON_TREE_DRAFT_NODE_LOGP,
    COMMON_TREE_DRAFT_NODE_PATH_ID,
    COMMON_TREE_DRAFT_NODE_FLAGS,
};

common_tree_draft_node_error common_tree_draft_nodes_validate(
        const common_tree_draft_node * nodes,
        size_t n_nodes,
        const common_tree_draft_config & config);

const char * common_tree_draft_node_error_name(common_tree_draft_node_error error);

