#include "tree-draft-node.h"

#include <cmath>
#include <limits>
#include <type_traits>

static_assert(std::is_standard_layout<common_tree_draft_node>::value, "draft node must be standard-layout");
static_assert(std::is_trivially_copyable<common_tree_draft_node>::value, "draft node must be trivially copyable");
static_assert(sizeof(common_tree_draft_node) == 32, "draft node ABI size changed");
static_assert(alignof(common_tree_draft_node) == alignof(uint64_t), "draft node ABI alignment changed");

static constexpr uint32_t COMMON_TREE_DRAFT_NODE_FLAG_MASK =
        COMMON_TREE_DRAFT_NODE_FLAG_FRONTIER |
        COMMON_TREE_DRAFT_NODE_FLAG_PRUNED |
        COMMON_TREE_DRAFT_NODE_FLAG_TERMINAL;

common_tree_draft_node_error common_tree_draft_nodes_validate(
        const common_tree_draft_node * nodes,
        size_t n_nodes,
        const common_tree_draft_config & config) {
    if (n_nodes == 0) {
        return COMMON_TREE_DRAFT_NODE_OK;
    }
    if (nodes == nullptr) {
        return COMMON_TREE_DRAFT_NODE_NULL_NODES;
    }
    if (n_nodes > config.tree_budget() || n_nodes > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        return COMMON_TREE_DRAFT_NODE_COUNT_RANGE;
    }

    for (size_t i = 0; i < n_nodes; ++i) {
        const common_tree_draft_node & node = nodes[i];

        if ((node.flags & ~COMMON_TREE_DRAFT_NODE_FLAG_MASK) != 0 ||
            ((node.flags & COMMON_TREE_DRAFT_NODE_FLAG_FRONTIER) != 0 &&
             (node.flags & (COMMON_TREE_DRAFT_NODE_FLAG_PRUNED | COMMON_TREE_DRAFT_NODE_FLAG_TERMINAL)) != 0)) {
            return COMMON_TREE_DRAFT_NODE_FLAGS;
        }
        if (node.path_id == 0) {
            return COMMON_TREE_DRAFT_NODE_PATH_ID;
        }

        if (i == 0) {
            if (node.parent != -1) {
                return COMMON_TREE_DRAFT_NODE_ROOT_PARENT;
            }
            if (node.depth != 0) {
                return COMMON_TREE_DRAFT_NODE_ROOT_DEPTH;
            }
            if (node.logp != 0.0f) {
                return COMMON_TREE_DRAFT_NODE_ROOT_LOGP;
            }
            continue;
        }

        if (node.parent < 0 || static_cast<size_t>(node.parent) >= i) {
            return COMMON_TREE_DRAFT_NODE_PARENT_RANGE;
        }
        if (nodes[node.parent].depth == std::numeric_limits<int32_t>::max() ||
            node.depth != nodes[node.parent].depth + 1) {
            return COMMON_TREE_DRAFT_NODE_DEPTH_MISMATCH;
        }
        if (node.depth <= 0 || static_cast<uint32_t>(node.depth) > config.max_depth()) {
            return COMMON_TREE_DRAFT_NODE_DEPTH_LIMIT;
        }
        if (node.token < 0) {
            return COMMON_TREE_DRAFT_NODE_TOKEN_RANGE;
        }
        if (!std::isfinite(node.logp) || node.logp > 0.0f) {
            return COMMON_TREE_DRAFT_NODE_LOGP;
        }
    }

    return COMMON_TREE_DRAFT_NODE_OK;
}

const char * common_tree_draft_node_error_name(common_tree_draft_node_error error) {
    switch (error) {
        case COMMON_TREE_DRAFT_NODE_OK:             return "ok";
        case COMMON_TREE_DRAFT_NODE_NULL_NODES:     return "null_nodes";
        case COMMON_TREE_DRAFT_NODE_COUNT_RANGE:    return "count_range";
        case COMMON_TREE_DRAFT_NODE_ROOT_PARENT:    return "root_parent";
        case COMMON_TREE_DRAFT_NODE_ROOT_DEPTH:     return "root_depth";
        case COMMON_TREE_DRAFT_NODE_ROOT_LOGP:      return "root_logp";
        case COMMON_TREE_DRAFT_NODE_PARENT_RANGE:   return "parent_range";
        case COMMON_TREE_DRAFT_NODE_DEPTH_MISMATCH: return "depth_mismatch";
        case COMMON_TREE_DRAFT_NODE_DEPTH_LIMIT:    return "depth_limit";
        case COMMON_TREE_DRAFT_NODE_TOKEN_RANGE:    return "token_range";
        case COMMON_TREE_DRAFT_NODE_LOGP:           return "logp";
        case COMMON_TREE_DRAFT_NODE_PATH_ID:        return "path_id";
        case COMMON_TREE_DRAFT_NODE_FLAGS:          return "flags";
    }
    return "unknown";
}

