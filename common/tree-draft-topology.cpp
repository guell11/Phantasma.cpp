#include "tree-draft-topology.h"

#include <limits>

common_tree_draft_topology_error common_tree_draft_topology_validate(const common_tree_draft_topology & topology) {
    if (topology.n_nodes < 0) {
        return COMMON_TREE_DRAFT_TOPOLOGY_NEGATIVE_NODE_COUNT;
    }
    if (topology.n_nodes == 0) {
        return COMMON_TREE_DRAFT_TOPOLOGY_OK;
    }
    if (topology.parent == nullptr) {
        return COMMON_TREE_DRAFT_TOPOLOGY_NULL_PARENT;
    }
    if (topology.depth == nullptr) {
        return COMMON_TREE_DRAFT_TOPOLOGY_NULL_DEPTH;
    }
    if (topology.tree_id == nullptr) {
        return COMMON_TREE_DRAFT_TOPOLOGY_NULL_TREE_ID;
    }

    for (int32_t i = 0; i < topology.n_nodes; ++i) {
        const int32_t parent = topology.parent[i];
        const int32_t depth = topology.depth[i];
        const int32_t tree_id = topology.tree_id[i];

        if (tree_id < 0) {
            return COMMON_TREE_DRAFT_TOPOLOGY_NEGATIVE_TREE_ID;
        }

        if (parent == -1) {
            if (depth != 0) {
                return COMMON_TREE_DRAFT_TOPOLOGY_ROOT_DEPTH;
            }
            for (int32_t j = 0; j < i; ++j) {
                if (topology.parent[j] == -1 && topology.tree_id[j] == tree_id) {
                    return COMMON_TREE_DRAFT_TOPOLOGY_MULTIPLE_ROOTS;
                }
            }
            continue;
        }

        if (parent < 0 || parent >= i) {
            return COMMON_TREE_DRAFT_TOPOLOGY_PARENT_RANGE;
        }
        if (topology.tree_id[parent] != tree_id) {
            return COMMON_TREE_DRAFT_TOPOLOGY_PARENT_TREE_MISMATCH;
        }
        if (topology.depth[parent] == std::numeric_limits<int32_t>::max() || depth != topology.depth[parent] + 1) {
            return COMMON_TREE_DRAFT_TOPOLOGY_DEPTH_MISMATCH;
        }
    }

    return COMMON_TREE_DRAFT_TOPOLOGY_OK;
}

const char * common_tree_draft_topology_error_name(common_tree_draft_topology_error error) {
    switch (error) {
        case COMMON_TREE_DRAFT_TOPOLOGY_OK:                       return "ok";
        case COMMON_TREE_DRAFT_TOPOLOGY_NULL_PARENT:              return "null_parent";
        case COMMON_TREE_DRAFT_TOPOLOGY_NULL_DEPTH:               return "null_depth";
        case COMMON_TREE_DRAFT_TOPOLOGY_NULL_TREE_ID:             return "null_tree_id";
        case COMMON_TREE_DRAFT_TOPOLOGY_NEGATIVE_NODE_COUNT:      return "negative_node_count";
        case COMMON_TREE_DRAFT_TOPOLOGY_ROOT_DEPTH:               return "root_depth";
        case COMMON_TREE_DRAFT_TOPOLOGY_PARENT_RANGE:             return "parent_range";
        case COMMON_TREE_DRAFT_TOPOLOGY_DEPTH_MISMATCH:           return "depth_mismatch";
        case COMMON_TREE_DRAFT_TOPOLOGY_PARENT_TREE_MISMATCH:     return "parent_tree_mismatch";
        case COMMON_TREE_DRAFT_TOPOLOGY_NEGATIVE_TREE_ID:         return "negative_tree_id";
        case COMMON_TREE_DRAFT_TOPOLOGY_MULTIPLE_ROOTS:           return "multiple_roots";
    }
    return "unknown";
}
