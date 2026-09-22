#pragma once

#include <cstdint>

// Packed Tree-Draft topology ABI. All arrays are contiguous 1-D int32 buffers
// with shape [n_nodes] and use the same topological node order in C++/CUDA/Triton.
struct common_tree_draft_topology {
    const int32_t * parent;
    const int32_t * depth;
    const int32_t * tree_id;
    int32_t n_nodes;
};

enum common_tree_draft_topology_error : int32_t {
    COMMON_TREE_DRAFT_TOPOLOGY_OK = 0,
    COMMON_TREE_DRAFT_TOPOLOGY_NULL_PARENT,
    COMMON_TREE_DRAFT_TOPOLOGY_NULL_DEPTH,
    COMMON_TREE_DRAFT_TOPOLOGY_NULL_TREE_ID,
    COMMON_TREE_DRAFT_TOPOLOGY_NEGATIVE_NODE_COUNT,
    COMMON_TREE_DRAFT_TOPOLOGY_ROOT_DEPTH,
    COMMON_TREE_DRAFT_TOPOLOGY_PARENT_RANGE,
    COMMON_TREE_DRAFT_TOPOLOGY_DEPTH_MISMATCH,
    COMMON_TREE_DRAFT_TOPOLOGY_PARENT_TREE_MISMATCH,
    COMMON_TREE_DRAFT_TOPOLOGY_NEGATIVE_TREE_ID,
    COMMON_TREE_DRAFT_TOPOLOGY_MULTIPLE_ROOTS,
};

// ABI rules:
// - n_nodes is in [0, INT32_MAX]; empty forests may use null array pointers.
// - roots have parent=-1 and depth=0.
// - non-roots have 0 <= parent[i] < i, the same tree_id as their parent, and
//   depth[i] = depth[parent[i]] + 1.
// - tree_id is non-negative and identifies one rooted tree. Tree ids need not
//   be dense or sorted; each distinct tree id has exactly one root.
// These rules make parent/depth/tree_id directly loadable as tl.int32 tensors.
common_tree_draft_topology_error common_tree_draft_topology_validate(const common_tree_draft_topology & topology);

const char * common_tree_draft_topology_error_name(common_tree_draft_topology_error error);
