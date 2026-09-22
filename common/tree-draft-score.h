#pragma once

#include "tree-draft-node.h"

#include <cstddef>
#include <cstdint>

enum common_tree_draft_score_status : uint32_t {
    COMMON_TREE_DRAFT_SCORE_OK = 0,
    COMMON_TREE_DRAFT_SCORE_NULL_BUFFER,
    COMMON_TREE_DRAFT_SCORE_EPSILON,
    COMMON_TREE_DRAFT_SCORE_NODE_INVALID,
    COMMON_TREE_DRAFT_SCORE_NUMERIC,
};

common_tree_draft_score_status common_tree_draft_score_cumulative_logp(
        const common_tree_draft_node * nodes,
        size_t n_nodes,
        const common_tree_draft_config & config,
        float epsilon,
        double * cumulative_logp,
        size_t out_capacity);

