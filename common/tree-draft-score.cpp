#include "tree-draft-score.h"

#include <algorithm>
#include <cmath>

common_tree_draft_score_status common_tree_draft_score_cumulative_logp(
        const common_tree_draft_node * nodes,
        size_t n_nodes,
        const common_tree_draft_config & config,
        float epsilon,
        double * cumulative_logp,
        size_t out_capacity) {
    if ((n_nodes > 0 && nodes == nullptr) || (n_nodes > 0 && cumulative_logp == nullptr) || out_capacity < n_nodes) {
        return COMMON_TREE_DRAFT_SCORE_NULL_BUFFER;
    }
    if (!(epsilon > 0.0f) || epsilon > 1.0f || !std::isfinite(epsilon)) {
        return COMMON_TREE_DRAFT_SCORE_EPSILON;
    }
    if (common_tree_draft_nodes_validate(nodes, n_nodes, config) != COMMON_TREE_DRAFT_NODE_OK) {
        return COMMON_TREE_DRAFT_SCORE_NODE_INVALID;
    }
    if (n_nodes == 0) {
        return COMMON_TREE_DRAFT_SCORE_OK;
    }

    cumulative_logp[0] = 0.0;
    const double log_floor = std::log(static_cast<double>(epsilon));
    for (size_t i = 1; i < n_nodes; ++i) {
        const double local = std::max(static_cast<double>(nodes[i].logp), log_floor);
        cumulative_logp[i] = cumulative_logp[nodes[i].parent] + local;
        if (!std::isfinite(cumulative_logp[i])) {
            return COMMON_TREE_DRAFT_SCORE_NUMERIC;
        }
    }
    return COMMON_TREE_DRAFT_SCORE_OK;
}

