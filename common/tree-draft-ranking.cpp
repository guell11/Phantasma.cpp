#include "tree-draft-ranking.h"

#include <algorithm>
#include <cmath>
#include <vector>

common_tree_draft_ranking_status common_tree_draft_length_normalize(
        const common_tree_draft_node * nodes,
        size_t n_nodes,
        const double * cumulative_scores,
        double alpha,
        double * length_scores,
        size_t out_capacity) {
    if ((n_nodes > 0 && (nodes == nullptr || cumulative_scores == nullptr || length_scores == nullptr)) || out_capacity < n_nodes) {
        return COMMON_TREE_DRAFT_RANKING_NULL_BUFFER;
    }
    if (!(alpha >= 0.0) || !std::isfinite(alpha)) {
        return COMMON_TREE_DRAFT_RANKING_ALPHA;
    }
    for (size_t i = 0; i < n_nodes; ++i) {
        if (!std::isfinite(cumulative_scores[i])) {
            return COMMON_TREE_DRAFT_RANKING_NUMERIC;
        }
        if (nodes[i].depth < 0) {
            return COMMON_TREE_DRAFT_RANKING_NODE_RANGE;
        }
        if (nodes[i].depth == 0 || alpha == 0.0) {
            length_scores[i] = cumulative_scores[i];
        } else {
            const double denom = std::pow(static_cast<double>(nodes[i].depth), alpha);
            if (!(denom > 0.0) || !std::isfinite(denom)) {
                return COMMON_TREE_DRAFT_RANKING_NUMERIC;
            }
            length_scores[i] = cumulative_scores[i] / denom;
        }
    }
    return COMMON_TREE_DRAFT_RANKING_OK;
}

common_tree_draft_ranking_status common_tree_draft_apply_sibling_diversity(
        const common_tree_draft_node * nodes,
        size_t n_nodes,
        const double * base_scores,
        double lambda,
        double * diversity_scores,
        size_t out_capacity) {
    if ((n_nodes > 0 && (nodes == nullptr || base_scores == nullptr || diversity_scores == nullptr)) || out_capacity < n_nodes) {
        return COMMON_TREE_DRAFT_RANKING_NULL_BUFFER;
    }
    if (!(lambda >= 0.0) || !std::isfinite(lambda)) {
        return COMMON_TREE_DRAFT_RANKING_LAMBDA;
    }
    for (size_t i = 0; i < n_nodes; ++i) {
        if (!std::isfinite(base_scores[i])) {
            return COMMON_TREE_DRAFT_RANKING_NUMERIC;
        }
        diversity_scores[i] = base_scores[i];
    }

    for (size_t i = 0; i < n_nodes; ++i) {
        if (nodes[i].parent < 0) {
            continue;
        }
        std::vector<size_t> siblings;
        for (size_t j = 0; j < n_nodes; ++j) {
            if (nodes[j].parent == nodes[i].parent) {
                siblings.push_back(j);
            }
        }
        std::sort(siblings.begin(), siblings.end(), [nodes, base_scores](size_t a, size_t b) {
            if (base_scores[a] != base_scores[b]) {
                return base_scores[a] > base_scores[b];
            }
            if (nodes[a].token != nodes[b].token) {
                return nodes[a].token < nodes[b].token;
            }
            return a < b;
        });
        size_t rank = 0;
        for (; rank < siblings.size(); ++rank) {
            if (siblings[rank] == i) {
                break;
            }
        }
        diversity_scores[i] = base_scores[i] - lambda * std::log1p(static_cast<double>(rank));
    }
    return COMMON_TREE_DRAFT_RANKING_OK;
}

common_tree_draft_ranking_status common_tree_draft_build_rank_entry(
        const common_tree_draft_node * nodes,
        size_t n_nodes,
        const double * cumulative_scores,
        const double * length_scores,
        const double * diversity_scores,
        uint32_t node_index,
        common_tree_draft_rank_entry * out) {
    if (out == nullptr || nodes == nullptr || cumulative_scores == nullptr || length_scores == nullptr || diversity_scores == nullptr) {
        return COMMON_TREE_DRAFT_RANKING_NULL_BUFFER;
    }
    if (node_index >= n_nodes) {
        return COMMON_TREE_DRAFT_RANKING_NODE_RANGE;
    }
    if (!std::isfinite(cumulative_scores[node_index]) || !std::isfinite(length_scores[node_index]) || !std::isfinite(diversity_scores[node_index])) {
        return COMMON_TREE_DRAFT_RANKING_NUMERIC;
    }
    *out = {
        cumulative_scores[node_index],
        length_scores[node_index],
        diversity_scores[node_index],
        nodes[node_index].depth,
        nodes[node_index].path_id,
        node_index,
    };
    return COMMON_TREE_DRAFT_RANKING_OK;
}

static bool common_tree_draft_rank_primary(const common_tree_draft_rank_entry & a, const common_tree_draft_rank_entry & b,
        common_tree_draft_score_mode mode, double * av, double * bv) {
    switch (mode) {
        case COMMON_TREE_DRAFT_SCORE_CUMULATIVE:        *av = a.cumulative_score; *bv = b.cumulative_score; return true;
        case COMMON_TREE_DRAFT_SCORE_LENGTH_NORMALIZED: *av = a.length_score;     *bv = b.length_score;     return true;
        case COMMON_TREE_DRAFT_SCORE_DIVERSITY:         *av = a.diversity_score;  *bv = b.diversity_score;  return true;
    }
    return false;
}

int common_tree_draft_rank_compare(
        const common_tree_draft_rank_entry & a,
        const common_tree_draft_rank_entry & b,
        common_tree_draft_score_mode mode,
        common_tree_draft_depth_preference depth_preference) {
    double av = 0.0;
    double bv = 0.0;
    if (!common_tree_draft_rank_primary(a, b, mode, &av, &bv)) {
        return 0;
    }
    if (av != bv) {
        return av > bv ? -1 : 1;
    }
    if (depth_preference == COMMON_TREE_DRAFT_DEPTH_SHALLOW_FIRST && a.depth != b.depth) {
        return a.depth < b.depth ? -1 : 1;
    }
    if (depth_preference == COMMON_TREE_DRAFT_DEPTH_DEEP_FIRST && a.depth != b.depth) {
        return a.depth > b.depth ? -1 : 1;
    }
    if (a.path_id != b.path_id) {
        return a.path_id < b.path_id ? -1 : 1;
    }
    if (a.node_index != b.node_index) {
        return a.node_index < b.node_index ? -1 : 1;
    }
    return 0;
}

