#pragma once

#include "tree-draft-node.h"

#include <cstddef>
#include <cstdint>

enum common_tree_draft_score_mode : uint32_t {
    COMMON_TREE_DRAFT_SCORE_CUMULATIVE = 0,
    COMMON_TREE_DRAFT_SCORE_LENGTH_NORMALIZED,
    COMMON_TREE_DRAFT_SCORE_DIVERSITY,
};

enum common_tree_draft_depth_preference : uint32_t {
    COMMON_TREE_DRAFT_DEPTH_NEUTRAL = 0,
    COMMON_TREE_DRAFT_DEPTH_SHALLOW_FIRST,
    COMMON_TREE_DRAFT_DEPTH_DEEP_FIRST,
};

struct common_tree_draft_rank_entry {
    double cumulative_score;
    double length_score;
    double diversity_score;
    int32_t depth;
    uint64_t path_id;
    uint32_t node_index;
};

enum common_tree_draft_ranking_status : uint32_t {
    COMMON_TREE_DRAFT_RANKING_OK = 0,
    COMMON_TREE_DRAFT_RANKING_NULL_BUFFER,
    COMMON_TREE_DRAFT_RANKING_ALPHA,
    COMMON_TREE_DRAFT_RANKING_LAMBDA,
    COMMON_TREE_DRAFT_RANKING_NODE_RANGE,
    COMMON_TREE_DRAFT_RANKING_NUMERIC,
    COMMON_TREE_DRAFT_RANKING_MODE,
};

common_tree_draft_ranking_status common_tree_draft_length_normalize(
        const common_tree_draft_node * nodes,
        size_t n_nodes,
        const double * cumulative_scores,
        double alpha,
        double * length_scores,
        size_t out_capacity);

common_tree_draft_ranking_status common_tree_draft_apply_sibling_diversity(
        const common_tree_draft_node * nodes,
        size_t n_nodes,
        const double * base_scores,
        double lambda,
        double * diversity_scores,
        size_t out_capacity);

common_tree_draft_ranking_status common_tree_draft_build_rank_entry(
        const common_tree_draft_node * nodes,
        size_t n_nodes,
        const double * cumulative_scores,
        const double * length_scores,
        const double * diversity_scores,
        uint32_t node_index,
        common_tree_draft_rank_entry * out);

int common_tree_draft_rank_compare(
        const common_tree_draft_rank_entry & a,
        const common_tree_draft_rank_entry & b,
        common_tree_draft_score_mode mode,
        common_tree_draft_depth_preference depth_preference);

