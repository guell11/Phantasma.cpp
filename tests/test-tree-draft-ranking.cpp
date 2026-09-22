#include "tree-draft-ranking.h"

#include <cassert>
#include <cmath>

int main() {
    const common_tree_draft_node nodes[] = {
        { -1, -1, 0, 0.0f, 10, 0 },
        { 0, 10, 1, -0.2f, 20, 0 },
        { 0, 11, 1, -0.3f, 30, 0 },
        { 1, 12, 2, -0.1f, 40, 0 },
    };
    const double cumulative[] = { 0.0, -0.2, -0.3, -0.3 };
    double length[4] = {};
    assert(common_tree_draft_length_normalize(nodes, 4, cumulative, 1.0, length, 4) == COMMON_TREE_DRAFT_RANKING_OK);
    assert(length[0] == 0.0);
    assert(std::fabs(length[3] - (-0.15)) < 1e-12);
    double length_raw[4] = {};
    assert(common_tree_draft_length_normalize(nodes, 4, cumulative, 0.0, length_raw, 4) == COMMON_TREE_DRAFT_RANKING_OK);
    for (int i = 0; i < 4; ++i) assert(length_raw[i] == cumulative[i]);

    double diversity[4] = {};
    assert(common_tree_draft_apply_sibling_diversity(nodes, 4, length, 0.5, diversity, 4) == COMMON_TREE_DRAFT_RANKING_OK);
    assert(diversity[1] == length[1]);
    assert(std::fabs(diversity[2] - (length[2] - 0.5 * std::log(2.0))) < 1e-12);

    common_tree_draft_rank_entry a = {};
    common_tree_draft_rank_entry b = {};
    assert(common_tree_draft_build_rank_entry(nodes, 4, cumulative, length, diversity, 1, &a) == COMMON_TREE_DRAFT_RANKING_OK);
    assert(common_tree_draft_build_rank_entry(nodes, 4, cumulative, length, diversity, 3, &b) == COMMON_TREE_DRAFT_RANKING_OK);
    assert(common_tree_draft_rank_compare(b, a, COMMON_TREE_DRAFT_SCORE_LENGTH_NORMALIZED, COMMON_TREE_DRAFT_DEPTH_NEUTRAL) < 0);

    common_tree_draft_rank_entry tied_a = { -1.0, -1.0, -1.0, 1, 50, 5 };
    common_tree_draft_rank_entry tied_b = { -1.0, -1.0, -1.0, 2, 40, 6 };
    assert(common_tree_draft_rank_compare(tied_a, tied_b, COMMON_TREE_DRAFT_SCORE_CUMULATIVE, COMMON_TREE_DRAFT_DEPTH_SHALLOW_FIRST) < 0);
    assert(common_tree_draft_rank_compare(tied_a, tied_b, COMMON_TREE_DRAFT_SCORE_CUMULATIVE, COMMON_TREE_DRAFT_DEPTH_DEEP_FIRST) > 0);
    assert(common_tree_draft_rank_compare(tied_a, tied_b, COMMON_TREE_DRAFT_SCORE_CUMULATIVE, COMMON_TREE_DRAFT_DEPTH_NEUTRAL) > 0);

    assert(common_tree_draft_length_normalize(nodes, 4, cumulative, -1.0, length, 4) == COMMON_TREE_DRAFT_RANKING_ALPHA);
    assert(common_tree_draft_apply_sibling_diversity(nodes, 4, length, -1.0, diversity, 4) == COMMON_TREE_DRAFT_RANKING_LAMBDA);
    return 0;
}

