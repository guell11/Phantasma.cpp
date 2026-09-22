#include "tree-draft-candidates.h"
#include "tree-draft-categorical.h"

#include <cassert>
#include <cmath>
#include <limits>

static void test_stable_top_k_after_normalization() {
    const float logits[] = { 1.0f, 3.0f, 3.0f, -std::numeric_limits<float>::infinity(), 2.0f };
    float scores[5] = {};
    float probs[5] = {};
    assert(common_tree_draft_categorical_normalize(logits, 5, 1.0f, scores, probs) == COMMON_TREE_DRAFT_CATEGORICAL_OK);

    common_tree_draft_candidate out[4] = {};
    size_t count = 0;
    assert(common_tree_draft_candidate_top_k(scores, probs, 5, 4, out, 4, &count) == COMMON_TREE_DRAFT_CANDIDATE_OK);
    assert(count == 4);
    assert(out[0].token == 1);
    assert(out[1].token == 2);
    assert(out[2].token == 4);
    assert(out[3].token == 0);
    for (size_t i = 0; i < count; ++i) {
        assert(std::isfinite(out[i].score));
        assert(out[i].probability > 0.0f);
    }
}

static void test_k_boundaries_and_mask_exclusion() {
    const float scores[] = { 4.0f, -std::numeric_limits<float>::infinity(), 2.0f };
    const float probs[] = { 0.8f, 0.0f, 0.2f };
    common_tree_draft_candidate out[3] = {};
    size_t count = 99;
    assert(common_tree_draft_candidate_top_k(scores, probs, 3, 0, out, 3, &count) == COMMON_TREE_DRAFT_CANDIDATE_OK);
    assert(count == 0);
    assert(common_tree_draft_candidate_top_k(scores, probs, 3, 9, out, 3, &count) == COMMON_TREE_DRAFT_CANDIDATE_OK);
    assert(count == 2);
    assert(out[0].token == 0 && out[1].token == 2);
}

static void test_invalid_rows() {
    common_tree_draft_candidate out[2] = {};
    size_t count = 0;
    const float bad_score[] = { 1.0f, std::numeric_limits<float>::quiet_NaN() };
    const float probs[] = { 0.5f, 0.5f };
    assert(common_tree_draft_candidate_top_k(bad_score, probs, 2, 2, out, 2, &count) == COMMON_TREE_DRAFT_CANDIDATE_INVALID_SCORE);

    const float scores[] = { 1.0f, 0.0f };
    const float bad_prob[] = { 1.1f, -0.1f };
    assert(common_tree_draft_candidate_top_k(scores, bad_prob, 2, 2, out, 2, &count) == COMMON_TREE_DRAFT_CANDIDATE_INVALID_PROBABILITY);
    assert(common_tree_draft_candidate_top_k(scores, probs, 2, 2, out, 1, &count) == COMMON_TREE_DRAFT_CANDIDATE_BUFFER_TOO_SMALL);
}

int main() {
    test_stable_top_k_after_normalization();
    test_k_boundaries_and_mask_exclusion();
    test_invalid_rows();
    return 0;
}

