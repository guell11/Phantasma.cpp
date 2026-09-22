#include "tree-draft-categorical.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>

static void test_stable_normalization_and_masking() {
    const float logits[] = { 1000.0f, 999.0f, -std::numeric_limits<float>::infinity(), 998.0f };
    float scaled[4] = {};
    float probs[4] = {};
    assert(common_tree_draft_categorical_normalize(logits, 4, 0.5f, scaled, probs) == COMMON_TREE_DRAFT_CATEGORICAL_OK);
    assert(scaled[0] == 2000.0f);
    assert(scaled[1] == 1998.0f);
    assert(std::isinf(scaled[2]) && scaled[2] < 0.0f);
    assert(probs[2] == 0.0f);
    assert(probs[0] > probs[1] && probs[1] > probs[3]);
    const double sum = static_cast<double>(probs[0]) + probs[1] + probs[2] + probs[3];
    assert(std::fabs(sum - 1.0) < 1e-6);
}

static void test_temperature_changes_distribution_not_order() {
    const float logits[] = { 3.0f, 2.0f, 1.0f };
    float scaled_cold[3] = {};
    float probs_cold[3] = {};
    float scaled_hot[3] = {};
    float probs_hot[3] = {};
    assert(common_tree_draft_categorical_normalize(logits, 3, 0.5f, scaled_cold, probs_cold) == COMMON_TREE_DRAFT_CATEGORICAL_OK);
    assert(common_tree_draft_categorical_normalize(logits, 3, 2.0f, scaled_hot, probs_hot) == COMMON_TREE_DRAFT_CATEGORICAL_OK);
    assert(probs_cold[0] > probs_hot[0]);
    assert(probs_cold[0] > probs_cold[1] && probs_cold[1] > probs_cold[2]);
    assert(probs_hot[0] > probs_hot[1] && probs_hot[1] > probs_hot[2]);
}

static void test_invalid_inputs() {
    float scaled[2] = {};
    float probs[2] = {};
    const float valid[] = { 1.0f, 2.0f };
    assert(common_tree_draft_categorical_normalize(valid, 2, 0.0f, scaled, probs) == COMMON_TREE_DRAFT_CATEGORICAL_TEMPERATURE);
    assert(common_tree_draft_categorical_normalize(valid, 2, std::numeric_limits<float>::infinity(), scaled, probs) == COMMON_TREE_DRAFT_CATEGORICAL_TEMPERATURE);

    const float nan_logits[] = { 1.0f, std::numeric_limits<float>::quiet_NaN() };
    assert(common_tree_draft_categorical_normalize(nan_logits, 2, 1.0f, scaled, probs) == COMMON_TREE_DRAFT_CATEGORICAL_INVALID_LOGIT);
    const float pos_inf[] = { 1.0f, std::numeric_limits<float>::infinity() };
    assert(common_tree_draft_categorical_normalize(pos_inf, 2, 1.0f, scaled, probs) == COMMON_TREE_DRAFT_CATEGORICAL_INVALID_LOGIT);
    const float masked[] = { -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity() };
    assert(common_tree_draft_categorical_normalize(masked, 2, 1.0f, scaled, probs) == COMMON_TREE_DRAFT_CATEGORICAL_NO_FINITE);
    assert(common_tree_draft_categorical_normalize(nullptr, 2, 1.0f, scaled, probs) == COMMON_TREE_DRAFT_CATEGORICAL_NULL_BUFFER);
}

int main() {
    test_stable_normalization_and_masking();
    test_temperature_changes_distribution_not_order();
    test_invalid_inputs();
    return 0;
}

