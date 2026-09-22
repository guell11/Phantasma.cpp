#include "tree-draft-top-p.h"

#include <cassert>
#include <cmath>

static void test_threshold_and_renormalization() {
    const common_tree_draft_candidate in[] = {
        { 1, 5.0f, 0.50f },
        { 2, 4.0f, 0.30f },
        { 3, 3.0f, 0.15f },
        { 4, 2.0f, 0.05f },
    };
    common_tree_draft_candidate out[4] = {};
    size_t count = 0;
    assert(common_tree_draft_top_p_filter(in, 4, 0.75f, out, 4, &count) == COMMON_TREE_DRAFT_TOP_P_OK);
    assert(count == 2);
    assert(out[0].token == 1 && out[1].token == 2);
    assert(std::fabs((out[0].probability + out[1].probability) - 1.0f) < 1e-6f);
    assert(std::fabs(out[0].probability - 0.625f) < 1e-6f);
}

static void test_partial_input_mass_is_normalized_before_top_p() {
    const common_tree_draft_candidate in[] = {
        { 1, 5.0f, 0.40f },
        { 2, 4.0f, 0.20f },
    };
    common_tree_draft_candidate out[2] = {};
    size_t count = 0;
    assert(common_tree_draft_top_p_filter(in, 2, 0.60f, out, 2, &count) == COMMON_TREE_DRAFT_TOP_P_OK);
    assert(count == 1);
    assert(out[0].token == 1);
    assert(out[0].probability == 1.0f);
}

static void test_minimum_one_and_invalid_inputs() {
    const common_tree_draft_candidate singleton[] = { { 7, 1.0f, 1.0f } };
    common_tree_draft_candidate out[2] = {};
    size_t count = 0;
    assert(common_tree_draft_top_p_filter(singleton, 1, 0.01f, out, 2, &count) == COMMON_TREE_DRAFT_TOP_P_OK);
    assert(count == 1 && out[0].token == 7);
    assert(common_tree_draft_top_p_filter(singleton, 1, 0.0f, out, 2, &count) == COMMON_TREE_DRAFT_TOP_P_THRESHOLD);

    const common_tree_draft_candidate zero[] = { { 1, 1.0f, 0.0f } };
    assert(common_tree_draft_top_p_filter(zero, 1, 1.0f, out, 2, &count) == COMMON_TREE_DRAFT_TOP_P_ZERO_MASS);

    const common_tree_draft_candidate unsorted[] = {
        { 2, 1.0f, 0.5f },
        { 1, 2.0f, 0.5f },
    };
    assert(common_tree_draft_top_p_filter(unsorted, 2, 1.0f, out, 2, &count) == COMMON_TREE_DRAFT_TOP_P_ORDER);
}

int main() {
    test_threshold_and_renormalization();
    test_partial_input_mass_is_normalized_before_top_p();
    test_minimum_one_and_invalid_inputs();
    return 0;
}

