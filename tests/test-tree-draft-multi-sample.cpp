#include "tree-draft-multi-sample.h"

#include <cassert>
#include <cmath>

static void test_without_replacement_and_replay() {
    const common_tree_draft_candidate candidates[] = {
        { 10, 4.0f, 0.50f },
        { 20, 3.0f, 0.30f },
        { 30, 2.0f, 0.15f },
        { 40, 1.0f, 0.05f },
    };
    common_tree_draft_child_sample first[3] = {};
    common_tree_draft_child_sample replay[3] = {};
    size_t n_first = 0;
    size_t n_replay = 0;
    assert(common_tree_draft_sample_without_replacement(candidates, 4, 3, 99, 7, 123, 2, first, 3, &n_first) == COMMON_TREE_DRAFT_MULTI_SAMPLE_OK);
    assert(common_tree_draft_sample_without_replacement(candidates, 4, 3, 99, 7, 123, 2, replay, 3, &n_replay) == COMMON_TREE_DRAFT_MULTI_SAMPLE_OK);
    assert(n_first == 3 && n_replay == 3);

    for (size_t i = 0; i < n_first; ++i) {
        assert(first[i].token == replay[i].token);
        assert(first[i].conditional_probability == replay[i].conditional_probability);
        assert(first[i].draw_ordinal == i);
        assert(first[i].conditional_probability > 0.0f && first[i].conditional_probability <= 1.0f);
        assert(std::fabs(std::exp(first[i].log_conditional_probability) - first[i].conditional_probability) < 1e-6f);
        for (size_t j = 0; j < i; ++j) {
            assert(first[i].token != first[j].token);
        }
    }
}

static void test_zero_count_and_exhausted_mass() {
    const common_tree_draft_candidate candidates[] = {
        { 1, 2.0f, 1.0f },
        { 2, 1.0f, 0.0f },
    };
    common_tree_draft_child_sample out[2] = {};
    size_t count = 99;
    assert(common_tree_draft_sample_without_replacement(candidates, 2, 0, 1, 2, 3, 1, out, 2, &count) == COMMON_TREE_DRAFT_MULTI_SAMPLE_OK);
    assert(count == 0);
    assert(common_tree_draft_sample_without_replacement(candidates, 2, 2, 1, 2, 3, 1, out, 2, &count) == COMMON_TREE_DRAFT_MULTI_SAMPLE_OK);
    assert(count == 1);
}

static void test_invalid_inputs() {
    common_tree_draft_child_sample out[2] = {};
    size_t count = 0;
    const common_tree_draft_candidate duplicate[] = {
        { 1, 2.0f, 0.5f },
        { 1, 1.0f, 0.5f },
    };
    assert(common_tree_draft_sample_without_replacement(duplicate, 2, 1, 1, 2, 3, 1, out, 2, &count) == COMMON_TREE_DRAFT_MULTI_SAMPLE_DUPLICATE_TOKEN);
    assert(common_tree_draft_sample_without_replacement(duplicate, 2, 3, 1, 2, 3, 1, out, 2, &count) == COMMON_TREE_DRAFT_MULTI_SAMPLE_COUNT_RANGE);
}

int main() {
    test_without_replacement_and_replay();
    test_zero_count_and_exhausted_mass();
    test_invalid_inputs();
    return 0;
}

