#include "tree-draft-sample.h"

#include <cassert>
#include <cmath>

static void test_deterministic_draw_and_provenance() {
    const common_tree_draft_candidate candidates[] = {
        { 10, 3.0f, 0.6f },
        { 20, 2.0f, 0.3f },
        { 30, 1.0f, 0.1f },
    };
    common_tree_draft_sample_result a = {};
    common_tree_draft_sample_result b = {};
    assert(common_tree_draft_sample_one(candidates, 3, 99, 5, 1234, 2, 7, &a) == COMMON_TREE_DRAFT_SAMPLE_OK);
    assert(common_tree_draft_sample_one(candidates, 3, 99, 5, 1234, 2, 7, &b) == COMMON_TREE_DRAFT_SAMPLE_OK);
    assert(a.token == b.token);
    assert(a.probability == b.probability);
    assert(a.log_probability == b.log_probability);
    assert(a.rng_address.seed == 99);
    assert(a.rng_address.request == 5);
    assert(a.rng_address.branch == 1234);
    assert(a.rng_address.lane == 7);
    assert(std::fabs(std::exp(a.log_probability) - a.probability) < 1e-6f);
}

static void test_singleton_and_partial_mass() {
    const common_tree_draft_candidate singleton[] = { { 77, 1.0f, 0.25f } };
    common_tree_draft_sample_result result = {};
    assert(common_tree_draft_sample_one(singleton, 1, 1, 2, 3, 1, 0, &result) == COMMON_TREE_DRAFT_SAMPLE_OK);
    assert(result.token == 77);
    assert(result.probability == 1.0f);
    assert(result.log_probability == 0.0f);
}

static void test_invalid_input() {
    common_tree_draft_sample_result result = {};
    assert(common_tree_draft_sample_one(nullptr, 0, 1, 2, 3, 1, 0, &result) == COMMON_TREE_DRAFT_SAMPLE_EMPTY);
    const common_tree_draft_candidate zero[] = { { 1, 1.0f, 0.0f } };
    assert(common_tree_draft_sample_one(zero, 1, 1, 2, 3, 1, 0, &result) == COMMON_TREE_DRAFT_SAMPLE_INVALID_PROBABILITY);
    const common_tree_draft_candidate valid[] = { { 1, 1.0f, 1.0f } };
    assert(common_tree_draft_sample_one(valid, 1, 1, 2, 0, 1, 0, &result) == COMMON_TREE_DRAFT_SAMPLE_RNG);
}

int main() {
    test_deterministic_draw_and_provenance();
    test_singleton_and_partial_mass();
    test_invalid_input();
    return 0;
}

