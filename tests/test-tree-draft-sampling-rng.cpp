#include "tree-draft-sampling-rng.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>

static void test_address_mapping_and_domain() {
    common_tree_draft_rng_address address = {};
    assert(common_tree_draft_sampling_rng_address(7, 11, 13, 3, 5, &address) == COMMON_TREE_DRAFT_SAMPLING_RNG_OK);
    assert(address.seed == 7);
    assert(address.request == 11);
    assert(address.step == ((static_cast<uint64_t>(COMMON_TREE_DRAFT_RNG_DOMAIN_DRAFT) << 32) | 3));
    assert(address.branch == 13);
    assert(address.lane == 5);
}

static void test_reordering_does_not_change_draws() {
    struct draw {
        uint64_t path;
        uint32_t depth;
        uint32_t ordinal;
        float value;
    };
    std::array<draw, 4> draws = {{
        { 101, 1, 0, 0.0f },
        { 102, 2, 0, 0.0f },
        { 101, 1, 1, 0.0f },
        { 103, 3, 7, 0.0f },
    }};

    for (auto & draw : draws) {
        assert(common_tree_draft_sampling_rng_uniform(99, 5, draw.path, draw.depth, draw.ordinal, &draw.value) == COMMON_TREE_DRAFT_SAMPLING_RNG_OK);
        assert(draw.value >= 0.0f && draw.value < 1.0f);
    }
    const auto reference = draws;

    std::reverse(draws.begin(), draws.end());
    for (const auto & draw : draws) {
        float replay = 0.0f;
        assert(common_tree_draft_sampling_rng_uniform(99, 5, draw.path, draw.depth, draw.ordinal, &replay) == COMMON_TREE_DRAFT_SAMPLING_RNG_OK);
        bool matched = false;
        for (const auto & expected : reference) {
            if (expected.path == draw.path && expected.depth == draw.depth && expected.ordinal == draw.ordinal) {
                assert(replay == expected.value);
                matched = true;
            }
        }
        assert(matched);
    }
}

static void test_counter_fields_are_independent() {
    float base = 0.0f;
    assert(common_tree_draft_sampling_rng_uniform(1, 2, 3, 4, 5, &base) == COMMON_TREE_DRAFT_SAMPLING_RNG_OK);
    float changed = 0.0f;
    assert(common_tree_draft_sampling_rng_uniform(2, 2, 3, 4, 5, &changed) == COMMON_TREE_DRAFT_SAMPLING_RNG_OK && changed != base);
    assert(common_tree_draft_sampling_rng_uniform(1, 3, 3, 4, 5, &changed) == COMMON_TREE_DRAFT_SAMPLING_RNG_OK && changed != base);
    assert(common_tree_draft_sampling_rng_uniform(1, 2, 4, 4, 5, &changed) == COMMON_TREE_DRAFT_SAMPLING_RNG_OK && changed != base);
    assert(common_tree_draft_sampling_rng_uniform(1, 2, 3, 5, 5, &changed) == COMMON_TREE_DRAFT_SAMPLING_RNG_OK && changed != base);
    assert(common_tree_draft_sampling_rng_uniform(1, 2, 3, 4, 6, &changed) == COMMON_TREE_DRAFT_SAMPLING_RNG_OK && changed != base);
}

int main() {
    test_address_mapping_and_domain();
    test_reordering_does_not_change_draws();
    test_counter_fields_are_independent();

    common_tree_draft_rng_address address = {};
    float value = 0.0f;
    assert(common_tree_draft_sampling_rng_address(1, 2, 0, 1, 0, &address) == COMMON_TREE_DRAFT_SAMPLING_RNG_INVALID_PATH);
    assert(common_tree_draft_sampling_rng_uniform(1, 2, 3, 1, 0, nullptr) == COMMON_TREE_DRAFT_SAMPLING_RNG_NULL_OUTPUT);
    return 0;
}

