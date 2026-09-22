#include "tree-draft-rng.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <future>
#include <map>
#include <tuple>
#include <vector>

using address_key = std::tuple<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t>;

static address_key key(const common_tree_draft_rng_address & address) {
    return { address.seed, address.request, address.step, address.branch, address.lane };
}

static std::map<address_key, uint64_t> evaluate(const std::vector<common_tree_draft_rng_address> & addresses) {
    std::map<address_key, uint64_t> result;
    for (const auto & address : addresses) {
        result.emplace(key(address), common_tree_draft_rng_u64(address));
    }
    return result;
}

static void test_deterministic_replay() {
    const common_tree_draft_rng_address address = {
        UINT64_C(0x0123456789abcdef),
        UINT64_C(0x1020304050607080),
        77,
        9,
        3,
    };

    const uint64_t expected = UINT64_C(0x63c2cf92409777ac);
    const float expected_uniform = 6537935.0f / 16777216.0f;

    for (int i = 0; i < 1000; ++i) {
        assert(common_tree_draft_rng_u64(address) == expected);
        assert(common_tree_draft_rng_uniform_f32(address) == expected_uniform);
    }

    assert(expected_uniform >= 0.0f);
    assert(expected_uniform < 1.0f);
}

static void test_execution_order_independence() {
    std::vector<common_tree_draft_rng_address> addresses;
    for (uint64_t request = 0; request < 5; ++request) {
        for (uint64_t step = 0; step < 7; ++step) {
            for (uint64_t branch = 0; branch < 4; ++branch) {
                for (uint64_t lane = 0; lane < 3; ++lane) {
                    addresses.push_back({ 123456789, request, step, branch, lane });
                }
            }
        }
    }

    const auto reference = evaluate(addresses);

    std::reverse(addresses.begin(), addresses.end());
    assert(evaluate(addresses) == reference);

    std::rotate(addresses.begin(), addresses.begin() + 37, addresses.end());
    assert(evaluate(addresses) == reference);
}

static void test_parallel_reordering_independence() {
    std::array<common_tree_draft_rng_address, 64> addresses{};
    for (uint64_t i = 0; i < addresses.size(); ++i) {
        addresses[i] = { 99887766, i % 8, i / 8, (i * 5) % 13, i % 4 };
    }

    std::array<uint64_t, 64> serial{};
    std::array<std::future<uint64_t>, 64> parallel{};

    for (size_t i = 0; i < addresses.size(); ++i) {
        serial[i] = common_tree_draft_rng_u64(addresses[i]);
    }

    for (size_t i = addresses.size(); i-- > 0;) {
        const auto address = addresses[i];
        parallel[i] = std::async(std::launch::async, [address]() {
            return common_tree_draft_rng_u64(address);
        });
    }

    for (size_t i = 0; i < addresses.size(); ++i) {
        assert(parallel[i].get() == serial[i]);
    }
}

static void test_address_fields_are_independent() {
    const common_tree_draft_rng_address base = { 1, 2, 3, 4, 5 };
    const uint64_t value = common_tree_draft_rng_u64(base);

    const std::array<common_tree_draft_rng_address, 5> variants = {{
        { 2, 2, 3, 4, 5 },
        { 1, 3, 3, 4, 5 },
        { 1, 2, 4, 4, 5 },
        { 1, 2, 3, 5, 5 },
        { 1, 2, 3, 4, 6 },
    }};

    for (const auto & variant : variants) {
        assert(common_tree_draft_rng_u64(variant) != value);
    }
}

int main() {
    static_assert(sizeof(common_tree_draft_rng_address) == COMMON_TREE_DRAFT_RNG_ADDRESS_WORDS * sizeof(uint64_t),
            "RNG address tensor ABI must stay tightly packed");

    test_deterministic_replay();
    test_execution_order_independence();
    test_parallel_reordering_independence();
    test_address_fields_are_independent();
    return 0;
}
